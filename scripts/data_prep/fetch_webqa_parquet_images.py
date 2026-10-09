"""Read selected WebQA images from fixed-revision HF Parquet column ranges."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import struct
import time
import urllib.error
import urllib.request

from PIL import Image, ImageChops, ImageStat
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
REPOSITORY = 'MrZilinXiao/mbeir_test_webqa_task2'
REVISION = 'bffbce531a1503ac57284131528dd1c7f7bfd268'
TARGETS = ['30218009', '30009006', '30179094', '30098171', '30029642']


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def requested_image_ids(path):
    return list(dict.fromkeys(str(json.loads(line)['image_id'])
                             for line in path.read_text().splitlines() if line.strip()))


def predicted_ordinals(source):
    seen = {}
    for row in source.values():
        for key in ('img_negFacts', 'img_posFacts'):
            for fact in row[key]:
                image_id = str(fact['image_id'])
                if image_id not in seen:
                    seen[image_id] = len(seen)
    return seen


class RangeCache:
    def __init__(self, url, size, directory, readonly_directory=None):
        self.url, self.size, self.directory = url, size, directory
        self.readonly_directory = readonly_directory
        directory.mkdir(parents=True, exist_ok=True)

    def get(self, start, end):
        path = self.directory / f'{start}-{end}.range'
        for candidate in [path] + ([self.readonly_directory/path.name] if self.readonly_directory else []):
            if candidate.exists():
                data = candidate.read_bytes()
                metadata = json.loads(candidate.with_suffix('.json').read_text())
                if (len(data) != end-start or metadata['url'] != self.url
                        or metadata['sha256'] != hashlib.sha256(data).hexdigest()):
                    raise ValueError(f'Cached range identity mismatch: {candidate}')
                return data
        request = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{end-1}'})
        expected = f'bytes {start}-{end-1}/{self.size}'
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != expected:
                        raise ValueError(f'Server did not honor requested range: {response.status}, {response.headers.get("Content-Range")}')
                    data = response.read()
                break
            except (OSError, urllib.error.URLError) as error:
                with (self.directory/'errors.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
                        start=start, end=end, attempt=attempt+1,
                        error_type=type(error).__name__, error=str(error)))+'\n')
                if attempt == 2:
                    raise
                time.sleep(attempt+1)
        if len(data) != end-start:
            raise ValueError(f'Incomplete range {start}:{end}: received {len(data)} bytes')
        path.write_bytes(data)
        save_json(path.with_suffix('.json'), dict(url=self.url, http_status=206,
            content_range=expected, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        return data


class SegmentReader(io.RawIOBase):
    """Expose only fetched header/footer/column segments to the Parquet decoder."""
    def __init__(self, size, segments):
        self.size, self.segments, self.position = size, segments, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset
        return self.position

    def read(self, length=-1):
        length = self.size-self.position if length < 0 else length
        for start, data in self.segments:
            if start <= self.position and self.position+length <= start+len(data):
                result = data[self.position-start:self.position-start+length]
                self.position += length
                return result
        raise ValueError(f'Parquet requested an unfetched range: {self.position}:{self.position+length}')


class Shard:
    def __init__(self, entry, output, reuse_cache=None):
        self.path, self.size = entry['path'], entry['size']
        url = f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{self.path}'
        readonly = reuse_cache/'ranges'/Path(self.path).stem if reuse_cache else None
        self.cache = RangeCache(url, self.size, output / 'ranges' / Path(self.path).stem, readonly)
        tail = self.cache.get(self.size-8, self.size)
        if tail[-4:] != b'PAR1':
            raise ValueError('Invalid Parquet footer magic')
        length = struct.unpack('<I', tail[:4])[0] + 8
        self.footer = self.cache.get(self.size-length, self.size)
        self.metadata = pq.read_metadata(pa.BufferReader(b'PAR1'+self.footer))

    def read(self, group_index, names):
        group = self.metadata.row_group(group_index)
        segments = [(0, b'PAR1'), (self.size-len(self.footer), self.footer)]
        for index in range(group.num_columns):
            column = group.column(index)
            if not any(column.path_in_schema == name or column.path_in_schema.startswith(name+'.') for name in names):
                continue
            start = min(value for value in (column.dictionary_page_offset, column.data_page_offset)
                        if value is not None and value >= 0)
            segments.append((start, self.cache.get(start, start+column.total_compressed_size)))
        reader = SegmentReader(self.size, segments)
        return pq.ParquetFile(reader, metadata=self.metadata, pre_buffer=False).read_row_group(
            group_index, columns=names, use_threads=False).to_pylist()


def source_id(row):
    return str(json.loads(row['src_content'])['image_id'])


def checked_image(row, expected_id):
    actual = source_id(row)
    path_id = Path(row['img_path']).stem
    embedded_id = Path(row['img']['path']).stem
    if (actual, path_id, embedded_id) != (expected_id, expected_id, expected_id):
        raise ValueError(f'Image identity mismatch: requested {expected_id}, received {actual}/{path_id}/{embedded_id}')
    data = row['img']['bytes']
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        return data, dict(width=image.width, height=image.height, mode=image.mode, format=image.format)


def compare_wikimedia(data, local_path):
    original = Path(local_path).read_bytes()
    with Image.open(io.BytesIO(data)) as mirror, Image.open(io.BytesIO(original)) as wiki:
        left, right = mirror.convert('RGB'), wiki.convert('RGB')
        same_dimensions = left.size == right.size
        pixels_equal = same_dimensions and ImageChops.difference(left, right).getbbox() is None
        resized = right.resize(left.size, Image.Resampling.LANCZOS)
        delta = ImageChops.difference(left, resized)
        return dict(wikimedia_path=str(Path(local_path).resolve()), wikimedia_bytes=len(original),
            wikimedia_dimensions=list(wiki.size), wikimedia_format=wiki.format,
            same_file_bytes=data == original, same_dimensions=same_dimensions,
            decoded_rgb_pixels_equal=pixels_equal,
            mean_absolute_rgb_difference_after_wikimedia_lanczos_resize=sum(ImageStat.Stat(delta).mean)/3,
            comparison_note='Resize comparison is a same-scene diagnostic, not proof of an identical historical file.')


def fetch(image_ids, output, source_path, reuse_cache=None, requests_path=None):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if reuse_cache:
        previous = json.loads((reuse_cache/'manifest.json').read_text())
        if (previous['repository'], previous['revision']) != (REPOSITORY, REVISION):
            raise ValueError('Reusable cache is not from the pinned repository/revision')
    catalog_path = output / 'catalog.json'
    if catalog_path.exists():
        catalog = json.loads(catalog_path.read_text())
    elif reuse_cache and (reuse_cache/'catalog.json').exists():
        catalog = json.loads((reuse_cache/'catalog.json').read_text())
        save_json(catalog_path, catalog)
    else:
        url = f'https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}/corpus?recursive=true&limit=1000'
        with urllib.request.urlopen(url, timeout=30) as response:
            catalog = json.load(response)
        save_json(catalog_path, catalog)
    entries = sorted((row for row in catalog if row['path'].endswith('.parquet')), key=lambda row: row['path'])
    guesses = predicted_ordinals(json.loads(source_path.read_text()))
    index_path = output / 'locations.json'
    locations = json.loads((reuse_cache/'locations.json').read_text()) if reuse_cache else {}
    if index_path.exists():
        locations.update(json.loads(index_path.read_text()))
    shards, starts, scanned = {}, {}, set()

    def shard(index):
        if index not in shards:
            shards[index] = Shard(entries[index], output, reuse_cache)
            print(json.dumps(dict(event='footer_ready', shard=index,
                rows=shards[index].metadata.num_rows)), flush=True)
        return shards[index]

    def read_ids(pair):
        index, group = pair
        rows = shard(index).read(group, ['src_content'])
        return pair, {source_id(row): dict(shard=index, path=entries[index]['path'], row_group=group, row_in_group=i)
                      for i, row in enumerate(rows)}

    def scan(pairs):
        pending = [pair for pair in pairs if pair not in scanned]
        with ThreadPoolExecutor(max_workers=6) as pool:
            for pair, found in pool.map(read_ids, pending):
                scanned.add(pair)
                for image_id, location in found.items():
                    locations.setdefault(image_id, location)
        save_json(index_path, locations)

    missing = [i for i in image_ids if i not in locations]
    maximum = max((guesses[i] for i in missing if i in guesses), default=-1)
    total = 0
    for index in range(len(entries)):
        if total > maximum:
            break
        starts[index] = total
        total += shard(index).metadata.num_rows
    predicted_groups, relevant = [], set()
    for image_id in missing:
        ordinal = guesses.get(image_id)
        if ordinal is None:
            continue
        for index, start in starts.items():
            meta = shard(index).metadata
            if start <= ordinal < start+meta.num_rows:
                relevant.add(index)
                offset = ordinal-start
                for group in range(meta.num_row_groups):
                    if offset < meta.row_group(group).num_rows:
                        predicted_groups.append((index, group))
                        break
                    offset -= meta.row_group(group).num_rows
                break
    scan(list(dict.fromkeys(predicted_groups)))
    for index in sorted(relevant) + [i for i in range(len(entries)) if i not in relevant]:
        if all(i in locations for i in image_ids):
            break
        current = shard(index)
        scan([(index, g) for g in range(current.metadata.num_row_groups)])
        print(json.dumps(dict(event='scanned_ids', shard=index,
            found=[i for i in image_ids if i in locations], missing=[i for i in image_ids if i not in locations])), flush=True)
    wiki_manifest = ROOT / 'cache/webqa_image_smoke/manifest.json'
    wiki = ({str(f['image_id']): f for r in json.loads(wiki_manifest.read_text())['examples'] for f in r['images']}
            if wiki_manifest.exists() else {})
    results = {}
    grouped = {}
    for image_id in image_ids:
        saved = output/(image_id+'.json')
        if saved.exists():
            record = json.loads(saved.read_text())
            target = Path(record.get('local_path', ''))
            if record['status'] == 'available' and target.is_file():
                data = target.read_bytes()
                checked_image(dict(src_content=record['src_content'], img_path=record['img_path'],
                                   img=dict(path=record['embedded_path'], bytes=data)), image_id)
                if hashlib.sha256(data).hexdigest() != record['sha256']:
                    raise ValueError(f'Saved image hash mismatch: {image_id}')
                results[image_id] = record
                continue
        if image_id in locations:
            location = locations[image_id]
            grouped.setdefault((location['shard'], location['row_group']), []).append(image_id)
        else:
            results[image_id] = dict(image_id=image_id, status='not_found_in_image_id_columns')
    image_directory = output / 'images'
    image_directory.mkdir(exist_ok=True)
    def read_images(pair, ids):
        index, group = pair
        rows = shard(index).read(group, ['src_content', 'img_path', 'img'])
        batch = []
        for image_id in ids:
            location = locations[image_id]
            row = rows[location['row_in_group']]
            try:
                data, properties = checked_image(row, image_id)
            except Exception as error:
                batch.append(dict(image_id=image_id, status='error', **location,
                                  error_type=type(error).__name__, error=str(error)))
                continue
            suffix = '.jpg' if properties['format'] == 'JPEG' else '.'+properties['format'].lower()
            target = image_directory / (image_id+suffix)
            target.write_bytes(data)
            result = dict(image_id=image_id, status='available', **location, **properties,
                local_path=str(target.resolve()), bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                src_content=row['src_content'], img_path=row['img_path'], embedded_path=row['img']['path'],
                predicted_ordinal=guesses.get(image_id), official_tsv_bytes_verified=False)
            if image_id in wiki:
                result['wikimedia_comparison'] = compare_wikimedia(data, wiki[image_id]['local_path'])
            batch.append(result)
            save_json(output / (image_id+'.json'), result)
            print(json.dumps(dict(event='image_saved', image_id=image_id, bytes=len(data), dimensions=[properties['width'],properties['height']])), flush=True)
        return batch

    def checkpoint(completed=False):
        available = sum(r['status'] == 'available' for r in results.values())
        manifest = dict(repository=REPOSITORY, revision=REVISION,
            completed_utc=datetime.now(timezone.utc).isoformat() if completed else None,
            updated_utc=datetime.now(timezone.utc).isoformat(), target_image_ids=image_ids,
            results=[results.get(i, dict(image_id=i, status='pending')) for i in image_ids],
            indexed_image_ids=len(locations), available_images=available,
            downloaded_range_bytes=sum(p.stat().st_size for p in (output/'ranges').glob('*/*.range')),
            reused_cache=str(reuse_cache.resolve()) if reuse_cache else None,
            scope='ID-verified public mirror images; not original TSV byte verification; no answer/model access')
        if requests_path:
            manifest['requests_file'] = str(requests_path.resolve())
            manifest['requests_sha256'] = hashlib.sha256(requests_path.read_bytes()).hexdigest()
        save_json(output/'manifest.json', manifest)
        print(json.dumps(dict(event='progress', available=available, targets=len(image_ids),
            completed_groups=done, total_groups=len(grouped),
            downloaded_range_bytes=manifest['downloaded_range_bytes'],
            elapsed_seconds=round(time.monotonic()-started, 1))), flush=True)
        return manifest

    done = 0
    checkpoint()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(read_images, pair, ids): (pair, ids) for pair, ids in grouped.items()}
        for future in as_completed(futures):
            pair, ids = futures[future]
            try:
                batch = future.result()
            except Exception as error:
                batch = [dict(image_id=i, status='error', **locations[i],
                              error_type=type(error).__name__, error=str(error)) for i in ids]
            for record in batch:
                results[record['image_id']] = record
                save_json(output/(record['image_id']+'.json'), record)
                if record['status'] == 'error':
                    with (output/'errors.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), **record))+'\n')
            done += 1
            checkpoint()
    manifest = checkpoint(completed=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--image-ids', nargs='+', default=TARGETS)
    selection.add_argument('--requests-file', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'cache/webqa_parquet_images')
    parser.add_argument('--source', type=Path, default=ROOT/'data/WebQA_train_val.json')
    parser.add_argument('--reuse-cache', type=Path)
    args = parser.parse_args()
    ids = requested_image_ids(args.requests_file) if args.requests_file else args.image_ids
    result = fetch(ids, args.output, args.source, args.reuse_cache, args.requests_file)
    raise SystemExit(0 if result['available_images'] == len(ids) else 1)
