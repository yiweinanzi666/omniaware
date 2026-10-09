"""Fetch the official Short videos by ZIP member ranges, without whole ZIPs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import threading
import urllib.request
import zipfile
import zlib


REVISION = 'ead1408f75b618502df9a1d8e0950166bf0a2a0b'
REPOSITORY = 'lmms-eval/Video-MME'
BASE_URL = f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/'
OUTPUT = Path(__file__).resolve().parent / 'data/videomme_short'
ANNOTATIONS = 'videomme/test-00000-of-00001.parquet'


def get_bytes(url, start=None, end=None):
    headers = {} if start is None else {'Range': f'bytes={start}-{end}'}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
        if start is not None:
            assert response.status == 206
            assert response.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
        result = response.read()
    if start is not None:
        assert len(result) == end - start + 1
    return result


class RemoteZip:
    def __init__(self, url, size):
        self.url = url
        self.size = size
        self.pos = 0

    def seek(self, offset, whence=0):
        self.pos = (offset, self.pos + offset, self.size + offset)[whence]
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def read(self, count=-1):
        end = self.size if count < 0 else min(self.size, self.pos + count)
        if end == self.pos:
            return b''
        result = get_bytes(self.url, self.pos, end - 1)
        self.pos = end
        return result


def prepare(output=OUTPUT):
    import pyarrow.parquet as pq

    output.mkdir(parents=True, exist_ok=True)
    raw = get_bytes(BASE_URL + ANNOTATIONS)
    (output / 'official_annotations.parquet').write_bytes(raw)
    columns = ['video_id', 'duration', 'domain', 'sub_category', 'url', 'videoID',
               'question_id', 'task_type', 'question', 'options']
    rows = pq.read_table(io.BytesIO(raw), columns=columns).to_pylist()
    questions = [row for row in rows if row['duration'] == 'short']
    video_ids = {row['videoID'] for row in questions}
    subtitles_raw = get_bytes(BASE_URL + 'subtitle.zip')
    (output / 'official_subtitle.zip').write_bytes(subtitles_raw)
    subtitle_dir = output / 'subtitles'
    subtitle_dir.mkdir(exist_ok=True)
    subtitles = []
    with zipfile.ZipFile(io.BytesIO(subtitles_raw)) as archive:
        assert archive.testzip() is None
        for info in archive.infolist():
            if not info.is_dir() and Path(info.filename).stem in video_ids:
                data = archive.read(info.filename)
                target = subtitle_dir / Path(info.filename).name
                target.write_bytes(data)
                subtitles.append({'video_id': Path(info.filename).stem, 'member': info.filename,
                                  'path': str(target), 'bytes': len(data),
                                  'sha256': hashlib.sha256(data).hexdigest()})
    catalog_url = f'https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}?recursive=true&limit=1000'
    catalog = json.loads(get_bytes(catalog_url))
    zip_entries = [entry for entry in catalog if entry['path'].startswith('videos_chunked_')]

    def inspect(entry):
        url = BASE_URL + entry['path']
        with zipfile.ZipFile(RemoteZip(url, entry['size'])) as archive:
            videos = [{'video_id': Path(info.filename).stem, 'member': info.filename,
                       'compressed_bytes': info.compress_size, 'bytes': info.file_size,
                       'crc32': info.CRC, 'compress_type': info.compress_type,
                       'header_offset': info.header_offset}
                      for info in archive.infolist()
                      if not info.is_dir() and Path(info.filename).stem in video_ids]
        return {'path': entry['path'], 'url': url, 'bytes': entry['size'], 'videos': videos}

    with ThreadPoolExecutor(max_workers=4) as pool:
        archives = list(pool.map(inspect, zip_entries))
    videos = [video for archive in archives for video in archive['videos']]
    assert {video['video_id'] for video in videos} == video_ids
    assert len(videos) == len(video_ids)
    with (output / 'questions.jsonl').open('w') as handle:
        for row in questions:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest = {
        'prepared_utc': datetime.now(timezone.utc).isoformat(),
        'repository': REPOSITORY, 'revision': REVISION,
        'selection': "official duration == 'short'; all matching questions and videos; no answer-dependent selection",
        'split': 'official_test', 'historical_omni_aware_id_list': False,
        'annotations': {'url': BASE_URL + ANNOTATIONS, 'bytes': len(raw),
                        'sha256': hashlib.sha256(raw).hexdigest(), 'answer_column_read': False},
        'subtitles': {'url': BASE_URL + 'subtitle.zip', 'bytes': len(subtitles_raw),
                      'sha256': hashlib.sha256(subtitles_raw).hexdigest(),
                      'short_files': subtitles,
                      'short_videos_without_subtitles': sorted(video_ids - {row['video_id'] for row in subtitles})},
        'question_count': len(questions), 'video_count': len(videos),
        'compressed_video_bytes': sum(video['compressed_bytes'] for video in videos),
        'video_bytes': sum(video['bytes'] for video in videos),
        'archive_bytes_not_downloaded': sum(archive['bytes'] for archive in archives),
        'archives': archives,
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in manifest.items() if key not in ('archives', 'subtitles')}, indent=2))
    print(json.dumps({'short_subtitle_files': len(subtitles),
                      'short_videos_without_subtitles': len(manifest['subtitles']['short_videos_without_subtitles'])}))
    return manifest


def save_video(archive, video, destination):
    path = destination / Path(video['member']).name
    reused = path.exists()
    data = path.read_bytes() if reused else archive.read(video['member'])
    assert len(data) == video['bytes'], path
    assert zlib.crc32(data) == video['crc32'], path
    if not reused:
        partial = path.with_name(path.name + '.partial')
        partial.write_bytes(data)
        partial.rename(path)
    return {'video_id': video['video_id'], 'member': video['member'],
            'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
            'crc32': video['crc32'], 'reused': reused,
            'completed_utc': datetime.now(timezone.utc).isoformat()}


def download(output=OUTPUT):
    manifest = json.loads((output / 'manifest.json').read_text())
    destination = output / 'videos'
    destination.mkdir(exist_ok=True)
    lock = threading.Lock()

    def log_attempt(record):
        record['recorded_utc'] = datetime.now(timezone.utc).isoformat()
        with lock, (output / 'download_attempts.jsonl').open('a') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')

    def fetch_archive(entry):
        records = []
        try:
            archive = zipfile.ZipFile(RemoteZip(entry['url'], entry['bytes']))
        except Exception as error:
            log_attempt({'stage': 'archive_directory', 'status': 'failed',
                         'archive_url': entry['url'], 'error': repr(error)})
            raise
        with archive:
            for video in entry['videos']:
                attempt = {'stage': 'video', 'video_id': video['video_id'],
                           'member': video['member'], 'archive_url': entry['url']}
                log_attempt({**attempt, 'status': 'started'})
                try:
                    record = save_video(archive, video, destination)
                except Exception as error:
                    log_attempt({**attempt, 'status': 'failed', 'error': repr(error)})
                    raise
                record['archive_url'] = entry['url']
                records.append(record)
                with lock, (output / 'downloads.jsonl').open('a') as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                log_attempt({**attempt, 'status': 'completed', 'reused': record['reused']})
        print(json.dumps({'archive': entry['path'], 'videos_completed': len(records)}), flush=True)
        return records

    with ThreadPoolExecutor(max_workers=16) as pool:
        groups = list(pool.map(fetch_archive, manifest['archives']))
    records = [record for group in groups for record in group]
    completion = {'completed_utc': datetime.now(timezone.utc).isoformat(),
                  'revision': manifest['revision'], 'video_count': len(records),
                  'video_bytes': sum(record['bytes'] for record in records),
                  'whole_zip_archives_downloaded': 0,
                  'all_video_bytes_and_zip_crc_verified': True,
                  'videos': records}
    (output / 'download_completion.json').write_text(json.dumps(completion, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in completion.items() if key != 'videos'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'download'])
    arguments = parser.parse_args()
    {'prepare': prepare, 'download': download}[arguments.action]()
