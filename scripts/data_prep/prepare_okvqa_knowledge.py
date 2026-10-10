"""Resume official DPR corpus bytes and stream the archived first-3M construction rule."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import requests

ROOT = Path(os.environ.get('OMNIAWARE_ASSETS') or os.environ.get('ERC_ASSET_ROOT')
            or Path(__file__).resolve().parents[2] / 'assets').resolve()  # legacy experiments/eab_first_token
CACHE = ROOT / 'cache/okvqa_knowledge_20261004'
DATA = ROOT / 'data/okvqa_knowledge_dpr_first3m'
URL = 'https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz'
SIZE = 4694541059
ETAG = '"7ccd39b61360f246b8258e0eb08f1376-299"'
MD5 = '612fe66e0b6b41ee28f806140226c563'
SOURCE = ROOT / 'cache/emnlp_source_20261004/self-aware-mrag-main/FlashRAG/tools/build_corpus_wiki_only_3m.py'


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def hashes(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open('rb') as f:
        while block := f.read(8 * 1024 * 1024):
            sha.update(block); md5.update(block)
    return dict(sha256=sha.hexdigest(), md5=md5.hexdigest())


def download():
    CACHE.mkdir(parents=True, exist_ok=True)
    partial = CACHE / 'psgs_w100.tsv.gz.part'
    target = CACHE / 'psgs_w100.tsv.gz'
    status = CACHE / 'download_status.json'
    previous = json.loads(status.read_text()) if status.exists() else {}
    if target.exists():
        assert previous.get('status') == 'COMPLETE' and target.stat().st_size == SIZE
        return target
    prefix = CACHE / 'psgs_w100.prefix_8m.gz.part'
    if not partial.exists() and prefix.exists():
        probe = json.loads((CACHE / 'download_probe.json').read_text())
        assert probe['status'] == 206 and probe['headers']['ETag'] == ETAG
        assert hashes(prefix)['sha256'] == probe['sha256']
        shutil.copyfile(prefix, partial)
    start = time.perf_counter(); initial = partial.stat().st_size if partial.exists() else 0
    record = dict(url=URL, expected_bytes=SIZE, expected_etag=ETAG, expected_md5=MD5,
        started_utc=datetime.now(timezone.utc).isoformat(), initial_bytes=initial,
        status='DOWNLOADING', downloaded_bytes=initial, attempts=[])
    write(status, record)
    for attempt in range(1, 11):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == SIZE:
            break
        assert offset < SIZE
        try:
            headers = {'Range': f'bytes={offset}-', 'If-Match': ETAG, 'Accept-Encoding': 'identity'}
            with requests.get(URL, headers=headers, stream=True, timeout=(15, 30)) as response:
                response.raise_for_status()
                assert response.status_code == 206
                assert response.headers.get('ETag') == ETAG
                assert response.headers.get('Content-Range') == f'bytes {offset}-{SIZE - 1}/{SIZE}'
                record['attempts'].append(dict(attempt=attempt, offset=offset, status=206,
                    content_range=response.headers['Content-Range']))
                tick = time.perf_counter()
                with partial.open('ab') as f:
                    for block in response.iter_content(1024 * 1024):
                        if not block:
                            continue
                        f.write(block)
                        record['downloaded_bytes'] = f.tell()
                        if time.perf_counter() - tick >= 5:
                            f.flush()
                            elapsed = time.perf_counter() - start
                            record.update(elapsed_seconds=elapsed,
                                MiB_per_second=(f.tell() - initial) / 2**20 / elapsed)
                            write(status, record); tick = time.perf_counter()
            assert partial.stat().st_size == SIZE
            break
        except (requests.RequestException, AssertionError) as error:
            record['last_error'] = f'{type(error).__name__}: {error}'
            record['downloaded_bytes'] = partial.stat().st_size if partial.exists() else 0
            write(status, record)
            if attempt == 10:
                record['status'] = 'FAILED'; write(status, record); raise
            time.sleep(min(attempt * 2, 20))
    assert partial.stat().st_size == SIZE
    record.update(status='VERIFYING', downloaded_bytes=SIZE)
    write(status, record)
    measured = hashes(partial)
    assert measured['md5'] == MD5, 'Official object metadata MD5 mismatch'
    partial.replace(target)
    record.update(status='COMPLETE', finished_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter() - start, path=str(target), **measured)
    write(status, record)
    return target


def build():
    raw = CACHE / 'psgs_w100.tsv.gz'
    downloaded = json.loads((CACHE / 'download_status.json').read_text())
    assert downloaded['status'] == 'COMPLETE' and raw.stat().st_size == SIZE
    DATA.mkdir(parents=True, exist_ok=True)
    target = DATA / 'corpus_wiki_first3m.jsonl'
    if target.exists():
        manifest = json.loads((DATA / 'manifest.json').read_text())
        assert hashes(target)['sha256'] == manifest['corpus_sha256']
        return
    temporary = target.with_suffix('.jsonl.part')
    digest = hashlib.sha256(); seen = kept = skipped = 0; start = time.perf_counter()
    with gzip.open(raw, 'rt', encoding='utf-8', newline='') as f, temporary.open('wb') as out:
        reader = csv.DictReader(f, delimiter='\t')
        assert reader.fieldnames == ['id', 'text', 'title']
        for idx, row in enumerate(reader):
            if idx >= 3000000:
                break
            seen += 1
            title, text = row['title'].strip(), row['text'].strip()
            if not title or not text:
                skipped += 1
                continue
            doc = dict(id=idx, source='wikipedia', title=title, contents=f'{title}\n{text}', image_url='')
            payload = (json.dumps(doc, ensure_ascii=False) + '\n').encode()
            out.write(payload); digest.update(payload); kept += 1
            if seen % 100000 == 0:
                write(DATA / 'preparation_status.json', dict(status='BUILDING', input_rows=seen,
                    documents=kept, skipped_empty=skipped, elapsed_seconds=time.perf_counter() - start))
        assert seen == 3000000
    temporary.replace(target)
    write(DATA / 'manifest.json', dict(status='CORPUS_COMPLETE_INDEX_PENDING', documents=kept,
        input_rows=seen, skipped_empty=skipped, corpus_bytes=target.stat().st_size,
        corpus_sha256=digest.hexdigest(), source_url=URL, compressed_source_sha256=downloaded['sha256'],
        legacy_rule_source=str(SOURCE) if SOURCE.exists() else None,
        legacy_rule_source_sha256=hashes(SOURCE)['sha256'] if SOURCE.exists() else None,
        rule='First 3,000,000 TSV data rows in source order; strip title/text; skip empty; zero-based row ID; title + newline + text',
        historical_byte_identity='Unverified: historical TSV/corpus checksum unavailable',
        annotation_answers_read=False, model_calls=0, elapsed_seconds=time.perf_counter() - start,
        finished_utc=datetime.now(timezone.utc).isoformat()))
    write(DATA / 'preparation_status.json', dict(status='COMPLETE', documents=kept))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=['download', 'build'], required=True)
    parser.add_argument('--build-after-download', action='store_true')
    args = parser.parse_args()
    if args.action == 'download':
        download()
        if args.build_after_download:
            build()
    else:
        build()
