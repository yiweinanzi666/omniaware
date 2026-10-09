"""Full DPR Wikipedia (all psgs_w100 rows) SQLite FTS5 BM25 index for OK-VQA retrieval, CPU only.

Same construction as the 3M index (prepare_okvqa_knowledge.build + prepare_okvqa_bm25_index.build): stripped title and
text, empty rows skipped, zero-based input-row id, contents = title + newline + text, unicode61 tokenizer, rowid =
id + 1, optimize + integrity-check. The only change is that every source row is kept instead of the first 3,000,000,
so doc ids of the 3M index are an exact prefix of this one. Streams the hash-verified gzip; writes no JSONL copy.
"""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'cache/okvqa_knowledge_20261004/psgs_w100.tsv.gz'
SOURCE_MANIFEST = ROOT / 'data/okvqa_knowledge_dpr_first3m/manifest.json'
OUTPUT = ROOT / 'data/okvqa_knowledge_dpr_full_bm25/index.sqlite'


def write(path, record):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def build(output=OUTPUT):
    expected = json.loads(SOURCE_MANIFEST.read_text())['compressed_source_sha256']
    source_hash = hashlib.sha256()
    with SOURCE.open('rb') as handle:
        while block := handle.read(8 * 1024 * 1024):
            source_hash.update(block)
    if source_hash.hexdigest() != expected:
        raise ValueError('DPR source gzip differs from the recorded SHA256')
    temporary = output.with_name(output.name + '.part')
    if output.exists() or temporary.exists():
        raise FileExistsError(f'index output already exists: {output} or {temporary}')
    output.parent.mkdir(parents=True, exist_ok=True)
    progress = output.with_name(output.name + '.progress.json')
    record = dict(status='BUILDING', source=str(SOURCE), source_sha256=expected, documents=0, skipped_empty=0,
                  started_utc=datetime.now(timezone.utc).isoformat(), sqlite_version=sqlite3.sqlite_version,
                  tokenizer='unicode61', ranker='SQLite FTS5 BM25', builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  rule='all psgs_w100 rows; strip title/text; skip empty; zero-based row id; title + newline + text',
                  annotation_answers_read=False, model_calls=0)
    write(progress, record)
    start = time.perf_counter()
    csv.field_size_limit(sys.maxsize)
    connection = sqlite3.connect(temporary)
    connection.execute('PRAGMA synchronous=OFF')
    connection.execute('PRAGMA journal_mode=OFF')
    connection.execute("CREATE VIRTUAL TABLE documents USING fts5(doc_id UNINDEXED, contents, tokenize='unicode61')")
    batch = []
    with gzip.open(SOURCE, 'rt', encoding='utf-8', newline='') as handle:
        reader = csv.reader(handle, delimiter='\t')
        if next(reader) != ['id', 'text', 'title']:
            raise ValueError('unexpected DPR header')
        for row, (_, text, title) in enumerate(reader):
            title, text = title.strip(), text.strip()
            if not title or not text:
                record['skipped_empty'] += 1
                continue
            batch.append((row + 1, row, title + '\n' + text))
            record['documents'] += 1
            if len(batch) == 20000:
                connection.executemany('INSERT INTO documents(rowid, doc_id, contents) VALUES (?, ?, ?)', batch)
                connection.commit()
                batch.clear()
            if record['documents'] % 1000000 == 0:
                write(progress, dict(record, elapsed_seconds=time.perf_counter() - start))
    if batch:
        connection.executemany('INSERT INTO documents(rowid, doc_id, contents) VALUES (?, ?, ?)', batch)
    connection.commit()
    write(progress, dict(record, status='OPTIMIZING', elapsed_seconds=time.perf_counter() - start))
    connection.execute("INSERT INTO documents(documents) VALUES ('optimize')")
    connection.commit()
    connection.execute("INSERT INTO documents(documents) VALUES ('integrity-check')")
    connection.commit()
    if connection.execute('SELECT COUNT(*) FROM documents').fetchone()[0] != record['documents']:
        raise ValueError('index document count differs from inserted rows')
    connection.close()
    temporary.replace(output)
    record.update(status='INDEX_COMPLETE', index=str(output), index_bytes=output.stat().st_size,
                  elapsed_seconds=time.perf_counter() - start, finished_utc=datetime.now(timezone.utc).isoformat())
    write(output.with_name(output.name + '.manifest.json'), record)
    print(json.dumps(record))


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    build()
