"""Build a separate CPU FTS5 index from a hash-verified public knowledge corpus."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import time


def write(path, record):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def build(corpus, source_manifest, output):
    expected = json.loads(source_manifest.read_text())
    temporary = output.with_name(output.name + '.part')
    if output.exists() or temporary.exists():
        raise FileExistsError(f'Index output already exists: {output} or {temporary}')
    output.parent.mkdir(parents=True, exist_ok=True)
    progress = output.with_name(output.name + '.progress.json')
    record = dict(status='BUILDING', corpus=str(corpus.resolve()),
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  corpus_sha256=expected['corpus_sha256'],
                  expected_documents=expected['documents'], documents=0,
                  sqlite_version=sqlite3.sqlite_version,
                  tokenizer='unicode61', ranker='SQLite FTS5 BM25',
                  experiment_condition='New text-only retrieval configuration',
                  builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  sqlite_auxiliary_threads=0,
                  annotation_answers_read=False, model_calls=0)
    write(progress, record)
    start = time.perf_counter()
    digest = hashlib.sha256()
    connection = sqlite3.connect(temporary)
    try:
        connection.execute('PRAGMA threads=0')
        connection.execute('PRAGMA synchronous=NORMAL')
        connection.execute("CREATE VIRTUAL TABLE documents USING fts5(doc_id UNINDEXED, contents, tokenize='unicode61')")
        batch = []
        with corpus.open('rb') as handle:
            for payload in handle:
                digest.update(payload)
                document = json.loads(payload)
                doc_id = document['id']
                if not isinstance(doc_id, int) or doc_id < 0:
                    raise ValueError(f'Invalid corpus document ID: {doc_id!r}')
                batch.append((doc_id + 1, doc_id, document['contents']))
                record['documents'] += 1
                if len(batch) == 5000:
                    connection.executemany('INSERT INTO documents(rowid, doc_id, contents) VALUES (?, ?, ?)', batch)
                    connection.commit()
                    batch.clear()
                if record['documents'] % 100000 == 0:
                    record['elapsed_seconds'] = time.perf_counter() - start
                    write(progress, record)
        if batch:
            connection.executemany('INSERT INTO documents(rowid, doc_id, contents) VALUES (?, ?, ?)', batch)
        if digest.hexdigest() != expected['corpus_sha256']:
            raise ValueError('Corpus SHA256 differs from its manifest')
        if record['documents'] != expected['documents']:
            raise ValueError('Corpus document count differs from its manifest')
        record['status'] = 'OPTIMIZING'
        write(progress, record)
        connection.execute("INSERT INTO documents(documents) VALUES ('optimize')")
        connection.commit()
        connection.execute("INSERT INTO documents(documents) VALUES ('integrity-check')")
        connection.commit()
        if connection.execute('SELECT COUNT(*) FROM documents').fetchone()[0] != expected['documents']:
            raise ValueError('Index document count differs from its manifest')
        connection.close()
        index_hash = hashlib.sha256()
        with temporary.open('rb') as handle:
            while block := handle.read(8 * 1024 * 1024):
                index_hash.update(block)
        temporary.replace(output)
        record.update(status='INDEX_COMPLETE_CANDIDATES_PENDING',
                      index=str(output.resolve()), index_bytes=output.stat().st_size,
                      index_sha256=index_hash.hexdigest(),
                      elapsed_seconds=time.perf_counter() - start,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write(output.with_name(output.name + '.manifest.json'), record)
        write(progress, record)
        return record
    except Exception as error:
        connection.close()
        record.update(status='FAILED', error=f'{type(error).__name__}: {error}')
        write(progress, record)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.corpus, args.source_manifest, args.output), indent=2))
