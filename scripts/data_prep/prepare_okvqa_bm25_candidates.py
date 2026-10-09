"""Retrieve frozen question-only top-5 BM25 candidates for OK-VQA train cohorts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

ROOT = Path(os.environ.get('OMNIAWARE_ASSETS') or os.environ['ERC_ASSET_ROOT']).resolve()  # legacy experiments/eab_first_token
SQL = ('SELECT doc_id, contents, bm25(documents) AS score FROM documents '
       'WHERE documents MATCH ? ORDER BY score ASC, doc_id ASC LIMIT 5')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


class QueryTokenizer:
    def __init__(self):
        self.connection = sqlite3.connect(':memory:')
        self.connection.execute("CREATE VIRTUAL TABLE tokens USING fts5(question, tokenize='unicode61')")
        self.connection.execute("CREATE VIRTUAL TABLE terms USING fts5vocab(tokens, instance)")

    def parse(self, question):
        self.connection.execute('DELETE FROM tokens')
        self.connection.execute('INSERT INTO tokens(question) VALUES (?)', (question,))
        terms = list(dict.fromkeys(row[0] for row in self.connection.execute('SELECT term FROM terms ORDER BY offset')))
        return terms, ' OR '.join('"' + term.replace('"', '""') + '"' for term in terms)


def prepare(index, output):
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = index.with_name(index.name + '.manifest.json')
    index_manifest = json.loads(manifest_path.read_text())
    if sha256(index) != index_manifest['index_sha256']:
        raise ValueError('Index SHA256 differs from its manifest')
    inputs = {name: ROOT / 'data/okvqa_local' / f'{name}_inputs.jsonl'
              for name in ('calibration', 'development')}
    protocol = dict(configuration='okvqa_dpr_first3m_fts5_question_only_top5_v1',
                    retrieval='New CPU text-only SQLite FTS5 BM25 configuration',
                    top_k=5, top_k_source='Archived run_okvqa_baselines.py line 766, --retrieval-topk default=5',
                    query_input='Original question string only',
                    query_tokenizer='SQLite FTS5 unicode61, native fts5vocab instance order',
                    query_rule='Deduplicate tokens preserving order; quote each token; join with OR; no stopword filtering or expansion',
                    tie_break='Ascending raw negative BM25 score, then ascending numeric document ID',
                    sql=SQL, sqlite_version=sqlite3.sqlite_version,
                    index_path=str(index.resolve()), index_sha256=index_manifest['index_sha256'],
                    corpus_sha256=index_manifest['corpus_sha256'],
                    input_sha256={name: sha256(path) for name, path in inputs.items()},
                    generator_sha256=sha256(Path(__file__)),
                    no_reranking=True, annotation_answers_read=False, model_calls=0,
                    confirmation_queries=0, zero_results='Record failure and stop; no QA fallback')
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError('Existing candidate protocol differs')
    else:
        write(protocol_path, protocol)
    protocol_hash = sha256(protocol_path)
    connection = sqlite3.connect(index.resolve().as_uri() + '?mode=ro', uri=True)
    connection.execute('PRAGMA query_only=ON')
    connection.execute('PRAGMA threads=0')
    tokenizer = QueryTokenizer()
    started = time.perf_counter()
    status = dict(status='RETRIEVING', protocol_sha256=protocol_hash,
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  completed=0, total=628, subsets={})
    write(output / 'progress.json', status)
    try:
        for subset, path in inputs.items():
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if len(rows) != (500 if subset == 'calibration' else 128):
                raise ValueError(f'Unexpected {subset} count')
            target = output / f'{subset}_candidates.jsonl'
            temporary = target.with_name(target.name + '.part')
            completed = []
            retained = target if target.exists() else temporary
            if retained.exists():
                completed = [json.loads(line) for line in retained.read_text().splitlines()]
                for source, result in zip(rows, completed, strict=False):
                    if result['question_id'] != source['question_id'] or result['protocol_sha256'] != protocol_hash or len(result['candidates']) != 5:
                        raise ValueError('Retained candidate identity or protocol differs')
                if len(completed) > len(rows):
                    raise ValueError('Too many retained candidate rows')
                if target.exists() and len(completed) != len(rows):
                    raise ValueError('Final candidate file has an incomplete cohort')
            status['completed'] += len(completed)
            if not target.exists():
                with temporary.open('a') as handle:
                    for row in rows[len(completed):]:
                        terms, query = tokenizer.parse(row['question'])
                        tick = time.perf_counter()
                        found = connection.execute(SQL, (query,)).fetchall() if query else []
                        elapsed = time.perf_counter() - tick
                        if len(found) != 5:
                            write(output / 'failure.json', dict(subset=subset, question_id=row['question_id'],
                                  query=query, candidates=len(found), status='RETRIEVAL_INCOMPLETE'))
                            raise ValueError(f'Expected five hits for question {row["question_id"]}, got {len(found)}')
                        candidates = [dict(rank=rank, doc_id=doc_id, fact_id=f'dpr_first3m:{doc_id}',
                                           text=contents, content_sha256=hashlib.sha256(contents.encode()).hexdigest(),
                                           bm25_score=score)
                                      for rank, (doc_id, contents, score) in enumerate(found, 1)]
                        result = dict(row, subset=subset, query_terms=terms, retrieval_query=query,
                                      retrieval_seconds=elapsed, protocol_sha256=protocol_hash,
                                      status='RETRIEVED', candidates=candidates)
                        handle.write(json.dumps(result, ensure_ascii=False) + '\n')
                        handle.flush()
                        completed.append(result)
                        status['completed'] += 1
                        status.update(current_subset=subset, elapsed_seconds=time.perf_counter() - started,
                                      latest_query_seconds=elapsed)
                        write(output / 'progress.json', status)
                temporary.replace(target)
            status['subsets'][subset] = dict(questions=len(completed), candidates=sum(len(r['candidates']) for r in completed),
                                           path=str(target.resolve()), sha256=sha256(target),
                                           query_seconds=sum(r['retrieval_seconds'] for r in completed))
        status.update(status='CANDIDATES_COMPLETE_QA_PENDING', elapsed_seconds=time.perf_counter() - started,
                      finished_utc=datetime.now(timezone.utc).isoformat(), annotation_answers_read=False,
                      model_calls=0, confirmation_queries=0, index_sha256=index_manifest['index_sha256'])
        write(output / 'manifest.json', status)
        write(output / 'progress.json', status)
        return status
    except Exception as error:
        status.update(status='FAILED', error=f'{type(error).__name__}: {error}')
        write(output / 'progress.json', status)
        raise
    finally:
        tokenizer.connection.close()
        connection.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.index, args.output), indent=2))
