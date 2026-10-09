"""Prepare official 5046 OK-VQA val inputs without annotations or model calls."""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sqlite3
import time
import zipfile

from PIL import Image
import requests

from .prepare_okvqa_bm25_candidates import QueryTokenizer, SQL, sha256, write

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'data/okvqa_val5046_assets'
ORIGINAL = ROOT / 'data/okvqa_local'
BASE = ROOT / 'data/okvqa_dpr_first3m_fts5_question_only_top5_v1/protocol.json'
LOCK = ROOT / 'runs/local_m4_20261004/supplements_queue_lock.json'
QUESTION_ZIP = ORIGINAL / 'original/OpenEnded_mscoco_val2014_questions.json.zip'
FORBIDDEN = {'answer', 'answers', 'annotation', 'annotations', 'references', 'gold_answer'}


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def write_rows(path, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    temporary.replace(path)


def selection():
    http = read(QUESTION_ZIP.with_suffix(QUESTION_ZIP.suffix + '.http.json'))
    assert sha256(QUESTION_ZIP) == http['sha256']
    with zipfile.ZipFile(QUESTION_ZIP) as archive:
        rows = json.loads(archive.read('OpenEnded_mscoco_val2014_questions.json'))['questions']
    assert len(rows) == len({r['question_id'] for r in rows}) == 5046
    assert len({r['image_id'] for r in rows}) == 5033
    assert all(set(r) == {'question', 'question_id', 'image_id'} for r in rows)
    assert rows == lines(ORIGINAL / 'confirmation_inputs.jsonl')
    return [dict(r, qid=str(r['question_id']), source_split='val2014', subset='validation') for r in rows]


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in ('images', 'image_records', 'candidate_records', 'failures'):
        (OUTPUT / name).mkdir(exist_ok=True)
    rows, base = selection(), read(BASE)
    assert base['sql'] == SQL and base['sqlite_version'] == sqlite3.sqlite_version
    source_files = (QUESTION_ZIP, QUESTION_ZIP.with_suffix(QUESTION_ZIP.suffix + '.http.json'),
                    ORIGINAL / 'confirmation_inputs.jsonl', BASE,
                    Path(__file__), ROOT / 'prepare_okvqa_bm25_candidates.py')
    protocol = dict(configuration=base['configuration'], source_split='val2014', questions=5046,
        unique_images=5033, query_input='Original question string only', top_k=5,
        query_tokenizer=base['query_tokenizer'], query_rule=base['query_rule'], tie_break=base['tie_break'],
        sql=SQL, sqlite_version=sqlite3.sqlite_version, index_path=base['index_path'],
        index_sha256=base['index_sha256'], corpus_sha256=base['corpus_sha256'],
        source_sha256={str(p): sha256(p) for p in source_files},
        official_image_url='https://s3.amazonaws.com/images.cocodataset.org/val2014/COCO_val2014_{image_id:012d}.jpg',
        image_policy='Exact official image ID only; no substitutes; validate JPEG decode and record byte/RGB hashes',
        no_reranking=True, annotation_answers_read=False, model_calls=0,
        resume='Per-question and per-image atomic records; reject changed source/config/content',
        immutable_shared_lock_sha256=sha256(LOCK))
    target = OUTPUT / 'protocol.json'
    if target.exists():
        assert read(target) == protocol, 'Frozen val asset protocol changed'
    else:
        write(target, protocol)
        write_rows(OUTPUT / 'selection.jsonl', rows)
    assert lines(OUTPUT / 'selection.jsonl') == rows
    return rows, protocol


def verify_image(path):
    with Image.open(path) as image:
        assert image.format == 'JPEG'
        image.load()
        rgb = image.convert('RGB')
        assert rgb.width > 0 and rgb.height > 0
        return dict(width=rgb.width, height=rgb.height,
                    rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())


def image_one(image_id):
    name = f'COCO_val2014_{image_id:012d}.jpg'
    target, record_path = OUTPUT / 'images' / name, OUTPUT / 'image_records' / f'{image_id}.json'
    url = 'https://s3.amazonaws.com/images.cocodataset.org/val2014/' + name
    if record_path.exists():
        prior = read(record_path)
        assert prior['image_id'] == image_id and prior['url'] == url
        assert sha256(target) == prior['sha256']
        assert all(prior[k] == v for k, v in verify_image(target).items())
        return prior
    attempts = []
    for attempt in range(1, 5):
        tick = time.perf_counter()
        try:
            with requests.get(url, timeout=(15, 45)) as response:
                response.raise_for_status()
                payload = response.content
                if response.headers.get('Content-Length') is not None:
                    assert len(payload) == int(response.headers['Content-Length'])
                temporary = target.with_suffix('.jpg.part')
                temporary.write_bytes(payload)
                metadata = verify_image(temporary)
                record = dict(image_id=image_id, url=url, final_url=response.url,
                    etag=response.headers.get('ETag'), bytes=len(payload), sha256=sha256(temporary),
                    local_path=str(target), downloaded_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.perf_counter()-tick, previous_failed_attempts=attempts, **metadata)
                temporary.replace(target)
                write(record_path, record)
                return record
        except Exception as error:
            attempts.append(dict(attempt=attempt, error=f'{type(error).__name__}: {error}',
                                 elapsed_seconds=time.perf_counter()-tick))
            write(OUTPUT / 'failures' / f'image_{image_id}.json',
                  dict(image_id=image_id, url=url, status='DOWNLOAD_OR_VALIDATION_FAILED', attempts=attempts))
            if attempt < 4:
                time.sleep(attempt)
    raise RuntimeError(f'Official image {image_id} remains unavailable; see failure record')


def images(workers=3):
    rows, _ = prepare()
    ids = sorted({r['image_id'] for r in rows})
    started, complete, failures = time.perf_counter(), [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(image_one, image_id): image_id for image_id in ids}
        for future in as_completed(jobs):
            try:
                complete.append(future.result())
            except Exception as error:
                failures.append(dict(image_id=jobs[future], error=f'{type(error).__name__}: {error}'))
            done = len(complete)+len(failures)
            if done % 50 == 0 or done == len(ids):
                progress = dict(status='DOWNLOADING_IMAGES', complete=len(complete), failed=len(failures),
                    total=len(ids), bytes=sum(r['bytes'] for r in complete), elapsed_seconds=time.perf_counter()-started,
                    workers=workers, model_calls=0)
                write(OUTPUT / 'image_progress.json', progress)
                print(json.dumps(progress), flush=True)
    if failures:
        write(OUTPUT / 'image_failures.json', dict(status='INCOMPLETE', failures=failures))
        raise RuntimeError(f'{len(failures)} official images unavailable')
    complete.sort(key=lambda r:r['image_id'])
    write(OUTPUT / 'image_manifest.json', dict(status='ALL_5033_OFFICIAL_IMAGES_VERIFIED', images=complete,
        unique_image_ids=5033, total_bytes=sum(r['bytes'] for r in complete),
        distinct_byte_hashes=len({r['sha256'] for r in complete}),
        distinct_rgb_hashes=len({r['rgb_sha256'] for r in complete}),
        duplicate_pixels={h:[r['image_id'] for r in complete if r['rgb_sha256']==h]
                          for h,n in __import__('collections').Counter(r['rgb_sha256'] for r in complete).items() if n>1},
        substitutions=0, annotation_answers_read=False, model_calls=0))


def init_retrieval(index):
    global CONNECTION, TOKENIZER
    os.nice(10)
    CONNECTION = sqlite3.connect(Path(index).as_uri() + '?mode=ro', uri=True)
    for sql in ('PRAGMA query_only=ON', 'PRAGMA threads=0', 'PRAGMA cache_size=-32768', 'PRAGMA mmap_size=0'):
        CONNECTION.execute(sql)
    TOKENIZER = QueryTokenizer()


def query_one(row):
    terms, query = TOKENIZER.parse(row['question'])
    tick, cpu = time.perf_counter(), time.process_time()
    found = CONNECTION.execute(SQL, (query,)).fetchall() if query else []
    assert len(found) == 5, f'Expected five hits for question {row["question_id"]}'
    return dict(qid=row['qid'], question_id=row['question_id'], image_id=row['image_id'], question=row['question'],
        query_terms=terms, retrieval_query=query, retrieval_seconds=time.perf_counter()-tick,
        retrieval_cpu_seconds=time.process_time()-cpu, worker_pid=os.getpid(),
        worker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        candidates=[dict(rank=i, doc_id=doc_id, fact_id=f'dpr_first3m:{doc_id}', text=text,
                        content_sha256=hashlib.sha256(text.encode()).hexdigest(), bm25_score=score)
                    for i,(doc_id,text,score) in enumerate(found,1)])


def validate_candidates(row, result, protocol_sha):
    assert not FORBIDDEN.intersection(result)
    assert result['protocol_sha256'] == protocol_sha
    assert all(result[k] == row[k] for k in ('qid','question_id','image_id','question'))
    found=result['candidates']
    assert len(found) == len({r['doc_id'] for r in found}) == 5
    assert [(r['bm25_score'],r['doc_id']) for r in found] == sorted((r['bm25_score'],r['doc_id']) for r in found)
    for rank,r in enumerate(found,1):
        assert r['rank']==rank and r['fact_id']==f'dpr_first3m:{r["doc_id"]}'
        assert hashlib.sha256(r['text'].encode()).hexdigest()==r['content_sha256']


def retrieve(workers=3, limit=None):
    rows, protocol = prepare()
    index=Path(protocol['index_path']);assert sha256(index)==protocol['index_sha256']
    protocol_sha=sha256(OUTPUT/'protocol.json');retained=[];pending=[]
    for row in rows:
        path=OUTPUT/'candidate_records'/f'{row["qid"]}.json'
        if path.exists():
            result=read(path);validate_candidates(row,result,protocol_sha);retained.append(result)
        else:pending.append(row)
    if limit is not None:pending=pending[:limit]
    started=time.perf_counter();done=[];rss={}
    with ProcessPoolExecutor(max_workers=workers, initializer=init_retrieval, initargs=(str(index),)) as pool:
        jobs={pool.submit(query_one,row):row for row in pending}
        for future in as_completed(jobs):
            row=jobs[future]
            try:
                result=dict(future.result(),protocol_sha256=protocol_sha,status='RETRIEVED')
                validate_candidates(row,result,protocol_sha)
                write(OUTPUT/'candidate_records'/f'{row["qid"]}.json',result)
                done.append(result);rss[result['worker_pid']]=max(rss.get(result['worker_pid'],0),result['worker_peak_rss_bytes'])
            except Exception as error:
                write(OUTPUT/'failures'/f'query_{row["qid"]}.json',dict(qid=row['qid'],status='RETRIEVAL_FAILED',error=f'{type(error).__name__}: {error}'))
                raise
            if len(done)%10==0 or len(done)==len(pending):
                elapsed=time.perf_counter()-started
                progress=dict(status='CPU_RETRIEVING',complete=len(retained)+len(done),total=5046,new_queries=len(done),
                    workers=workers,elapsed_seconds=elapsed,worker_peak_rss_bytes=rss,
                    sum_worker_peak_rss_bytes=sum(rss.values()),
                    cpu_core_equivalents=sum(r['retrieval_cpu_seconds'] for r in done)/elapsed,
                    projected_remaining_seconds=(5046-len(retained)-len(done))*elapsed/len(done),
                    annotation_answers_read=False,model_calls=0)
                write(OUTPUT/'retrieval_progress.json',progress);print(json.dumps(progress),flush=True)
    if limit is not None:write(OUTPUT/'retrieval_benchmark.json',progress)


def finalize():
    rows,protocol=prepare();image_manifest=read(OUTPUT/'image_manifest.json')
    assert image_manifest['status']=='ALL_5033_OFFICIAL_IMAGES_VERIFIED'
    by_id={r['image_id']:r for r in image_manifest['images']}
    assert set(by_id)=={r['image_id'] for r in rows}
    inputs=[];candidates=[];protocol_sha=sha256(OUTPUT/'protocol.json')
    for row in rows:
        image=by_id[row['image_id']];assert sha256(Path(image['local_path']))==image['sha256']
        inp=dict(row,image_path=image['local_path'],image_sha256=image['sha256'])
        result=read(OUTPUT/'candidate_records'/f'{row["qid"]}.json');validate_candidates(row,result,protocol_sha)
        inputs.append(inp);candidates.append(dict(inp,**{k:v for k,v in result.items() if k not in inp}))
    write_rows(OUTPUT/'validation_inputs.jsonl',inputs)
    write_rows(OUTPUT/'validation_candidates.jsonl',candidates)
    for name,expected in read(LOCK)['files_sha256'].items():assert sha256(ROOT/name)==expected,name
    assert sha256(LOCK)==protocol['immutable_shared_lock_sha256']
    ready=dict(status='OKVQA_VAL5046_INPUT_ASSETS_READY',questions=5046,query_images=5033,candidates=25230,
        protocol_sha256=protocol_sha,files_sha256={name:sha256(OUTPUT/name) for name in
        ('selection.jsonl','validation_inputs.jsonl','validation_candidates.jsonl','image_manifest.json','protocol.json')},
        total_image_bytes=image_manifest['total_bytes'],no_image_substitutions=True,shared_locked_files_unchanged=158,
        annotation_answers_read=False,model_calls=0,official_vqa_scoring_performed=False,
        scope='Original public val questions/images and unchanged question-only DPR3M FTS5 top5; inference/scoring pending')
    write(OUTPUT/'ready.json',ready);print(json.dumps(ready,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('prepare','images','retrieve','finalize'),default='prepare')
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    assert 1<=args.workers<=4
    if args.action=='prepare':prepare()
    elif args.action=='images':images(args.workers)
    elif args.action=='retrieve':retrieve(args.workers,args.limit)
    else:finalize()
