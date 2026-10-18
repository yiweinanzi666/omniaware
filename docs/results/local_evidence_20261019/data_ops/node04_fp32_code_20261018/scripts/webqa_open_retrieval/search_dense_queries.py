"""Persist per-modality top5 with index, query-vector and source-ID bindings.

The default requires the complete text corpus or all currently available images.
--engineering-prefix marks small integration checks and cannot produce full-corpus
results. BGE and CLIP scores remain separate; this stage makes no routing decision.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import time

import faiss
import numpy as np

from . import build_dense_index as B

TOP_K, BATCH_SIZE = 5, 32


def bind_inputs(index_dir, query_dir, engineering_prefix=False):
    index_dir, query_dir = Path(index_dir), Path(query_dir)
    plan, complete = B.read(index_dir/'plan.json'), B.read(index_dir/'completion.json')
    kind, dimension = plan['kind'], plan['dimension']
    if kind not in ('bge','clip') or dimension != (1024 if kind=='bge' else 768):
        raise ValueError('Index encoder/dimension mismatch')
    if complete['status'] != 'INDEX_COMPLETE' or complete['plan_sha256'] != B.sha(index_dir/'plan.json'):
        raise ValueError('Index completion/plan binding changed')
    if complete['index_sha256'] != B.sha(index_dir/'index.faiss'):
        raise ValueError('Index bytes changed')
    if complete['ids_sha256'] != B.sha(index_dir/'ids.txt') or complete['ids_sha256'] != plan['ordered_ids_sha256']:
        raise ValueError('Index source-ID ordering changed')
    ids = (index_dir/'ids.txt').read_text().splitlines()
    if len(ids) != plan['vectors'] or len(ids) != complete['vectors'] or len(set(ids)) != len(ids):
        raise ValueError('Index source-ID count/uniqueness mismatch')
    availability = {}
    if not engineering_prefix:
        if not plan['full_encoded_inputs'] or plan['pilot_shards_per_input'] is not None:
            raise ValueError('Engineering prefix is not a full-corpus search input')
        if kind=='bge':
            if len(ids) != 544489:
                raise ValueError('Full text corpus requires544489 exact-text documents')
        else:
            expected = {f'image:{i}' for i in range(30000000,30389750)}
            missing = expected-set(ids)
            diagnosis = B.HERE/'global_image_assets/tsv_gaps/CORRUPTION_DIAGNOSIS.json'
            unavailable = {'image:'+r['image_id'] for r in B.read(diagnosis)
                           if r.get('header',{}).get('format')=='JPEG' and 'truncated' in r.get('decode','')}
            if set(ids)-expected or missing-unavailable:
                raise ValueError('Image corpus misses assets beyond the documented unavailable JPEGs')
            availability = dict(missing_image_ids=sorted(missing),diagnosis_sha256=B.sha(diagnosis),
                                scope='All available images, with every omitted ID recorded; retain all questions')
    queries_plan = B.read(query_dir/'plan.json')
    queries_complete = B.read(query_dir/'completion.json')
    receipt = B.read(query_dir/f'{kind}.json')
    if queries_complete['plan_sha256'] != B.sha(query_dir/'plan.json') or receipt['plan_sha256'] != queries_complete['plan_sha256']:
        raise ValueError('Query plan binding changed')
    saved = queries_complete['encoders'][kind]
    if saved['receipt_sha256'] != B.sha(query_dir/f'{kind}.json'):
        raise ValueError('Query encoder receipt changed')
    if saved['array_sha256'] != receipt['array_sha256'] or receipt['array_sha256'] != B.sha(query_dir/f'{kind}.npz'):
        raise ValueError('Query vectors changed')
    if queries_plan['model_files_sha256'][kind] != plan['model_files_sha256']:
        raise ValueError('Query and corpus encoder assets differ')
    with np.load(query_dir/f'{kind}.npz',allow_pickle=False) as archive:
        qids, vectors = archive['qids'].tolist(), archive['vectors']
    if qids != queries_plan['ordered_qids'] or len(qids) != queries_plan['questions'] or len(set(qids)) != len(qids):
        raise ValueError('Query IDs/count/order differ from frozen plan')
    if vectors.dtype != np.float32 or vectors.shape != (len(qids),dimension) or not np.isfinite(vectors).all():
        raise ValueError('Invalid query vector array')
    if not np.allclose(np.linalg.norm(vectors,axis=1),1.,atol=2e-6,rtol=0):
        raise ValueError('Query vectors are not normalized')
    binding = dict(kind=kind,dimension=dimension,index_directory=str(index_dir.resolve()),
        query_directory=str(query_dir.resolve()),index_plan_sha256=complete['plan_sha256'],
        index_completion_sha256=B.sha(index_dir/'completion.json'),index_sha256=complete['index_sha256'],
        ids_sha256=complete['ids_sha256'],query_plan_sha256=queries_complete['plan_sha256'],
        query_completion_sha256=B.sha(query_dir/'completion.json'),query_receipt_sha256=saved['receipt_sha256'],
        query_array_sha256=receipt['array_sha256'],questions=len(qids),vectors=len(ids),
        engineering_prefix=engineering_prefix,global_pool_complete=plan['global_pool_complete'],
        availability=availability,top_k=TOP_K,query_batch_size=BATCH_SIZE,threads=2,
        faiss=faiss.__version__,numpy=np.__version__,score='raw unit-vector inner product, kept separate by encoder',
        tie_rule='FAISS native top5 selection; returned neighbors ordered by descending score then source ID',
        source_sha256={str(p):B.sha(p) for p in (Path(__file__),Path(B.__file__),B.HERE/'encode_global_text.py')},
        labels_read=False,model_calls=0,MPS_calls=0)
    return binding,ids,qids,np.ascontiguousarray(vectors)


def search(index_dir,query_dir,output,engineering_prefix=False):
    started=time.perf_counter()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with (output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan,ids,qids,queries=bind_inputs(index_dir,query_dir,engineering_prefix)
        if (output/'plan.json').exists():
            if B.read(output/'plan.json') != plan:
                raise ValueError('Search inputs changed; preserve the existing retrieval artifact')
        else:
            B.save(output/'plan.json',plan)
        if (output/'completion.json').exists():
            complete=B.read(output/'completion.json')
            if complete['plan_sha256'] != B.sha(output/'plan.json') or complete['rankings_sha256'] != B.sha(output/'rankings.jsonl'):
                raise ValueError('Completed retrieval bindings changed')
            return dict(complete,new_search_batches=0)
        if any((output/name).exists() for name in ('rankings.jsonl','rankings.jsonl.tmp')):
            raise ValueError('Retrieval output exists without completion; preserve it for inspection')
        faiss.omp_set_num_threads(2)
        index=faiss.read_index(str(Path(index_dir)/'index.faiss'))
        if not isinstance(index,faiss.IndexFlatIP) or index.d != plan['dimension'] or index.ntotal != len(ids):
            raise ValueError('Serialized index type/shape differs from metadata')
        tick=time.perf_counter();rows=[];batches=0
        for start in range(0,len(qids),BATCH_SIZE):
            distances,neighbors=index.search(queries[start:start+BATCH_SIZE],TOP_K)
            batches+=1
            for offset,(scores,indices) in enumerate(zip(distances,neighbors,strict=True)):
                if any(i<0 or i>=len(ids) for i in indices) or len(set(indices.tolist())) != TOP_K or not np.isfinite(scores).all():
                    raise ValueError('Invalid FAISS neighbors or scores')
                ranked=sorted((dict(source_id=ids[int(i)],index_row=int(i),cosine=float(score))
                               for i,score in zip(indices,scores,strict=True)),key=lambda r:(-r['cosine'],r['source_id']))
                rows.append(dict(qid=qids[start+offset],encoder=plan['kind'],
                                 candidates=[dict(rank=rank,**r) for rank,r in enumerate(ranked,1)]))
        search_seconds=time.perf_counter()-tick
        temporary=output/'rankings.jsonl.tmp'
        temporary.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
        temporary.replace(output/'rankings.jsonl')
        complete=dict(status='MODAL_TOP5_COMPLETE',plan_sha256=B.sha(output/'plan.json'),
            rankings_sha256=B.sha(output/'rankings.jsonl'),questions=len(qids),encoder=plan['kind'],
            search_batches=batches,search_seconds=search_seconds,wall_seconds=time.perf_counter()-started,
            engineering_prefix=engineering_prefix,global_pool_complete=plan['global_pool_complete'],
            availability=plan['availability'],completed_utc=datetime.now(timezone.utc).isoformat(),
            labels_read=False,model_calls=0,MPS_calls=0,
            scope='Engineering prefix only' if engineering_prefix else 'Per-modality full available corpus; fusion/calibration and QA not performed')
        B.save(output/'completion.json',complete)
        return dict(complete,new_search_batches=batches)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,required=True)
    parser.add_argument('--queries',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--engineering-prefix',action='store_true')
    args=parser.parse_args()
    print(json.dumps(search(args.index,args.queries,args.output,args.engineering_prefix)),flush=True)
