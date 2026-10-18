"""Check the query/index connection using real saved vectors and isolated faults."""
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import faiss
import numpy as np

from . import search_dense_queries as S


def rejected(operation):
    try:
        operation()
    except ValueError as exc:
        return str(exc)
    raise AssertionError('Mismatched inputs accepted')


def main():
    faiss.omp_set_num_threads(2)
    base=S.B.HERE
    checks=[]
    for kind,encoded in (('bge','bge_text_cpu'),('clip','clip_images_cpu_wave0')):
        index_dir=base/f'faiss_{kind}_engineering1024'
        _,chunks,source_ids,_=S.B.inspect_inputs(kind,[base/encoded],8)
        original=np.concatenate([S.B.load_vectors(chunk,kind) for chunk in chunks])
        for cohort,folder in (('calibration500','webqa_calibration500_queries_cpu'),('train256','webqa_train256_queries_cpu')):
            query_dir=base/folder
            output=base/'dense_top5_engineering'/f'{kind}_{cohort}'
            result=S.search(index_dir,query_dir,output,True)
            rows=[json.loads(line) for line in (output/'rankings.jsonl').read_text().splitlines()]
            with np.load(query_dir/f'{kind}.npz') as archive:
                queries=archive['vectors'];qids=archive['qids'].tolist()
            assert [r['qid'] for r in rows]==qids and all(r['encoder']==kind for r in rows)
            brute=queries@original.T
            exact=0;max_error=0.
            for i,row in enumerate(rows):
                candidates=row['candidates']
                assert [c['rank'] for c in candidates]==[1,2,3,4,5]
                positions=np.asarray([c['index_row'] for c in candidates])
                assert len(set(positions.tolist()))==5
                assert [c['source_id'] for c in candidates]==[source_ids[j] for j in positions]
                scores=np.asarray([c['cosine'] for c in candidates])
                max_error=max(max_error,float(np.max(np.abs(scores-brute[i,positions]))))
                assert np.all(brute[i,positions]>=np.partition(brute[i],-5)[-5]-2e-6)
                exact+=set(positions.tolist())==set(np.argsort(-brute[i],kind='stable')[:5].tolist())
            assert max_error<2e-6
            files=[p for p in output.iterdir() if p.suffix!='.lock']
            before={p.name:S.B.sha(p) for p in files}
            with patch.object(faiss,'read_index',side_effect=AssertionError('Completed search reloaded index')):
                resumed=S.search(index_dir,query_dir,output,True)
            assert resumed['new_search_batches']==0
            assert before=={p.name:S.B.sha(p) for p in files}
            checks.append(dict(encoder=kind,cohort=cohort,queries=len(rows),index_vectors=1024,
                max_cosine_error=max_error,exact_top5_sets=exact,top5_cutoff_check=True,
                source_id_rows_exact=True,reentry_search_batches=0,reentry_byte_changes=0,
                completion_sha256=S.B.sha(output/'completion.json'),search_seconds=result['search_seconds']))
    index_dir=base/'faiss_bge_engineering1024'
    query_dir=base/'webqa_calibration500_queries_cpu'
    faults={'prefix_as_full':rejected(lambda:S.bind_inputs(index_dir,query_dir))}
    with tempfile.TemporaryDirectory() as folder:
        temp=Path(folder);index_copy=temp/'index';index_copy.mkdir()
        for name in ('plan.json','completion.json','index.faiss'):
            (index_copy/name).symlink_to((index_dir/name).resolve())
        (index_copy/'ids.txt').write_text('\n'.join(reversed((index_dir/'ids.txt').read_text().splitlines()))+'\n')
        faults['swapped_source_ids']=rejected(lambda:S.bind_inputs(index_copy,query_dir,True))
        query_copy=temp/'queries';query_copy.mkdir()
        for name in ('bge.npz',):
            (query_copy/name).symlink_to((query_dir/name).resolve())
        for name in ('plan.json','completion.json','bge.json'):
            (query_copy/name).write_bytes((query_dir/name).read_bytes())
        plan=S.B.read(query_copy/'plan.json');plan['model_files_sha256']['bge']['model.safetensors']='different-encoder'
        S.B.save(query_copy/'plan.json',plan)
        for name in ('completion.json','bge.json'):
            row=S.B.read(query_copy/name);row['plan_sha256']=S.B.sha(query_copy/'plan.json');S.B.save(query_copy/name,row)
        complete=S.B.read(query_copy/'completion.json')
        complete['encoders']['bge']['receipt_sha256']=S.B.sha(query_copy/'bge.json')
        S.B.save(query_copy/'completion.json',complete)
        faults['internally_bound_wrong_encoder']=rejected(lambda:S.bind_inputs(index_dir,query_copy,True))
        plan=S.B.read(query_dir/'plan.json');plan['ordered_qids']=list(reversed(plan['ordered_qids']))
        S.B.save(query_copy/'plan.json',plan)
        for name in ('completion.json','bge.json'):
            row=S.B.read(query_dir/name);row['plan_sha256']=S.B.sha(query_copy/'plan.json');S.B.save(query_copy/name,row)
        complete=S.B.read(query_copy/'completion.json');complete['encoders']['bge']['receipt_sha256']=S.B.sha(query_copy/'bge.json')
        S.B.save(query_copy/'completion.json',complete)
        faults['query_id_vector_order_mismatch']=rejected(lambda:S.bind_inputs(index_dir,query_copy,True))
    receipt=dict(status='PASS',checked_utc=datetime.now(timezone.utc).isoformat(),real_prefix_checks=checks,
        mismatch_rejections=faults,labels_read=False,model_calls=0,MPS_calls=0,full_global_search_complete=False,
        source_sha256={str(p):S.B.sha(p) for p in (Path(__file__),Path(S.__file__))},
        scope='Real1024-vector engineering indices only; no routing, relevance fit, benchmark retrieval or QA result')
    S.B.save(base/'DENSE_SEARCH_INDEPENDENT_VALIDATION.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
