"""Fit frozen per-modality top5 ECDFs and merge the WebQA Phase1 sources."""
import argparse
from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time

from . import search_dense_queries as S

HERE=S.B.HERE
PROTOCOL=HERE/'DENSE_ECDF_PROTOCOL.md'
RULE='webqa-global-bge-clip-top5-ecdf-v1'
KINDS=('bge','clip')
read,sha,save=S.B.read,S.B.sha,S.B.save


def load_retrieval(directory,kind,query_directory,engineering_prefix):
    directory=Path(directory);plan=read(directory/'plan.json');complete=read(directory/'completion.json')
    if (plan['kind']!=kind or plan['engineering_prefix'] is not engineering_prefix
            or Path(plan['query_directory']).resolve()!=Path(query_directory).resolve()):
        raise ValueError('Retrieval encoder, cohort or engineering scope differs')
    binding,ids,qids,_=S.bind_inputs(plan['index_directory'],plan['query_directory'],engineering_prefix)
    if binding!=plan or complete['status']!='MODAL_TOP5_COMPLETE' or complete['plan_sha256']!=sha(directory/'plan.json'):
        raise ValueError('Retrieval completion/input binding changed')
    if complete['rankings_sha256']!=sha(directory/'rankings.jsonl'):
        raise ValueError('Retrieval rankings changed')
    rows=[json.loads(line) for line in (directory/'rankings.jsonl').read_text().splitlines()]
    if [r['qid'] for r in rows]!=qids or len(rows)!=complete['questions']:
        raise ValueError('Retrieval query IDs/order changed')
    for row in rows:
        candidates=row['candidates']
        if (row['encoder']!=kind or [c['rank'] for c in candidates]!=[1,2,3,4,5]
                or len({c['source_id'] for c in candidates})!=5):
            raise ValueError('Retrieval candidate count/rank/identity differs')
        for c in candidates:
            if (type(c['index_row']) is not int or not 0<=c['index_row']<len(ids)
                    or ids[c['index_row']]!=c['source_id'] or not math.isfinite(c['cosine']) or abs(c['cosine'])>1.00001):
                raise ValueError('Retrieval candidate is not bound to a valid source row/score')
        if candidates!=sorted(candidates,key=lambda c:(-c['cosine'],c['source_id'])):
            raise ValueError('Retrieval score order differs')
    identity=dict(directory=str(directory.resolve()),plan_sha256=sha(directory/'plan.json'),
        completion_sha256=sha(directory/'completion.json'),rankings_sha256=complete['rankings_sha256'],
        index_plan_sha256=plan['index_plan_sha256'],index_sha256=plan['index_sha256'],
        ids_sha256=plan['ids_sha256'],availability=plan['availability'],vectors=plan['vectors'])
    return rows,identity


def percentile(values,score):
    return bisect_right(values,score)/len(values)


def select(qid,candidates,pools):
    scored=[]
    for kind in KINDS:
        for c in candidates[kind]:
            tie=hashlib.sha256(('phase1-webqa-dense-ecdf-v1:'+qid+'\0'+c['source_id']).encode()).hexdigest()
            scored.append(dict(source_id=c['source_id'],index_row=c['index_row'],encoder=kind,
                modality='text' if kind=='bge' else 'image',modality_rank=c['rank'],cosine=c['cosine'],
                r_j=percentile(pools[kind],c['cosine']),tie_hash=tie))
    scored.sort(key=lambda c:(-c['r_j'],c['modality_rank'],c['tie_hash'],c['source_id']))
    return dict(qid=qid,candidates=scored,selected_sources=[dict(c,rank=i) for i,c in enumerate(scored[:5],1)])


def freeze(path,value):
    if path.exists():
        if read(path)!=value:raise ValueError(f'Frozen dense calibration artifact changed: {path}')
    else:save(path,value)


def sources():
    return {str(p):sha(p) for p in (PROTOCOL,Path(__file__),Path(S.__file__),Path(S.B.__file__))}


def fit(bge,clip,output,engineering_prefix=False):
    started=time.perf_counter();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with (output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        loaded={kind:load_retrieval(path,kind,HERE/'webqa_calibration500_queries_cpu',engineering_prefix)
                for kind,path in zip(KINDS,(bge,clip),strict=True)}
        qids=[r['qid'] for r in loaded['bge'][0]]
        if len(qids)!=500 or qids!=[r['qid'] for r in loaded['clip'][0]]:
            raise ValueError('ECDF requires the same500 calibration questions in both modalities')
        plan=dict(rule=RULE,engineering_prefix=engineering_prefix,calibration_qids=qids,
            retrievals={kind:loaded[kind][1] for kind in KINDS},source_sha256=sources(),labels_read=False,
            fit_population='All five retrieved query-document scores per calibration query and encoder;2500 per encoder')
        freeze(output/'plan.json',plan)
        if (output/'completion.json').exists():
            receipt=read(output/'completion.json')
            if receipt['plan_sha256']!=sha(output/'plan.json') or receipt['calibration_sha256']!=sha(output/'calibration.json'):
                raise ValueError('ECDF completion binding changed')
            return dict(receipt,new_fit=False)
        pools={kind:sorted(c['cosine'] for r in loaded[kind][0] for c in r['candidates']) for kind in KINDS}
        calibration=dict(rule=RULE,engineering_prefix=engineering_prefix,plan_sha256=sha(output/'plan.json'),
                         calibration_qids=qids,pools=pools,counts={k:len(v) for k,v in pools.items()})
        freeze(output/'calibration.json',calibration)
        receipt=dict(status='MODAL_TOP5_ECDF_COMPLETE',plan_sha256=sha(output/'plan.json'),
            calibration_sha256=sha(output/'calibration.json'),questions=500,counts=calibration['counts'],
            engineering_prefix=engineering_prefix,wall_seconds=time.perf_counter()-started,
            model_calls=0,MPS_calls=0,labels_read=False,completed_utc=datetime.now(timezone.utc).isoformat())
        save(output/'completion.json',receipt)
        return dict(receipt,new_fit=True)


def merge(calibration_directory,bge,clip,output,engineering_prefix=False):
    started=time.perf_counter();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with (output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cal_dir=Path(calibration_directory);cal_plan=read(cal_dir/'plan.json');cal=read(cal_dir/'calibration.json')
        receipt=read(cal_dir/'completion.json')
        if (cal_plan['source_sha256']!=sources() or cal['rule']!=RULE or cal['engineering_prefix'] is not engineering_prefix
                or cal['plan_sha256']!=sha(cal_dir/'plan.json') or receipt['plan_sha256']!=cal['plan_sha256']
                or receipt['calibration_sha256']!=sha(cal_dir/'calibration.json')):
            raise ValueError('ECDF rule/input/completion binding changed')
        loaded={kind:load_retrieval(path,kind,HERE/'webqa_train256_queries_cpu',engineering_prefix)
                for kind,path in zip(KINDS,(bge,clip),strict=True)}
        qids=[r['qid'] for r in loaded['bge'][0]]
        if len(qids)!=256 or qids!=[r['qid'] for r in loaded['clip'][0]] or set(qids)&set(cal['calibration_qids']):
            raise ValueError('Expected the same256 target questions, disjoint from calibration')
        for kind in KINDS:
            if any(loaded[kind][1][key]!=cal_plan['retrievals'][kind][key]
                   for key in ('index_plan_sha256','index_sha256','ids_sha256','availability','vectors')):
                raise ValueError('Calibration and target use different retrieval corpora')
            values=cal['pools'][kind]
            if len(values)!=2500 or cal['counts'][kind]!=2500 or values!=sorted(values) or not all(math.isfinite(v) for v in values):
                raise ValueError('Invalid calibration score population')
        plan=dict(rule=RULE,engineering_prefix=engineering_prefix,source_sha256=sources(),
            calibration_directory=str(cal_dir.resolve()),calibration_plan_sha256=sha(cal_dir/'plan.json'),
            calibration_sha256=sha(cal_dir/'calibration.json'),calibration_completion_sha256=sha(cal_dir/'completion.json'),
            retrievals={kind:loaded[kind][1] for kind in KINDS},questions=256,ordered_qids=qids,
            selection='ECDF descending, within-modality rank, fixed query/source SHA, source ID; total5 from10',labels_read=False)
        freeze(output/'plan.json',plan)
        if (output/'completion.json').exists():
            complete=read(output/'completion.json')
            if complete['plan_sha256']!=sha(output/'plan.json') or complete['selected_sha256']!=sha(output/'selected.jsonl'):
                raise ValueError('Completed source selection changed')
            return dict(complete,new_merge=False)
        rows=[select(qid,{kind:loaded[kind][0][i]['candidates'] for kind in KINDS},cal['pools'])
              for i,qid in enumerate(qids)]
        target=output/'selected.jsonl'
        content=''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows)
        if target.exists():
            if target.read_text()!=content:raise ValueError('Preserve existing source selection without a matching completion')
        else:
            temporary=output/'selected.jsonl.tmp'
            temporary.write_text(content)
            temporary.replace(target)
        complete=dict(status='JOINT_TOP5_COMPLETE',plan_sha256=sha(output/'plan.json'),selected_sha256=sha(output/'selected.jsonl'),
            questions=256,selected_modality_counts=dict(Counter(c['modality'] for r in rows for c in r['selected_sources'])),
            engineering_prefix=engineering_prefix,wall_seconds=time.perf_counter()-started,
            model_calls=0,MPS_calls=0,labels_read=False,completed_utc=datetime.now(timezone.utc).isoformat(),
            scope='Source selection only; no U, answer generation, accuracy or evidence-utility claim')
        save(output/'completion.json',complete)
        return dict(complete,new_merge=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('fit','merge'))
    parser.add_argument('--bge',type=Path,required=True)
    parser.add_argument('--clip',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--engineering-prefix',action='store_true')
    args=parser.parse_args()
    if args.action=='merge' and args.calibration is None:parser.error('merge requires --calibration')
    result=(fit(args.bge,args.clip,args.output,args.engineering_prefix) if args.action=='fit' else
            merge(args.calibration,args.bge,args.clip,args.output,args.engineering_prefix))
    print(json.dumps(result,ensure_ascii=False),flush=True)
