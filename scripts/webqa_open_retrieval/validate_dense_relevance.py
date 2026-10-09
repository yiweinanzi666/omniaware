"""Check real saved retrievals with independent counting and isolated bad bindings."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import numpy as np

from . import calibrate_dense_relevance as C


def rejected(operation):
    try:operation()
    except ValueError as exc:return str(exc)
    raise AssertionError('Mismatched calibration/retrieval accepted')


def main():
    base=C.HERE;raw=base/'dense_top5_engineering';cal=base/'dense_relevance_engineering/calibration'
    output=base/'dense_relevance_engineering/train256'
    fit=C.fit(raw/'bge_calibration500',raw/'clip_calibration500',cal,True)
    merged=C.merge(cal,raw/'bge_train256',raw/'clip_train256',output,True)
    rows=[json.loads(line) for line in (output/'selected.jsonl').read_text().splitlines()]
    reference={kind:[json.loads(line) for line in (raw/f'{kind}_train256/rankings.jsonl').read_text().splitlines()]
               for kind in C.KINDS}
    calibration={kind:[json.loads(line) for line in (raw/f'{kind}_calibration500/rankings.jsonl').read_text().splitlines()]
                 for kind in C.KINDS}
    pools={kind:np.asarray([c['cosine'] for r in calibration[kind] for c in r['candidates']]) for kind in C.KINDS}
    maximum=0.;selected=0
    for i,row in enumerate(rows):
        assert all(reference[k][i]['qid']==row['qid'] for k in C.KINDS)
        expected=[]
        for kind in C.KINDS:
            for source in reference[kind][i]['candidates']:
                relevance=float(np.count_nonzero(pools[kind]<=source['cosine'])/pools[kind].size)
                tie=hashlib.sha256(('phase1-webqa-dense-ecdf-v1:'+row['qid']+'\0'+source['source_id']).encode()).hexdigest()
                expected.append((relevance,source['rank'],tie,source['source_id']))
        expected.sort(key=lambda x:(-x[0],*x[1:]))
        assert [r['source_id'] for r in row['candidates']]==[r[3] for r in expected]
        assert [r['source_id'] for r in row['selected_sources']]==[r[3] for r in expected[:5]]
        assert [r['rank'] for r in row['selected_sources']]==[1,2,3,4,5]
        for actual,wanted in zip(row['candidates'],expected,strict=True):maximum=max(maximum,abs(actual['r_j']-wanted[0]))
        selected+=len(row['selected_sources'])
    assert maximum==0 and len(rows)==256 and selected==1280
    files=[p for parent in (cal,output) for p in parent.iterdir() if p.name!='worker.lock']
    before={str(p):C.sha(p) for p in files}
    with patch.object(C,'select',side_effect=AssertionError('Completed merge reselected sources')):
        assert not C.fit(raw/'bge_calibration500',raw/'clip_calibration500',cal,True)['new_fit']
        assert not C.merge(cal,raw/'bge_train256',raw/'clip_train256',output,True)['new_merge']
    assert before=={str(p):C.sha(p) for p in files}
    faults=dict(prefix_as_full=rejected(lambda:C.load_retrieval(raw/'bge_calibration500','bge',base/'webqa_calibration500_queries_cpu',False)),
        wrong_encoder=rejected(lambda:C.load_retrieval(raw/'clip_calibration500','bge',base/'webqa_calibration500_queries_cpu',True)),
        evaluation_used_as_calibration=rejected(lambda:C.load_retrieval(raw/'bge_train256','bge',base/'webqa_calibration500_queries_cpu',True)))
    with tempfile.TemporaryDirectory() as folder:
        wrong=Path(folder)/'cal';shutil.copytree(cal,wrong)
        plan=C.read(wrong/'plan.json');plan['retrievals']['bge']['index_sha256']='different-corpus';C.save(wrong/'plan.json',plan)
        mapping=C.read(wrong/'calibration.json');mapping['plan_sha256']=C.sha(wrong/'plan.json');C.save(wrong/'calibration.json',mapping)
        receipt=C.read(wrong/'completion.json');receipt['plan_sha256']=mapping['plan_sha256'];receipt['calibration_sha256']=C.sha(wrong/'calibration.json');C.save(wrong/'completion.json',receipt)
        faults['internally_rebound_different_corpus']=rejected(lambda:C.merge(wrong,raw/'bge_train256',raw/'clip_train256',Path(folder)/'out',True))
    proof=dict(status='PASS_ENGINEERING_PREFIX_ONLY',checked_utc=datetime.now(timezone.utc).isoformat(),
        calibration_queries=500,calibration_scores_per_encoder=2500,target_queries=len(rows),
        transformed_candidate_scores=2560,selected_sources=selected,maximum_percentile_error=maximum,
        all_selection_orders_match=True,selected_modality_counts=merged['selected_modality_counts'],
        selected_image_count_histogram=dict(Counter(sum(c['modality']=='image' for c in r['selected_sources']) for r in rows)),
        reentry_new_fits=0,reentry_new_merges=0,reentry_byte_changes=0,mismatch_rejections=faults,
        model_calls=0,MPS_calls=0,labels_read=False,full_corpus_calibration_complete=False,
        fit_receipt_sha256=C.sha(cal/'completion.json'),merge_receipt_sha256=C.sha(output/'completion.json'),
        source_sha256={str(p):C.sha(p) for p in (Path(__file__),Path(C.__file__),C.PROTOCOL,
                         base/'test_dense_relevance.py',base/'DENSE_RELEVANCE_TESTS.log')},
        scope='Actual1024-vector prefix retrievals only; no recall/accuracy/utility interpretation or formal calibrated scores')
    C.save(base/'DENSE_RELEVANCE_INDEPENDENT_VALIDATION.json',proof)
    print(json.dumps(proof,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
