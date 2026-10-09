"""Encode the frozen mixed500 calibration queries for both full dense indices."""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch
import transformers

from experiments.eab_first_token.multimodal_retrieval import encode_bge
from experiments.eab_first_token.prepare_webqa import QUERY_INSTRUCTION
from experiments.eab_first_token.score_visual_candidates import normalize_features
from .encode_global_text import HERE, ROOT, sha, save

COHORT = HERE/'webqa_calibration500/queries.jsonl'
OUTPUT = HERE/'webqa_calibration500_queries_cpu'


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selection = json.loads((COHORT.parent/'plan.json').read_text())
        if sha(COHORT) != selection['output_sha256']['queries.jsonl']:
            raise ValueError('Frozen calibration query file changed')
        rows = [json.loads(line) for line in COHORT.read_text().splitlines()]
        if len(rows) != 500 or len({r['qid'] for r in rows}) != 500:
            raise ValueError('Expected500 distinct calibration questions')
        if any(set(r) != {'qid','question','stratum','source_split'} or r['source_split'] != 'train' for r in rows):
            raise ValueError('Expected label-free train query schema')
        questions = [r['question'] for r in rows]
        ids = np.asarray([r['qid'] for r in rows])
        corpus_plans = {'bge':HERE/'bge_text_cpu/plan.json', 'clip':HERE/'clip_images_cpu_wave0/plan.json'}
        models = {'bge':ROOT/'cache/bge-large-en-v1.5', 'clip':ROOT/'cache/clip-vit-large-patch14'}
        expected = {name:json.loads(p.read_text())['model_files_sha256'] for name,p in corpus_plans.items()}
        for name, assets in expected.items():
            if {asset:sha(models[name]/asset) for asset in assets} != assets:
                raise ValueError(f'{name} model differs from corpus encoder')
        plan = dict(cohort_sha256=sha(COHORT), selection_plan_sha256=sha(COHORT.parent/'plan.json'), questions=500,
            ordered_qids=ids.tolist(), corpus_plan_sha256={n:sha(p) for n,p in corpus_plans.items()},
            model_files_sha256=expected, device='cpu', dtype='float32', attention='sdpa', threads=2, batch_size=4,
            bge_input=QUERY_INSTRUCTION+'<exact question>', bge_max_length=512,
            clip_input='Exact question without added template', clip_max_length=77,
            pooling={'bge':'CLS L2 normalized','clip':'text projection L2 normalized'},
            torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
            source_sha256={str(p):sha(p) for p in (Path(__file__), HERE/'encode_global_text.py',
                ROOT/'multimodal_retrieval.py', ROOT/'prepare_webqa.py', ROOT/'score_visual_candidates.py')},
            labels_read=False, MPS_calls=0, scope='Both encoders for every query; no stratum-based routing or accuracy scoring')
        if (OUTPUT/'plan.json').exists():
            if json.loads((OUTPUT/'plan.json').read_text()) != plan:
                raise ValueError('Calibration encoding plan changed')
        else:
            save(OUTPUT/'plan.json',plan)
        plan_sha, receipts, new_loads = sha(OUTPUT/'plan.json'), {}, 0
        for name, dimension, max_length in (('bge',1024,512),('clip',768,77)):
            path, receipt_path = OUTPUT/f'{name}.npz', OUTPUT/f'{name}.json'
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if receipt['plan_sha256'] != plan_sha or receipt['array_sha256'] != sha(path):
                    raise ValueError(f'{name} completed array binding changed')
                with np.load(path) as cached:
                    if not np.array_equal(cached['qids'],ids) or cached['vectors'].shape != (500,dimension):
                        raise ValueError(f'{name} query identities differ')
                receipts[name] = receipt
                continue
            if path.exists():
                raise ValueError(f'Array exists without completion receipt: {path}')
            started = time.perf_counter()
            tokenizer = transformers.AutoTokenizer.from_pretrained(models[name],local_files_only=True)
            cls = transformers.AutoModel if name=='bge' else transformers.CLIPModel
            model = cls.from_pretrained(models[name],local_files_only=True,dtype=torch.float32,attn_implementation='sdpa').eval()
            new_loads += 1
            loaded = time.perf_counter()
            texts = [QUERY_INSTRUCTION+q for q in questions] if name=='bge' else questions
            raw_lengths = [len(tokenizer(t,truncation=False)['input_ids']) for t in texts]
            with torch.inference_mode():
                if name=='bge':
                    vectors,lengths = encode_bge(texts,tokenizer,model,4)
                    scalar,_ = encode_bge(texts[:1],tokenizer,model,1)
                else:
                    batches,lengths = [],[]
                    for start in range(0,500,4):
                        batch=tokenizer(texts[start:start+4],padding=True,truncation=True,max_length=77,return_tensors='pt')
                        lengths.extend(batch['attention_mask'].sum(1).tolist())
                        batches.append(normalize_features(model.get_text_features(**batch).pooler_output))
                    vectors=torch.cat(batches)
                    batch=tokenizer(texts[:1],truncation=True,max_length=77,return_tensors='pt')
                    scalar=normalize_features(model.get_text_features(**batch).pooler_output)
            error = float((vectors[0]-scalar[0]).abs().max())
            array = vectors.numpy()
            norm_error = float(np.abs(np.linalg.norm(array,axis=1)-1).max())
            if array.shape != (500,dimension) or not np.isfinite(array).all() or norm_error>2e-6 or error>1e-5:
                raise ValueError(f'{name} numerical/query batch validation failed')
            temporary=path.with_suffix('.npz.tmp')
            with temporary.open('wb') as stream:
                np.savez(stream,vectors=array,qids=ids,input_lengths=np.asarray(lengths,dtype=np.int32),
                         untruncated_lengths=np.asarray(raw_lengths,dtype=np.int32))
            temporary.replace(path)
            receipt=dict(status='QUERY_ENCODING_COMPLETE',plan_sha256=plan_sha,array_sha256=sha(path),
                queries=500,dimension=dimension,encoding_forwards=125,validation_forwards=1,
                model_load_seconds=loaded-started,encoding_validation_write_seconds=time.perf_counter()-loaded,
                truncated_queries=sum(n>max_length for n in raw_lengths),max_norm_error=norm_error,
                scalar_vs_batch_max_absolute_error=error,labels_read=False,MPS_calls=0,
                completed_utc=datetime.now(timezone.utc).isoformat())
            save(receipt_path,receipt);receipts[name]=receipt
            print(json.dumps(dict(encoder=name,**receipt)),flush=True)
            del model,tokenizer,vectors,scalar
        completion=dict(status='CALIBRATION_QUERY_VECTORS_COMPLETE',plan_sha256=plan_sha,
            encoders={name:dict(receipt_sha256=sha(OUTPUT/f'{name}.json'),array_sha256=r['array_sha256']) for name,r in receipts.items()},
            labels_read=False,MPS_calls=0,scope='Query encodings only; full-index retrieval and score calibration pending')
        if (OUTPUT/'completion.json').exists():
            if json.loads((OUTPUT/'completion.json').read_text()) != completion:
                raise ValueError('Completed query receipt changed')
        else:
            save(OUTPUT/'completion.json',completion)
        print(json.dumps(dict(status=completion['status'],new_model_loads=new_loads)))


if __name__=='__main__':
    main()
