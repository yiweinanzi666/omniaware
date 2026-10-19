"""Reconstruct the longest multi-image prompt in retained S1; no scoring or answer rewriting."""
import json, pathlib, time, sys
root=pathlib.Path('/home/user/Futaoran/omniaware_v2_selfcheck_20261018')
sys.path[:0]=[str(root),str(root/'src')]
import torch
from omniaware import data, paths, pipeline, signals
from omniaware.qwen_eab import QwenEAB
torch.set_num_threads(4)
old=pathlib.Path('/home/user/Futaoran/omniaware_v2_exec/runs/v2_s1_webqa_full/webqa_full.jsonl')
records=[json.loads(x) for x in old.read_text().splitlines()]
choices=[(a['input']['prompt_tokens'],r['qid'],name,a) for r in records for name,a in r['actions'].items()
         if a.get('available') and a.get('input',{}).get('n_images',0)>1]
_,qid,name,action=max(choices,key=lambda x:(x[0],x[1],x[2]))
examples=data.TRAIN_LOADERS['webqa_full']()
e=next(e for e in examples if e['qid']==qid)
pool={c['cid']:c for c in e['candidates']}
chunks=[pool[cid] for cid in action['chunks']]
del examples,records,choices
start=time.monotonic()
q=QwenEAB(str(paths.QWEN_8B),dtype=torch.float32,max_pixels=262144,max_patches=1)
torch.cuda.reset_peak_memory_stats()
enc=q.encode(*pipeline.build_segments(e,chunks,'default'))
trace=signals.input_trace(enc,q.image_token_id)
assert trace['prompt_sha256']==action['input']['prompt_sha256'], (trace,action['input'])
result=q.run(enc,None,pipeline.MAX_NEW_TOKENS['webqa'],decode_static=True)[0]['static']
torch.cuda.synchronize()
report={'ok':True,'diagnostic_only':True,'selection':'longest recorded multi-image S1 prompt',
        'qid':qid,'action':name,'dtype':str(q.model.dtype),'input':trace,
        'max_pixels':262144,'max_new_tokens':pipeline.MAX_NEW_TOKENS['webqa'],
        'generated_tokens':len(result.tokens),'text':result.text,
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
        'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
        'total_seconds':time.monotonic()-start}
(paths.RUNS/'multiimage_memory_probe.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
