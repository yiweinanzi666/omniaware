"""Locate the first full/segmented divergence; all modifications are process-local diagnostics."""
import datetime, json, pathlib, sys, time
ROOT=pathlib.Path('/home/user/Futaoran/omniaware_v2_selfcheck_20261018')
OUT=pathlib.Path('/home/user/Futaoran/omniaware_selfcheck_logs_20261018')
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from scripts import verify_real_model as vrm
from omniaware import pipeline,qwen_eab,paths
from omniaware.pipeline import Variant
torch.set_num_threads(4)
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
q=qwen_eab.QwenEAB(str(paths.QWEN_8B),dtype=torch.bfloat16,device='cuda:0')
chunks=[{'cid':'t1','modality':'text','title':'Background','text':'A short placeholder passage.'},
        {'cid':'t2','modality':'text','title':'Other','text':'A second placeholder passage.'}]
enc=q.encode(*pipeline.build_segments({'dataset':'okvqa','question':'What is shown in the picture?','query_images':[]},chunks))
arms={name:Variant(name,1.2,rows=name,form='pos',gate=False).spec(enc,{'t1':1.,'t2':.8},1.,0.,q.n_layers) for name in ('ans1','ans2')}
report={'started':datetime.datetime.now().astimezone().isoformat(),'diagnostic_only':True,'modes':[]}

captures={};phase='full';handles=[]
def hook_for(name):
    def hook(module, inputs, output):
        if isinstance(output,tuple): output=output[0]
        if torch.is_tensor(output) and output.ndim==3:
            captures.setdefault(name,{}).setdefault(phase,[]).append(output.detach().float().cpu())
    return hook
for name,module in q.model.named_modules():
    if '.language_model.layers.0.' in name and isinstance(module,torch.nn.Linear):
        handles.append(module.register_forward_hook(hook_for(name)))
    if name.endswith('embed_tokens') or name.endswith('lm_head'):
        handles.append(module.register_forward_hook(hook_for(name)))
with sdpa_kernel(SDPBackend.MATH):
    vrm._last_logits(q,enc,enc.input_ids,enc.mm_token_type_ids)
    phase='segmented'
    q.run(enc,None,1,decode_static=False)
for h in handles:h.remove()
diffs={}
for name,phases in captures.items():
    if 'full' not in phases or 'segmented' not in phases:continue
    full=phases['full'][0];seg=torch.cat(phases['segmented'],dim=1)
    if full.shape==seg.shape:
        diffs[name]={'max_diff':float((full-seg).abs().max()),'different_values':int((full!=seg).sum()),'shape':list(full.shape)}
report['first_layer_differences']=diffs
del captures
original_linear=torch.nn.Linear.forward
def precise_linear(self,input):
    return F.linear(input.float(),self.weight.float(),self.bias.float() if self.bias is not None else None).to(input.dtype)
for mode in ['bf16_fp32_linear','fp32_model']:
    start=time.time()
    if mode=='bf16_fp32_linear':torch.nn.Linear.forward=precise_linear
    else:
        torch.nn.Linear.forward=original_linear
        q.model.float()
    with sdpa_kernel(SDPBackend.MATH):
        row={'mode':mode,'segmented':vrm.segmented_check(q,enc),'reference':vrm.reference_check(q,enc,arms,steps=1)}
    row['seconds']=time.time()-start
    report['modes'].append(row)
    (OUT/'layer_diagnostic.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(row),flush=True)
torch.nn.Linear.forward=original_linear
report['ended']=datetime.datetime.now().astimezone().isoformat()
(OUT/'layer_diagnostic.json').write_text(json.dumps(report,indent=2))
