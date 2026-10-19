"""Read-only model diagnostic: distinguish input divergence from down-projection divergence."""
import json, pathlib, sys
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
root=pathlib.Path('/home/user/Futaoran/omniaware_v2_selfcheck_20261018')
out=pathlib.Path('/home/user/Futaoran/omniaware_selfcheck_logs_20261018')
sys.path[:0]=[str(root),str(root/'src')]
from scripts import verify_real_model as vrm
from omniaware import pipeline,qwen_eab,paths
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
q=qwen_eab.QwenEAB(str(paths.QWEN_8B),dtype=torch.bfloat16,device='cuda:0')
chunks=[{'cid':'t1','modality':'text','title':'Background','text':'A short placeholder passage.'},
        {'cid':'t2','modality':'text','title':'Other','text':'A second placeholder passage.'}]
enc=q.encode(*pipeline.build_segments({'dataset':'okvqa','question':'What is shown in the picture?','query_images':[]},chunks))
captures={};phase='full';handles=[]
def hook_for(name):
    def hook(module,inputs,output):
        for kind,tensor in [('input',inputs[0]),('output',output)]:
            captures.setdefault(name+':'+kind,{}).setdefault(phase,[]).append(tensor.detach().float().cpu())
    return hook
for name,module in q.model.named_modules():
    if '.language_model.layers.0.mlp.' in name and isinstance(module,torch.nn.Linear):
        handles.append(module.register_forward_hook(hook_for(name)))
with sdpa_kernel(SDPBackend.MATH):
    vrm._last_logits(q,enc,enc.input_ids,enc.mm_token_type_ids)
    phase='segmented'
    q.run(enc,None,1,decode_static=False)
for h in handles:h.remove()
diffs={}
for name,p in captures.items():
    a=p['full'][0];b=torch.cat(p['segmented'],dim=1)
    diffs[name]={'shape':list(a.shape),'max_abs_diff':float((a-b).abs().max()),'different_values':int((a!=b).sum())}
report={'diagnostic_only':True,'dtype':'bfloat16','backend':'math_sdpa','first_layer_mlp':diffs}
(out/'mlp_input_diagnostic.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
