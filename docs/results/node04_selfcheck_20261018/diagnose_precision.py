"""Diagnostic only: compare numerical backends without altering repository source or thresholds."""
import contextlib
import datetime
import json
import pathlib
import subprocess
import sys
import time
import traceback

ROOT = pathlib.Path('/home/user/Futaoran/omniaware_v2_selfcheck_20261018')
OUT = pathlib.Path('/home/user/Futaoran/omniaware_selfcheck_logs_20261018')
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from scripts import verify_real_model as vrm
from omniaware import pipeline, qwen_eab, paths
from omniaware.pipeline import Variant

torch.set_num_threads(4)
torch.set_float32_matmul_precision('highest')
report = {'started': datetime.datetime.now().astimezone().isoformat(),
          'source': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
          'torch': torch.__version__, 'tolerance': vrm.TOLERANCE, 'diagnostic_only': True, 'modes': []}
q = qwen_eab.QwenEAB(str(paths.QWEN_8B), dtype=torch.bfloat16, device='cuda:0')
chunks = [{'cid': 't1', 'modality': 'text', 'title': 'Background', 'text': 'A short placeholder passage.'},
          {'cid': 't2', 'modality': 'text', 'title': 'Other', 'text': 'A second placeholder passage.'}]
segs, split = pipeline.build_segments({'dataset': 'okvqa', 'question': 'What is shown in the picture?', 'query_images': []}, chunks)
enc = q.encode(segs, split)
arms = {name: Variant(name, 1.2, rows=name, form='pos', gate=False).spec(enc, {'t1': 1., 't2': .8}, 1., 0., q.n_layers)
        for name in ('ans1', 'ans2')}
original_sdpa = qwen_eab._ORIG_SDPA

def fp32_sdpa(module, query, key, value, attention_mask, **kwargs):
    original_dtype = query.dtype
    mask = attention_mask
    if mask is not None and mask.dtype != torch.bool:
        mask = mask.float()
    output, attn = original_sdpa(module, query.float(), key.float(), value.float(), mask, **kwargs)
    return output.to(original_dtype), attn

modes = ['default', 'no_reduced_precision', 'math', 'flash', 'fp32_attention']
for mode in modes:
    start = time.time()
    row = {'mode': mode}
    try:
        if mode != 'default':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        qwen_eab._ORIG_SDPA = fp32_sdpa if mode == 'fp32_attention' else original_sdpa
        backend = (sdpa_kernel(SDPBackend.MATH) if mode in ('math', 'fp32_attention') else
                   sdpa_kernel(SDPBackend.FLASH_ATTENTION) if mode == 'flash' else contextlib.nullcontext())
        with backend:
            row['segmented'] = vrm.segmented_check(q, enc)
            row['reference'] = vrm.reference_check(q, enc, arms, steps=1)
        row['ok'] = row['segmented']['ok'] and row['reference']['ok']
    except Exception:
        row['error'] = traceback.format_exc()
        row['ok'] = False
    finally:
        qwen_eab._ORIG_SDPA = original_sdpa
        qwen_eab.STATE.reset()
        torch.cuda.empty_cache()
    row['seconds'] = time.time() - start
    report['modes'].append(row)
    (OUT / 'precision_diagnostic.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(row), flush=True)
report['ended'] = datetime.datetime.now().astimezone().isoformat()
(OUT / 'precision_diagnostic.json').write_text(json.dumps(report, indent=2), encoding='utf8')
