import json,pathlib,datetime
base=pathlib.Path('/home/user/Futaoran/omniaware_v2_format_20261018/runs/fp32_format_20261018')
cache=pathlib.Path('/home/user/Futaoran/omniaware_v2_cache_fp32_format_20261018')
ops=pathlib.Path('/home/user/Futaoran/omniaware_selfcheck_logs_20261018/format_revision/retry1')
r={'at':datetime.datetime.now().astimezone().isoformat(),
   'phase':(ops/'phase.txt').read_text().strip(),'exit':(ops/'validation.exit').read_text().strip() if (ops/'validation.exit').exists() else None,
   'cache':{},'runs':{}}
for p in cache.glob('*.json'):
    r['cache'][p.name]={'entries':len(json.loads(p.read_text())),'bytes':p.stat().st_size,'mtime':p.stat().st_mtime}
for dataset in ['webqa_full','okvqa_gs']:
    d=base/('v2_s1_'+dataset);answers=d/(dataset+'.jsonl');log=d/(dataset+'.log')
    r['runs'][dataset]={'answers':sum(1 for x in answers.open()) if answers.exists() else 0,
                       'log_tail':log.read_text()[-600:] if log.exists() else None}
print(json.dumps(r,ensure_ascii=False),flush=True)
