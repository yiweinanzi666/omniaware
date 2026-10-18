"""Inspect retained S1 provenance without changing any experiment file."""
import json,pathlib,hashlib
base=pathlib.Path('/home/user/Futaoran/omniaware_v2_exec/runs')
report={}
for dataset in ['webqa_full','okvqa_gs']:
    root=base/('v2_s1_'+dataset)
    envpath=root/(dataset+'.env.ndjson')
    env=[json.loads(s) for s in envpath.read_text().splitlines()]
    prompts=[json.loads(s) for s in (root/(dataset+'.prompts.ndjson')).read_text().splitlines()]
    log=(root/(dataset+'.log')).read_text()
    report[dataset]={'env':[{k:e.get(k) for k in ['git_commit','git_dirty','model','gpus','python','torch','transformers']} for e in env],
       'prompt_rows':len(prompts),'prompt_examples':prompts[:3],
       'log_has_env':'[env]' in log,'log_has_done':'[done]' in log,
       'format':json.loads((root/(dataset+'.answer_format.json')).read_text()),
       'completion':json.loads((root/(dataset+'.merged.json')).read_text()),
       'sha256':{p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in root.iterdir() if p.is_file() and p.suffix!='.lock'}}
dest=pathlib.Path('/home/user/Futaoran/omniaware_selfcheck_logs_20261018/s1_provenance_review.json')
dest.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(str(dest))
