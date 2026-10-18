import datetime,json,pathlib,subprocess
base=pathlib.Path('/home/user/Futaoran');logs=base/'omniaware/assets/manifests/v2_run_20261017';stage=base/'omniaware_v2_download_20261017'
out={'time':datetime.datetime.now().astimezone().isoformat(),'downloads':{},'completed_manifests':[],'commands':[]}
for name,rel in {'qwen8b':'cache/qwen3-vl-8b-instruct','reranker':'cache/bge-reranker-v2-m3','nli':'cache/e5_citation_support/deberta_nli','bartscore':'cache/bartscore'}.items():
 files=[p for p in (stage/rel).rglob('*') if p.is_file()]
 out['downloads'][name]={'bytes_on_disk':sum(p.stat().st_size for p in files),'files':len(files),'incomplete_bytes':sum(p.stat().st_size for p in files if p.name.endswith('.incomplete'))}
for p in logs.glob('*verified_manifest.json'):out['completed_manifests'].append(str(p))
for p in sorted(logs.glob('*.command.json')):
 result=p.with_name(p.name.replace('.command.json','.result.json'))
 rec=json.loads((result if result.exists() else p).read_text())
 rec['state']='exited' if result.exists() else 'check_process'
 log=pathlib.Path(rec['log']);rec['log_tail']=log.read_text(errors='replace')[-1000:] if log.exists() else ''
 out['commands'].append(rec)
out['tmux']=subprocess.run(['tmux','list-windows','-t','v2-prep-20261017'],capture_output=True,text=True).stdout
print(json.dumps(out,indent=2))
with (logs/('progress_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json')).open('x') as f:json.dump(out,f,indent=2)
