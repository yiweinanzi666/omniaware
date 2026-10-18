import datetime,json,os,pathlib,subprocess,time
ROOT=pathlib.Path('/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017')
jobs={'qwen8b':ROOT/'qwen8b_retry1_exit_code.txt','reranker':ROOT/'reranker_retry1_exit_code.txt'}
attempts={x:0 for x in jobs}
while jobs:
 for name,exit_path in list(jobs.items()):
  if (ROOT/(name+'_verified_manifest.json')).exists():del jobs[name];continue
  if not exit_path.exists():continue
  if exit_path.read_text().strip()=='0':del jobs[name];continue
  consoles=list(ROOT.glob(name+'*console.log'))
  if any('hash conflict' in p.read_text(errors='replace') for p in consoles):
   print(name+': HASH CONFLICT; STOP',flush=True);del jobs[name];continue
  if attempts[name]>=3:
   print(name+': retries exhausted; STOP',flush=True);del jobs[name];continue
  attempts[name]+=1;tag=name+'_network_retry'+str(attempts[name]);end=ROOT/(tag+'_exit.txt')
  cmd=f"export PYTHONPATH=/home/user/Futaoran/omniaware_v2_bootstrap_20261017; /home/user/anaconda3/bin/python -u {ROOT}/prepare_v3.py {name} > {ROOT}/{tag}_console.log 2>&1; echo $? > {end}"
  rec={'time':datetime.datetime.now().astimezone().isoformat(),'command':cmd}
  print(json.dumps(rec),flush=True)
  subprocess.run(['tmux','new-window','-d','-t','v2-prep-20261017','-n',tag,'bash -c '+__import__('shlex').quote(cmd)],check=True)
  jobs[name]=end
 if jobs:time.sleep(60)
