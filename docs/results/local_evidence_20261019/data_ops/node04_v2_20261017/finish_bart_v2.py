import datetime, getpass, hashlib, json, os, pathlib, shlex, time
from remote_ops import connect
ROOT=pathlib.Path(__file__).resolve().parent
p=ROOT/'bart_score.pth'
password=getpass.getpass('SSH password: ')
log=ROOT/'bart_transfer.jsonl'
def record(x):
 x['time']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with log.open('a',encoding='utf8') as f:f.write(json.dumps(x)+'\n')
 print(json.dumps(x),flush=True)
record({'state':'waiting_for_gdown_atomic_completion','local':str(p)})
while not p.is_file():time.sleep(30)
sha=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
record({'state':'local_complete','bytes':p.stat().st_size,'sha256':sha})
target='/home/user/Futaoran/omniaware_v2_download_20261017/cache/bartscore/bart_score.pth'
for attempt in range(1,5):
 try:
  jump,node=connect(password);s=node.open_sftp()
  try:s.stat(target)
  except FileNotFoundError:pass
  else:raise RuntimeError('Final target already exists; stop for hash comparison')
  tmp=target+'.upload_20261017_v2_'+str(attempt)
  with s.open(tmp,'wx') as f:
   f.set_pipelined(True)
   with p.open('rb') as src:
    while data:=src.read(1024*1024):f.write(data)
  _,out,err=node.exec_command('sha256sum '+shlex.quote(tmp))
  remote_sha=out.read().decode().split()[0]
  if remote_sha!=sha:raise RuntimeError('SHA256 mismatch')
  s.rename(tmp,target)
  record({'state':'uploaded_verified','remote':target,'bytes':p.stat().st_size,'sha256':sha})
  cmd="tmux new-window -d -t v2-prep-20261017 -n bartscore-finish \"bash -c 'export PYTHONPATH=/home/user/Futaoran/omniaware_v2_bootstrap_20261017; /home/user/anaconda3/bin/python -u /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/prepare_v3.py bartscore > /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/bartscore_finish.log 2>&1'\""
  _,out,err=node.exec_command(cmd);rc=out.channel.recv_exit_status()
  record({'command':cmd,'exit_code':rc,'stderr':err.read().decode()})
  s.put(str(log),'/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/bart_transfer.jsonl')
  node.close();jump.close();break
 except Exception as e:
  record({'state':'transfer_failed','attempt':attempt,'error':str(e)})
  if 'already exists' in str(e) or 'SHA256' in str(e):raise
  if attempt==4:raise
  time.sleep(30)
