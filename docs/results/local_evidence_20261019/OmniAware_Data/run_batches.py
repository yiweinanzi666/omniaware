from pathlib import Path
import subprocess,sys,os,time,json
R=Path(__file__).resolve().parent;M=R/'manifests';L=R/'logs';env=os.environ.copy();env['PYTHONPATH']=str(R/'tools');env['PYTHONUTF8']='1';env['PYTHONIOENCODING']='utf-8';env['OMNIAWARE_ASSETS']=str(R/'assets')
error=M/'runtime_preparation.done.json'
if error.exists():error.unlink()
procs={};logs=[]
for name in ['download_dpr','download_videorag','finalize_downloads']:
 log=(L/(name+'_supervised.log')).open('a',encoding='utf8');logs.append(log);procs[name]=subprocess.Popen([sys.executable,'-u',str(R/(name+'.py'))],stdout=log,stderr=log,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
finished=set()
while len(finished)<len(procs):
 for name,p in procs.items():
  code=p.poll()
  if code is not None and name not in finished:
   finished.add(name)
   if code:
    error.write_text(json.dumps({'status':'failed','error':f'{name} exited {code}; see logs/{name}_supervised.log'}));raise SystemExit(code)
 time.sleep(5)
for log in logs:log.close()
print('ALL_BATCHES_COMPLETE',flush=True)
