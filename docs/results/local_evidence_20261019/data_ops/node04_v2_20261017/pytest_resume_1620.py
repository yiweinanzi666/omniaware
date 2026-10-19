import datetime,json,os,pathlib,subprocess,sys,hashlib
L=pathlib.Path('/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017')
R=pathlib.Path('/home/user/Futaoran/omniaware_v2_exec')
os.environ.update(OMNIAWARE_ASSETS='/home/user/Futaoran/omniaware_v2_assets_20261017_resume',OMNIAWARE_CACHE='/home/user/Futaoran/omniaware_v2_cache_20261017_resume',OMNIAWARE_WEBQA_OPEN_DIR='/home/user/Futaoran/omniaware/assets/webqa_open',NO_PROXY='localhost,127.0.0.1,::1',no_proxy='localhost,127.0.0.1,::1',CUDA_VISIBLE_DEVICES='')
cmd=[sys.executable,'-m','pytest','-p','no:warnings']
r={'start':datetime.datetime.now().astimezone().isoformat(),'command':cmd,'cwd':str(R),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()}
with (L/'pytest_resume_1620.log').open('x') as f:
    r['exit_code']=subprocess.run(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT).returncode
r['end']=datetime.datetime.now().astimezone().isoformat()
r['log_sha256']=hashlib.sha256((L/'pytest_resume_1620.log').read_bytes()).hexdigest()
with (L/'pytest_resume_1620.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r),flush=True)
sys.exit(r['exit_code'])
