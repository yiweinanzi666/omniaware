import pathlib,os,sys,json,subprocess,datetime,hashlib,time
B=pathlib.Path('/home/user/Futaoran');L=B/'omniaware/assets/manifests/v2_run_20261017';R=B/'omniaware_v2_exec_lf'
p=L/'publish_webqa_1631.json'
for _ in range(240):
    if p.exists():break
    time.sleep(5)
assert p.exists() and json.loads(p.read_text())['ok'],'WebQA acceptance incomplete or failed'
os.environ.update(OMNIAWARE_ASSETS=str(B/'omniaware_v2_assets_20261017_resume'),OMNIAWARE_CACHE=str(B/'omniaware_v2_cache_20261017_resume'),OMNIAWARE_WEBQA_OPEN_DIR=str(B/'omniaware/assets/webqa_open'),PYTHONPATH=str(R/'src'),NO_PROXY='localhost,127.0.0.1,::1',no_proxy='localhost,127.0.0.1,::1')
cmd=[sys.executable,'-m','scripts.setup.check_assets'];r={'start':datetime.datetime.now().astimezone().isoformat(),'command':cmd,'cwd':str(R)}
with (L/'check_assets_1640.log').open('x') as f:r['exit_code']=subprocess.run(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT).returncode
r['end']=datetime.datetime.now().astimezone().isoformat();r['log_sha256']=hashlib.sha256((L/'check_assets_1640.log').read_bytes()).hexdigest()
with (L/'check_assets_1640.json').open('x') as f:json.dump(r,f,indent=2)
with (L/'pip_freeze_1640.txt').open('x') as f:subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
print(json.dumps(r),flush=True)
