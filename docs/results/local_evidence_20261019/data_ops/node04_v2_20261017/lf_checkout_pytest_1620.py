import pathlib,subprocess,json,datetime,os,sys,hashlib,traceback
B=pathlib.Path('/home/user/Futaoran');O=B/'omniaware_v2_exec';R=B/'omniaware_v2_exec_lf';L=B/'omniaware/assets/manifests/v2_run_20261017'
r={'start':datetime.datetime.now().astimezone().isoformat(),'commands':[],'original_checkout_preserved':True}
def run(cmd,cwd,log):
    rec={'command':cmd,'cwd':str(cwd),'start':datetime.datetime.now().astimezone().isoformat(),'log':str(L/log)}
    with (L/log).open('x') as f:rec['exit_code']=subprocess.run(cmd,cwd=cwd,stdout=f,stderr=subprocess.STDOUT).returncode
    rec['end']=datetime.datetime.now().astimezone().isoformat();rec['log_sha256']=hashlib.sha256((L/log).read_bytes()).hexdigest();r['commands'].append(rec)
    if rec['exit_code']:raise RuntimeError(str(rec))
try:
    assert not R.exists()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=O,text=True).strip()
    assert commit=='7ccf676e9804b2dfdf571d5b94fe8c9aba843acb'
    run(['git','-c','core.autocrlf=false','worktree','add','--detach',str(R),commit],O,'lf_checkout_1620.log')
    (R/'external').symlink_to(O/'external',target_is_directory=True)
    checks={}
    for n,h in [('eval_1022.py','6862a280454b576062695724b3417c9e4ce43af5447596ff4efca15d8d1e3f5c'),('eval_webqa.md','c6dfcb9a7f8aed27c393e3d1296e038f37f10483de66b771963770482fea1ab3')]:
        checks[n]=hashlib.sha256((R/'third_party/webqa'/n).read_bytes()).hexdigest();assert checks[n]==h
    r['official_source_hashes']=checks;r['commit']=commit
    run([sys.executable,'-m','pip','install','--no-deps','--no-build-isolation','-e',str(R)],R,'lf_editable_install_1620.log')
    os.environ.update(OMNIAWARE_ASSETS=str(B/'omniaware_v2_assets_20261017_resume'),OMNIAWARE_CACHE=str(B/'omniaware_v2_cache_20261017_resume'),OMNIAWARE_WEBQA_OPEN_DIR=str(B/'omniaware/assets/webqa_open'),NO_PROXY='localhost,127.0.0.1,::1',no_proxy='localhost,127.0.0.1,::1',CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(R/'src'))
    with (L/'env_resume_lf_1620.sh').open('x') as f:f.write((L/'env_resume_1608.sh').read_text()+'export PYTHONPATH='+str(R/'src')+'\n')
    run([sys.executable,'-m','pytest','-p','no:warnings'],R,'pytest_resume_lf_1620.log')
    r['ok']=True
except Exception:r['ok']=False;r['traceback']=traceback.format_exc()
r['end']=datetime.datetime.now().astimezone().isoformat()
with (L/'lf_checkout_pytest_1620.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r),flush=True)
sys.exit(0 if r['ok'] else 1)
