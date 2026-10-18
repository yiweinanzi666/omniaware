"""Operations only; no changes to experiment code or frozen settings."""
import datetime, hashlib, json, os, pathlib, subprocess, sys, time

BASE=pathlib.Path('/home/user/Futaoran')
REPO=BASE/'omniaware_v2_exec'
ASSETS=BASE/'omniaware/assets'
LOG=ASSETS/'manifests/v2_run_20261017'
STAGE=BASE/'omniaware_v2_download_20261017'
ENV=BASE/'omniaware_v2_python311'
LOG.mkdir(parents=True,exist_ok=True)
os.environ.update(HTTPS_PROXY='http://127.0.0.1:18761', HTTP_PROXY='http://127.0.0.1:18761', HF_ENDPOINT='https://hf-mirror.com', HF_HUB_DISABLE_XET='1', PIP_INDEX_URL='https://pypi.org/simple', PIP_DISABLE_PIP_VERSION_CHECK='1')
os.environ.update(HF_HUB_DOWNLOAD_TIMEOUT='120', HF_HUB_ETAG_TIMEOUT='60')

def run(cmd,tag,env=None,cwd=REPO,check=True):
    stamp=datetime.datetime.now().strftime('%H%M%S_%f'); path=LOG/(tag+'_'+stamp+'.log')
    rec={'command':cmd,'cwd':str(cwd),'start':datetime.datetime.now(datetime.timezone.utc).isoformat(),'log':str(path)}
    print(json.dumps(rec),flush=True)
    with path.open('x') as f:
        p=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
        rec['pid']=p.pid; (LOG/(tag+'_'+stamp+'.command.json')).write_text(json.dumps(rec,indent=2))
        rc=p.wait()
    rec.update(exit_code=rc,end=datetime.datetime.now(datetime.timezone.utc).isoformat(),bytes=path.stat().st_size,sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest())
    (LOG/(tag+'_'+stamp+'.result.json')).write_text(json.dumps(rec,indent=2))
    print(json.dumps(rec),flush=True)
    if check and rc:raise RuntimeError(f'{tag}: exit {rc}; see {path}')
    return rc

def publish(name):
    mapping={'qwen8b':'cache/qwen3-vl-8b-instruct','reranker':'cache/bge-reranker-v2-m3','nli':'cache/e5_citation_support/deberta_nli/b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7','bartscore':'cache/bartscore'}
    src=STAGE/mapping[name];dst=ASSETS/mapping[name]
    records=[]
    for p in sorted(src.rglob('*')):
        if not p.is_file() or '.cache' in p.relative_to(src).parts:continue
        target=dst/p.relative_to(src);sha=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
        if target.exists() and hashlib.file_digest(target.open('rb'),'sha256').hexdigest()!=sha:
            raise RuntimeError(f'Existing file hash conflict, untouched: {target}')
        records.append({'source':str(p),'path':str(target),'bytes':p.stat().st_size,'sha256':sha})
    for rec in records:
        target=pathlib.Path(rec['path']);target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():os.link(rec['source'],target)
    (LOG/(name+'_verified_manifest.json')).write_text(json.dumps(records,indent=2))
    print('PUBLISHED '+name,flush=True)

def models(name):
    env=os.environ.copy();env['OMNIAWARE_ASSETS']=str(STAGE)
    names=['bart_cnn_config','bartscore'] if name=='bartscore' else [name]
    run([sys.executable,'-m','scripts.setup.download_models','--only',*names],name,env=env)
    publish(name)

def environment():
    os.environ.pop('PYTHONPATH',None)
    tools=BASE/'omniaware_v2_tools_20261017'
    run([sys.executable,'-m','pip','install','--target',str(tools),'uv'],'install_uv')
    uv=str(tools/'bin/uv')
    run([uv,'python','install','3.11','--install-dir',str(BASE/'omniaware_v2_runtime_20261017')],'python311')
    runtimes=sorted((BASE/'omniaware_v2_runtime_20261017').glob('cpython-3.11*/bin/python3.11'))
    if not runtimes:raise RuntimeError('Python 3.11 runtime missing')
    run([uv,'venv','--seed','--python',str(runtimes[-1]),str(ENV)],'create_venv')
    py=str(ENV/'bin/python')
    run([py,'-m','pip','install','torch','torchvision','--index-url','https://download.pytorch.org/whl/cu128'],'install_torch')
    run([py,'-m','pip','install','-e','.[all]'],'install_all')
    run([py,'-m','spacy','download','en_core_web_sm'],'spacy')
    run([py,'scripts/fetch_external.py'],'external')
    run([py,'-m','pip','check'],'pip_check')
    run([py,'-c','import sys,torch,transformers,spacy,faiss; print(sys.version); print(torch.__version__,transformers.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count()); assert sys.version_info[:2]==(3,11); assert transformers.__version__=="5.5.3"; assert torch.cuda.is_available(); spacy.load("en_core_web_sm")'],'environment_verify')
    (LOG/'environment_complete.json').write_text(json.dumps({'python':py,'completed':datetime.datetime.now(datetime.timezone.utc).isoformat()}))

if __name__=='__main__':
    if sys.argv[1]=='environment':environment()
    else:models(sys.argv[1])
