import pathlib,json,datetime,hashlib,subprocess,sys,os,importlib.metadata
B=pathlib.Path('/home/user/Futaoran');A=B/'omniaware/assets';L=A/'manifests/v2_run_20261017';R=B/'omniaware_v2_exec_lf'
os.environ.update(OMNIAWARE_ASSETS=str(B/'omniaware_v2_assets_20261017_resume'),OMNIAWARE_CACHE=str(B/'omniaware_v2_cache_20261017_resume'),OMNIAWARE_WEBQA_OPEN_DIR=str(A/'webqa_open'),NO_PROXY='localhost,127.0.0.1,::1',no_proxy='localhost,127.0.0.1,::1')
sys.path.insert(0,str(R/'src'));sys.path.insert(0,str(R))
r={'start':datetime.datetime.now().astimezone().isoformat(),'command':sys.argv,'python':sys.version,'packages':{},'model_manifests':{},'loaders':{},'server_ready':False}
for p in ['torch','torchvision','transformers','spacy','faiss-cpu','en-core-web-sm','Pillow','numpy','pytest']:
    try:r['packages'][p]=importlib.metadata.version(p)
    except importlib.metadata.PackageNotFoundError:r['packages'][p]=None
from scripts.setup.download_models import model_complete
for name in ['qwen8b','reranker','nli','bartscore']:
    manifest=L/(name+'_verified_manifest.json');records=json.loads(manifest.read_text());bad=[]
    for item in records:
        p=pathlib.Path(item['path'])
        if not p.is_file() or p.stat().st_size!=item['bytes']:bad.append(str(p))
    r['model_manifests'][name]={'manifest':str(manifest),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'files':len(records),'bytes':sum(x['bytes'] for x in records),'current_size_errors':bad,'verification':'Original publish SHA256 verified; current sizes rechecked.'}
r['model_structural_checks']={n:model_complete(A/'cache'/n) for n in ['qwen3-vl-8b-instruct','bge-large-en-v1.5','clip-vit-large-patch14','bge-reranker-v2-m3']}
from omniaware import data
ex=data.load_okvqa_gs();r['loaders']['okvqa_gs']={'count':len(ex),'meta':ex.meta,'ok':len(ex)==5046 and not ex.meta.get('dropped')}
r['pip_check']=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True).__dict__
r['pip_check']={k:r['pip_check'][k] for k in ['args','returncode','stdout','stderr']}
r['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
r['source_status']=subprocess.check_output(['git','-c','core.autocrlf=false','status','--porcelain'],cwd=R,text=True)
r['pending']=['WebQA strict decode fails on 10 official source images; diagnostic permissive images not adopted','Full code review','Real 8B selfcheck','S1/S2 gates']
r['end']=datetime.datetime.now().astimezone().isoformat()
with (L/'preparation_inventory_1631.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r),flush=True)
