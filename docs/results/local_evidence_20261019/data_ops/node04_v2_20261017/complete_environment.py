import importlib.util,json,os,pathlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('prep',ROOT/'prepare_v3.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
py=str(p.ENV/'bin/python')
# Validate installed dependencies independently of reference repository retrieval.
p.run([py,'-m','pip','check'],'pip_check')
p.run([py,'-c','import sys,torch,transformers,spacy,faiss; print(sys.version); print(torch.__version__,transformers.__version__); print(torch.cuda.is_available(),torch.cuda.device_count()); assert sys.version_info[:2]==(3,11); assert transformers.__version__=="5.5.3"; assert torch.cuda.is_available(); spacy.load("en_core_web_sm")'],'environment_verify')
manifest=json.loads((p.REPO/'third_party/external_repos.json').read_text())
for rec in manifest['repos']:
 name=rec['name'];target=p.REPO/'external'/name;ref=rec['ref']
 if not ref:raise RuntimeError('Unpinned external repository: '+name)
 if target.exists() and not (target/'.git').exists():raise RuntimeError('Existing non-git target: '+str(target))
 if not target.exists():
  target.mkdir(parents=True)
  p.run(['git','init','-q'],'external_'+name+'_init',cwd=target)
  p.run(['git','remote','add','origin',rec['url']],'external_'+name+'_remote',cwd=target)
 old=subprocess.run(['git','rev-parse','--verify','HEAD'],cwd=target,capture_output=True,text=True)
 if old.returncode==0:
  if old.stdout.strip()!=ref:raise RuntimeError('Existing ref mismatch: '+str(target))
  continue
 # This server's older Git rejects --filter. A shallow fetch without filtering preserves the pinned content.
 p.run(['git','fetch','--depth','1','origin',ref],'external_'+name+'_fetch',cwd=target)
 p.run(['git','checkout','--detach',ref],'external_'+name+'_checkout',cwd=target)
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=target,text=True).strip()
 if head!=ref:raise RuntimeError('Commit verification failed: '+name)
(ROOT/'environment_complete.json').write_text(json.dumps({'python':py,'all_dependencies_verified':True,'external_refs_verified':True},indent=2))
print('ENVIRONMENT_COMPLETE',flush=True)
