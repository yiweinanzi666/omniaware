import pathlib,json,subprocess,hashlib,datetime
R=pathlib.Path('/home/user/Futaoran/omniaware_v2_exec');L=pathlib.Path('/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017')
r={'time':datetime.datetime.now().astimezone().isoformat(),'status':subprocess.check_output(['git','status','--porcelain'],cwd=R,text=True),'config':subprocess.run(['git','config','--get','core.autocrlf'],cwd=R,capture_output=True,text=True).stdout,'files':[]}
for name in ['eval_1022.py','webqa_eval.py']:
    p=R/'third_party/webqa'/name
    if not p.exists():continue
    b=p.read_bytes();g=subprocess.check_output(['git','show','HEAD:third_party/webqa/'+name],cwd=R)
    r['files'].append({'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'git_blob_sha256':hashlib.sha256(g).hexdigest(),'lf_sha256':hashlib.sha256(b.replace(b'\r\n',b'\n')).hexdigest(),'crlf_sha256':hashlib.sha256(b.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')).hexdigest(),'crlf':b.count(b'\r\n'),'lf':b.count(b'\n')})
with (L/'source_hash_probe_1620.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r))
