import os,pathlib,json,datetime,collections
root=pathlib.Path('D:/WWW2027')
groups=collections.defaultdict(lambda:{'files':0,'bytes':0})
errors=[]
for base,dirs,files in os.walk(root,followlinks=False,onerror=lambda e:errors.append(str(e))):
    dirs[:]=[d for d in dirs if d not in ('.git','node_modules') and not pathlib.Path(base,d).is_symlink()]
    for name in files:
        p=pathlib.Path(base,name)
        if p.is_symlink():continue
        rel=p.relative_to(root)
        key='/'.join(rel.parts[:3] if rel.parts[0]=='OmniAware_Data' and len(rel.parts)>3 else rel.parts[:2])
        try:size=p.stat().st_size
        except OSError as e:errors.append(str(e));continue
        groups[key]['files']+=1;groups[key]['bytes']+=size
r={'time':datetime.datetime.now().astimezone().isoformat(),'root':str(root),'scope':'Task data/cache only; preserve code/reports/other files. No deletion performed.','groups':dict(sorted(groups.items(),key=lambda x:-x[1]['bytes'])),'errors':errors}
out=root/'data_ops/node04_v2_20261017/local_cleanup_inventory_1620.json'
with out.open('x',encoding='utf8') as f:json.dump(r,f,indent=2,ensure_ascii=False)
print(json.dumps(dict(list(r['groups'].items())[:25]),ensure_ascii=True))
