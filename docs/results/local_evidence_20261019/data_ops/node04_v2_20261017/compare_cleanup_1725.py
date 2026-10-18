import pathlib,json,gzip,hashlib,collections,datetime
L=pathlib.Path('D:/WWW2027/data_ops/node04_v2_20261017');roots=json.loads((L/'cleanup_hashes_1635_local.json').read_text())['groups']
counts=collections.defaultdict(lambda:{'matched_files':0,'matched_bytes':0,'retained_files':0,'retained_bytes':0});retained=[];total=0
eligible=L/'cleanup_eligible_1725.jsonl'
with (L/'cleanup_hashes_1635_local.jsonl').open() as a,gzip.open(L/'cleanup_hashes_1645_server.jsonl.gz','rt') as b,eligible.open('x') as out:
    for la,lb in zip(a,b,strict=True):
        x,y=json.loads(la),json.loads(lb);assert (x['g'],x['r'])==(y['g'],y['r']);total+=1
        good='error' not in x and 'error' not in y and x['b']==y['b'] and x.get('sha256')==y.get('sha256')
        if good:
            out.write(json.dumps({'g':x['g'],'r':x['r'],'b':x['b'],'m':x['m'],'sha256':x['sha256'],'server_m':y['m']},separators=(',',':'))+'\n');counts[x['g']]['matched_files']+=1;counts[x['g']]['matched_bytes']+=x['b']
        else:
            counts[x['g']]['retained_files']+=1;counts[x['g']]['retained_bytes']+=x.get('b',0)
            retained.append({'local':x,'server':y,'reason':y.get('error') or x.get('error') or 'SHA256 differs'})
assert total==394890
r={'time':datetime.datetime.now().astimezone().isoformat(),'candidate_files':total,'groups':roots,'group_summary':dict(counts),'matched_files':sum(x['matched_files'] for x in counts.values()),'matched_bytes':sum(x['matched_bytes'] for x in counts.values()),'retained_files':len(retained),'retained':retained,'eligible_manifest_sha256':hashlib.sha256(eligible.read_bytes()).hexdigest(),'deletion_performed':False,'deletion_gate':'Requires successful GitHub push and unchanged local file size/mtime, with resolved paths restricted to the approved task-data roots.'}
with (L/'cleanup_comparison_1725.json').open('x',encoding='utf8') as f:json.dump(r,f,indent=2,ensure_ascii=False)
print(json.dumps({k:v for k,v in r.items() if k not in ('retained','groups')},ensure_ascii=True))
