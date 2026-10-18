import pathlib,json,hashlib,datetime,concurrent.futures,time,os,sys
L=pathlib.Path('/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017');B=pathlib.Path('/home/user/Futaoran')
roots=['WebQA','WebQA','omniaware/assets/webqa_open','omniaware/assets/data/coco/val2014','omniaware/assets/data/okvqa_local/original','omniaware/assets/downloads/okvqa_gs','omniaware/assets/cache/qwen3-vl-8b-instruct','omniaware/assets/cache/bge-reranker-v2-m3','omniaware/assets/cache/e5_citation_support/deberta_nli/b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7','omniaware/assets/cache/bartscore']
plan=L/'cleanup_file_plan_1635.jsonl';old=L/'cleanup_hashes_1635_server.jsonl';out=L/'cleanup_hashes_1645_server.jsonl'
assert not pathlib.Path('/proc/2279047').exists(),'Previous hash worker is still active'
start=datetime.datetime.now().astimezone().isoformat();count=total=errors=reused=0;t=time.monotonic()
def check(row):
    p=B/roots[row['g']]/row['r'];r={'g':row['g'],'r':row['r']}
    try:
        st=p.stat();r.update(b=st.st_size,m=st.st_mtime_ns)
        if st.st_size!=row['b']:r['error']='size mismatch';return r
        with p.open('rb') as f:r['sha256']=hashlib.file_digest(f,'sha256').hexdigest()
        after=p.stat();assert (after.st_size,after.st_mtime_ns)==(st.st_size,st.st_mtime_ns),'file changed during hashing'
    except Exception as e:r['error']=str(e)
    return r
with plan.open() as src,old.open() as previous,out.open('x') as dst,concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    for line in previous:
        row=json.loads(line);expected=json.loads(next(src));assert (row['g'],row['r'])==(expected['g'],expected['r'])
        dst.write(line);count+=1;reused+=1;total+=row.get('b',0);errors+='error' in row
    dst.flush();print(json.dumps({'reused_verified_records':reused,'reused_bytes':total}),flush=True)
    batch=[]
    for line in src:
        batch.append(json.loads(line))
        if len(batch)<1000:continue
        for row in pool.map(check,batch):
            dst.write(json.dumps(row,separators=(',',':'))+'\n');count+=1;total+=row.get('b',0);errors+='error' in row
        batch=[];dst.flush();print(json.dumps({'side':'server','files':count,'bytes':total,'errors':errors,'seconds':round(time.monotonic()-t)}),flush=True)
    for row in pool.map(check,batch):
        dst.write(json.dumps(row,separators=(',',':'))+'\n');count+=1;total+=row.get('b',0);errors+='error' in row
r={'start':start,'end':datetime.datetime.now().astimezone().isoformat(),'files':count,'bytes':total,'errors':errors,'reused_records':reused,'source_partial_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256(plan.read_bytes()).hexdigest(),'deletion_performed':False}
with (L/'cleanup_hashes_1645_server.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r),flush=True)
