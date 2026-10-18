"""Read-only cross-host SHA256 inventory for explicitly scoped task-data cleanup."""
import os,sys,pathlib,json,hashlib,datetime,concurrent.futures,time,re
local=sys.platform=='win32'
L=pathlib.Path('D:/WWW2027/data_ops/node04_v2_20261017' if local else '/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017')
base='D:/WWW2027' if local else '/home/user/Futaoran'
groups=[
 ['WebQA','WebQA','raw'],
 ['WebQA/chunks','WebQA','chunks'],
 ['OmniAware_Data/assets/webqa_open','omniaware/assets/webqa_open','open'],
 ['OmniAware_Data/assets/data/coco/val2014','omniaware/assets/data/coco/val2014','jpg'],
 ['OmniAware_Data/assets/data/okvqa_local/original','omniaware/assets/data/okvqa_local/original','data'],
 ['OmniAware_Data/assets/data/okvqa_gs','omniaware/assets/downloads/okvqa_gs','csv'],
 ['OmniAware_Data/transfer/models/qwen3-vl-8b-instruct','omniaware/assets/cache/qwen3-vl-8b-instruct','model'],
 ['OmniAware_Data/transfer/models/bge-reranker-v2-m3','omniaware/assets/cache/bge-reranker-v2-m3','model'],
 ['OmniAware_Data/transfer/models/deberta_nli','omniaware/assets/cache/e5_citation_support/deberta_nli/b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7','model'],
 ['data_ops/node04_v2_20261017','omniaware/assets/cache/bartscore','bart'],
]
plan=L/'cleanup_file_plan_1635.jsonl'
def selected(rel,kind):
    p=pathlib.PurePosixPath(rel);n=p.name
    if kind=='raw':return len(p.parts)==1 and n in {'imgs.tsv','imgs.lineidx','WebQA_train_val.json','WebQA_test.json','WebQA_train_val.json.gz','WebQA_test.json.gz'}
    if kind=='chunks':return len(p.parts)==1 and bool(re.fullmatch(r'imgs\.7z\.\d{3}',n)) and 1<=int(n[-3:])<=51
    if kind=='bart':return rel=='bart_score.pth'
    if kind=='jpg':return p.suffix.lower()=='.jpg'
    if kind=='csv':return n in ('okvqa_train_corpus.csv','okvqa_full_corpus.csv')
    if kind=='data':return p.suffix.lower()=='.zip'
    if kind=='model':return not any(x.startswith('.') for x in p.parts) and p.suffix.lower() in ('.safetensors','.bin','.pth','.json','.model','.txt')
    if kind=='open':return n not in ('assets.jsonl','bad_images.jsonl','missing_images.jsonl') and not any(x.startswith('.') for x in p.parts) and p.suffix.lower() in ('.jpg','.jpeg','.png','.jsonl','.json','.f16','.npy','.index','.done')
    return False
if local:
    with plan.open('x',encoding='utf8') as f:
        for i,(lr,rr,kind) in enumerate(groups):
            root=pathlib.Path(base,lr)
            if not root.is_dir():continue
            for d,ds,fs in os.walk(root,followlinks=False):
                ds[:]=[n for n in ds if not pathlib.Path(d,n).is_symlink() and not n.startswith('.')]
                if kind in ('raw','chunks','bart'):ds[:]=[]
                for n in fs:
                    p=pathlib.Path(d,n);rel=p.relative_to(root).as_posix()
                    if not selected(rel,kind) or p.is_symlink():continue
                    st=p.stat();f.write(json.dumps({'g':i,'r':rel,'b':st.st_size,'m':st.st_mtime_ns},separators=(',',':'))+'\n')
    print('PLAN_READY '+str(plan),flush=True)
side='local' if local else 'server'
out=L/('cleanup_hashes_1635_'+side+'.jsonl')
def check(row):
    p=pathlib.Path(base,groups[row['g']][0 if local else 1],row['r'])
    result={'g':row['g'],'r':row['r']}
    try:
        st=p.stat();result.update(b=st.st_size,m=st.st_mtime_ns)
        if st.st_size!=row['b']:result['error']='size mismatch';return result
        with p.open('rb') as f:result['sha256']=hashlib.file_digest(f,'sha256').hexdigest()
        after=p.stat();assert (after.st_size,after.st_mtime_ns)==(st.st_size,st.st_mtime_ns),'file changed during hashing'
    except Exception as e:result['error']=str(e)
    return result
start=datetime.datetime.now().astimezone().isoformat();count=0;total=0;errors=0;t=time.monotonic()
with plan.open(encoding='utf8') as src,out.open('x',encoding='utf8') as dst,concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    batch=[]
    for line in src:
        batch.append(json.loads(line))
        if len(batch)<1000:continue
        for row in pool.map(check,batch):
            dst.write(json.dumps(row,separators=(',',':'))+'\n');count+=1;total+=row.get('b',0);errors+='error' in row
        batch=[];dst.flush();print(json.dumps({'side':side,'files':count,'bytes':total,'errors':errors,'seconds':round(time.monotonic()-t)}),flush=True)
    for row in pool.map(check,batch):
        dst.write(json.dumps(row,separators=(',',':'))+'\n');count+=1;total+=row.get('b',0);errors+='error' in row
r={'start':start,'end':datetime.datetime.now().astimezone().isoformat(),'side':side,'files':count,'bytes':total,'errors':errors,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'groups':groups,'plan_sha256':hashlib.sha256(plan.read_bytes()).hexdigest(),'deletion_performed':False}
with (L/('cleanup_hashes_1635_'+side+'.json')).open('x',encoding='utf8') as f:json.dump(r,f,indent=2)
print(json.dumps(r),flush=True)
