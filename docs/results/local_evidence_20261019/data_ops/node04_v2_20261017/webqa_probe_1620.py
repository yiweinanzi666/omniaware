import pathlib,json,os,datetime
A=pathlib.Path('/home/user/Futaoran/omniaware/assets')
L=A/'manifests/v2_run_20261017'
I=A/'webqa_open/images'
raw=json.loads((A/'data/WebQA_train_val.json').read_text())
names=set(e.name for e in os.scandir(I))
missing={}
for q,v in raw.items():
    if v['split'] not in ('train','val'):continue
    for key in ('img_posFacts','img_negFacts'):
        for f in v[key]:
            iid=str(f['image_id'])
            if iid+'.jpg' not in names:missing.setdefault(iid,[]).append({'qid':q,'split':v['split'],'kind':key})
r={'time':datetime.datetime.now().astimezone().isoformat(),'missing':missing,'existing_image_filenames':len(names),'bad_manifest':[]}
if (I/'bad_images.jsonl').exists():
    r['bad_manifest']=[json.loads(l) for l in (I/'bad_images.jsonl').open()]
offsets=[int(x) for x in pathlib.Path('/home/user/Futaoran/WebQA/imgs.lineidx').read_text().splitlines()]
r['tsv_bytes']=pathlib.Path('/home/user/Futaoran/WebQA/imgs.tsv').stat().st_size
r['lineidx_rows']=len(offsets)
r['sample_ids']=[]
with pathlib.Path('/home/user/Futaoran/WebQA/imgs.tsv').open('rb') as f:
    for i in [0,1,2,len(offsets)//4,len(offsets)//2,len(offsets)-1]:
        f.seek(offsets[i]);r['sample_ids'].append([i,f.read(30).split(b'\t')[0].decode()])
with (L/'webqa_probe_1620.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r),flush=True)
