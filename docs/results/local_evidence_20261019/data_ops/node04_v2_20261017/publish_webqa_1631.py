import pathlib,json,datetime,hashlib,os,sys,subprocess,traceback
from PIL import Image,ImageFile
B=pathlib.Path('/home/user/Futaoran');A=B/'omniaware/assets';L=A/'manifests/v2_run_20261017';R=B/'omniaware_v2_exec_lf'
r={'start':datetime.datetime.now().astimezone().isoformat(),'command':sys.argv,'authorization':'User explicitly approved official LOAD_TRUNCATED_IMAGES=True decoding for these 10 images; preserve sources and record handling.','published':[],'loaders':{}}
try:
    trial=json.loads((L/'webqa_decode_trial_1620.json').read_text());ImageFile.LOAD_TRUNCATED_IMAGES=False
    for row in trial['images']:
        assert row['ok'];p=pathlib.Path(row['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
        with Image.open(p) as im:im.load()
        dest=A/'webqa_open/images'/(row['image_id']+'.jpg');assert not dest.exists()
    for row in trial['images']:
        dest=A/'webqa_open/images'/(row['image_id']+'.jpg');os.link(row['path'],dest)
        r['published'].append(dict(row,destination=str(dest)))
    os.environ.update(OMNIAWARE_ASSETS=str(B/'omniaware_v2_assets_20261017_resume'),OMNIAWARE_CACHE=str(B/'omniaware_v2_cache_20261017_resume'),OMNIAWARE_WEBQA_OPEN_DIR=str(A/'webqa_open'))
    sys.path.insert(0,str(R/'src'))
    from omniaware import data
    for ds in ('webqa_full','webqa_open_img','okvqa_gs'):
        u=data.IMAGE_SUBSET_OF.get(ds,ds);ex=data.LOADERS[u]();expected=5046 if ds=='okvqa_gs' else 4966
        rec={'universe':u,'count':len(ex),'meta':ex.meta};assert len(ex)==expected and not ex.meta.get('dropped')
        if u in data.TRAIN_LOADERS:
            tr=data.TRAIN_LOADERS[u]();rec.update(train_count=len(tr),train_meta=tr.meta);assert len(tr)==36766 and not tr.meta.get('dropped')
        r['loaders'][ds]=rec
    raw=json.loads((A/'data/WebQA_train_val.json').read_text());names={e.name for e in os.scandir(A/'webqa_open/images')};missing=[];facts=0
    for q,v in raw.items():
        if v['split'] not in ('train','val'):continue
        for key in ('img_posFacts','img_negFacts'):
            for fact in v[key]:
                facts+=1
                if str(fact['image_id'])+'.jpg' not in names:missing.append([q,fact['image_id']])
    r['image_fact_closure']={'checked_facts':facts,'missing':missing};assert not missing
    r['retrieval_note']='Existing open-retrieval rankings and embeddings preserved; the previously undecodable images were excluded by the original indexing process. This repair fills distractor image facts and does not regenerate frozen retrieval caches.'
    r['ok']=True
except Exception:r['ok']=False;r['traceback']=traceback.format_exc()
r['end']=datetime.datetime.now().astimezone().isoformat()
with (L/'publish_webqa_1631.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r),flush=True)
sys.exit(0 if r['ok'] else 1)
