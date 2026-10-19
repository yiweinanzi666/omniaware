import datetime,hashlib,json,pathlib,sys,zipfile
from PIL import Image
A=pathlib.Path('/home/user/Futaoran/omniaware/assets'); C=A.parent/'cache'; L=A/'manifests/v2_run_20261017'
out={'started':datetime.datetime.now().astimezone().isoformat(),'assets':{},'checks':{}}
def inventory(p):
 d={'path':str(p),'exists':p.exists()}
 if p.is_file():
  d.update(bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest())
  if p.suffix=='.jsonl':
   with p.open('rb') as f:d['lines']=sum(1 for _ in f)
 elif p.is_dir():
  fs=[x for x in p.rglob('*') if x.is_file()];d.update(files=len(fs),bytes=sum(x.stat().st_size for x in fs))
 return d
items={'webqa_trainval':A/'data/WebQA_train_val.json','webqa_test':A/'data/WebQA_test.json','webqa_images':A/'cache/webqa_full4966_images/images','webqa_open_images':A/'webqa_open/images','okvqa_annotations':A/'data/okvqa_local/original/mscoco_val2014_annotations.json.zip','okvqa_questions':A/'data/okvqa_local/original/OpenEnded_mscoco_val2014_questions.json.zip','okvqa_inputs':A/'data/okvqa_val5046_assets/validation_inputs.jsonl','coco_images':A/'data/coco/val2014'}
for name in ['pool_stats.json','text_documents.jsonl','image_corpus.jsonl']:items['webqa_open_'+name]=A/'webqa_open'/name
for split in ['train','val']:
 for kind in ['bge','clip']:items[f'webqa_rankings_{split}_{kind}']=A/f'webqa_open/dense_top5_full/webqa_{split}/{kind}/rankings.jsonl'
for name in ['corpus.jsonl','emb_112724.f16','emb_112724.done','candidates_full.jsonl']:items['gs_'+name]=C/'okvqa_gs112k'/name
for name in ['qwen3-vl-8b-instruct','bge-large-en-v1.5','clip-vit-large-patch14','bge-reranker-v2-m3','e5_citation_support/deberta_nli','bartscore']:items['model_'+name]=A/'cache'/name
for key,p in items.items():
 print('INVENTORY '+key,flush=True);out['assets'][key]=inventory(p)
check=out['checks'];gs=C/'okvqa_gs112k'
check['corpus_112724']=out['assets']['gs_corpus.jsonl'].get('lines')==112724
check['embedding_bytes']=out['assets']['gs_emb_112724.f16'].get('bytes')==230858752
check['embedding_done']=(gs/'emb_112724.done').read_text().strip()=='112724'
ids=[];bad=[]
with (gs/'candidates_full.jsonl').open() as f:
 for line in f:
  r=json.loads(line);ids.append(str(r['question_id']));c=r['candidates']
  if len(c)!=20 or [x['rank'] for x in c]!=list(range(1,21)):bad.append(r['question_id'])
check['candidates']={'rows':len(ids),'unique_question_ids':len(set(ids)),'bad_qids':bad,'pass':len(ids)==len(set(ids))==5046 and not bad}
with zipfile.ZipFile(items['okvqa_questions']) as z:
 q=json.loads(z.read(next(x for x in z.namelist() if x.endswith('.json'))))['questions']
imids={r['image_id'] for r in q};missing=[];broken=[]
for iid in sorted(imids):
 p=A/f'data/coco/val2014/COCO_val2014_{iid:012d}.jpg'
 if not p.exists():missing.append(str(p));continue
 try:
  with Image.open(p) as img:img.load()
 except Exception as e:broken.append({'path':str(p),'error':str(e)})
check['okvqa_images']={'questions':len(q),'distinct_image_ids':len(imids),'missing':missing,'decode_errors':broken,'pass':len(q)==5046 and len(imids)==5033 and not missing and not broken}
out['ended']=datetime.datetime.now().astimezone().isoformat()
p=L/'V2_ASSET_CHECK_20261017_initial.json'
with p.open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(check),flush=True)
print('REPORT '+str(p),flush=True)
