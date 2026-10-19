"""Resume exact official first-3M construction with large sequential I/O buffers."""
from pathlib import Path
import csv,gzip,io,json,hashlib,time
from datetime import datetime,timezone
R=Path(r'E:\OmniAware_Data');C=R/'assets/cache/okvqa_knowledge_20261004';D=R/'assets/data/okvqa_knowledge_dpr_first3m';D.mkdir(parents=True,exist_ok=True)
def save(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
 for attempt in range(120):
  try:t.replace(p);return
  except PermissionError:
   if attempt==119:raise
   time.sleep(.1)
raw=C/'psgs_w100.tsv.gz';downloaded=json.loads((C/'download_status.json').read_text());assert downloaded['status']=='COMPLETE' and raw.stat().st_size==4694541059
out=D/'corpus_wiki_first3m.jsonl';part=out.with_suffix('.jsonl.part')
if out.exists():raise SystemExit(0)
h=hashlib.sha256();kept=0;position=0;last=None;start=time.time()
if part.exists():
 with part.open('rb',buffering=8*2**20) as f:
  for line in f:
   if not line.endswith(b'\n'):break
   h.update(line);kept+=1;position+=len(line);last=line
 with part.open('r+b') as f:f.truncate(position)
resume=json.loads(last)['id']+1 if last else 0;seen=resume;skipped=resume-kept
with raw.open('rb',buffering=8*2**20) as source,gzip.GzipFile(fileobj=source) as gz,io.TextIOWrapper(gz,encoding='utf8',newline='') as f,part.open('ab',buffering=8*2**20) as target:
 reader=csv.DictReader(f,delimiter='\t');assert reader.fieldnames==['id','text','title']
 for idx,row in enumerate(reader):
  if idx<resume:continue
  if idx>=3000000:break
  seen+=1;title,text=row['title'].strip(),row['text'].strip()
  if not title or not text:skipped+=1;continue
  doc=dict(id=idx,source='wikipedia',title=title,contents=f'{title}\n{text}',image_url='');payload=(json.dumps(doc,ensure_ascii=False)+'\n').encode();target.write(payload);h.update(payload);kept+=1
  if seen%100000==0:save(D/'preparation_status.json',dict(status='BUILDING',input_rows=seen,documents=kept,skipped_empty=skipped,resumed_rows=resume,elapsed_seconds=time.time()-start))
 assert seen==3000000
part.replace(out)
save(D/'manifest.json',dict(status='CORPUS_COMPLETE_INDEX_PENDING',documents=kept,input_rows=seen,skipped_empty=skipped,corpus_bytes=out.stat().st_size,corpus_sha256=h.hexdigest(),source_url=downloaded['url'],compressed_source_sha256=downloaded['sha256'],legacy_rule_source=None,legacy_rule_source_sha256=None,rule='First 3,000,000 TSV data rows in source order; strip title/text; skip empty; zero-based row ID; title + newline + text',historical_byte_identity='Unverified: historical TSV/corpus checksum unavailable',annotation_answers_read=False,model_calls=0,elapsed_seconds=time.time()-start,finished_utc=datetime.now(timezone.utc).isoformat(),io_optimization='Resumed only complete JSONL lines, exact original transformation, 8 MiB sequential buffers',resumed_rows=resume))
save(D/'preparation_status.json',dict(status='COMPLETE',documents=kept));print('DPR first3m complete')
