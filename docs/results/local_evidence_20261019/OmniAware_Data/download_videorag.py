"""Reproducible 20% VideoRAG corpus with complete benchmark evidence priority.
Run again to resume; successful validated downloads are never redownloaded.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import defaultdict,Counter
import json,math,hashlib,random,re,tarfile,subprocess,sys,os,time,shutil,threading
R=Path(r'E:\OmniAware_Data');U=R/'omniaware/external/UniversalRAG';D=U/'dataset/videorag';V=D/'videos';V.mkdir(exist_ok=True)
M=R/'manifests';M.mkdir(exist_ok=True);L=R/'logs/video';L.mkdir(parents=True,exist_ok=True)
SEED=20261015;TARGET=math.ceil(8621*.2); NODE=r'C:\Users\mengguai\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe'
TOPICS={'food':'cook recipe food chicken meat cake flour onion rice butter oil sauce bread grill bake','crafts':'fabric sew sewing stitch quilt yarn crochet knitting paper glue paint draw','automotive':'car brake wheel tire engine vehicle motorcycle truck','home':'house wall floor door plumbing faucet pipe screw drill wood furniture clean','garden':'plant garden soil tree flower seed lawn','beauty':'hair makeup skin nail beauty shampoo','fitness':'exercise workout muscle fitness yoga stretch weight sport golf ball','technology':'computer software phone camera laptop screen windows android install','music':'guitar piano chord music song drum bass','education':'math equation science lesson study physics chemistry'}
def save(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
 for attempt in range(120):
  try:t.replace(p);return
  except PermissionError:
   if attempt==119:raise
   time.sleep(.1)
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
 p=M/'videorag_plan.json'
 if p.exists():return json.loads(p.read_text(encoding='utf8'))
 urls=(D/'videos.txt').read_text().splitlines();ids=sorted({re.search(r'[?&]v=([\w-]{11})',u).group(1) for u in urls if u.strip()})
 q=json.loads((U/'dataset/query/videorag_synth.json').read_text(encoding='utf8'));gold={Path(v).stem for a in q for v in a['gt_videos']}
 texts={}
 with tarfile.open(D/'scripts.tar.gz') as tar:
  for m in tar.getmembers():
   if m.isfile():
    texts[Path(m.name).stem]=tar.extractfile(m).read().decode('utf8',errors='replace')
    out=D/m.name;out.parent.mkdir(parents=True,exist_ok=True)
    if not out.exists():out.write_text(texts[Path(m.name).stem],encoding='utf8')
 lengths=sorted(len(t.split()) for t in texts.values());cuts=[lengths[int(len(lengths)*x)] for x in [.25,.5,.75]]
 groups=defaultdict(list);strata={}
 for i in ids:
  words=Counter(re.findall('[a-z]+',texts.get(i,'').lower()));scores={k:sum(words[w] for w in s.split()) for k,s in TOPICS.items()};topic=max(scores,key=scores.get) if max(scores.values()) else 'other';n=sum(words.values());key=f'{topic}|length{sum(n>c for c in cuts)}';strata[i]=key
  if i not in gold:groups[key].append(i)
 rng=random.Random(SEED)
 for k in sorted(groups):rng.shuffle(groups[k])
 n=TARGET-len(gold);total=sum(map(len,groups.values()));raw={k:n*len(v)/total for k,v in groups.items()};alloc={k:math.floor(x) for k,x in raw.items()}
 for k in sorted(raw,key=lambda k:(-(raw[k]-alloc[k]),k))[:n-sum(alloc.values())]:alloc[k]+=1
 selected=sorted(gold)+[i for k in sorted(groups) for i in groups[k][:alloc[k]]]
 backups={k:v[alloc[k]:] for k,v in groups.items()};record={'seed':SEED,'source_url':'https://github.com/wgcyeo/UniversalRAG','source_commit':'d29ab3acaebfbaebdc7f095457076585bec88065','population':len(ids),'target_successful_videos':TARGET,'fraction':.2,'benchmark_questions':len(q),'benchmark_evidence_videos':sorted(gold),'selected_ids':selected,'strata':strata,'backup_ids_by_stratum':backups,'method':'Keep every benchmark source video, then allocate distractors proportionally across caption-keyword topic x caption-length quartile strata using largest remainders. Python Random(seed), sorted input IDs. Answers and model results are never used. Replace unavailable distractors within the same stratum; unavailable gold is always reported.','length_word_count_cuts':cuts,'topic_keywords':TOPICS,'input_sha256':{str(x.relative_to(U)):digest(x) for x in [D/'videos.txt',D/'scripts.tar.gz',U/'dataset/query/videorag_synth.json']},'limitation':'Reduced retrieval corpus changes benchmark difficulty; do not label as full-corpus results.'};save(p,record);return record
lock=threading.Lock()
def worker(i):
 recp=M/f'video_records/{i}.json';recp.parent.mkdir(exist_ok=True)
 if recp.exists():
  rec=json.loads(recp.read_text())
  p=V/rec.get('filename','__missing__')
  if rec.get('status')=='complete' and p.exists() and p.stat().st_size==rec.get('bytes'):return rec
 if shutil.disk_usage(R).free<35*2**30:return {'id':i,'status':'blocked','error':'Less than 35 GiB free; storage guard stopped new downloads'}
 cmd=[sys.executable,'-m','yt_dlp','--js-runtimes','node:'+NODE,'--no-playlist','--socket-timeout','20','--retries','2','--fragment-retries','2','--extractor-retries','1','--no-progress','--abort-on-unavailable-fragments','--cache-dir',str(R/'cache/yt-dlp'),'--continue','--no-overwrites','--write-info-json','--no-clean-info-json','-f','bestvideo[height<=480][vcodec^=avc1][protocol=https]+bestaudio[ext=m4a][language^=en]/bestvideo[height<=480][vcodec!^=av01][protocol=https]+bestaudio[protocol=https]/bestvideo[height<=480]+bestaudio/best[height<=480]','--merge-output-format','mp4','-o',str(V/'%(id)s.%(ext)s'),'https://www.youtube.com/watch?v='+i]
 env=os.environ.copy();env['PYTHONPATH']=str(R/'tools');env['PYTHONIOENCODING']='utf-8'
 error=''
 with (L/f'{i}.log').open('a',encoding='utf8') as log:
  try:r=subprocess.run(cmd,stdout=log,stderr=log,env=env,timeout=300,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));error=f'yt-dlp exit {r.returncode}' if r.returncode else ''
  except Exception as e:error=str(e)
 if error and '403' in (L/f'{i}.log').read_text(encoding='utf8',errors='replace')[-3000:]:
  fallback=cmd.copy();fallback[fallback.index('-f')+1]='bestvideo[height<=480][protocol^=m3u8]+bestaudio[protocol^=m3u8]/bestvideo[height<=480][protocol^=m3u8]'
  with (L/f'{i}.log').open('a',encoding='utf8') as log:
   try:
    log.write('\n[automatic retry] public HLS format after direct HTTP 403\n');log.flush();rr=subprocess.run(fallback,stdout=log,stderr=log,env=env,timeout=300,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));error=f'yt-dlp fallback exit {rr.returncode}' if rr.returncode else ''
   except Exception as e:error=str(e)
 p=next((V/f'{i}{ext}' for ext in ['.mp4','.mkv','.webm','.mov'] if (V/f'{i}{ext}').exists()),None)
 rec={'id':i,'source_url':'https://www.youtube.com/watch?v='+i,'status':'failed','error':error,'log':str(L/f'{i}.log')}
 if p and not error:
  try:
   check=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_type,width,height','-of','json',str(p)],capture_output=True,text=True,timeout=40,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));meta=json.loads(check.stdout);assert check.returncode==0 and any(s.get('codec_type')=='video' for s in meta['streams']);rec.update(status='complete',filename=p.name,bytes=p.stat().st_size,sha256=digest(p),media=meta,error=None)
  except Exception as e:rec['error']='media validation: '+str(e)
 if rec['status']!='complete':rec['detail']=(L/f'{i}.log').read_text(encoding='utf8',errors='replace')[-2500:]
 save(recp,rec);return rec

def main():
 p=plan();records={};attempted=set();start=time.time();status=M/'video_download_status.json';queue=p['selected_ids'];counts=Counter();gold=set(p['benchmark_evidence_videos']);storage_stop=False
 def batch(ids):
  nonlocal storage_stop
  with ThreadPoolExecutor(max_workers=20) as ex:
   fs={ex.submit(worker,i):i for i in ids}
   for f in as_completed(fs):
    i=fs[f];attempted.add(i)
    try:rec=f.result()
    except Exception as e:rec={'id':i,'status':'failed','error':str(e)}
    records[i]=rec;counts[rec['status']]+=1
    if rec['status']=='blocked':storage_stop=True
    good=[r for r in records.values() if r['status']=='complete'];save(status,{'state':'downloading','completed':len(good),'target':TARGET,'attempted':len(attempted),'failed':sum(r['status']=='failed' for r in records.values()),'bytes':sum(r['bytes'] for r in good),'gold_completed':sum(r['id'] in gold for r in good),'gold_total':len(gold),'elapsed_seconds':round(time.time()-start),'last_id':i})
 batch(queue)
 # One retry pass for failures. Retain resumable partial bytes.
 if not storage_stop:batch([i for i,r in list(records.items()) if r['status']=='failed'])
 # Deterministic same-stratum replacements; at most 30% of original corpus attempted.
 backup={k:list(v) for k,v in p['backup_ids_by_stratum'].items()};replacement=[]
 for round_no in range(4):
  good={i for i,r in records.items() if r['status']=='complete'};need=TARGET-len(good)
  if need<=0 or storage_stop:break
  if len(attempted)>=math.ceil(p['population']*.3):break
  gaps=Counter(p['strata'][i] for i in p['selected_ids'] if i not in good)
  nextids=[]
  for k in sorted(gaps):
   pool=backup.get(k,[])
   while pool and gaps[k] and len(nextids)<need:
    i=pool.pop(0)
    if i not in attempted:nextids.append(i);gaps[k]-=1
  if not nextids:break
  replacement.extend(nextids);batch(nextids)
 good={i for i,r in records.items() if r['status']=='complete'};q=json.loads((U/'dataset/query/videorag_synth.json').read_text(encoding='utf8'));covered=[x for x in q if all(Path(v).stem in good for v in x['gt_videos'])]
 save(M/'videorag_results.json',{'population':p['population'],'target':TARGET,'complete':len(good),'fraction':len(good)/p['population'],'bytes':sum(records[i]['bytes'] for i in good),'gold_complete':len(good&gold),'gold_total':len(gold),'questions_total':len(q),'questions_covered':len(covered),'missing_gold_ids':sorted(gold-good),'completed_ids':sorted(good),'replacement_ids':replacement,'failed':[r for r in records.values() if r['status']!='complete'],'storage_stop':storage_stop,'seed':SEED})
 save(M/'videorag_available_questions.json',covered);save(status,{'state':'complete' if len(good)>=TARGET else 'incomplete','completed':len(good),'target':TARGET,'questions_covered':len(covered),'questions_total':len(q),'bytes':sum(records[i]['bytes'] for i in good),'failures':len(records)-len(good)})
 print(json.dumps(json.loads(status.read_text())),flush=True)
if __name__=='__main__':main()
