"""Prepare portable inputs and manifests after local batch downloaders complete."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os,sys,time,json,csv,hashlib,subprocess,shutil,traceback
R=Path(r'E:\OmniAware_Data');A=R/'assets';U=R/'omniaware/external/UniversalRAG';M=R/'manifests';M.mkdir(exist_ok=True);LOG=R/'logs';LOG.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'tools'));import pandas as pd
START=time.time(); summary={};errors=[]
def save(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
 for attempt in range(120):
  try:t.replace(p);return
  except PermissionError:
   if attempt==119:raise
   time.sleep(.1)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def await_file(p,limit=7200):
 while not p.exists():
  if time.time()-START>limit:raise TimeoutError(str(p))
  time.sleep(15)
def okvqa():
 if (M/'okvqa.json').exists():
  summary['okvqa']=json.loads((M/'okvqa.json').read_text(encoding='utf8'));return
 await_file(R/'okvqa_image_failures.json',3600)
 from PIL import Image
 raw=A/'data/okvqa_local/original';records=[];bad=[];counts={};paths={}
 for split in ['train2014','val2014']:
  q=json.loads((raw/f'OpenEnded_mscoco_{split}_questions.json').read_text(encoding='utf8'))['questions'];ann=json.loads((raw/f'mscoco_{split}_annotations.json').read_text(encoding='utf8'))['annotations'];counts[split]={'questions':len(q),'annotations':len(ann),'images_expected':len({a['image_id'] for a in q})}
  for i in sorted({a['image_id'] for a in q}):paths[f'{split}/{i}']=A/'data/coco'/split/f'COCO_{split}_{i:012d}.jpg'
  p=A/'data'/('okvqa_val5046_assets' if split=='val2014' else 'okvqa_train9009_assets')/('validation_inputs.jsonl' if split=='val2014' else 'training_inputs.jsonl');p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('w',encoding='utf8') as out:
   for x in sorted(q,key=lambda x:x['question_id']):
    rec={'question_id':x['question_id'],'question':x['question'],'image_id':x['image_id'],'image_path':f'data/coco/{split}/COCO_{split}_{x["image_id"]:012d}.jpg'};out.write(json.dumps(rec,ensure_ascii=False)+'\n')
 def check(item):
  key,p=item
  try:
   with Image.open(p) as im:im.verify()
   return {'id':key,'path':str(p.relative_to(R)).replace('\\','/'),'bytes':p.stat().st_size,'sha256':sha(p),'status':'complete','source_url':'http://images.cocodataset.org/'+p.parent.name+'/'+p.name}
  except Exception as e:return {'id':key,'status':'failed','error':str(e)}
 with ThreadPoolExecutor(max_workers=12) as e:
  for rec in e.map(check,paths.items()):records.append(rec)
 bad=[r for r in records if r['status']!='complete'];save(M/'okvqa_images.json',records)
 info={'splits':counts,'images_complete':len(records)-len(bad),'images_expected':len(records),'bytes':sum(x.get('bytes',0) for x in records),'failures':bad,'question_coverage':(len(records)-len(bad))/len(records),'source':'https://okvqa.allenai.org/download.html','image_selection':'All unique COCO images referenced by official OK-VQA train and val questions; unrelated COCO images omitted. No question subsampling.'};save(M/'okvqa.json',info);summary['okvqa']=info

def gs():
 if (M/'gs_corpora.json').exists():
  summary['gs']=json.loads((M/'gs_corpora.json').read_text(encoding='utf8'));return
 infos=[]
 for name,key,count,fid in [('okvqa_train_corpus.csv','gs112k',112724,'1O8HkDzWIMc-uzLgG3BqHz-g2qtGeRcT6'),('okvqa_full_corpus.csv','gs168k_full',168306,'1x9unz58_Lw7oLgTO73nslQ92xr4un6bl')]:
  p=A/'data/okvqa_gs'/name;await_file(p,1800);dest=R/'cache'/('okvqa_'+key)/'corpus.jsonl';dest.parent.mkdir(parents=True,exist_ok=True);n=0;rows=0
  with p.open(encoding='utf8',newline='') as f,dest.open('w',encoding='utf8') as out:
   it=csv.DictReader(f);assert 'text' in it.fieldnames
   for row in it:
    rows+=1;text=' '.join((row['text'] or '').split())
    if text:out.write(json.dumps({'id':n,'contents':text},ensure_ascii=False)+'\n');n+=1
  assert rows==count,(name,rows,count)
  infos.append({'file':name,'rows':rows,'nonempty_passages':n,'bytes':p.stat().st_size,'sha256':sha(p),'prepared_path':str(dest.relative_to(R)),'prepared_sha256':sha(dest),'source_url':'https://drive.google.com/file/d/'+fid+'/view','usage':'Training-only GS main corpus' if key=='gs112k' else 'Appendix comparison only; includes evaluation-question retrieval results'})
 save(M/'gs_corpora.json',infos);summary['gs']=infos

def hybrid():
 if (M/'hybridqa.json').exists():
  summary['hybridqa']=json.loads((M/'hybridqa.json').read_text(encoding='utf8'));return
 import zipfile,importlib.util
 s=R/'sources/WikiTables-WithLinks';await_file(s/'SOURCE_DOWNLOAD.json',3600);await_file(R/'sources/HybridQA/SOURCE_DOWNLOAD.json',3600)
 d=U/'dataset/HybridQA';q=U/'dataset/query/hybridqa.json';questions=json.loads(q.read_text(encoding='utf8'));needed=sorted({x['gt_tables'] for x in questions});tables=[];texts=[];missing=[];missingtext=[]
 spec=importlib.util.spec_from_file_location('format_hybrid_table',d/'extract_table.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 with zipfile.ZipFile(R/'sources/WikiTables-WithLinks.zip') as z:
  names=set(z.namelist());prefix=z.namelist()[0].split('/')[0]
  for t in needed:
   n=f'{prefix}/tables_tok/{t}.json';p=f'{prefix}/request_tok/{t}.json'
   if n in names:tables.append({'psg_id':'hybridqa-'+t,'text':mod.format_table_as_text(json.loads(z.read(n)))})
   else:missing.append(t)
   if p in names:
    for idx,(wiki,text) in enumerate(json.loads(z.read(p)).items()):texts.append({'psg_id':f'hybridqa-{t}_{idx}','text':text})
   else:missingtext.append(t)
  tablefiles=sum('/tables_tok/' in n and n.endswith('.json') for n in names);passagefiles=sum('/request_tok/' in n and n.endswith('.json') for n in names)
 td=pd.DataFrame(tables).set_index('psg_id');tx=pd.DataFrame(texts).set_index('psg_id');td.to_parquet(d/'hybridqa_table.parquet');tx.to_parquet(d/'hybridqa_text.parquet');shutil.copy2(d/'hybridqa_table.parquet',U/'dataset/table.parquet');shutil.copy2(d/'hybridqa_text.parquet',U/'dataset/paragraph.parquet')
 info={'suite_questions':len(questions),'suite_tables':len(td),'suite_paragraphs':len(tx),'questions_with_all_source_files':sum(x['gt_tables'] not in set(missing+missingtext) for x in questions),'missing_tables':missing,'missing_passage_files':missingtext,'source_table_files':tablefiles,'source_passage_files':passagefiles,'raw_storage':'Complete ZIP; suite files read directly from ZIP to preserve Windows-incompatible filenames','sources':[json.loads((s/'SOURCE_DOWNLOAD.json').read_text()),json.loads((R/'sources/HybridQA/SOURCE_DOWNLOAD.json').read_text())]};save(M/'hybridqa.json',info);summary['hybridqa']=info

def dpr():
 p=A/'data/okvqa_knowledge_dpr_first3m/preparation_status.json'
 while True:
  await_file(p,3600);d=json.loads(p.read_text())
  if d.get('status')=='COMPLETE':break
  if time.time()-START>3600:raise TimeoutError('DPR build')
  time.sleep(15)
 info=json.loads((p.parent/'manifest.json').read_text());assert info['documents']==3000000;assert sha(p.parent/'corpus_wiki_first3m.jsonl')==info['corpus_sha256'];summary['dpr']={k:info[k] for k in ['documents','corpus_bytes','corpus_sha256','source_url']};summary['dpr']['raw_gzip_bytes']=4694541059;summary['dpr']['raw_md5']='612fe66e0b6b41ee28f806140226c563'

def video():
 p=M/'videorag_results.json';await_file(p,14400);r=json.loads(p.read_text());good=set(r['completed_ids']);dfiles=[]
 for name in ['video','clip']:
  raw=U/'dataset'/f'{name}.full.parquet';current=U/'dataset'/f'{name}.parquet'
  if not raw.exists():shutil.copy2(current,raw)
  df=pd.read_parquet(raw)
  def vid(i):return Path(i.rsplit('_',1)[0] if name=='clip' else i).stem
  keep=[i for i in df.index if 'videorag' in i and vid(i) in good];df.loc[keep].to_parquet(current);dfiles.append({'name':name,'rows':len(keep),'sha256':sha(current)})
 available=json.loads((M/'videorag_available_questions.json').read_text());save(U/'dataset/query/videorag_synth_available.json',available)
 summary['videorag']={k:r[k] for k in ['population','target','complete','fraction','bytes','gold_complete','gold_total','questions_total','questions_covered','missing_gold_ids','storage_stop','seed']};summary['videorag']['failed_attempts']=len(r['failed']);summary['videorag']['indexes']=dfiles
 from collections import Counter
 plan=json.loads((M/'videorag_plan.json').read_text(encoding='utf8'));pop=Counter(plan['strata'].values());actual=Counter(plan['strata'][i] for i in good)
 save(M/'videorag_strata_coverage.json',{k:{'population':n,'downloaded':actual[k],'coverage':actual[k]/n} for k,n in sorted(pop.items())})

def report():
 summary['errors']=errors;summary['webqa']='Excluded from dataset downloading';summary['generated_at_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());summary['remaining_free_bytes']=shutil.disk_usage(R).free
 total=0;count=0
 with (M/'files_inventory.jsonl').open('w',encoding='utf8') as out:
  for p in R.rglob('*'):
   if p.is_file():
    rel=p.relative_to(R)
    if '.git' in rel.parts or rel.parts[0]=='tools' or p==M/'files_inventory.jsonl':continue
    n=p.stat().st_size;total+=n;count+=1;out.write(json.dumps({'path':rel.as_posix(),'bytes':n})+'\n')
 summary['total_data_and_source_bytes']=total;summary['file_count_excluding_tools_and_git']=count;save(M/'SUMMARY.json',summary)
 lines=['# OmniAware 数据下载报告','',f'目录：{R}','',f'总文件容量（不含工具和 Git 对象）：{total/2**30:.2f} GiB；剩余 {summary["remaining_free_bytes"]/2**30:.2f} GiB。','', 'WebQA 未下载；模型权重、GPU 编码、检索索引训练不属于此次数据下载。','', '| 数据 | 下载数量／覆盖率 |','|---|---|']
 if 'okvqa' in summary:
  x=summary['okvqa'];lines.append(f'| OK-VQA | train 9,009 题 + val 5,046 题；图片 {x["images_complete"]}/{x["images_expected"]}；{x["bytes"]/2**30:.2f} GiB |')
 if 'gs' in summary:
  for x in summary['gs']:lines.append(f'| {x["file"]} | {x["rows"]:,} 行，原始 CSV + 可直接编码的 JSONL |')
 if 'dpr' in summary:lines.append('| DPR Wikipedia | 原始压缩包完整；按项目规则构建前 3,000,000 段，完整保留项目要求的 wiki3m |')
 if 'hybridqa' in summary:
  x=summary['hybridqa'];lines.append(f'| HybridQA | 完整官方原始仓库；UniversalRAG {x["questions_with_all_source_files"]}/{x["suite_questions"]} 题证据文件齐全，{x["suite_tables"]} 表、{x["suite_paragraphs"]:,} 段落 |')
 if 'videorag' in summary:
  x=summary['videorag'];lines.append(f'| VideoRAG-Synth | 视频 {x["complete"]}/{x["population"]}（{x["fraction"]:.2%}），{x["bytes"]/2**30:.2f} GiB；评测题覆盖 {x["questions_covered"]}/{x["questions_total"]}，gold 视频 {x["gold_complete"]}/{x["gold_total"]} |')
 lines+=['','## 可复现抽样','', '视频种子为 20261015。先保留全部 178 个评测源视频，再按字幕关键词主题 × 字幕长度四分位进行比例分层抽取候选视频，目标 1,725 个（约 20%）。失效候选在同一分层中按固定顺序补选；失效的评测源视频单独报告。未使用答案内容、模型输出或实验分数选择样本。','', '视频采用最高 480p，保存音轨和官方套件提供的字幕。该候选库是缩减设置，会改变检索难度；实验应明确报告子集设置，不与完整候选库结果直接等同。','', '详细 ID、抽样规则、源地址和输入哈希：manifests/videorag_plan.json。成功文件哈希和媒体校验：manifests/video_records/。失败视频：manifests/videorag_results.json。图片逐文件 SHA-256 和可读性：manifests/okvqa_images.json。总体机器可读汇总：manifests/SUMMARY.json。','', '## 使用','', '运行前设置 OMNIAWARE_ASSETS 为 assets 的绝对路径，OMNIAWARE_CACHE 为 cache，OMNIAWARE_UNIVERSALRAG_DIR 为 omniaware/external/UniversalRAG。Linux 迁移时保留目录结构；OK-VQA 输入图片路径已写为相对资产根目录的路径。','', 'GS-112K 是默认训练来源语料；GS-168K 包含评测题检索内容，仅供项目规定的附录对照。','', '下一步仍需安装实验环境和模型，并执行 BGE／CLIP／UniversalRAG 特征编码、共享候选池导出及视频抽帧；本次没有执行模型实验。', '', '原始视频评测问题文件保持完整。只有证据可用的问题写入 dataset/query/videorag_synth_available.json；若覆盖率低于 100%，必须显式采用可用问题子集或补齐缺失视频后再运行，不能静默跳题。','', '## 自动处理失败项','']
 if errors:lines += [f'- {e}' for e in errors]
 else:lines+=['处理脚本全部完成。个别源视频若已删除、私有或要求登录，详见上述失败清单与汇总中的缺失 ID。']
 (R/'下载报告.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print(json.dumps(summary,ensure_ascii=False),flush=True)

def run(name,fn):
 try:fn();save(M/f'{name}_preparation.done.json',{'status':'complete'})
 except Exception as e:
  errors.append(f'{name}: {e}');(LOG/f'{name}_finalize_error.log').write_text(traceback.format_exc(),encoding='utf8');save(M/f'{name}_preparation.done.json',{'status':'failed','error':str(e)})
with ThreadPoolExecutor(max_workers=5) as ex:list(ex.map(lambda x:run(*x),[('okvqa',okvqa),('gs',gs),('hybridqa',hybrid),('dpr',dpr),('video',video)]))
report()


