from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv,json,hashlib,sys
R=Path(r'E:\OmniAware_Data');sys.path.insert(0,str(R/'tools'));import gdown
D=R/'assets/data/okvqa_gs';D.mkdir(parents=True,exist_ok=True)
def job(x):
 fid,name,n=x;p=D/name
 if not p.exists():gdown.download(id=fid,output=str(p),resume=True,quiet=True,timeout=(20,60),use_cookies=False)
 with p.open(encoding='utf8',newline='') as f:
  it=csv.DictReader(f);assert 'text' in it.fieldnames;assert sum(1 for _ in it)==n
 return name
with ThreadPoolExecutor(max_workers=2) as e:list(e.map(job,[('1O8HkDzWIMc-uzLgG3BqHz-g2qtGeRcT6','okvqa_train_corpus.csv',112724),('1x9unz58_Lw7oLgTO73nslQ92xr4un6bl','okvqa_full_corpus.csv',168306)]))
