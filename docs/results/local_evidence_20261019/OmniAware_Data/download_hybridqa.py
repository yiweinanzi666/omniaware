from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests,zipfile,time,json,hashlib
R=Path(r'E:\OmniAware_Data');D=R/'sources';D.mkdir(exist_ok=True)
def job(repo):
 name=repo.split('/')[-1];dest=D/name;marker=dest/'SOURCE_DOWNLOAD.json'
 if marker.exists():return
 p=D/f'{name}.zip';part=p.with_suffix('.zip.part');sha=requests.get('https://api.github.com/repos/'+repo+'/commits/master',timeout=30).json()['sha'];u=f'https://codeload.github.com/{repo}/zip/{sha}'
 if not p.exists():
  for a in range(5):
   try:
    with requests.get(u,stream=True,timeout=(20,60)) as r:
     r.raise_for_status()
     with part.open('wb') as f:
      for b in r.iter_content(2**20):f.write(b)
    with zipfile.ZipFile(part) as z:assert z.testzip() is None
    part.replace(p);break
   except Exception:
    if a==4:raise
    time.sleep(3)
 with zipfile.ZipFile(p) as z:assert z.testzip() is None;entries=len(z.namelist())
 dest.mkdir(exist_ok=True)
 with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 marker.write_text(json.dumps({'repo':repo,'commit':sha,'url':u,'archive_sha256':digest,'archive_bytes':p.stat().st_size,'archive_entries':entries,'raw_storage':'Complete original ZIP, including filenames invalid on Windows; access directly with zipfile. Previously extracted legal paths are auxiliary only.'},indent=2))
with ThreadPoolExecutor(max_workers=2) as e:list(e.map(job,['wenhuchen/WikiTables-WithLinks','wenhuchen/HybridQA']))
