"""Diagnostic only: official WebQA decoder option; never publish or run experiments."""
import pathlib,json,hashlib,base64,io,datetime
from PIL import Image,ImageFile
W=pathlib.Path('/home/user/Futaoran/WebQA')
L=pathlib.Path('/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017')
D=L/'webqa_official_decode_trial_1620';D.mkdir()
source=json.loads((L/'webqa_raw_check_1620_server.json').read_text())
offsets=[int(x) for x in (W/'imgs.lineidx').read_text().splitlines()]
ImageFile.LOAD_TRUNCATED_IMAGES=True
r={'start':datetime.datetime.now().astimezone().isoformat(),'diagnostic_only':True,'published':False,'policy':'Official WebQA README recommends PIL ImageFile.LOAD_TRUNCATED_IMAGES=True; requires a dataset policy decision before adoption.','images':[]}
with (W/'imgs.tsv').open('rb') as f:
    for src in source['images']:
        iid=src['image_id'];f.seek(offsets[int(iid)-30000000]);k,b64=f.readline().rstrip(b'\r\n').split(b'\t',1);assert k.decode()==iid
        raw=base64.b64decode(b64,validate=True);assert hashlib.sha256(raw).hexdigest()==src['sha256']
        row={'image_id':iid,'source_sha256':src['sha256']}
        try:
            im=Image.open(io.BytesIO(raw)).convert('RGB');p=D/(iid+'.jpg');im.save(p,format='JPEG',quality=95)
            ImageFile.LOAD_TRUNCATED_IMAGES=False
            with Image.open(p) as verify:verify.load()
            ImageFile.LOAD_TRUNCATED_IMAGES=True
            row.update(ok=True,size=im.size,path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        except Exception as e:row.update(ok=False,error=str(e))
        r['images'].append(row)
r['end']=datetime.datetime.now().astimezone().isoformat()
with (L/'webqa_decode_trial_1620.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r),flush=True)
