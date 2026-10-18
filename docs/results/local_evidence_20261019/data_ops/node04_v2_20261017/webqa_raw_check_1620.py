import pathlib,json,hashlib,base64,io,datetime,sys
from PIL import Image
local=sys.platform=='win32'
W=pathlib.Path('D:/WWW2027/WebQA' if local else '/home/user/Futaoran/WebQA')
L=pathlib.Path('D:/WWW2027/data_ops/node04_v2_20261017' if local else '/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017')
ids=['30278219','30045608','30366390','30298493','30014375','30039288','30233625','30161353','30172755','30039112']
offsets=[int(x) for x in (W/'imgs.lineidx').read_text().splitlines()]
results=[]
with (W/'imgs.tsv').open('rb') as f:
    for iid in ids:
        f.seek(offsets[int(iid)-30000000]);line=f.readline();k,b64=line.rstrip(b'\r\n').split(b'\t',1);assert k.decode()==iid
        raw=base64.b64decode(b64,validate=True)
        r={'image_id':iid,'raw_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'head_hex':raw[:32].hex(),'tail_hex':raw[-24:].hex()}
        try:
            im=Image.open(io.BytesIO(raw));r['format']=im.format;r['size']=im.size;im.load();r['strict_decode_ok']=True
        except Exception as e:r['strict_decode_ok']=False;r['error']=str(e)
        results.append(r)
r={'time':datetime.datetime.now().astimezone().isoformat(),'source':str(W/'imgs.tsv'),'images':results}
name='webqa_raw_check_1620_'+('local' if local else 'server')+'.json'
with (L/name).open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r))
