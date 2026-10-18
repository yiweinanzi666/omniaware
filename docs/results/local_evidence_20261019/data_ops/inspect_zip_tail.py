import struct, urllib.request
u='https://s3.amazonaws.com/images.cocodataset.org/zips/val2014.zip'
size=6645013297
start=size-8*1024*1024
req=urllib.request.Request(u,headers={'Range':f'bytes={start}-{size-1}','User-Agent':'Mozilla/5.0'})
with urllib.request.urlopen(req,timeout=180) as r: b=r.read(); print(r.status,len(b),r.headers.get('Content-Range'))
for sig in (b'PK\x05\x06',b'PK\x06\x06',b'PK\x06\x07'):
    pos=[]; off=0
    while True:
        p=b.find(sig,off)
        if p<0: break
        pos.append(start+p); off=p+1
    print(sig,pos[-10:])
if b'PK\x05\x06' in b:
    p=b.rfind(b'PK\x05\x06'); print('eocd',struct.unpack_from('<4s4H2LH',b,p))
if b'PK\x06\x06' in b:
    p=b.rfind(b'PK\x06\x06'); print('zip64eocd',struct.unpack_from('<4sQ2H2L4Q',b,p))
if b'PK\x06\x07' in b:
    p=b.rfind(b'PK\x06\x07'); print('zip64locator',struct.unpack_from('<4sQQL',b,p))
