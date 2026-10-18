import urllib.request
u='https://s3.amazonaws.com/images.cocodataset.org/zips/val2014.zip'
req=urllib.request.Request(u,headers={'Range':'bytes=0-1048575','User-Agent':'Mozilla/5.0'})
with urllib.request.urlopen(req,timeout=60) as r:
    b=r.read(); print(r.status,len(b),r.headers.get('Content-Range'),r.headers.get('Content-Type'),flush=True)
