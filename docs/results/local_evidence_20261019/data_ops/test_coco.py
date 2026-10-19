import urllib.request
u='https://images.cocodataset.org/zips/val2014.zip'
for method in ('HEAD','GET'):
    try:
        req=urllib.request.Request(u,method=method,headers={'Range':'bytes=0-1048575','User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req,timeout=60) as r:
            data=r.read(1024*1024)
            print(method,r.status,r.headers.get('Content-Length'),r.headers.get('Content-Range'),r.headers.get('Content-Type'),len(data),flush=True)
    except Exception as e: print(method,type(e).__name__,repr(e),flush=True)
