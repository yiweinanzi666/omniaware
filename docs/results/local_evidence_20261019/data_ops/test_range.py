import time
import urllib.request

u = "https://drive.usercontent.google.com/download?id=1bcINVUPzlZRs_4ZImisZXmQOZFkQCNem&export=download&confirm=t"
t0 = time.time()
req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-4194303"})
with urllib.request.urlopen(req, timeout=30) as response:
    data = response.read()
    print(response.status, len(data), response.headers.get("Content-Range"), round(time.time() - t0, 2), flush=True)
