import time
import urllib.request

u = "https://huggingface.co/datasets/Ancci/webqa_large/resolve/main/imgs.7z.001?download=true"
t0 = time.time()
req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-67108863"})
with urllib.request.urlopen(req, timeout=60) as response:
    data = response.read()
    print(response.status, len(data), response.headers.get("Content-Range"), round(time.time() - t0, 2), flush=True)
