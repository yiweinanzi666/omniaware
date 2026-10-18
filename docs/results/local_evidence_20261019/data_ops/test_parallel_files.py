import concurrent.futures
import time
import urllib.request

names = [f"imgs.7z.{i:03d}" for i in range(1, 9)]

def one(name):
    u = f"https://huggingface.co/datasets/Ancci/webqa_large/resolve/main/{name}?download=true"
    req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-4194303"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
        return name, len(data), time.time() - t0, r.status, r.geturl().split("?")[0]

t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    out = list(pool.map(one, names))
elapsed = time.time() - t0
print(out)
print("total_bytes", sum(x[1] for x in out), "elapsed", round(elapsed, 2), "aggregate_mib_s", round(sum(x[1] for x in out)/elapsed/1024/1024, 3), flush=True)
