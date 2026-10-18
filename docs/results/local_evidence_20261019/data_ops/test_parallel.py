import concurrent.futures
import time
import urllib.request

base = "https://huggingface.co/datasets/Ancci/webqa_large/resolve/main/imgs.7z.001?download=true"

def one(i):
    start = i * 4 * 1024 * 1024
    end = start + 4 * 1024 * 1024 - 1
    req = urllib.request.Request(base, headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={start}-{end}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
        return i, len(data), time.time() - t0, r.status

t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    out = list(pool.map(one, range(8)))
print(out)
print("total_bytes", sum(x[1] for x in out), "elapsed", round(time.time()-t0,2), "aggregate_mib_s", round(sum(x[1] for x in out)/(time.time()-t0)/1024/1024,3), flush=True)
