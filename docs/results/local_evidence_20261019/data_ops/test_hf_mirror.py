import time
import urllib.request

urls = [
    "https://hf-mirror.com/datasets/Ancci/webqa_large/resolve/main/imgs.7z.001?download=true",
    "https://huggingface.co/datasets/Ancci/webqa_large/resolve/main/imgs.7z.001?download=true&xet=0",
]
for u in urls:
    t0 = time.time()
    try:
        req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-4194303"})
        with urllib.request.urlopen(req, timeout=90) as response:
            data = response.read()
            print(response.status, len(data), response.headers.get("Content-Range"), round(time.time() - t0, 2), response.geturl()[:100], flush=True)
    except Exception as exc:
        print(type(exc).__name__, repr(exc), round(time.time() - t0, 2), flush=True)
