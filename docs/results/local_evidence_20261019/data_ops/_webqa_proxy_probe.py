import os
import time
import urllib.request


URL = "https://huggingface.co/datasets/Ancci/webqa_large/resolve/main/imgs.7z.051?download=true"
started = time.time()
received = 0
try:
    request = urllib.request.Request(
        URL,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Range": "bytes=0-1048575",
            "Accept-Encoding": "identity",
        },
    )
    with urllib.request.urlopen(request, timeout=35) as response:
        while True:
            block = response.read(256 * 1024)
            if not block:
                break
            received += len(block)
    elapsed = time.time() - started
    print(
        f"{os.environ.get('PROBE_NODE')}\t{received}\t{elapsed:.2f}\t"
        f"{received / elapsed / 1024 / 1024:.3f} MiB/s",
        flush=True,
    )
except Exception as exc:
    print(
        f"{os.environ.get('PROBE_NODE')}\tERROR\t{type(exc).__name__}: {exc}\t"
        f"{time.time() - started:.2f}s",
        flush=True,
    )
