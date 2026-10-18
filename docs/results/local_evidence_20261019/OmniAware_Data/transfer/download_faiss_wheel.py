from pathlib import Path
import hashlib, requests, time

URL = "https://files.pythonhosted.org/packages/00/62/2b174dd024921d04f8d18e9e83285dac46772932164e27a438e6a07263f9/faiss_cpu-1.9.0.post1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
OUT = Path(__file__).resolve().parent / "models/extra_linux_wheels/faiss_cpu-1.9.0.post1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
OUT.parent.mkdir(parents=True, exist_ok=True)
total = 27_471_791
chunk = 1_048_576
for attempt in range(120):
    pos = OUT.stat().st_size if OUT.exists() else 0
    if pos >= total:
        break
    end = min(total - 1, pos + chunk - 1)
    try:
        r = requests.get(URL, headers={"Range": f"bytes={pos}-{end}"}, timeout=(15, 60))
        r.raise_for_status()
        cr = r.headers.get("Content-Range", "")
        if r.status_code != 206 or not cr.startswith(f"bytes {pos}-"):
            raise RuntimeError(f"bad range {r.status_code} {cr}")
        data = r.content
        expected = end - pos + 1
        if len(data) != expected:
            raise RuntimeError(f"short chunk {len(data)} != {expected}")
        with OUT.open("ab") as f:
            f.write(data)
        print(f"{OUT.stat().st_size}/{total}", flush=True)
    except Exception as e:
        print("retry", type(e).__name__, str(e)[:100], flush=True)
        time.sleep(2)
else:
    raise SystemExit("download did not finish")
if OUT.stat().st_size != total:
    raise SystemExit(f"bad size {OUT.stat().st_size}")
print("sha256", hashlib.sha256(OUT.read_bytes()).hexdigest(), flush=True)
