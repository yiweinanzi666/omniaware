import base64, io, inspect, concurrent.futures as cf
from PIL import Image, ImageFile
from PIL import __version__ as pv
print("PIL", pv, "LOAD_TRUNCATED_IMAGES =", ImageFile.LOAD_TRUNCATED_IMAGES)
buf = io.BytesIO(); Image.new("RGB", (64, 64), (200, 10, 10)).save(buf, format="JPEG", quality=95)
full = buf.getvalue(); trunc = full[: int(len(full) * 0.7)]
try:
    Image.open(io.BytesIO(base64.b64encode(trunc).decode() and trunc)).convert("RGB")
    print("truncated JPEG: decoded without error")
except Exception as e:
    print("truncated JPEG ->", type(e).__name__, e)
src = inspect.getsource(cf.Executor.map)
print("--- Executor.map source lines that matter:")
for line in src.splitlines():
    if "submit" in line or "fs =" in line or "zip(*iterables)" in line:
        print("   ", line.strip())
