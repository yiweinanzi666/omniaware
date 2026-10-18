import hashlib
import urllib.request

file_id = "1bcINVUPzlZRs_4ZImisZXmQOZFkQCNem"
urls = {
    "drive": f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
    "hf": "https://huggingface.co/datasets/Ancci/webqa_large/resolve/main/imgs.7z.001?download=true",
}
for name, url in urls.items():
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-1048575"})
        with urllib.request.urlopen(req, timeout=60) as response:
            data = response.read()
            print(name, response.status, response.geturl(), response.headers.get("Content-Range"), response.headers.get("Content-Type"), len(data), hashlib.sha256(data).hexdigest())
    except Exception as exc:
        print(name, type(exc).__name__, repr(exc))
