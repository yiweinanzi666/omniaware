import urllib.request

id = "1bcINVUPzlZRs_4ZImisZXmQOZFkQCNem"
urls = [
    f"https://drive.google.com/uc?export=download&id={id}&confirm=t",
    f"https://drive.google.com/uc?id={id}&export=download&confirm=t",
    f"https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t",
    f"https://drive.googleusercontent.com/download?id={id}&export=download&confirm=t",
    f"https://docs.google.com/uc?export=download&id={id}&confirm=t",
    f"https://drive.usercontent.google.com/download?export=download&confirm=t&id={id}&authuser=0",
]
for u in urls:
    try:
        req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-1048575"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read(4096)
            print(r.status, r.geturl(), r.headers.get("Content-Range"), r.headers.get("Content-Type"), len(data), data[:20])
    except Exception as e:
        print(type(e).__name__, repr(e))
