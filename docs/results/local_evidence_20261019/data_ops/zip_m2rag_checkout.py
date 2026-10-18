from pathlib import Path
import zipfile
root = Path(r"D:\WWW2027\OmniAware_Data\omniaware\external\M2RAG")
out = Path(r"D:\WWW2027\OmniAware_Data\transfer\models\m2rag_official_checkout.zip")
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in root.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(root).as_posix())
print(out, out.stat().st_size)
