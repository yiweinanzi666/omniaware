import json
from huggingface_hub import HfApi
man = json.load(open("src/omniaware/eval/citation/checker_asset_manifest.json"))
print("manifest keys:", list(man.keys()))
print("manifest files:", [(f["file"], f.get("sha256","")[:12]) for f in man["files"]])
info = HfApi().model_info(man["repo"] if "repo" in man else "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                          revision=man["revision"], files_metadata=True)
for s in info.siblings:
    lfs = s.lfs.sha256 if s.lfs else None
    recs = [f for f in man["files"] if f["file"] == s.rfilename]
    tag = "in-manifest" if recs else "NOT-in-manifest"
    same = ""
    if recs and lfs:
        same = "sha==LFS" if recs[0].get("sha256") == lfs else "sha!=LFS"
    print(f"{s.rfilename:40s} {tag:16s} {same}  size={s.size}")
