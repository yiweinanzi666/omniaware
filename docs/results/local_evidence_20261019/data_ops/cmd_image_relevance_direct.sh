cd /home/user/Futaoran/omniaware_v2_exec
export OMNIAWARE_ASSETS=/home/user/Futaoran/omniaware/assets
export OMNIAWARE_CACHE=/home/user/Futaoran/omniaware_v2_exec/cache
nohup /home/user/Futaoran/omniaware_v2_python311/bin/python - <<'PY' > /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/image_relevance_direct_20261017.log 2>&1 &
from omniaware import data, run
ex = data.LOADERS["webqa_full"]()
imgs = sorted((e for e in ex if data.is_image_question(e)), key=lambda e: str(e["qid"]))[:50]
print("EXAMPLES", len(imgs), flush=True)
raw = run.load_relevance(imgs, run.paths.LOCAL_CACHE / "relevance_webqa_full.json", write=True)
print("DONE", len(raw), flush=True)
PY
echo PID=$!
