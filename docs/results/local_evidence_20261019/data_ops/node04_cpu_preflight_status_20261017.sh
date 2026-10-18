audit=/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/preflight_0242
date -Iseconds
for name in start.txt commit.txt git_status.txt exit_code.txt end.txt; do
  if [ -f "$audit/$name" ]; then
    printf '%s\n' "$name"
    cat "$audit/$name"
  fi
done
tail -35 "$audit/pytest.log"
cd /home/user/Futaoran/omniaware_v2_exec || exit 1
sed -n '143,160p' scripts/verify_real_model.py
/home/user/anaconda3/bin/python -u - <<'PY'
import importlib.metadata as m
import pathlib
import json
import sys
print('PYTHON', sys.version)
for name in ('torch', 'transformers', 'pytest', 'spacy', 'faiss-cpu', 'pyflakes'):
    try:
        print('VERSION', name, m.version(name))
    except m.PackageNotFoundError:
        print('MISSING_PACKAGE', name)
root = pathlib.Path('/home/user/Futaoran/omniaware')
for name in ('qwen3-vl-8b-instruct', 'bge-reranker-v2-m3', 'bartscore'):
    model = root / 'assets/cache' / name
    print('MODEL', name, [(p.name, p.stat().st_size) for p in model.glob('*') if p.is_file()])
for pattern in ('assets/webqa_open/emb/*.manifest.json', 'assets/webqa_open/dense_top5_full/*/*/rankings.jsonl'):
    for path in root.glob(pattern):
        print('ARTIFACT', str(path), path.stat().st_size)
marker = root / 'runs/qwen2b_fallback_dual_l20_smoke/okvqa_gs.merged.json'
if marker.is_file():
    print('FALLBACK_SMOKE_MARKER', marker.read_text())
PY
