cd /home/user/Futaoran/omniaware_v2_exec || exit 1
pwd
git rev-parse HEAD
git status --short
sed -n '140,166p' scripts/verify_real_model.py
/home/user/anaconda3/bin/python - <<'PY'
import importlib, json, pathlib, sys
print('PYTHON', sys.version)
for name in ('torch', 'transformers', 'pytest', 'spacy', 'faiss', 'pyflakes'):
    try:
        module = importlib.import_module(name)
        print('IMPORT', name, getattr(module, '__version__', 'unknown'))
    except (ImportError, OSError) as exc:
        print('IMPORT_FAILED', name, type(exc).__name__, str(exc))
root = pathlib.Path('/home/user/Futaoran/omniaware')
marker = root / 'runs/qwen2b_fallback_dual_l20_smoke/okvqa_gs.merged.json'
if marker.exists():
    print('SMOKE_MARKER', marker.read_text())
for name in ('webqa_image_encode.log', 'webqa_search_val.log', 'webqa_search_train.log'):
    path = root / 'logs' / name
    if path.exists():
        print('LOG_TAIL', str(path))
        print('\n'.join(path.read_text(errors='replace').splitlines()[-8:]))
for pattern in ('assets/webqa_open/emb/*.manifest.json', 'assets/webqa_open/dense_top5_full/*/*/rankings.jsonl'):
    for path in root.glob(pattern):
        print('ARTIFACT', path, path.stat().st_size)
PY
find /home/user/Futaoran -maxdepth 3 -type f -name '*pytest*' -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' 2>/dev/null
