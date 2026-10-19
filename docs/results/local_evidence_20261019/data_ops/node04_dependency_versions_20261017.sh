/home/user/anaconda3/bin/python - <<'PY'
import importlib.metadata as m
for name in ('numpy','scipy','scikit-learn','torch','transformers','pillow','torchvision'):
    try: print(name, m.version(name))
    except Exception as exc: print(name, type(exc).__name__, exc)
PY
python -m pip show numpy scipy scikit-learn 2>/dev/null | sed -n '1,80p'
conda list numpy scipy scikit-learn 2>/dev/null | tail -20 || true
