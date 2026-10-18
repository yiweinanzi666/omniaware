/home/user/Futaoran/omniaware_v2_python311/bin/python - <<'PY'
from importlib.metadata import version
for p in ['numpy','scipy','scikit-learn','transformers','torch','pytest']:
 try: print(p,version(p))
 except Exception as e: print(p,e)
PY
