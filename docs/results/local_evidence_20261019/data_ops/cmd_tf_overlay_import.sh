base=/home/user/Futaoran/omniaware_numpy126_overlay2
tf=/home/user/Futaoran/omniaware_transformers55_overlay
export PYTHONPATH="$base/lib/python3.12/site-packages:$tf/lib/python3.12/site-packages:\${PYTHONPATH:-}"
/home/user/anaconda3/bin/python - <<'PY'
import importlib.metadata
print('metadata',importlib.metadata.version('transformers'))
import transformers
print('import',transformers.__version__,transformers.__file__)
PY
