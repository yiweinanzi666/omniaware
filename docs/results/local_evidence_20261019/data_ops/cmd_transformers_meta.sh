kill 2280749 2280723 2>/dev/null || true
base=/home/user/Futaoran/omniaware_numpy126_overlay2
export PYTHONPATH="$base/lib/python3.12/site-packages:\${PYTHONPATH:-}"
/home/user/anaconda3/bin/python - <<'PY'
from importlib.metadata import version, distribution
print('TRANSFORMERS_VERSION',version('transformers'))
print('TRANSFORMERS_LOCATION',distribution('transformers').locate_file(''))
PY
