base=/home/user/Futaoran/omniaware_numpy126_overlay2
export PYTHONPATH="$base/lib/python3.12/site-packages:\${PYTHONPATH:-}"
/home/user/anaconda3/bin/python - <<'PY'
import transformers
print(transformers.__version__, transformers.__file__)
PY
