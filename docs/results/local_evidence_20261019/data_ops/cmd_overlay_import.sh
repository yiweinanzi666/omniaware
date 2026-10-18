base=/home/user/Futaoran/omniaware_numpy126_overlay2
export PYTHONPATH="$base/lib/python3.12/site-packages:\${PYTHONPATH:-}"
/home/user/anaconda3/bin/python - <<'PY'
import numpy, scipy, sklearn
print('VERSIONS',numpy.__version__,scipy.__version__,sklearn.__version__)
print('NUMPY_PATH',numpy.__file__)
print('SCIPY_PATH',scipy.__file__)
print('SKLEARN_PATH',sklearn.__file__)
PY
