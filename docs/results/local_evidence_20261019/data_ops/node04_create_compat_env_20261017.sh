set -eu
env=/home/user/Futaoran/omniaware_env_312_numpy126
if [ ! -x "$env/bin/python" ]; then
  /home/user/anaconda3/bin/conda create -y --offline -p "$env" python=3.12 numpy=1.26.4
fi
"$env/bin/python" - <<'PY'
import numpy
print('NUMPY',numpy.__version__,numpy.__file__)
PY
