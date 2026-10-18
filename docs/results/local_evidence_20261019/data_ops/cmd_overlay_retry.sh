set -e
base=/home/user/Futaoran/omniaware_numpy126_overlay
rm -rf "$base"
mkdir -p "$base"
/home/user/anaconda3/bin/python - <<'PY'
from conda_package_handling import api
api.extract('/home/user/anaconda3/pkgs/numpy-1.26.4-py312hc5e2394_0.conda', dest_dir='/home/user/Futaoran/omniaware_numpy126_overlay')
api.extract('/home/user/anaconda3/pkgs/numpy-base-1.26.4-py312h0da6c21_0.conda', dest_dir='/home/user/Futaoran/omniaware_numpy126_overlay')
PY
export PYTHONPATH="$base/lib/python3.12/site-packages:\${PYTHONPATH:-}"
/home/user/anaconda3/bin/python - <<'PY'
import numpy, scipy, sklearn
print('VERSIONS',numpy.__version__,scipy.__version__,sklearn.__version__)
print('NUMPY_PATH',numpy.__file__)
PY
