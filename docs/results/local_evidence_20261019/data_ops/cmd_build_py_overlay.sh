set -e
tf=/home/user/Futaoran/omniaware_transformers55_overlay
rm -rf "$tf"
mkdir -p "$tf/lib/python3.12/site-packages"
cp -a /home/user/Futaoran/omniaware_v2_python311/lib/python3.11/site-packages/transformers "$tf/lib/python3.12/site-packages/"
cp -a /home/user/Futaoran/omniaware_v2_python311/lib/python3.11/site-packages/transformers-5.5.3.dist-info "$tf/lib/python3.12/site-packages/"
export PYTHONPATH="/home/user/Futaoran/omniaware_numpy126_overlay2/lib/python3.12/site-packages:$tf/lib/python3.12/site-packages:\${PYTHONPATH:-}"
/home/user/anaconda3/bin/python - <<'PY'
from importlib.metadata import version
import numpy, scipy, sklearn, transformers
print('VERSIONS',numpy.__version__,scipy.__version__,sklearn.__version__,transformers.__version__)
print('TRANSFORMERS_PATH',transformers.__file__)
PY
