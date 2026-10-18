export PYTHONPATH=/home/user/anaconda3/pkgs/numpy-1.26.4-py312hc5e2394_0/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}
/home/user/anaconda3/bin/python - <<'PY'
import numpy, scipy, sklearn
print('NUMPY',numpy.__version__,numpy.__file__)
print('SCIPY',scipy.__version__)
print('SKLEARN',sklearn.__version__)
PY
