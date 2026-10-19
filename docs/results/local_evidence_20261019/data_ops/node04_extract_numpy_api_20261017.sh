set -eu
/home/user/anaconda3/bin/python - <<'PY'
import inspect
from conda_package_handling import api
print(inspect.signature(api.extract))
print(inspect.getsource(api.extract))
PY
