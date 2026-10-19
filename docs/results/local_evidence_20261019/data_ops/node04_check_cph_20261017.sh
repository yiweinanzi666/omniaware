/home/user/anaconda3/bin/python - <<'PY'
import importlib.util
for n in ('conda_package_handling','conda_package_handling.api','zstandard','libarchive'):
 print(n,importlib.util.find_spec(n))
PY
find /home/user/anaconda3/bin -maxdepth 1 -iname '*conda*' -printf '%f\n' | sort
