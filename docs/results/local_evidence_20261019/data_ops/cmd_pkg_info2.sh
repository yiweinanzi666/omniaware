/home/user/anaconda3/bin/python - <<'PY'
import zipfile,glob,os
for p in glob.glob('/home/user/anaconda3/pkgs/numpy*1.26.4*.conda'):
 print(p,os.path.getsize(p))
 with zipfile.ZipFile(p) as z: print(z.namelist())
PY
