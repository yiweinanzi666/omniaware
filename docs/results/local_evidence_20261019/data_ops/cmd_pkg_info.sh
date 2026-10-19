ls -lh /home/user/anaconda3/pkgs/numpy-1.26.4-py312hc5e2394_0.conda /home/user/anaconda3/pkgs/numpy-base-1.26.4-py312h0da6c21_0.conda
python - <<'PY'
import zipfile,glob,os
for p in glob.glob('/home/user/anaconda3/pkgs/numpy*1.26.4*.conda'):
 print(p,os.path.getsize(p))
 with zipfile.ZipFile(p) as z: print(z.namelist())
PY
