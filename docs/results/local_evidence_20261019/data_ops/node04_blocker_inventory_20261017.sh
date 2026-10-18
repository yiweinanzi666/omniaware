echo 'BASE'
/home/user/anaconda3/bin/python - <<'PY'
import importlib.metadata as m
for n in ('numpy','scipy','scikit-learn','torch','transformers','pytest','spacy','faiss-cpu'):
 try: print(n,m.version(n))
 except Exception: print(n,'MISSING')
PY
echo 'CONDA_CACHE'
find /home/user/anaconda3/pkgs -maxdepth 1 -type f \( -iname 'numpy-*.conda' -o -iname 'scipy-*.conda' -o -iname 'transformers-*.conda' \) -printf '%f\n' | sort | tail -40
echo 'PIP_CACHE'
find /home/user/.cache/pip /home/user/.cache -type f \( -iname '*transformers*5.5*' -o -iname '*numpy*1.26*' -o -iname '*scipy*1.13*' \) -printf '%p %s\n' 2>/dev/null | head -50
echo 'M2RAG'
ls -la /home/user/Futaoran/omniaware/assets/cache/official_m2rag | head -20
echo 'DISK'
df -h /home/user
