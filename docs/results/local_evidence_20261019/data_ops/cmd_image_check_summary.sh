/home/user/Futaoran/omniaware_v2_python311/bin/python - <<'PY'
import json
p='/home/user/Futaoran/omniaware_v2_exec/runs/image_evidence_check.json'
d=json.load(open(p))
print('mode',d.get('mode'),'questions',len(d.get('questions',[])))
print('summary')
for k,v in d['summary'].items(): print(k,v)
print('findings')
for x in d['findings']: print('-',x)
PY
