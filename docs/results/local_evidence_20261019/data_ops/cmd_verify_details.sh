cd /home/user/Futaoran/omniaware_v2_exec
/home/user/Futaoran/omniaware_v2_python311/bin/python - <<'PY'
import json,pprint
d=json.load(open('runs/verify_real_model_8b.json'))
pprint.pp(d['checks']['segmented'])
pprint.pp(d['checks']['reference'])
PY
