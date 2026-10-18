cd /home/user/Futaoran/omniaware_v2_exec
/home/user/Futaoran/omniaware_v2_python311/bin/python - <<'PY'
import json
d=json.load(open('runs/verify_real_model_8b.json'))
print('ok',d['ok'],'dtype',d['dtype'],'max_pixels',d['max_pixels'])
for k,v in d['checks'].items():
 print(k, v.get('ok') if isinstance(v,dict) else None)
 if k=='reference':
  print({a:x['first_logits_max_abs_diff'] for a,x in v['text']['arms'].items()})
PY
git status --short
git rev-parse HEAD
