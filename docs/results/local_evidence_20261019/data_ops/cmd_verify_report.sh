cd /home/user/Futaoran/omniaware_v2_exec
/home/user/Futaoran/omniaware_v2_python311/bin/python - <<'PY'
import json
p='runs/verify_real_model_8b.json'
d=json.load(open(p))
print('TOP_OK',d.get('ok'))
for k,v in d.items():
 if isinstance(v,dict) and 'ok' in v:
  print(k,'ok=',v.get('ok'))
  for kk in ('max_abs_diff','first_token_max_abs_diff','t_end_attention_max_abs_diff','argmax_match','tokens_match','error'):
   if kk in v: print(' ',kk,v[kk])
print('keys',list(d))
PY
tail -80 /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/verify_real_model_8b_20261017.log
