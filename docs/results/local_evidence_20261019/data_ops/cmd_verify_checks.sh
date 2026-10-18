cd /home/user/Futaoran/omniaware_v2_exec
/home/user/Futaoran/omniaware_v2_python311/bin/python - <<'PY'
import json,pprint
d=json.load(open('runs/verify_real_model_8b.json'))
for k,v in d['checks'].items():
 print('\\nCHECK',k)
 if isinstance(v,dict):
  print('ok',v.get('ok'))
  if k=='reference':
   for arm,x in v.items():
    if isinstance(x,dict): print(arm,{a:x.get(a) for a in ('ok','first_token_max_abs_diff','t_end_attention_max_abs_diff','tokens_match','argmax_match')})
  else:
   for a,b in v.items():
    if a=='ok' or not isinstance(b,(dict,list)): print(a,b)
    elif isinstance(b,dict): print(a,{kk:b.get(kk) for kk in ('ok','max_abs_diff','argmax_match','error')})
 print()
PY
