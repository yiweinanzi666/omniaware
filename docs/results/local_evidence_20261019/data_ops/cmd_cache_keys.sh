/home/user/Futaoran/omniaware_v2_python311/bin/python - <<'PY'
import json
p='/home/user/Futaoran/omniaware_v2_exec/cache/relevance_webqa_full.json'
d=json.load(open(p))
print(type(d), len(d))
for k in list(d)[:5]: print(k, d[k].keys() if isinstance(d[k],dict) else type(d[k]))
print('target', 'd5bbc7c80dba11ecb1e81171463288e9' in d)
PY
