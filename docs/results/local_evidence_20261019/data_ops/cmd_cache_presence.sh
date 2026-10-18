/home/user/Futaoran/omniaware_v2_python311/bin/python - <<'PY'
import json
from omniaware import data
d=json.load(open('/home/user/Futaoran/omniaware_v2_exec/cache/relevance_webqa_full.json'))
ex=data.LOADERS['webqa_full']()
imgs=sorted([e for e in ex if data.is_image_question(e)],key=lambda e:str(e['qid']))[:50]
print('present',sum(e['qid'] in d for e in imgs))
print('missing',[e['qid'] for e in imgs if e['qid'] not in d][:10])
print('firstcache',list(d)[-60:][:10])
PY
