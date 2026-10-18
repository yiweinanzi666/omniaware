cd /home/user/Futaoran/omniaware_v2_exec
export OMNIAWARE_ASSETS=/home/user/Futaoran/omniaware/assets
export OMNIAWARE_CACHE=/home/user/Futaoran/omniaware_v2_exec/cache
/home/user/Futaoran/omniaware_v2_python311/bin/python - <<'PY'
from omniaware import data
ex=data.LOADERS['webqa_full']()
imgs=[e for e in ex if data.is_image_question(e)]
imgs=sorted(imgs,key=lambda e:str(e['qid']))[:50]
pos={e['qid']:i for i,e in enumerate(ex)}
print('image_count',len(imgs),'maxpos',max(pos[e['qid']] for e in imgs),'minpos',min(pos[e['qid']] for e in imgs))
print('first',[(e['qid'],pos[e['qid']]) for e in imgs[:10]])
PY
