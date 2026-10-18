"""Fill only the diagnostic's missing relevance entries in the new isolated cache."""
import sys,pathlib
root=pathlib.Path('/home/user/Futaoran/omniaware_v2_selfcheck_20261018')
sys.path[:0]=[str(root),str(root/'src')]
from omniaware import data,paths,run
from scripts.analysis import image_evidence_check as check
examples=data.LOADERS['webqa_full']()
pool=sorted((e for e in examples if data.is_image_question(e)),key=lambda e:str(e['qid']))[:50]
assert len(pool)==50
run.load_relevance(pool,paths.LOCAL_CACHE/'relevance_webqa_full.json')
raise SystemExit(check.main(['--n','50','--output',str(paths.RUNS/'image_evidence_check.json')]))
