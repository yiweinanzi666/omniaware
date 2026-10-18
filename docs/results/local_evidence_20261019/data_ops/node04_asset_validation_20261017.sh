cd /home/user/Futaoran/omniaware_v2_exec || exit 1
export OMNIAWARE_ASSETS=/home/user/Futaoran/omniaware/assets
export OMNIAWARE_CACHE=/home/user/Futaoran/omniaware/cache
export OMNIAWARE_WEBQA_OPEN_DIR=/home/user/Futaoran/omniaware/assets/webqa_open
/home/user/anaconda3/bin/python -u - <<'PY'
import sys, json, pathlib, hashlib, time
sys.path.insert(0, 'src')
from omniaware import data
root = pathlib.Path('/home/user/Futaoran/omniaware')
for ds in ('webqa_full', 'webqa_open_img', 'okvqa_gs'):
    start = time.monotonic()
    try:
        universe = data.IMAGE_SUBSET_OF.get(ds, ds)
        examples = data.LOADERS[universe]()
        print('LOADER', json.dumps({'dataset': ds, 'universe': universe, 'count': len(examples), 'meta': examples.meta, 'seconds': time.monotonic()-start}), flush=True)
    except Exception as exc:
        print('LOADER_FAILED', ds, type(exc).__name__, str(exc), flush=True)
for pattern in ('assets/webqa_open/dense_top5_full/*/*/rankings.jsonl', 'cache/okvqa_gs112k/corpus.jsonl', 'cache/okvqa_gs112k/candidates_full.jsonl'):
    for path in root.glob(pattern):
        digest = hashlib.sha256()
        count = 0
        with path.open('rb') as handle:
            for line in handle:
                digest.update(line)
                count += 1
        print('VERIFIED_FILE', json.dumps({'path': str(path), 'bytes': path.stat().st_size, 'lines': count, 'sha256': digest.hexdigest()}), flush=True)
PY
ps -eo pid,etime,pcpu,pmem,args | grep -E 'pytest|node04_preflight_detail|python -u -' | grep -v grep
