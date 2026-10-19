import datetime, hashlib, json, os, pathlib, subprocess, sys

ROOT = pathlib.Path('/home/user/Futaoran/omniaware_v2_exec')
LOG = pathlib.Path('/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017')
os.environ.update(OMNIAWARE_ASSETS=str(LOG.parents[1]), OMNIAWARE_CACHE='/home/user/Futaoran/omniaware/cache', OMNIAWARE_WEBQA_OPEN_DIR='/home/user/Futaoran/omniaware/assets/webqa_open')
os.chdir(ROOT)

def run(cmd, name):
    stamp = datetime.datetime.now().strftime('%H%M%S_%f')
    dest = LOG / (name + '_' + stamp + '.log')
    rec = {'command': cmd, 'cwd': str(ROOT), 'start': datetime.datetime.now().astimezone().isoformat(), 'log': str(dest)}
    with dest.open('x') as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    rec.update(end=datetime.datetime.now().astimezone().isoformat(), exit_code=proc.returncode, bytes=dest.stat().st_size, sha256=hashlib.file_digest(dest.open('rb'), 'sha256').hexdigest())
    with dest.with_suffix('.json').open('x') as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec), flush=True)

code = '''import json
from omniaware import data
for ds in ("okvqa_gs", "webqa_full", "webqa_open_img"):
    universe = data.IMAGE_SUBSET_OF.get(ds, ds)
    print("LOADING", ds, universe, flush=True)
    try:
        ex = data.LOADERS[universe]()
        expected = 5046 if ds == "okvqa_gs" else 4966
        print(json.dumps({"dataset": ds, "universe": universe, "count": len(ex), "meta": ex.meta, "count_ok": len(ex) == expected}), flush=True)
        if len(ex) != expected or ex.meta.get("dropped"):
            print("DATASET_STOP", ds, flush=True)
            continue
        if universe in data.TRAIN_LOADERS:
            tr = data.TRAIN_LOADERS[universe]()
            print(json.dumps({"dataset": ds, "train_count": len(tr), "train_meta": tr.meta}), flush=True)
    except Exception as e:
        print(json.dumps({"dataset": ds, "error": type(e).__name__, "message": str(e)}), flush=True)
'''
run([sys.executable, '-u', '-c', code], 'loader_acceptance')
run([sys.executable, '-u', '-m', 'scripts.setup.check_assets'], 'check_assets')
