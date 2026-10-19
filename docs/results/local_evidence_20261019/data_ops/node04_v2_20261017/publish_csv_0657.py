import datetime, hashlib, json, os, pathlib
base = pathlib.Path('/home/user/Futaoran')
stage = base / 'omniaware_v2_download_20261017/source_csv_0657'
dest = base / 'omniaware/assets/downloads/okvqa_gs'
log = base / 'omniaware/assets/manifests/v2_run_20261017/source_csv_verified_0657.json'
expected = {'okvqa_train_corpus.csv': 'f6646ad30d05166cfd32c281566a0032656035761e7d33acaf3267bfbb9d4997', 'okvqa_full_corpus.csv': '4ab2352a2bff6413ac74f1a18ef7ffcc8a15b5f77df97a5b6afd8375448128ae'}
rec = {'start': datetime.datetime.now().astimezone().isoformat(), 'files': [], 'existing_retrieval_cache_changed': False}
for name, sha in expected.items():
    p = stage / name
    actual = hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
    if actual != sha:
        raise RuntimeError('Staged source hash mismatch: ' + name)
    target = dest / name
    if target.exists() and hashlib.file_digest(target.open('rb'), 'sha256').hexdigest() != sha:
        raise RuntimeError('Existing destination hash mismatch: ' + name)
dest.mkdir(parents=True, exist_ok=True)
for name, sha in expected.items():
    p, target = stage / name, dest / name
    if not target.exists():
        os.link(p, target)
    rec['files'].append({'path': str(target), 'bytes': target.stat().st_size, 'sha256': hashlib.file_digest(target.open('rb'), 'sha256').hexdigest(), 'expected_sha256': sha})
rec['end'] = datetime.datetime.now().astimezone().isoformat()
with log.open('x') as f:
    json.dump(rec, f, indent=2)
print(json.dumps(rec), flush=True)
