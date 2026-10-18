import datetime, hashlib, json, pathlib, subprocess

BASE = pathlib.Path('/home/user/Futaoran')
A = BASE / 'omniaware/assets'
L = A / 'manifests/v2_run_20261017'
R = BASE / 'omniaware_v2_exec'
out = {'start': datetime.datetime.now().astimezone().isoformat(), 'commands': [], 'files': {}, 'gpu_gate': 'STOP: unresolved asset and pytest failures'}

def command(args, cwd=R):
    rec = {'command': args, 'cwd': str(cwd), 'start': datetime.datetime.now().astimezone().isoformat()}
    r = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=45)
    rec.update(end=datetime.datetime.now().astimezone().isoformat(), exit_code=r.returncode, stdout=r.stdout, stderr=r.stderr)
    out['commands'].append(rec)

command(['git', 'status', '--porcelain=v1'])
command(['git', 'rev-parse', 'HEAD'])
command(['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu', '--format=csv,noheader'])
command(['tmux', 'list-windows', '-t', 'v2-prep-20261017'])
corpus = A.parent / 'cache/okvqa_gs112k/corpus.jsonl'
out['corpus'] = {'path': str(corpus), 'bytes': corpus.stat().st_size, 'sha256': hashlib.file_digest(corpus.open('rb'), 'sha256').hexdigest()}
out['missing_image_status'] = {name: (A / 'cache/webqa_full4966_images/images' / name).is_file() for name in ['30278219.jpg', '30045608.jpg', '30172755.jpg']}
for p in [L / 'environment_complete.json', L / 'pytest_064746_161869.result.json', L / 'source_csv_verified_0657.json', L / 'bartscore_verified_manifest.json', L / 'bartscore_finish.log', A / 'cache/bartscore/bart_score.pth', BASE / 'omniaware_v2_download_20261017/cache/bartscore/bart_score.pth', BASE / 'omniaware_v2_download_20261017/cache/bartscore/bart_score.pth.upload_20261017_v2_1']:
    item = {'exists': p.is_file()}
    if p.is_file():
        stat = p.stat()
        item.update(bytes=stat.st_size, modified=datetime.datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat())
        if p.suffix == '.json':
            item['content'] = json.loads(p.read_text())
    out['files'][str(p)] = item
out['end'] = datetime.datetime.now().astimezone().isoformat()
dest = L / ('heartbeat_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.json')
with dest.open('x') as f:
    json.dump(out, f, indent=2)
print(json.dumps({'report': str(dest), 'bytes': dest.stat().st_size, 'sha256': hashlib.file_digest(dest.open('rb'), 'sha256').hexdigest(), 'state': out}), flush=True)
