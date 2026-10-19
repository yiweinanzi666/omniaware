import datetime, hashlib, json, os, pathlib, sys, traceback

L = pathlib.Path('/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017')
R = pathlib.Path('/home/user/Futaoran/omniaware_v2_exec')
os.environ['OMNIAWARE_ASSETS'] = '/home/user/Futaoran/omniaware/assets'
os.environ['OMNIAWARE_WEBQA_OPEN_DIR'] = '/home/user/Futaoran/omniaware/assets/webqa_open'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.chdir(R)
sys.path.insert(0, str(R / 'src'))
record = {'start': datetime.datetime.now().astimezone().isoformat(), 'command': [sys.executable, str(pathlib.Path(__file__).resolve())], 'cwd': str(R), 'purpose': 'CPU-only BARTScore asset integrity and strict weight loading; no experiment inference'}
try:
    from omniaware.eval.webqa_bart import load_parabank
    scorer, metadata = load_parabank(device='cpu')
    record['metadata'] = metadata
    expected = '4c076ee386795a831e0021788148becf57136127d6512ca350c7973f36c0e573'
    record['ok'] = (metadata['weights']['sha256'] == expected and metadata['weights']['bytes'] == 1625565295 and not metadata['strict_state_dict_missing_keys'] and not metadata['strict_state_dict_unexpected_keys'])
    record['exit_code'] = 0 if record['ok'] else 1
except Exception:
    record.update(ok=False, exit_code=1, traceback=traceback.format_exc())
record['end'] = datetime.datetime.now().astimezone().isoformat()
dest = L / 'bartscore_cpu_asset_check_0750.json'
with dest.open('x') as f:
    json.dump(record, f, indent=2)
print(json.dumps({'report': str(dest), 'ok': record['ok'], 'exit_code': record['exit_code'], 'sha256': hashlib.file_digest(dest.open('rb'), 'sha256').hexdigest()}), flush=True)
sys.exit(record['exit_code'])
