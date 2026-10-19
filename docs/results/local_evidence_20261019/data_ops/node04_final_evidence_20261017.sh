date -Iseconds
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
tail -30 /home/user/Futaoran/omniaware/logs/verify_real_model_qwen2b_20261016.log
/home/user/anaconda3/bin/python -u - <<'PY'
import pathlib,json,hashlib
root=pathlib.Path('/home/user/Futaoran/omniaware')
run=root/'runs/qwen2b_fallback_dual_l20_smoke'
marker=json.loads((run/'okvqa_gs.merged.json').read_text())
config=json.loads((run/'okvqa_gs_config.json').read_text())
print('FALLBACK_CONFIG',json.dumps({k:v for k,v in config.items() if k in ('model','model_path','v2','n_eval','n_calib','max_new_tokens','selector','split','n_dev')}))
for field in ('output','config'):
    path=run/marker[field]
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    print('SMOKE_HASH',field,actual,'matches_marker',actual==marker[field+'_sha256'])
for pattern in ('assets/webqa_open/dense_top5_full/*/*/rankings.jsonl',):
    for path in root.glob(pattern):
        digest=hashlib.sha256()
        lines=0
        with path.open('rb') as stream:
            for line in stream:
                digest.update(line)
                lines+=1
        print('RANKING',json.dumps({'path':str(path),'lines':lines,'bytes':path.stat().st_size,'sha256':digest.hexdigest()}))
PY
