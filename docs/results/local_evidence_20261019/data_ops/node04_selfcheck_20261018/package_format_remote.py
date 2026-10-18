"""Snapshot finished format verification evidence, preserving failures and omitting caches/locks."""
import pathlib,tarfile,json,hashlib,subprocess
root=pathlib.Path('/home/user/Futaoran/omniaware_v2_format_20261018')
ops=pathlib.Path('/home/user/Futaoran/omniaware_selfcheck_logs_20261018')
logroot=ops/'format_revision'
assert (logroot/'retry1/validation.exit').is_file()
assert (logroot/'retry1/review.exit').is_file()
assert not (ops/'format_completed_evidence.tar').exists()
files=[]
for p in (root/'runs/fp32_format_20261018').rglob('*'):
    if p.is_file() and p.suffix not in ['.lock','.tmp']:files.append((p,p.relative_to(root)))
for p in logroot.rglob('*'):
    if p.is_file() and p.suffix not in ['.lock','.tar']:files.append((p,pathlib.Path('docs/results/node04_format_validation_20261018')/p.relative_to(logroot)))
for name in ['run_format_validation.sh','run_format_validation_retry1.sh','run_format_review.sh','restore_pinned_metric_bytes.py','audit_format_results.py','prepare_format_image_review.py','format_webqa_manual_review.json','format_okvqa_manual_review.json']:
    files.append((ops/name,pathlib.Path('docs/results/node04_format_validation_20261018')/name))
manifest={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'files':{str(rel):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p,rel in files}}
m=ops/'format_completed_manifest.json';m.write_text(json.dumps(manifest,indent=2))
with tarfile.open(ops/'format_completed_evidence.tar','w') as tar:
    for p,rel in files:tar.add(p,arcname=str(rel))
    tar.add(m,arcname='docs/results/node04_format_validation_20261018/MANIFEST.json')
print(json.dumps({'files':len(files),'bytes':sum(x['bytes'] for x in manifest['files'].values())}))
