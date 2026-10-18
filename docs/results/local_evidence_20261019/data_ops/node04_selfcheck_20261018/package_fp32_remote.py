"""Collect complete validation evidence; excludes locks/caches and never changes source or old runs."""
import pathlib,tarfile,json,hashlib
root=pathlib.Path('/home/user/Futaoran/omniaware_v2_selfcheck_20261018')
ops=pathlib.Path('/home/user/Futaoran/omniaware_selfcheck_logs_20261018')
out=ops/'fp32_revision'
files=[]
for p in (root/'runs/fp32_20261018').rglob('*'):
    if p.is_file() and p.suffix not in ['.lock','.tmp']:files.append((p,p.relative_to(root)))
for p in out.rglob('*'):
    if p.is_file() and p.suffix not in ['.lock','.tar']:files.append((p,pathlib.Path('docs/results/node04_fp32_validation_20261018')/p.relative_to(out)))
for name in ['cpu_fp32_revision.sh','probe_fp32_multimage.py','run_fp32_validation.sh','continue_fp32_s1.sh','prepare_image_review.py','audit_fp32_results.py']:
    files.append((ops/name,pathlib.Path('docs/results/node04_fp32_validation_20261018')/name))
manifest={'source_commit':'a11e234fc2741fc122eb0a7bb6b5d7cd6f9f2ee8','files':{str(rel):{'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} for p,rel in files}}
manifestpath=ops/'fp32_manifest.json';manifestpath.write_text(json.dumps(manifest,indent=2))
with tarfile.open(ops/'fp32_completed_evidence.tar','w') as tar:
    for p,rel in files:tar.add(p,arcname=str(rel))
    tar.add(manifestpath,arcname='docs/results/node04_fp32_validation_20261018/MANIFEST.json')
print(json.dumps({'files':len(files),'bytes':sum(x['bytes'] for x in manifest['files'].values())}))
