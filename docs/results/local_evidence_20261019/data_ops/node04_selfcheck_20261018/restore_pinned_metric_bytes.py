"""Restore pinned checkout bytes only after confirming CRLF conversion is the sole difference."""
import pathlib,subprocess,hashlib,json
root=pathlib.Path('/home/user/Futaoran/omniaware_v2_format_20261018')
audit=pathlib.Path('/home/user/Futaoran/omniaware_selfcheck_logs_20261018/format_revision')
pins={'eval_1022.py':'6862a280454b576062695724b3417c9e4ce43af5447596ff4efca15d8d1e3f5c','eval_webqa.md':'c6dfcb9a7f8aed27c393e3d1296e038f37f10483de66b771963770482fea1ab3'}
report={}
for name,pin in pins.items():
    p=root/'third_party/webqa'/name
    before=p.read_bytes();blob=subprocess.check_output(['git','show','HEAD:third_party/webqa/'+name],cwd=root)
    assert hashlib.sha256(blob).hexdigest()==pin
    assert before.replace(b'\r\n',b'\n')==blob
    backup=audit/(name+'.checkout-crlf');assert not backup.exists();backup.write_bytes(before)
    p.write_bytes(blob)
    report[name]={'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'crlf_only':True,'backup':str(backup)}
(audit/'checkout_byte_repair.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
