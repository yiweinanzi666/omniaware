"""Delete only user-approved task-data files matched by two-host SHA256 evidence."""
import pathlib,json,hashlib,datetime,sys,os,shutil
L=pathlib.Path('D:/WWW2027/data_ops/node04_v2_20261017');W=pathlib.Path('D:/WWW2027').resolve()
C=json.loads((L/'cleanup_comparison_1725.json').read_text(encoding='utf8'));P=L/'cleanup_eligible_1725.jsonl'
assert hashlib.sha256(P.read_bytes()).hexdigest()==C['eligible_manifest_sha256']
roots=[(W/g[0]).resolve() for g in C['groups']]
assert all(r.is_relative_to(W) and r!=W for r in roots)
assert C['matched_files']>0 and C['deletion_performed'] is False
execute='--execute' in sys.argv
if execute:
    gate=json.loads((L/'github_push_verified.json').read_text(encoding='utf8'))
    assert gate['branch']=='results/node04-preparation-20261017' and gate['commit']==gate['remote_sha'] and gate['push_exit_code']==0
    assert json.loads((L/'publish_webqa_1631.json').read_text())['ok']
    assert json.loads((L/'lf_checkout_pytest_1631.json').read_text())['ok']
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');stem=('cleanup_executed_' if execute else 'cleanup_dryrun_')+stamp
count=deleted=bytes_deleted=skipped=0;before=shutil.disk_usage(W).free;start=datetime.datetime.now().astimezone().isoformat()
with P.open() as src,(L/(stem+'.jsonl')).open('x',encoding='utf8') as journal:
    for line in src:
        row=json.loads(line);count+=1;rel=pathlib.PureWindowsPath(row['r']);entry={'g':row['g'],'r':row['r'],'bytes':row['b'],'sha256':row['sha256']}
        try:
            assert not rel.is_absolute() and not rel.drive and '..' not in rel.parts
            root=roots[row['g']];p=(root/row['r']).resolve(strict=True)
            assert p.is_relative_to(root) and p!=root and p.is_relative_to(W)
            assert p.suffix.lower() not in ('.py','.ps1','.sh','.md','.log','.git','.bundle')
            assert not p.name in ('assets.jsonl','bad_images.jsonl','missing_images.jsonl')
            assert row['g']!=9 or row['r']=='bart_score.pth'
            st=p.stat();assert st.st_size==row['b'] and st.st_mtime_ns==row['m'],'changed since SHA256 verification'
            assert p.is_file()
            if execute:p.unlink();deleted+=1;bytes_deleted+=row['b'];entry['action']='deleted'
            else:entry['action']='eligible'
        except Exception as e:entry.update(action='retained',reason=str(e));skipped+=1
        journal.write(json.dumps(entry,ensure_ascii=False,separators=(',',':'))+'\n')
        if count%10000==0:
            journal.flush();print(json.dumps({'checked':count,'deleted':deleted,'deleted_bytes':bytes_deleted,'retained':skipped}),flush=True)
assert count==C['matched_files']
r={'start':start,'end':datetime.datetime.now().astimezone().isoformat(),'execute':execute,'eligible_files_checked':count,'deleted_files':deleted,'deleted_logical_bytes':bytes_deleted,'retained_after_verification':skipped,'unmatched_files_preserved':C['retained_files'],'free_bytes_before':before,'free_bytes_after':shutil.disk_usage(W).free,'journal':str(L/(stem+'.jsonl')),'directories_deleted':0,'code_and_reports_preserved':True,'eligible_manifest_sha256':C['eligible_manifest_sha256']}
with (L/(stem+'.json')).open('x',encoding='utf8') as f:json.dump(r,f,indent=2,ensure_ascii=False)
print(json.dumps(r),flush=True)
