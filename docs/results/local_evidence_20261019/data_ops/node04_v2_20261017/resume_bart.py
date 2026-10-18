import datetime,hashlib,json,pathlib,subprocess,sys,time
root=pathlib.Path(__file__).resolve().parent
cmd=[sys.executable,'-m','gdown','--no-cookies','--continue','https://drive.google.com/uc?id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m','-O',str(root/'bart_score.pth')]
for attempt in range(1,5):
 stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
 log=root/f'bart_resume_{stamp}_{attempt}.log'
 rec={'command':cmd,'start':datetime.datetime.now().astimezone().isoformat(),'log':str(log)}
 with log.open('x') as f:p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);rc=p.wait()
 rec.update(exit_code=rc,end=datetime.datetime.now().astimezone().isoformat(),log_sha256=hashlib.file_digest(log.open('rb'),'sha256').hexdigest())
 log.with_suffix('.json').write_text(json.dumps(rec,indent=2),encoding='utf8')
 print(json.dumps(rec),flush=True)
 if rc==0:break
 if attempt<4:time.sleep(30)
else:sys.exit(1)
