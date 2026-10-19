from pathlib import Path
import hashlib, importlib.util, re
doc = Path.home() / "Downloads" / "CODEX_FINAL_PROMPT_node04_with_login.md"
cred = re.search(r"密码[：:]\s*([^\s]+)", doc.read_text(encoding="utf-8")).group(1).strip(chr(96)+"'\"")
helper = Path(r"D:\WWW2027\OmniAware_Data\transfer\paramiko_resume_upload.py")
spec = importlib.util.spec_from_file_location("node04_connection", helper)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
pairs = [
 (Path(r"D:\WWW2027\data_ops\eval_1022_official.py"), "/home/user/Futaoran/omniaware_v2_exec/third_party/webqa/eval_1022.py"),
 (Path(r"D:\WWW2027\data_ops\eval_webqa_official.md"), "/home/user/Futaoran/omniaware_v2_exec/third_party/webqa/eval_webqa.md"),
]
jump,node,sftp = module.connect(cred)
try:
 for local,remote in pairs:
  sftp.put(str(local),remote)
  print(remote, sftp.stat(remote).st_size, hashlib.sha256(local.read_bytes()).hexdigest())
finally:
 sftp.close(); node.close(); jump.close()
