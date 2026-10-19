from pathlib import Path
import hashlib, importlib.util, re
doc = Path.home() / "Downloads" / "CODEX_FINAL_PROMPT_node04_with_login.md"
cred = re.search(r"密码[：:]\s*([^\s]+)", doc.read_text(encoding="utf-8")).group(1).strip(chr(96)+"'\"")
helper = Path(r"D:\WWW2027\OmniAware_Data\transfer\paramiko_resume_upload.py")
spec = importlib.util.spec_from_file_location("node04_connection", helper)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
local = Path(r"D:\WWW2027\OmniAware_Data\transfer\models\m2rag_official_checkout.zip")
remote = "/home/user/Futaoran/m2rag_official_checkout.zip"
h = hashlib.sha256(local.read_bytes()).hexdigest()
jump,node,sftp = module.connect(cred)
try:
    sftp.put(str(local), remote)
    print({"remote": remote, "size": sftp.stat(remote).st_size, "sha256": h})
finally:
    sftp.close(); node.close(); jump.close()
