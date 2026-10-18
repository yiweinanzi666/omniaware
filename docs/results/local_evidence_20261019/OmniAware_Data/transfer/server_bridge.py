"""Interactive SSH bridge; credentials remain in memory and are never logged."""
from __future__ import annotations

import getpass
import hashlib
import json
import shlex
import sys
import time
from pathlib import Path

from paramiko_resume_upload import connect


def main():
    password = getpass.getpass("SSH password: ")
    jump, node, sftp = connect(password)
    jump.get_transport().set_keepalive(15)
    node.get_transport().set_keepalive(15)
    print("BRIDGE_READY", flush=True)
    try:
        for line in sys.stdin:
            try:
                req = json.loads(line)
                action = req.get("action", "exec")
                if action == "exit":
                    return
                if node.get_transport() is None or not node.get_transport().is_active():
                    for obj in (sftp, node, jump):
                        try:
                            obj.close()
                        except Exception:
                            pass
                    jump, node, sftp = connect(password)
                    jump.get_transport().set_keepalive(15)
                    node.get_transport().set_keepalive(15)
                if action == "exec":
                    _, stdout, stderr = node.exec_command(req["cmd"], timeout=req.get("timeout", 50))
                    output = stdout.read().decode("utf-8", "replace")
                    error = stderr.read().decode("utf-8", "replace")
                    print(json.dumps({"exit_code": stdout.channel.recv_exit_status(), "output": output,
                                      "stderr": error}, ensure_ascii=False), flush=True)
                elif action == "upload":
                    local, remote = Path(req["local"]), req["remote"]
                    parent = str(Path(remote).parent).replace("\\", "/")
                    _, out, err = node.exec_command("mkdir -p " + shlex.quote(parent))
                    if out.channel.recv_exit_status():
                        raise RuntimeError(err.read().decode())
                    sftp.put(str(local), remote)
                    local_sha = hashlib.file_digest(local.open("rb"), "sha256").hexdigest()
                    _, out, err = node.exec_command("sha256sum " + shlex.quote(remote))
                    remote_sha = out.read().decode().split()[0]
                    if local_sha != remote_sha:
                        raise RuntimeError("upload SHA-256 mismatch")
                    print(json.dumps({"uploaded": remote, "bytes": local.stat().st_size,
                                      "sha256": local_sha}), flush=True)
                elif action == "download":
                    local, remote = Path(req["local"]), req["remote"]
                    local.parent.mkdir(parents=True, exist_ok=True)
                    sftp.get(remote, str(local))
                    print(json.dumps({"downloaded": str(local), "bytes": local.stat().st_size}), flush=True)
                else:
                    raise ValueError("unknown action")
            except Exception as exc:
                print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True)
    finally:
        for obj in (sftp, node, jump):
            obj.close()


if __name__ == "__main__":
    main()
