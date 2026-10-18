"""Run a bounded read-only node04 audit using the existing local login document.

Authentication is read into memory; it is never printed or included in reports.
"""
from pathlib import Path
import argparse
import importlib.util
import re
import sys
import time
import hashlib
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command_file", type=Path, nargs="?")
    parser.add_argument("--collect", type=Path)
    args = parser.parse_args()
    document = Path.home() / "Downloads" / "CODEX_FINAL_PROMPT_node04_with_login.md"
    content = document.read_text(encoding="utf-8")
    match = re.search(r"密码[：:]\s*([^\s]+)", content)
    if match is None:
        raise RuntimeError("Login document has no recognized credential field")
    credential = match.group(1).strip("`'\"")
    helper = Path(__file__).parents[1] / "OmniAware_Data/transfer/paramiko_resume_upload.py"
    spec = importlib.util.spec_from_file_location("node04_connection", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    jump, node, sftp = module.connect(credential)
    try:
        if args.collect:
            remote = "/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/preflight_0242"
            names = ("command.sh", "command.txt", "start.txt", "end.txt", "commit.txt", "git_status.txt",
                     "exit_code.txt", "pytest.log", "pytest.xml", "targeted.sh", "targeted_start.txt",
                     "targeted_end.txt", "targeted_exit_code.txt", "targeted.log", "targeted.xml")
            args.collect.mkdir(parents=True, exist_ok=True)
            records = []
            for name in names:
                target = args.collect / name
                if target.exists():
                    raise FileExistsError(f"Preserving existing evidence: {target}")
                sftp.get(remote + "/" + name, str(target))
                records.append({"remote": remote + "/" + name, "local": str(target),
                                "bytes": target.stat().st_size,
                                "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
            print(json.dumps(records, ensure_ascii=False, indent=2))
            return 0
        if args.command_file is None:
            parser.error("command_file or --collect is required")
        command = args.command_file.read_text(encoding="utf-8")
        _, stdout, stderr = node.exec_command(command, timeout=55)
        channel = stdout.channel
        deadline = time.monotonic() + 55
        while True:
            if channel.recv_ready():
                sys.stdout.write(channel.recv(65536).decode("utf-8", "replace"))
                sys.stdout.flush()
            if channel.recv_stderr_ready():
                sys.stderr.write(channel.recv_stderr(65536).decode("utf-8", "replace"))
                sys.stderr.flush()
            if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                return channel.recv_exit_status()
            if time.monotonic() > deadline:
                raise TimeoutError("Remote command exceeded 55 seconds; partial output retained")
            time.sleep(0.05)
    finally:
        sftp.close()
        node.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
