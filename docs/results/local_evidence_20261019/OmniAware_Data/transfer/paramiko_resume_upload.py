from __future__ import annotations

import concurrent.futures
import getpass
import hashlib
import os
import pathlib
import sys
import threading
import time

import paramiko


ROOT = pathlib.Path(r"D:\OmniAware_Data\transfer")
REMOTE_ROOT = "/home/user/Futaoran"
JUMP_HOST = "172.30.0.6"
TARGET_HOST = "node04"
USER = "user"


TASKS = [
    (ROOT / "clip_model.safetensors", f"{REMOTE_ROOT}/omniaware/assets/cache/clip-vit-large-patch14/model.safetensors"),
]
for i in range(2, 24):
    TASKS.append(
        (
            ROOT / "okvqa_parts2" / f"part_{i:03d}",
            f"{REMOTE_ROOT}/okvqa_parts/part_{i:03d}",
        )
    )

LOCK = threading.Lock()


def connect(password: str) -> tuple[paramiko.SSHClient, paramiko.SSHClient, paramiko.SFTPClient]:
    jump = paramiko.SSHClient()
    jump.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    jump.connect(
        JUMP_HOST,
        username=USER,
        password=password,
        look_for_keys=False,
        allow_agent=False,
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
    )
    source = jump.get_transport().sock.getsockname()
    channel = jump.get_transport().open_channel("direct-tcpip", (TARGET_HOST, 22), source)
    node = paramiko.SSHClient()
    node.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    node.connect(
        TARGET_HOST,
        username=USER,
        password=password,
        sock=channel,
        look_for_keys=False,
        allow_agent=False,
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
    )
    sftp = node.open_sftp()
    return jump, node, sftp


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(8 * 1024 * 1024)
            if not b:
                return h.hexdigest()
            h.update(b)


def upload(task: tuple[pathlib.Path, str], password: str) -> dict[str, object]:
    local, remote = task
    size = local.stat().st_size
    jump = node = None
    sftp = None
    start = time.time()
    try:
        jump, node, sftp = connect(password)
        try:
            remote_size = sftp.stat(remote).st_size
        except FileNotFoundError:
            remote_size = 0
        if remote_size > size:
            raise RuntimeError(f"remote larger than local: {remote_size} > {size}")
        if remote_size == size:
            return {"local": str(local), "remote": remote, "size": size, "resumed": False, "skipped": True}

        with local.open("rb") as src, sftp.file(remote, "ab") as dst:
            src.seek(remote_size)
            try:
                dst.set_pipelined(True)
            except Exception:
                pass
            done = remote_size
            last_report = time.time()
            while done < size:
                data = src.read(4 * 1024 * 1024)
                if not data:
                    raise RuntimeError(f"local EOF at {done}, expected {size}")
                dst.write(data)
                done += len(data)
                now = time.time()
                if now - last_report >= 15:
                    rate = (done - remote_size) / max(1e-6, now - start) / 1024 / 1024
                    with LOCK:
                        print(f"PROGRESS {local.name} {done}/{size} ({done/size:.1%}) {rate:.2f} MiB/s", flush=True)
                    last_report = now
            dst.flush()
        final_size = sftp.stat(remote).st_size
        if final_size != size:
            raise RuntimeError(f"remote size mismatch after upload: {final_size} != {size}")
        return {
            "local": str(local),
            "remote": remote,
            "size": size,
            "resumed": remote_size > 0,
            "remote_before": remote_size,
            "sha256": sha256(local),
            "elapsed_s": round(time.time() - start, 1),
        }
    finally:
        for obj in (sftp, node, jump):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass


def main() -> int:
    password = getpass.getpass("SSH password: ")
    max_workers = int(os.environ.get("OMNIAWARE_UPLOAD_WORKERS", "4"))
    selected = [t for t in TASKS if t[0].is_file()]
    print(f"TASKS {len(selected)} workers={max_workers}", flush=True)
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(upload, task, password): task for task in selected}
        for fut in concurrent.futures.as_completed(futs):
            task = futs[fut]
            try:
                print("DONE", fut.result(), flush=True)
            except Exception as exc:
                failures += 1
                print(f"FAILED {task[0]} -> {task[1]}: {type(exc).__name__}: {exc}", flush=True)
    print(f"SUMMARY failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
