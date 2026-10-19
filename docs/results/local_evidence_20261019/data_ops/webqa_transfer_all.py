"""Resume and verify every WebQA archive volume on node04."""

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_prefix(path, url, expected_bytes):
    size = path.stat().st_size
    if not 0 < size < expected_bytes:
        raise RuntimeError(f"invalid prefix size for {path.name}: {size}")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for start in {0, max(0, size - 65536)}:
        end = min(size, start + 65536) - 1
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        with opener.open(request, timeout=60) as response:
            content_range = f"bytes {start}-{end}/{expected_bytes}"
            if response.status != 206 or response.headers.get("Content-Range") != content_range:
                raise RuntimeError(f"unexpected source range for {path.name}")
            source_bytes = response.read()
        with path.open("rb") as existing:
            existing.seek(start)
            if existing.read(len(source_bytes)) != source_bytes or len(source_bytes) != end - start + 1:
                raise RuntimeError(f"prefix mismatch for {path.name} at byte {start}")


def transfer(record, args):
    name = record["name"]
    expected_bytes = int(record["bytes"])
    expected_sha256 = record["sha256"].lower()
    output = args.output_dir / name
    url = f"{args.url_root.rstrip('/')}/{name}"

    if output.exists() and output.stat().st_size == expected_bytes:
        if sha256_file(output) != expected_sha256:
            raise RuntimeError(f"complete remote file has wrong SHA-256: {name}")
        return {"name": name, "status": "already_verified", "bytes": expected_bytes}

    uploading = args.output_dir / f"{name}.uploading"
    if not output.exists() and uploading.exists():
        check_prefix(uploading, url, expected_bytes)
        shutil.copyfile(uploading, output)

    if output.exists():
        check_prefix(output, url, expected_bytes)

    log_path = args.log_dir / f"pull_{name}.log"
    command = [
        sys.executable,
        str(args.pull_script),
        url,
        str(output),
        str(expected_bytes),
        expected_sha256,
        "--workers",
        str(args.workers_per_file),
        "--block-size",
        str(args.block_size),
    ]
    with log_path.open("a", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"pull failed for {name}; see {log_path}")
    if output.stat().st_size != expected_bytes or sha256_file(output) != expected_sha256:
        raise RuntimeError(f"remote verification failed for {name}")
    return {"name": name, "status": "transferred_verified", "bytes": expected_bytes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--url-root", required=True)
    parser.add_argument("--pull-script", type=Path, default=Path(__file__).with_name("webqa_parallel_pull.py"))
    parser.add_argument("--concurrent-files", type=int, default=4)
    parser.add_argument("--workers-per-file", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=1024 * 1024)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    records = json.loads(args.manifest.read_text(encoding="utf-8-sig"))["records"]
    if len(records) != 51 or len({row["name"] for row in records}) != 51:
        raise RuntimeError("expected 51 distinct WebQA volumes")
    for record in records:
        if not record["name"].startswith("imgs.7z.") or len(record["sha256"]) != 64:
            raise RuntimeError(f"invalid manifest record: {record}")

    with (args.output_dir / ".webqa_transfer.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another WebQA transfer is already running") from exc

        records.sort(key=lambda row: int(row["bytes"]) - min(
            (args.output_dir / row["name"]).stat().st_size
            if (args.output_dir / row["name"]).exists() else 0,
            int(row["bytes"]),
        ))
        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrent_files) as pool:
            futures = {pool.submit(transfer, record, args): record["name"] for record in records}
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    print(json.dumps(future.result()), flush=True)
                except Exception as exc:
                    errors.append(name)
                    print(json.dumps({"name": name, "status": "error", "error": str(exc)}), flush=True)
        if errors:
            raise RuntimeError(f"{len(errors)} volumes failed: {', '.join(errors)}")


if __name__ == "__main__":
    main()
