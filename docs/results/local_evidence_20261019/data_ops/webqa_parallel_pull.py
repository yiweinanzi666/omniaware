"""Pull one verified WebQA chunk over HTTP using resumable fixed ranges."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pending_ranges(prefix, expected_bytes, block_size, completed):
    ranges = []
    cursor = prefix
    for start, end in sorted(completed):
        if start < cursor or end <= start or end > expected_bytes:
            raise RuntimeError("invalid or overlapping completed range")
        while cursor < start:
            next_end = min(cursor + block_size, start)
            ranges.append((cursor, next_end - 1))
            cursor = next_end
        cursor = end
    while cursor < expected_bytes:
        next_end = min(cursor + block_size, expected_bytes)
        ranges.append((cursor, next_end - 1))
        cursor = next_end
    return ranges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("expected_bytes", type=int)
    parser.add_argument("expected_sha256")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    temp = Path(f"{args.output}.parallel")
    state = Path(f"{args.output}.ranges.jsonl")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not temp.exists():
        prefix = args.output.stat().st_size if args.output.exists() else 0
        if prefix > args.expected_bytes:
            raise RuntimeError("existing prefix exceeds expected size")
        if prefix:
            shutil.copyfile(args.output, temp)
        else:
            temp.touch()
        with temp.open("r+b") as target:
            target.truncate(args.expected_bytes)
        state.write_text(
            json.dumps({"event": "start", "prefix": prefix, "expected": args.expected_bytes}) + "\n",
            encoding="utf-8",
        )
    else:
        first = json.loads(state.read_text(encoding="utf-8").splitlines()[0])
        prefix = int(first["prefix"])
        if int(first["expected"]) != args.expected_bytes or temp.stat().st_size != args.expected_bytes:
            raise RuntimeError("resume metadata does not match requested object")

    completed = []
    for line in state.read_text(encoding="utf-8").splitlines()[1:]:
        record = json.loads(line)
        if record.get("event") == "block_complete":
            completed.append((int(record["start"]), int(record["end"]) + 1))
    ranges = pending_ranges(prefix, args.expected_bytes, args.block_size, completed)

    lock = threading.Lock()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    fd = os.open(temp, os.O_RDWR)

    def fetch(one_range):
        start, end = one_range
        expected = end - start + 1
        failures = 0
        while True:
            try:
                request = urllib.request.Request(
                    args.url,
                    headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                )
                with opener.open(request, timeout=args.timeout) as response:
                    content_range = response.headers.get("Content-Range", "")
                    if response.status != 206 or content_range != f"bytes {start}-{end}/{args.expected_bytes}":
                        raise RuntimeError(f"unexpected response: {response.status} {content_range!r}")
                    data = response.read()
                if len(data) != expected:
                    raise RuntimeError(f"short range: {len(data)} != {expected}")
                os.pwrite(fd, data, start)
                with lock:
                    with state.open("a", encoding="utf-8") as log:
                        log.write(json.dumps({"event": "block_complete", "start": start, "end": end}) + "\n")
                        log.flush()
                        os.fsync(log.fileno())
                return start, expected
            except Exception as exc:
                failures += 1
                with lock:
                    print(f"retry start={start} failures={failures} error={type(exc).__name__}: {exc}", flush=True)
                time.sleep(min(30, max(2, failures)))

    done_bytes = prefix + sum(end - start for start, end in completed)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for start, count in pool.map(fetch, ranges):
                done_bytes += count
                print(json.dumps({"event": "progress", "bytes": done_bytes, "expected": args.expected_bytes, "last_start": start}), flush=True)
    finally:
        os.close(fd)

    digest = sha256_file(temp)
    if digest.lower() != args.expected_sha256.lower():
        raise RuntimeError(f"SHA-256 mismatch: {digest} != {args.expected_sha256}")
    os.replace(temp, args.output)
    with state.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"event": "verified", "sha256": digest, "bytes": args.expected_bytes}) + "\n")
    print(json.dumps({"event": "verified", "sha256": digest, "bytes": args.expected_bytes}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
