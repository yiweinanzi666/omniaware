"""Bounded-range HTTPS downloader used for small official data objects."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path


def download(url: str, target: Path, expected: int | None = None, block: int = 4 * 1024 * 1024) -> dict[str, object]:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    started = time.time()
    errors: list[str] = []
    for attempt in range(1, 61):
        have = part.stat().st_size if part.exists() else 0
        if expected is not None and have > expected:
            part.rename(part.with_name(part.name + ".oversize"))
            have = 0
        if expected is not None and have == expected:
            os.replace(part, target)
            break
        end = (expected - 1) if expected is not None else have + block - 1
        if expected is not None:
            end = min(end, have + block - 1)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept-Encoding": "identity", "Range": f"bytes={have}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                if response.status != 206:
                    raise RuntimeError(f"HTTP {response.status} for range {have}-{end}")
                ctype = response.headers.get("Content-Type", "").lower()
                if ctype.startswith("text/html"):
                    raise RuntimeError("received HTML instead of a binary object")
                with part.open("ab" if have else "wb") as out:
                    while True:
                        data = response.read(1024 * 1024)
                        if not data:
                            break
                        out.write(data)
            if expected is None or part.stat().st_size >= expected:
                if expected is not None and part.stat().st_size != expected:
                    raise RuntimeError(f"short object {part.stat().st_size}/{expected}")
                os.replace(part, target)
                break
        except Exception as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
        time.sleep(min(30, 2 + attempt))
    if not target.exists() or (expected is not None and target.stat().st_size != expected):
        raise RuntimeError(json.dumps({"url": url, "target": str(target), "errors": errors[-5:]}, ensure_ascii=False))
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return {"url": url, "path": str(target), "bytes": target.stat().st_size, "sha256": digest, "seconds": round(time.time() - started, 2), "errors": errors}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("target", type=Path)
    ap.add_argument("--expected", type=int, default=None)
    args = ap.parse_args()
    print(json.dumps(download(args.url, args.target, args.expected), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
