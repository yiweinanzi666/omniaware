"""Acquire pinned experiment assets locally, verifying upstream hashes before publishing files."""
from __future__ import annotations

import concurrent.futures
import argparse
import hashlib
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent / "models"
REPOS = [
    ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", "bge-reranker-v2-m3"),
    ("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7", "deberta_nli"),
    ("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b", "qwen3-vl-8b-instruct"),
]


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def valid(path, meta):
    if not path.is_file() or path.stat().st_size != meta["size"]:
        return False
    if "lfs" in meta:
        return sha(path) == meta["lfs"]["sha256"]
    content = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest() == meta["blobId"]


def acquire(repo, revision, folder, meta):
    name = meta["rfilename"]
    dest = folder / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if valid(dest, meta):
        print("VERIFIED_EXISTING", repo, name, flush=True)
    else:
        part = dest.with_name(dest.name + ".partial")
        if dest.exists():
            dest.replace(part)
        urls = [f"https://modelscope.cn/models/{repo}/resolve/master/{name}",
                f"https://huggingface.co/{repo}/resolve/{revision}/{name}",
                f"https://hf-mirror.com/{repo}/resolve/{revision}/{name}"]
        # Small files are fetched from the pinned revision directly.
        if meta["size"] < 20_000_000:
            urls = urls[1:] + urls[:1]
        for attempt in range(12):
            offset = part.stat().st_size if part.exists() else 0
            if offset == meta["size"]:
                if valid(part, meta):
                    part.replace(dest)
                    break
                part.replace(part.with_suffix(f".bad-{int(time.time())}"))
                offset = 0
            if offset > meta["size"]:
                raise ValueError(f"oversized partial: {part}")
            base = urls[attempt % len(urls)]
            url = base + f"?download=true&resume_offset={offset}"
            try:
                response = requests.get(url, headers={"Range": f"bytes={offset}-"},
                                        stream=True, timeout=(15, 35))
                response.raise_for_status()
                cr = response.headers.get("Content-Range", "")
                if response.status_code == 206:
                    if not cr.startswith(f"bytes {offset}-") or not cr.endswith(f"/{meta['size']}"):
                        raise ValueError(f"unexpected Content-Range {cr}")
                elif offset or response.status_code != 200:
                    raise ValueError(f"range ignored at {offset}: {response.status_code}")
                start = report = time.monotonic()
                received = offset
                print("DOWNLOADING", repo, name, "offset", offset, "source", base.split('/')[2], flush=True)
                with response, part.open("ab" if offset else "wb") as out:
                    for chunk in response.iter_content(1024 * 1024):
                        out.write(chunk)
                        received += len(chunk)
                        if received > meta["size"]:
                            raise ValueError("received beyond declared size")
                        now = time.monotonic()
                        if now - report > 30:
                            print("PROGRESS", repo, name, received, "/", meta["size"],
                                  "MiB/s", round((received - offset) / (now-start) / 2**20, 2), flush=True)
                            report = now
                if valid(part, meta):
                    part.replace(dest)
                    break
                if received == meta["size"]:
                    part.replace(part.with_suffix(f".bad-{int(time.time())}"))
                    raise ValueError("upstream hash mismatch; quarantined")
            except Exception as exc:
                print("RETRY", repo, name, type(exc).__name__, str(exc)[:180], flush=True)
                time.sleep(min(3 * (attempt + 1), 20))
        else:
            raise RuntimeError(f"could not obtain {repo}/{name}")
    record = {"repo": repo, "revision": revision, "file": name, "bytes": dest.stat().st_size,
              "sha256": sha(dest), "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    marker = dest.with_name(dest.name + ".verified.json")
    marker.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("COMPLETE", json.dumps(record), flush=True)


def download_repo(repo, revision, dirname):
    folder = ROOT / dirname
    folder.mkdir(parents=True, exist_ok=True)
    cached = folder / "upstream_manifest.json"
    data = json.loads(cached.read_text(encoding="utf-8")) if cached.exists() else None
    if data is None or data.get("sha") != revision:
        for attempt in range(6):
            try:
                endpoint = ("https://hf-mirror.com", "https://huggingface.co")[attempt % 2]
                response = requests.get(f"{endpoint}/api/models/{repo}/revision/{revision}",
                                        params={"blobs": "true"}, timeout=35)
                response.raise_for_status()
                data = response.json()
                break
            except requests.RequestException:
                if attempt == 5:
                    raise
                time.sleep(3)
    if data.get("sha") != revision:
        raise ValueError(f"unexpected revision for {repo}")
    (folder / "upstream_manifest.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    files = [f for f in data["siblings"] if "/" not in f["rfilename"] and
             f["rfilename"].endswith((".json", ".txt", ".model", ".safetensors"))]
    files.sort(key=lambda f: (f["size"] > 20_000_000, f["rfilename"]))
    small = [f for f in files if f["size"] <= 20_000_000]
    large = [f for f in files if f["size"] > 20_000_000]
    for meta in small:
        acquire(repo, revision, folder, meta)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as weights:
        for future in concurrent.futures.as_completed([weights.submit(acquire, repo, revision, folder, m) for m in large]):
            future.result()
    (folder / "MODEL_COMPLETE.json").write_text(json.dumps({"repo": repo, "revision": revision,
                                                            "files": len(files)}), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    selected = [entry for entry in REPOS if not args.only or entry[2] in args.only]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(download_repo, *entry): entry[0] for entry in selected}
        for fut in concurrent.futures.as_completed(futures):
            try:
                fut.result()
                print("REPO_COMPLETE", futures[fut], flush=True)
            except Exception as exc:
                print("REPO_FAILED", futures[fut], repr(exc), flush=True)
