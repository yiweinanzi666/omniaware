"""Inventory and resumably download the public WebQA Google Drive chunks.

This script only operates on the official public folder.  It never stores
credentials and it writes each chunk to a temporary .part path before making
it visible as the final file.  Existing files with the expected size are
skipped, so rerunning it does not redownload completed chunks.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path


FOLDER_URL = "https://drive.google.com/drive/folders/19ApkbD5w0I5sV1IeQ9EofJRyAjKnA7tb"
DOWNLOAD_URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"
HF_API_URL = "https://huggingface.co/api/datasets/Ancci/webqa_large/tree/main?recursive=true"
HF_DOWNLOAD_URL = "https://huggingface.co/datasets/Ancci/webqa_large/resolve/main/{name}?download=true"
CHUNK_RE = re.compile(r"imgs\.7z\.(\d{3})")
ROW_RE = re.compile(
    r'<tr[^>]*data-id="([^"]+)"[^>]*>[\s\S]*?'
    r'<strong[^>]*>\s*(imgs\.7z\.\d{3})\s*</strong>',
    re.IGNORECASE,
)


_PROXY_URL: str | None = None
_URL_OPENER = urllib.request.build_opener()


def configure_proxy(proxy_url: str | None) -> None:
    """Use the task-local HTTP proxy when one is explicitly supplied."""

    global _PROXY_URL, _URL_OPENER
    _PROXY_URL = proxy_url or None
    if _PROXY_URL:
        _URL_OPENER = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": _PROXY_URL, "https": _PROXY_URL})
        )
    else:
        _URL_OPENER = urllib.request.build_opener()


def fetch(url: str, method: str = "GET") -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(url, method=method, headers={"User-Agent": "Mozilla/5.0"})
    with _URL_OPENER.open(req, timeout=60) as response:
        body = response.read() if method != "HEAD" else b""
        return response.status, {k.lower(): v for k, v in response.headers.items()}, body


def inventory() -> list[dict[str, object]]:
    status, _, body = fetch(FOLDER_URL)
    if status != 200:
        raise RuntimeError(f"folder request returned HTTP {status}")
    text = body.decode("utf-8", "replace")
    found: dict[str, str] = {}
    for file_id, name in ROW_RE.findall(text):
        name = html.unescape(name)
        found[name] = file_id
    if not found:
        raise RuntimeError("no WebQA chunks were found in the public folder HTML")
    records: list[dict[str, object]] = []
    for name, file_id in sorted(found.items()):
        url = DOWNLOAD_URL.format(id=urllib.parse.quote(file_id, safe=""))
        status, headers, _ = fetch(url, method="HEAD")
        size = int(headers.get("content-length", "0"))
        records.append(
            {
                "name": name,
                "id": file_id,
                "url": url,
                "http_status": status,
                "content_length": size,
                "content_disposition": headers.get("content-disposition", ""),
                "last_modified": headers.get("last-modified", ""),
            }
        )
    return records


def hf_inventory() -> list[dict[str, object]]:
    status, _, body = fetch(HF_API_URL)
    if status != 200:
        raise RuntimeError(f"Hugging Face tree request returned HTTP {status}")
    entries = json.loads(body.decode("utf-8"))
    records: list[dict[str, object]] = []
    for entry in entries:
        name = str(entry.get("path", ""))
        if CHUNK_RE.fullmatch(name):
            records.append({
                "name": name,
                "id": None,
                "url": HF_DOWNLOAD_URL.format(name=urllib.parse.quote(name, safe=".")),
                "http_status": 200,
                "content_length": int(entry.get("size", 0)),
                "content_disposition": "",
                "last_modified": "",
            })
    if not records:
        raise RuntimeError("no WebQA chunks were found in the Hugging Face mirror")
    return sorted(records, key=lambda r: str(r["name"]))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Expose the signed CDN URL so each range does not repeat the redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


def resolve_download_url(url: str) -> str:
    """Resolve one public URL to its current signed object URL.

    Hugging Face returns a short lived signed URL for every request.  Following
    that redirect for every 32 MiB range doubles the number of proxy handshakes;
    resolve it once per worker and refresh it only when it expires.
    """

    handlers: list[urllib.request.BaseHandler] = []
    if _PROXY_URL:
        handlers.append(urllib.request.ProxyHandler({"http": _PROXY_URL, "https": _PROXY_URL}))
    handlers.append(_NoRedirect())
    opener = urllib.request.build_opener(*handlers)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Encoding": "identity",
            "Range": "bytes=0-0",
        },
    )
    try:
        with opener.open(request, timeout=60) as response:
            return response.geturl()
    except urllib.error.HTTPError as exc:
        if exc.code in (301, 302, 303, 307, 308):
            location = exc.headers.get("Location")
            if location:
                return urllib.parse.urljoin(url, location)
        raise


def parse_content_range(value: str) -> tuple[int, int, int] | None:
    """Return the byte range and total advertised by a 206 response."""

    match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", value.strip(), re.IGNORECASE)
    if not match or match.group(3) == "*":
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def download_one(record: dict[str, object], out_dir: Path, log_dir: Path, progress_path: Path) -> dict[str, object]:
    name = str(record["name"])
    expected = int(record["content_length"])
    url = str(record["url"])
    final_path = out_dir / name
    part_path = out_dir / f"{name}.part"
    log_path = log_dir / f"{name}.log"
    if expected <= 0:
        return {"name": name, "status": "failed", "error": "missing Content-Length"}
    if final_path.exists() and final_path.stat().st_size == expected:
        result = {"name": name, "status": "skipped_complete", "bytes": expected}
        with progress_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
        return result
    if final_path.exists() and final_path.stat().st_size != expected:
        # Preserve an unexpected pre-existing file; never overwrite it.
        backup = final_path.with_suffix(final_path.suffix + ".unexpected")
        if not backup.exists():
            final_path.rename(backup)
    started = time.time()
    log_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    completed = False
    consecutive_failures = 0
    active_url: str | None = None
    range_size = int(record.get("range_size", 32 * 1024 * 1024))
    log_path.write_text(
        json.dumps({"event": "file_start", "name": name, "expected": expected, "time": time.time()}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # Keep retrying until this file is complete.  A Range response shorter than
    # requested is valid progress (the proxy commonly caps responses at 4 MiB),
    # so it must never consume a finite attempt budget.
    while True:
        actual = part_path.stat().st_size if part_path.exists() else 0
        if actual and part_path.exists():
            with part_path.open("rb") as probe:
                prefix = probe.read(64).lstrip().lower()
            if prefix.startswith(b"<!doctype") or prefix.startswith(b"<html"):
                # A failed Drive response can be a small HTML quota/error page.
                # It is never a valid archive prefix, so restart that temp file.
                part_path.open("wb").close()
                actual = 0
        if actual > expected:
            part_path.rename(part_path.with_suffix(part_path.suffix + ".oversize"))
            actual = 0
        if actual == expected:
            completed = True
            break
        try:
            request_start = actual
            if active_url is None:
                active_url = resolve_download_url(url)
            request = urllib.request.Request(
                active_url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept-Encoding": "identity",
                    "Connection": "keep-alive",
                    # A bounded range avoids quota/error pages and permits
                    # immediate continuation when a proxy caps the response.
                    "Range": f"bytes={actual}-{min(expected - 1, actual + range_size - 1)}",
                },
            )
            with _URL_OPENER.open(request, timeout=120) as response:
                if response.status != 206:
                    raise RuntimeError(f"unexpected HTTP response {response.status} at {actual}")
                content_range = parse_content_range(response.headers.get("Content-Range", ""))
                if content_range is None:
                    raise RuntimeError(f"missing or invalid Content-Range at {actual}: {response.headers.get('Content-Range', '')!r}")
                range_start, range_end, range_total = content_range
                if range_start != actual or range_end < range_start or range_total != expected:
                    raise RuntimeError(
                        f"unexpected Content-Range at {actual}: "
                        f"bytes {range_start}-{range_end}/{range_total}"
                    )
                advertised_length = response.headers.get("Content-Length")
                expected_response_bytes = range_end - range_start + 1
                if advertised_length is not None and int(advertised_length) != expected_response_bytes:
                    raise RuntimeError(
                        f"Content-Length mismatch at {actual}: "
                        f"{advertised_length} != {expected_response_bytes}"
                    )
                mode = "ab" if actual else "wb"
                received = 0
                with part_path.open(mode) as output:
                    while True:
                        # Flush progress in small blocks so monitoring sees
                        # movement even when a range is several MiB long.
                        block = response.read(1 * 1024 * 1024)
                        if not block:
                            break
                        if actual + len(block) > expected:
                            raise RuntimeError(f"response exceeded expected size at {actual}")
                        output.write(block)
                        actual += len(block)
                        received += len(block)
                if received == 0:
                    raise RuntimeError(f"empty range response at {range_start}")
                if received < expected_response_bytes:
                    with log_path.open("a", encoding="utf-8") as log:
                        log.write(
                            f"short range at {range_start}: received {received} "
                            f"of {expected_response_bytes}; continuing without backoff\n"
                        )
            consecutive_failures = 0
            if actual == expected:
                completed = True
                break
        except Exception as exc:  # network failures are expected to be resumable
            actual = part_path.stat().st_size if part_path.exists() else 0
            consecutive_failures += 1
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403, 404, 410):
                # Signed CDN URLs expire; obtain a fresh one on the next pass.
                active_url = None
            errors.append(f"failure {consecutive_failures}: {type(exc).__name__}: {exc}; bytes={actual}")
            with log_path.open("a", encoding="utf-8") as log:
                log.write(errors[-1] + "\n")
            if actual > request_start:
                # A timeout after a partial body is still useful progress.  The
                # next loop validates and resumes at the new contiguous prefix.
                consecutive_failures = 0
                continue
            # Back off only on actual failures.  Successful short ranges are
            # followed immediately, which removes the old 2–30 second delay.
            time.sleep(min(30, max(2, consecutive_failures)))
    elapsed = max(time.time() - started, 0.001)
    actual = part_path.stat().st_size if part_path.exists() else 0
    if completed and actual == expected:
        os.replace(part_path, final_path)
        result = {"name": name, "status": "downloaded", "bytes": actual, "seconds": round(elapsed, 2), "avg_mib_s": round(actual / elapsed / 1024 / 1024, 3)}
    else:
        log_path.write_text("\n".join(errors) + "\n", encoding="utf-8")
        result = {"name": name, "status": "incomplete", "bytes": actual, "expected": expected, "seconds": round(elapsed, 2), "errors": errors[-3:]}
    with progress_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=r"D:\\WebQA\\chunks")
    parser.add_argument("--manifest", default=r"D:\\WWW2027\\data_ops\\WEBQA_DRIVE_MANIFEST.json")
    parser.add_argument("--progress", default=r"D:\\WWW2027\\data_ops\\WEBQA_DRIVE_PROGRESS.jsonl")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--source", choices=("drive", "hf"), default="drive")
    parser.add_argument("--range-size", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--proxy-url", default=None, help="task-local HTTP(S) proxy, e.g. http://127.0.0.1:7897")
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    configure_proxy(args.proxy_url)
    out_dir = Path(args.out)
    manifest_path = Path(args.manifest)
    progress_path = Path(args.progress)
    log_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]]
    inventory_fn = hf_inventory if args.source == "hf" else inventory
    if manifest_path.exists():
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
            candidate = cached.get("records", [])
            if isinstance(candidate, list) and len(candidate) >= 1 and all("name" in r and "url" in r and "content_length" in r for r in candidate):
                records = candidate
            else:
                records = inventory_fn()
        except Exception:
            records = inventory_fn()
    else:
        records = inventory_fn()
    write_json(manifest_path, {"folder_url": FOLDER_URL, "records": records, "count": len(records), "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    for record in records:
        record["range_size"] = args.range_size
    print(json.dumps({"count": len(records), "total_bytes": sum(int(r["content_length"]) for r in records), "manifest": str(manifest_path)}, ensure_ascii=False), flush=True)
    if args.inventory_only:
        return 0
    lock = threading.Lock()
    # A fresh progress file marks the beginning of this controlled run.
    with progress_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "run_start", "workers": args.workers, "time": time.time()}, ensure_ascii=False) + "\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(download_one, record, out_dir, log_dir, progress_path) for record in records]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            with lock:
                print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
