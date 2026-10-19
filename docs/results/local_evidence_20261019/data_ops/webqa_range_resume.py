"""Download disjoint ranges, then append them in order to the existing prefix.

Only this process writes archive prefixes. Range files survive interruptions;
they are removed only after their bytes have been flushed into the prefix.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path


def acquire_lock(root: Path):
    handle = (root / ".webqa-download.lock").open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("Another WebQA downloader already holds the output lock")
    return handle


class RangeFile:
    def __init__(self, record, root, block_size, emit, resolve, opener, parse_range, stop):
        self.name = str(record["name"])
        if Path(self.name).name != self.name:
            raise ValueError("Invalid manifest filename")
        self.total = int(record["content_length"])
        self.url = str(record["url"])
        self.final = root / self.name
        self.part = root / (self.name + ".part")
        self.stage = root / ".ranges" / self.name
        self.lock = threading.Lock()
        self.url_lock = threading.Lock()
        self.emit, self.resolve, self.opener = emit, resolve, opener
        self.parse_range, self.stop = parse_range, stop
        self.active_url = None
        self.url_time = 0.0
        self.started = time.monotonic()
        self.done = False
        self.initial = self.part.stat().st_size if self.part.exists() else 0
        self.ranges = []
        if self.final.exists():
            if self.final.stat().st_size != self.total:
                raise ValueError(f"Preserving unexpected final file: {self.name}")
            self.done = True
            return
        if self.initial > self.total:
            raise ValueError(f"Preserving oversized prefix: {self.name}")
        if self.initial:
            with self.part.open("rb") as source:
                if source.read(64).lstrip().lower().startswith((b"<html", b"<!doctype")):
                    raise ValueError(f"Preserving invalid HTML prefix: {self.name}")
        self.stage.mkdir(parents=True, exist_ok=True)
        plan_path = self.stage / "plan.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            if plan["total"] != self.total or plan["url"] != self.url:
                raise ValueError(f"Range plan source mismatch: {self.name}")
            base, block_size = int(plan["base"]), int(plan["block_size"])
            if not 0 <= base <= self.initial <= self.total or block_size < 1:
                raise ValueError(f"Range plan prefix mismatch: {self.name}")
        else:
            base = self.initial
            plan = dict(total=self.total, url=self.url, base=base, block_size=block_size)
            temp = self.stage / "plan.json.tmp"
            with temp.open("w", encoding="utf-8") as output:
                json.dump(plan, output)
                output.flush()
                os.fsync(output.fileno())
            temp.replace(plan_path)
        self.ranges = [(start, min(start + block_size, self.total))
                       for start in range(base, self.total, block_size)]
        self.commit()

    def segment(self, start, end):
        return self.stage / f"{start:012d}-{end:012d}.range"

    def ready_segment(self, start, end):
        return self.stage / f"{start:012d}-{end:012d}.ready"

    def commit(self):
        with self.lock:
            if self.done:
                return
            current = self.part.stat().st_size if self.part.exists() else 0
            for start, end in self.ranges:
                if end <= current:
                    continue
                if start > current:
                    break
                segment = self.ready_segment(start, end)
                if not segment.exists() or segment.stat().st_size != end - start:
                    break
                overlap = current - start
                with segment.open("rb") as source:
                    if overlap:
                        with self.part.open("rb") as prefix:
                            prefix.seek(start)
                            if prefix.read(overlap) != source.read(overlap):
                                raise ValueError(f"Interrupted append differs: {self.name} at {start}")
                    with self.part.open("ab") as output:
                        shutil.copyfileobj(source, output, 1024 * 1024)
                        output.flush()
                        os.fsync(output.fileno())
                current = end
                segment.unlink()
            if current == self.total:
                if self.final.exists():
                    raise FileExistsError(self.final)
                self.part.rename(self.final)
                self.done = True
                self.emit(dict(event="file_complete", name=self.name, status="downloaded",
                               bytes=current, new_bytes=current-self.initial,
                               seconds=round(time.monotonic()-self.started, 2)))

    def get_url(self):
        with self.url_lock:
            if self.active_url is None or time.monotonic() - self.url_time > 2400:
                self.active_url = self.resolve(self.url)
                self.url_time = time.monotonic()
            return self.active_url

    def fetch(self, bounds):
        start, end = bounds
        segment = self.segment(start, end)
        ready = self.ready_segment(start, end)
        failures = 0
        while not self.stop.is_set():
            with self.lock:
                if self.done or (self.part.exists() and self.part.stat().st_size >= end):
                    return
                if ready.exists():
                    if ready.stat().st_size != end-start:
                        raise ValueError(f"Invalid completed range: {ready.name}")
                    break
                received = segment.stat().st_size if segment.exists() else 0
            if received > end - start:
                raise ValueError(f"Preserving oversized range: {segment.name}")
            if received == end - start:
                with segment.open("r+b") as output:
                    output.flush()
                    os.fsync(output.fileno())
                segment.rename(ready)
                break
            request_start = start + received
            try:
                request = urllib.request.Request(self.get_url(), headers={
                    "User-Agent": "Mozilla/5.0", "Accept-Encoding": "identity",
                    "Range": f"bytes={request_start}-{end-1}",
                })
                with self.opener.open(request, timeout=120) as response:
                    bounds_received = self.parse_range(response.headers.get("Content-Range", ""))
                    if response.status != 206 or bounds_received is None:
                        raise ValueError(f"HTTP {response.status}: invalid Range response")
                    first, last, total = bounds_received
                    if first != request_start or not first <= last < end or total != self.total:
                        raise ValueError(f"Content-Range mismatch: {bounds_received}")
                    length = last - first + 1
                    advertised = response.headers.get("Content-Length")
                    if advertised is not None and int(advertised) != length:
                        raise ValueError("Content-Length mismatch")
                    if response.headers.get("Content-Encoding", "identity") != "identity":
                        raise ValueError("Unexpected Content-Encoding")
                    body_bytes = 0
                    with segment.open("ab", buffering=0) as output:
                        while body_bytes < length and not self.stop.is_set():
                            block = response.read1(min(64 * 1024, length-body_bytes))
                            if not block:
                                break
                            output.write(block)
                            body_bytes += len(block)
                    if not body_bytes:
                        raise OSError("Empty range response")
                failures = 0
            except ValueError:
                # Never retry a response/plan integrity mismatch as if it were a network error.
                raise
            except Exception as exc:
                failures += 1
                if (isinstance(exc, urllib.error.HTTPError) and exc.code in (401,403,404,410)) or failures >= 5:
                    with self.url_lock:
                        self.active_url = None
                self.emit(dict(event="range_retry", name=self.name, start=start,
                               offset=start+(segment.stat().st_size if segment.exists() else 0),
                               failures=failures, error_type=type(exc).__name__,
                               http_status=getattr(exc, "code", None)))
                if segment.exists() and segment.stat().st_size > received:
                    failures = 0
                    continue
                retry_after = None
                if isinstance(exc, urllib.error.HTTPError) and exc.code in (429,503):
                    retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else min(30, 2**min(failures,5))
                self.stop.wait(delay)
        if not self.stop.is_set():
            self.commit()


def download_records(records, root, progress_path, workers, block_size, resolve, opener, parse_range):
    log_lock = threading.Lock()
    stop = threading.Event()
    log = root / "logs" / "parallel_ranges.jsonl"
    log.parent.mkdir(exist_ok=True)

    def emit(event):
        event["time"] = time.time()
        with log_lock:
            with log.open("a", encoding="utf-8") as output:
                output.write(json.dumps(event) + "\n")
            if event.get("event") in ("file_complete", "run_start", "run_end", "fatal_error"):
                with progress_path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(event) + "\n")
                print(json.dumps(event), flush=True)

    if len({r["name"] for r in records}) != len(records):
        raise ValueError("Duplicate manifest filenames")
    files = [RangeFile(r, root, block_size, emit, resolve, opener, parse_range, stop) for r in records]
    queues = deque((f, deque(bounds for bounds in f.ranges
                             if bounds[1] > (f.part.stat().st_size if f.part.exists() else 0)))
                   for f in files if not f.done)
    jobs = []
    while queues:
        f, pending = queues.popleft()
        if pending:
            jobs.append((f, pending.popleft()))
        if pending:
            queues.append((f,pending))
    emit(dict(event="run_start", mode="parallel_ranges", workers=workers, ranges=len(jobs),
              block_size=block_size, pid=os.getpid()))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(f.fetch, bounds) for f,bounds in jobs]
        try:
            for future in concurrent.futures.as_completed(futures):
                future.result()
        except BaseException as exc:
            stop.set()
            for future in futures:
                future.cancel()
            emit(dict(event="fatal_error", error_type=type(exc).__name__, detail=str(exc)))
            raise
    for f in files:
        f.commit()
    complete = sum(f.done for f in files)
    emit(dict(event="run_end", complete=complete, total=len(files)))
    return 0 if complete == len(files) else 1
