"""Serve completed WebQA chunks over HTTP with single-range support."""

from __future__ import annotations

import argparse
import http.server
import re
from pathlib import Path


RANGE_RE = re.compile(r"bytes=(\d+)-(\d*)$")


class ChunkHandler(http.server.BaseHTTPRequestHandler):
    server_version = "WebQARange/1.0"

    def do_HEAD(self) -> None:  # noqa: N802
        self._serve(send_body=False)

    def do_GET(self) -> None:  # noqa: N802
        self._serve(send_body=True)

    def _serve(self, *, send_body: bool) -> None:
        name = self.path.split("?", 1)[0].lstrip("/")
        if not re.fullmatch(r"imgs\.7z\.\d{3}", name):
            self.send_error(404)
            return
        root: Path = self.server.chunk_root  # type: ignore[attr-defined]
        path = root / name
        if not path.is_file():
            self.send_error(404)
            return

        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        header = self.headers.get("Range")
        if header:
            match = RANGE_RE.fullmatch(header.strip())
            if not match:
                self.send_error(416)
                return
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else size - 1
            if start >= size or end < start:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            end = min(end, size - 1)
            status = 206

        length = end - start + 1
        self.send_response(status)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not send_body:
            return

        with path.open("rb") as source:
            source.seek(start)
            remaining = length
            while remaining:
                block = source.read(min(1024 * 1024, remaining))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} {format % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--root", type=Path, default=Path(r"D:\WebQA\chunks"))
    args = parser.parse_args()
    server = http.server.ThreadingHTTPServer((args.bind, args.port), ChunkHandler)
    server.chunk_root = args.root.resolve()  # type: ignore[attr-defined]
    server.serve_forever()


if __name__ == "__main__":
    main()
