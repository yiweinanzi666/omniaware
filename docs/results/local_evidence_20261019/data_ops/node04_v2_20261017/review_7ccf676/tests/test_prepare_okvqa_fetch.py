"""prepare_okvqa.fetch: resume a partial download with Range, restart when the server ignores Range, retry on errors."""
import http.server
import threading

import pytest

from scripts.setup import prepare_okvqa

PAYLOAD = bytes(range(256)) * 400


def _server(ignore_range=False, fail_first=0):
    state = {"requests": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            state["requests"] += 1
            if state["requests"] <= fail_first:
                self.send_error(500)
                return
            rng = self.headers.get("Range")
            if rng and not ignore_range:
                start = int(rng.split("=")[1].split("-")[0])
                body, code = PAYLOAD[start:], 206
            else:
                body, code = PAYLOAD, 200
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/val.zip", state


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(prepare_okvqa.time, "sleep", lambda s: None)


def test_partial_download_is_resumed_with_range(tmp_path):
    srv, url, _ = _server()
    try:
        target = tmp_path / "val.zip"
        part = tmp_path / "val.zip.part"
        part.write_bytes(PAYLOAD[:40000])                    # an earlier run was interrupted here
        assert prepare_okvqa.fetch(url, target) == target
        assert target.read_bytes() == PAYLOAD
        assert not part.exists()
    finally:
        srv.shutdown()


def test_server_ignoring_range_restarts_the_download(tmp_path):
    srv, url, _ = _server(ignore_range=True)
    try:
        target = tmp_path / "val.zip"
        (tmp_path / "val.zip.part").write_bytes(PAYLOAD[:1000])
        prepare_okvqa.fetch(url, target)
        assert target.read_bytes() == PAYLOAD                # not the stale bytes followed by a full copy
    finally:
        srv.shutdown()


def test_transient_server_error_is_retried(tmp_path, no_sleep):
    srv, url, state = _server(fail_first=1)
    try:
        target = tmp_path / "val.zip"
        prepare_okvqa.fetch(url, target, retries=3)
        assert target.read_bytes() == PAYLOAD and state["requests"] == 2
    finally:
        srv.shutdown()


def test_persistent_failure_stops_with_a_resume_hint(tmp_path, no_sleep):
    srv, url, _ = _server(fail_first=100)
    try:
        with pytest.raises(SystemExit, match="rerun the command to resume"):
            prepare_okvqa.fetch(url, tmp_path / "val.zip", retries=2)
        assert not (tmp_path / "val.zip").exists()           # nothing is renamed to the final name
    finally:
        srv.shutdown()
