"""Local HTTP regression checks for preserving and extending download prefixes."""
import hashlib
import http.server
import json
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

from webqa_drive_downloader import parse_content_range
from webqa_range_resume import RangeFile, acquire_lock, download_records

DATA = bytes(range(256))*8192


class RangeServer(http.server.BaseHTTPRequestHandler):
    requests = []
    mode = "short"

    def log_message(self, *args):
        pass

    def do_GET(self):
        start, end = map(int, self.headers["Range"][6:].split("-"))
        self.requests.append((start,end))
        if self.mode == "html":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"<html>error</html>")
            return
        cap = min(end, start+65535)
        self.send_response(206)
        first = start+1 if self.mode == "wrong_start" else start
        self.send_header("Content-Range", f"bytes {first}-{cap}/{len(DATA)}")
        self.send_header("Content-Length", str(cap-start+1))
        self.end_headers()
        # Half of some advertised responses is deliberately lost in transit.
        payload = DATA[start:cap+1]
        if start % 3 == 0:
            payload = payload[:max(1,len(payload)//2)]
        self.wfile.write(payload)


class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1",0),RangeServer)
        threading.Thread(target=cls.server.serve_forever,daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/object"
        cls.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.record = dict(name="imgs.7z.001",content_length=len(DATA),url=self.url)
        self.part = self.root/"imgs.7z.001.part"
        self.prefix = 12345
        self.part.write_bytes(DATA[:self.prefix])
        RangeServer.mode = "short"
        RangeServer.requests = []

    def tearDown(self):
        self.temp.cleanup()

    def make_file(self):
        return RangeFile(self.record,self.root,262144,lambda e:None,lambda u:u,
                         self.opener,parse_content_range,threading.Event())

    def run_download(self):
        return download_records([self.record],self.root,self.root/"progress.jsonl",8,262144,
                                lambda u:u,self.opener,parse_content_range)

    def test_parallel_short_reads_preserve_prefix(self):
        self.assertEqual(self.run_download(),0)
        final = self.root/"imgs.7z.001"
        self.assertEqual(hashlib.sha256(final.read_bytes()).digest(),hashlib.sha256(DATA).digest())
        self.assertTrue(all(start>=self.prefix for start,end in RangeServer.requests))
        first_count = len(RangeServer.requests)
        self.assertEqual(self.run_download(),0)
        self.assertEqual(first_count,len(RangeServer.requests))

    def test_resume_completed_and_partial_ranges(self):
        f = self.make_file()
        first,second = f.ranges[:2]
        f.segment(*first).write_bytes(DATA[first[0]:first[0]+7777])
        f.ready_segment(*second).write_bytes(DATA[second[0]:second[1]])
        self.assertEqual(self.run_download(),0)
        self.assertEqual((self.root/"imgs.7z.001").read_bytes(),DATA)
        self.assertTrue(all(not second[0]<=start<second[1] for start,end in RangeServer.requests))
        self.assertTrue(all(start>=first[0]+7777 for start,end in RangeServer.requests))

    def test_recover_interrupted_append(self):
        f = self.make_file()
        start,end = f.ranges[0]
        f.ready_segment(start,end).write_bytes(DATA[start:end])
        with self.part.open("ab") as output:
            output.write(DATA[start:start+5555])
        self.assertEqual(self.run_download(),0)
        self.assertEqual((self.root/"imgs.7z.001").read_bytes(),DATA)
        self.assertTrue(all(pos>=end for pos,_ in RangeServer.requests))

    def test_reject_wrong_range_and_html_without_touching_prefix(self):
        for mode in ("html","wrong_start"):
            RangeServer.mode = mode
            f = self.make_file()
            before = self.part.read_bytes()
            with self.assertRaises(ValueError):
                f.fetch(f.ranges[0])
            self.assertEqual(self.part.read_bytes(),before)
            self.assertFalse(f.segment(*f.ranges[0]).exists())

    def test_lock_prevents_second_writer(self):
        first = acquire_lock(self.root)
        try:
            with self.assertRaises(RuntimeError):
                acquire_lock(self.root)
        finally:
            first.close()
        acquire_lock(self.root).close()


if __name__ == "__main__":
    unittest.main()
