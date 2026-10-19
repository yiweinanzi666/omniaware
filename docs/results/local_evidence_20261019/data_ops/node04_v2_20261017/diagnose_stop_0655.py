import datetime, hashlib, http.server, importlib.util, json, os, pathlib, subprocess, threading, urllib.request

LOG = pathlib.Path('/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017')
spec = importlib.util.spec_from_file_location('prep', LOG / 'prepare_v3.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
old = p.ASSETS / 'cache/official_m2rag'
official = p.REPO / 'external/M2RAG'
rec = {'start': datetime.datetime.now().astimezone().isoformat(), 'gpu_gate': 'STOP: pytest failures; no real-weight verification, S1, S2 or S4', 'source_modified': False, 'cached_root': str(old), 'official_root': str(official)}
rec['git'] = {}
for name, root in [('legacy', old), ('official', official), ('experiment', p.REPO)]:
    rec['git'][name] = {}
    for key, args in [('head', ['rev-parse', 'HEAD']), ('toplevel', ['rev-parse', '--show-toplevel']), ('status', ['status', '--porcelain=v1'])]:
        r = subprocess.run(['git', *args], cwd=root, text=True, capture_output=True)
        rec['git'][name][key] = {'command': ['git', *args], 'cwd': str(root), 'exit_code': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}
rec['source_comparisons'] = []
for rel in ['src/utils/registry.py', 'src/summarizer/base.py', 'src/summarizer/multi_modal/base.py', 'src/summarizer/multi_modal/end_to_end.py']:
    row = {'relative_path': rel}
    for name, root in [('legacy', old), ('official', official)]:
        f = root / rel
        row[name] = {'exists': f.exists()}
        if f.exists():
            row[name].update(bytes=f.stat().st_size, sha256=hashlib.file_digest(f.open('rb'), 'sha256').hexdigest())
    row['same_sha256'] = row['legacy'].get('sha256') == row['official'].get('sha256')
    rec['source_comparisons'].append(row)

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'local-probe')
    def log_message(self, *args):
        pass

server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
url = 'http://127.0.0.1:' + str(server.server_port) + '/probe'
rec['loopback_proxy_probe'] = {'url': url, 'proxy_bypass': urllib.request.proxy_bypass('127.0.0.1')}
for name, opener in [('preparation_environment', urllib.request.build_opener()), ('local_direct', urllib.request.build_opener(urllib.request.ProxyHandler({})))]:
    try:
        with opener.open(url, timeout=10) as response:
            rec['loopback_proxy_probe'][name] = {'status': response.status, 'body': response.read().decode()}
    except Exception as e:
        rec['loopback_proxy_probe'][name] = {'error': type(e).__name__, 'message': str(e)}
server.shutdown()
server.server_close()
rec['end'] = datetime.datetime.now().astimezone().isoformat()
dest = LOG / 'pytest_stop_diagnosis_0655.json'
with dest.open('x') as f:
    json.dump(rec, f, indent=2)
print(json.dumps(rec), flush=True)
