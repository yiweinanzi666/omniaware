"""Serve the experiment monitor locally; retain source telemetry for inspection."""
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import base64
import zlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import threading
import time
from continuation_status import collect as continuation_status

HERE = Path(__file__).resolve().parent
LATEST = {'connection': '正在连接服务器', 'sample': None, 'history': []}
HISTORY = deque(maxlen=200)
LOCAL_STATUS = {}
PAUSE = HERE.parent / 'v319_continuation/HARDWARE_PAUSE.json'
SCORING = HERE.parent / 'cuda_controls512/parallel/scoring'
PREPARATION = HERE.parent / 'webqa_full4966_preparation'
CE = HERE.parent / 'baseline_full_preparation'
try:
    saved = json.loads((HERE / 'LATEST.json').read_text())
    HISTORY.extend(saved.get('history', [])[-200:])
    LATEST = dict(saved, connection='正在连接服务器', history=list(HISTORY))
except (OSError, ValueError, TypeError):
    pass


def scoring_status():
    """Read local scoring evidence independently of the remote GPU telemetry."""
    state = {'status': 'NOT_STARTED', 'checked_at': time.time()}
    launch_path = SCORING / 'RETRY_LAUNCH.json'
    if not launch_path.exists():
        return state
    try:
        launch = json.loads(launch_path.read_text())
        pid, started = int(launch['pid']), datetime.fromisoformat(launch['started_utc']).timestamp()
        process = subprocess.run(['ps', '-p', str(pid), '-o', 'stat=,command='],
                                 capture_output=True, text=True, timeout=2)
        fields = process.stdout.strip().split(None, 1)
        alive = (process.returncode == 0 and len(fields) == 2 and not fields[0].startswith('Z')
                 and str(SCORING / 'score_parallel_cuda.py') in fields[1])
        state.update(status='RUNNING' if alive else 'EXITED_NEEDS_CHECK', pid=pid, process_alive=alive,
                     started_utc=launch['started_utc'])
        progress_path = SCORING / 'PROGRESS.json'
        if progress_path.exists():
            try:
                progress = json.loads(progress_path.read_text())
                if datetime.fromisoformat(progress['updated_utc']).timestamp() >= started:
                    state['progress'] = progress
            except (ValueError, KeyError):
                pass  # The scorer writes this informational file non-atomically.
        result_path = SCORING / 'OFFICIAL_RESULT.json'
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text())
                score_path = SCORING / 'OFFICIAL_SCORES.jsonl'
                current_freeze = hashlib.sha256((SCORING / 'SCORE_FREEZE.json').read_bytes()).hexdigest()
                valid = (result['n'] == 512 and result['logical_predictions'] == 2560
                         and datetime.fromisoformat(result['completed_utc']).timestamp() >= started
                         and result['scoring_source_freeze_sha256'] == launch['score_freeze_sha256'] == current_freeze
                         and result['scores_sha256'] == hashlib.sha256(score_path.read_bytes()).hexdigest()
                         and result['bart_costs'][0]['metadata']['pid'] == pid)
                if valid:
                    state.update(status='COMPLETE', completed_utc=result['completed_utc'],
                                 result_path=str(result_path), scores_sha256=result['scores_sha256'])
                else:
                    state['result_issue'] = '结果文件与本次评分启动记录或评分文件不一致'
            except (OSError, ValueError, KeyError, IndexError, TypeError):
                state['result_issue'] = '结果文件尚未完整写入或需要检查'
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        state.update(status='CHECK_REQUIRED', error=str(error))
    return state


def extension_scoring(directory, script):
    path = HERE.parent / directory / 'scoring'
    if not (path / 'LIVE_STATUS.json').exists():
        return {'status': 'NOT_STARTED'}
    try:
        state = json.loads((path / 'LIVE_STATUS.json').read_text())
        if state['status'] == 'COMPLETE':
            for key, default in (('result', 'OFFICIAL_RESULT.json'), ('scores', 'OFFICIAL_SCORES.jsonl'),
                                 ('verification', 'FINAL_VERIFICATION.json')):
                artifact = Path(state.get(key + '_path', path / default)).resolve()
                artifact.relative_to(path.resolve())
                if hashlib.sha256(artifact.read_bytes()).hexdigest() != state[key + '_sha256']:
                    raise ValueError('评分产物校验不一致')
        elif state['status'] == 'RUNNING':
            process = subprocess.run(['ps', '-p', str(state['pid']), '-o', 'stat=,command='],
                                     capture_output=True, text=True, timeout=2)
            fields = process.stdout.strip().split(None, 1)
            if not (process.returncode == 0 and len(fields) == 2 and not fields[0].startswith('Z')
                    and script in fields[1]):
                state['status'] = 'EXITED_NEEDS_CHECK'
        return state
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        return {'status': 'CHECK_REQUIRED', 'error': str(error)}


def preparation_status():
    """Count only complete, hash-bound CPU preparation receipts; never estimate encoder rows."""
    state = {'status': 'NOT_STARTED', 'questions': 4966, 'checked_at': time.time(),
             'stages': {name: {'status': 'PENDING', 'completed_questions': None}
                        for name in ('encode_bge', 'encode_clip', 'search_bge', 'search_clip', 'selection', 'sources')}}
    if not (PREPARATION / 'ENCODE_LAUNCH.json').exists():
        return state
    try:
        hashes = {}

        def read(path):
            return json.loads(Path(path).read_text())

        def sha(path):
            path = Path(path)
            if path not in hashes:
                hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
            return hashes[path]

        def require(condition, message):
            if not condition:
                raise ValueError(message)

        def complete(name, path):
            state['stages'][name] = {'status': 'COMPLETE', 'completed_questions': 4966,
                                     'receipt_sha256': sha(path)}

        launch_path = PREPARATION / 'ENCODE_LAUNCH.json'
        launch, binding = read(launch_path), read(PREPARATION / 'QUERY_BINDING.json')
        require(binding['questions'] == 4966 and launch['queries_sha256'] == binding['queries_sha256']
                == sha(PREPARATION / 'queries.jsonl'), '问题文件与本次编码启动记录不一致')
        for name, digest in launch['source_sha256'].items():
            require(sha(PREPARATION / name) == digest, '准备脚本与编码启动记录不一致')
        qids = [json.loads(line)['qid'] for line in (PREPARATION / 'queries.jsonl').read_text().splitlines()]
        require(len(qids) == len(set(qids)) == 4966, '问题数量或唯一题号不一致')
        vectors = PREPARATION / 'query_vectors_cpu'
        query_complete = False
        if (vectors / 'plan.json').exists():
            plan = read(vectors / 'plan.json')
            require(plan['questions'] == 4966 and plan['ordered_qids'] == qids
                    and plan['cohort_sha256'] == binding['queries_sha256']
                    and plan['selection_plan_sha256'] == sha(PREPARATION / 'QUERY_BINDING.json'), '编码计划与当前问题不一致')
            for kind, dimension in (('bge', 1024), ('clip', 768)):
                path = vectors / (kind + '.json')
                if path.exists():
                    receipt = read(path)
                    require(receipt['status'] == 'QUERY_ENCODING_COMPLETE' and receipt['queries'] == 4966
                            and receipt['dimension'] == dimension and receipt['plan_sha256'] == sha(vectors / 'plan.json')
                            and receipt['array_sha256'] == sha(vectors / (kind + '.npz')), kind + ' 编码回执或向量校验不一致')
                    complete('encode_' + kind, path)
            if (vectors / 'completion.json').exists():
                receipt = read(vectors / 'completion.json')
                require(receipt['status'] == 'VALIDATION_QUERY_VECTORS_COMPLETE'
                        and receipt['plan_sha256'] == sha(vectors / 'plan.json'), '编码完成记录不一致')
                for kind in ('bge', 'clip'):
                    require(receipt['encoders'][kind]['receipt_sha256'] == sha(vectors / (kind + '.json'))
                            and receipt['encoders'][kind]['array_sha256'] == sha(vectors / (kind + '.npz')), '编码完成记录与单项回执不一致')
                query_complete = True
        for name, folder, status, artifact, hash_key in (
                ('search_bge', 'top5/bge', 'MODAL_TOP5_COMPLETE', 'rankings.jsonl', 'rankings_sha256'),
                ('search_clip', 'top5/clip', 'MODAL_TOP5_COMPLETE', 'rankings.jsonl', 'rankings_sha256'),
                ('selection', 'selection', 'JOINT_TOP5_COMPLETE', 'selected.jsonl', 'selected_sha256'),
                ('sources', 'sources', 'SOURCE_METADATA_JOIN_COMPLETE_U_PENDING', 'sources.jsonl', None)):
            directory = PREPARATION / folder
            path = directory / 'completion.json'
            if not path.exists():
                continue
            receipt, plan = read(path), read(directory / 'plan.json')
            require(query_complete and receipt['status'] == status and receipt['questions'] == 4966
                    and plan['questions'] == 4966 and receipt['plan_sha256'] == sha(directory / 'plan.json'), name + ' 完成数量或计划不一致')
            artifact_hashes = receipt['artifacts_sha256'] if hash_key is None else {artifact: receipt[hash_key]}
            require(all(sha(directory / p) == digest for p, digest in artifact_hashes.items()), name + ' 输出文件校验不一致')
            require([json.loads(line)['qid'] for line in (directory / artifact).read_text().splitlines()] == qids,
                    name + ' 题号数量或顺序不一致')
            if name.startswith('search_'):
                require(plan['query_plan_sha256'] == sha(vectors / 'plan.json')
                        and plan['query_completion_sha256'] == sha(vectors / 'completion.json'), '检索与本批编码不一致')
            elif name == 'selection':
                for kind in ('bge', 'clip'):
                    source = PREPARATION / 'top5' / kind
                    require(plan['retrievals'][kind]['plan_sha256'] == sha(source / 'plan.json')
                            and plan['retrievals'][kind]['completion_sha256'] == sha(source / 'completion.json')
                            and plan['retrievals'][kind]['rankings_sha256'] == sha(source / 'rankings.jsonl'), '融合与本批检索不一致')
            else:
                source = PREPARATION / 'selection'
                require(plan['cohort_sha256'] == binding['queries_sha256']
                        and plan['selection_plan_sha256'] == sha(source / 'plan.json')
                        and plan['selection_completion_sha256'] == sha(source / 'completion.json')
                        and plan['selected_sha256'] == sha(source / 'selected.jsonl'), '证据拼接与本批融合不一致')
            complete(name, path)
        sidecar_path = PREPARATION / 'PREPARATION_LIVE_STATUS.json'
        sidecar = read(sidecar_path) if sidecar_path.exists() else {}
        paths = (sidecar.get('launch_paths') if sidecar.get('phase') == 'search' else None)
        paths = paths or {'active': sidecar.get('launch_path', str(launch_path))}
        processes = []
        for path in paths.values():
            path = Path(path).resolve()
            path.relative_to(PREPARATION.resolve())
            active = read(path)
            require(active['queries_sha256'] == binding['queries_sha256']
                    and active['source_sha256'] == launch['source_sha256'], '阶段启动记录与冻结输入不一致')
            if str(path) == sidecar.get('launch_path') and sidecar.get('launch_sha256'):
                require(sha(path) == sidecar['launch_sha256'], '阶段启动回执校验不一致')
            command = active['command']
            require(any('webqa_full4966_preparation.' in arg for arg in command), '运行命令不属于本准备任务')
            process = subprocess.run(['ps', '-p', str(active['pid']), '-o', 'stat=,command='], capture_output=True, text=True, timeout=2)
            fields = process.stdout.strip().split(None, 1)
            alive = (process.returncode == 0 and len(fields) == 2 and not fields[0].startswith('Z')
                     and ' '.join(command[1:]) in fields[1])
            processes.append({'pid': active['pid'], 'alive': alive})
        state.update(processes=processes, phase=sidecar.get('phase', 'encoding'), encoder=sidecar.get('encoder'),
                     completed_stages=sum(x['status'] == 'COMPLETE' for x in state['stages'].values()),
                     query_completion_verified=query_complete)
        state['status'] = ('COMPLETE' if state['completed_stages'] == 6 else
                           'FAILED' if sidecar.get('status') == 'FAILED' else
                           'RUNNING' if any(p['alive'] for p in processes) else
                           'WAITING' if query_complete else 'EXITED_NEEDS_CHECK')
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        state.update(status='CHECK_REQUIRED', error=str(error))
    return state


def ce_status():
    """Read committed CPU CE records independently of the server's GPU telemetry."""
    state = {'status': 'NOT_STARTED', 'questions': 4966, 'completed_questions': 0,
             'candidate_pairs': 49660, 'completed_pairs': 0, 'checked_at': time.time(),
             'device': 'Mac CPU', 'threads': 2, 'incomplete_lines': 0}
    if not (CE / 'CE_FULL_LAUNCH.json').exists():
        return state
    try:
        def read(path):
            return json.loads(path.read_text())

        def sha(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()

        def objsha(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                             separators=(',', ':')).encode()).hexdigest()

        def require(condition, message):
            if not condition:
                raise ValueError(message)

        launch, plan = read(CE / 'CE_FULL_LAUNCH.json'), read(CE / 'CE_PLAN.json')
        plan_sha = sha(CE / 'CE_PLAN.json')
        require(launch['ce_plan_sha256'] == plan_sha and plan['runner_sha256'] == sha(CE / 'run_ce.py'),
                'CE 脚本或计划与启动记录不一致')
        require(plan['questions'] == launch['questions'] == 4966
                and plan['candidate_pairs'] == launch['candidate_pairs'] == 49660
                and launch['device'] == plan['device'] == 'cpu' and launch['threads'] == plan['threads'] == 2,
                'CE 全量规模或运行设备不一致')
        require(sha(CE / 'candidate_sources.jsonl') == plan['candidate_sources_sha256']
                and sha(CE / 'CANDIDATE_PLAN.json') == plan['candidate_plan_sha256'], 'CE 候选输入已改变')
        items = [json.loads(line) for line in (CE / 'candidate_sources.jsonl').read_text().splitlines()]
        qids = plan['ordered_qids']
        require(len(qids) == len(set(qids)) == 4966 and [x['qid'] for x in items] == qids, 'CE 题号或顺序不一致')
        command = launch['command']
        require(command[-3:] == [str(CE / 'run_ce.py'), '--mode', 'full'], 'CE 启动命令不是全量任务')
        pid = int(launch['pid'])
        process = subprocess.run(['ps', '-p', str(pid), '-o', 'stat=,command='],
                                 capture_output=True, text=True, timeout=2)
        fields = process.stdout.strip().split(None, 1)
        alive = (process.returncode == 0 and len(fields) == 2 and not fields[0].startswith('Z')
                 and shlex.split(fields[1])[1:] == command[1:])
        state.update(status='RUNNING' if alive else 'EXITED_NEEDS_CHECK', pid=pid, process_alive=alive,
                     started_utc=launch['started_utc'], plan_sha256=plan_sha, log_path=launch['log'])
        folder = CE / 'ce_full'
        if not (folder / 'EXECUTION_PLAN.json').exists():
            return state
        execution = read(folder / 'EXECUTION_PLAN.json')
        require(execution == {'ce_plan_sha256': plan_sha, 'mode': 'full', 'ordered_qids': qids},
                'CE 执行计划与冻结题号不一致')
        execution_sha = sha(folder / 'EXECUTION_PLAN.json')
        data = (folder / 'rankings.jsonl').read_bytes() if (folder / 'rankings.jsonl').exists() else b''
        lines = data.splitlines(keepends=True)
        if lines and not lines[-1].endswith(b'\n'):
            state['incomplete_lines'] = 1
            lines.pop()
        saved = []
        for i, line in enumerate(lines):
            row = json.loads(line)
            require(i < 4966 and row['qid'] == qids[i] and row['execution_plan_sha256'] == execution_sha
                    and row['input_row_sha256'] == objsha(items[i]), 'CE 记录不是本次输入的连续完整前缀')
            ids = [s['source_id'] for s in items[i]['sources']]
            require(len(ids) == len(set(ids)) == 10 and row['source_ids'] == ids
                    and row['pair_count'] == 10 and row['physical_forwards'] == 2
                    and len(row['logits']) == 10 and all(math.isfinite(x) for x in row['logits'])
                    and len(row['input_token_counts']) == 10
                    and all(1 <= n <= 512 for n in row['input_token_counts']) and len(row['selected_sources']) == 5
                    and len(row['ranked_source_ids']) == 10 and set(row['ranked_source_ids']) == set(ids),
                    'CE 单题候选或调用记录不完整')
            order = sorted(range(10), key=lambda j: (-row['logits'][j], ids[j]))
            expected = [dict(items[i]['sources'][j], original_joint_rank=items[i]['sources'][j]['rank'],
                             rank=rank, ce_logit=row['logits'][j]) for rank, j in enumerate(order[:5], 1)]
            require(row['ranked_source_ids'] == [ids[j] for j in order] and row['selected_sources'] == expected,
                    'CE 来源顺序与完整候选 logit 不一致')
            saved.append(row)
        state.update(completed_questions=len(saved), completed_pairs=10 * len(saved),
                     committed_forwards=sum(row['physical_forwards'] for row in saved))
        receipt_path = folder / 'COMPLETE.json'
        if receipt_path.exists():
            receipt = read(receipt_path)
            require(receipt['status'] == 'CPU_CE_RERANK_COMPLETE' and receipt['mode'] == 'full'
                    and receipt['questions'] == len(saved) == 4966 and receipt['pairs'] == 49660
                    and not state['incomplete_lines'] and receipt['execution_plan_sha256'] == execution_sha
                    and receipt['rankings_sha256'] == hashlib.sha256(data).hexdigest()
                    and receipt['sources_sha256'] == sha(folder / 'sources.jsonl')
                    and receipt['physical_forwards'] == state['committed_forwards']
                    and receipt['references_read'] is False and receipt['QA_calls'] == 0
                    and datetime.fromisoformat(receipt['completed_utc']) >= datetime.fromisoformat(launch['started_utc']),
                    'CE 完成回执、全量计数或文件 SHA 不一致')
            sources = [json.loads(line) for line in (folder / 'sources.jsonl').read_text().splitlines()]
            require(len(sources) == 4966 and all(s == {'qid': item['qid'], 'question': item['question'],
                                                       'sources': row['selected_sources']}
                                               for s, item, row in zip(sources, items, saved, strict=True)),
                    'CE 最终来源文件与已提交重排记录不一致')
            state.update(status='COMPLETE', completed_utc=receipt['completed_utc'],
                         receipt_sha256=sha(receipt_path), sources_sha256=receipt['sources_sha256'])
        elif len(saved) == 4966:
            state['status'] = 'FINALIZING' if alive else 'EXITED_NEEDS_CHECK'
    except (OSError, ValueError, KeyError, IndexError, TypeError, subprocess.TimeoutExpired) as error:
        state.update(status='CHECK_REQUIRED', error=str(error))
    return state


def hardware_pause():
    if not PAUSE.exists():
        return {'active': False}
    return json.loads(PAUSE.read_text())


def receive():
    global LATEST
    command = ['ssh', '-S', '/tmp/www2027-card-37028', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
               '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2',
               '-p', '37028', 'root@connect.bjb1.seetacloud.com',
               '/root/autodl-tmp/www2027-py312-cu130/bin/python -B -u /root/autodl-tmp/v318_resume_20261009/sample_remote.py']
    while True:
        if hardware_pause().get('active'):
            LATEST = dict(LATEST, connection='已暂停，等待硬件升级')
            time.sleep(5)
            continue
        with (HERE / 'SSH.log').open('a') as errors:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, text=True)
            for line in process.stdout:
                try:
                    value = json.loads(zlib.decompress(base64.b64decode(line[3:])) if line.startswith('z1:') else line)
                except ValueError:
                    continue
                if 'error' in value:
                    LATEST = dict(LATEST, connection='采样错误', error=value['error'])
                    continue
                try:
                    value['continuation'] = continuation_status()
                    target = HERE.parent / 'v319_continuation/FIVE_HOUR_TARGET.json'
                    if target.exists():
                        value['five_hour_target'] = json.loads(target.read_text())
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    value['continuation'] = {'error': str(error), 'groups': []}
                if HISTORY and HISTORY[-1].get('boot_id') != value.get('boot_id'):
                    HISTORY.clear()
                HISTORY.append(value)
                LATEST = {'connection': '实时连接', 'sample': value, 'received_at': time.time(), 'history': list(HISTORY)}
                (HERE / 'LATEST.json').write_text(json.dumps(LATEST, ensure_ascii=False))
                with (HERE / 'METRICS.jsonl').open('a') as archive:
                    archive.write(json.dumps(value, ensure_ascii=False) + '\n')
            process.wait()
        LATEST = dict(LATEST, connection='连接中断，正在重连')
        time.sleep(5)


def refresh_local_status():
    """Validate local artifacts once per interval, not once per browser request."""
    global LOCAL_STATUS
    while True:
        LOCAL_STATUS = dict(scoring=scoring_status(),
                            order_scoring=extension_scoring('e1_cuda_extension', 'score_official.py'),
                            paper_scoring=extension_scoring('phase1_paper_cuda', '/scoring/score.py'),
                            cpu_preparation=preparation_status(), cpu_ce=ce_status(),
                            continuation=continuation_status(), local_artifacts_checked_at=time.time())
        time.sleep(30)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ('/data', '/api/status'):
            status = dict(LATEST, **LOCAL_STATUS, hardware_pause=hardware_pause())
            sample = status.get('sample')
            if sample is not None:
                age = max(0.0, time.time() - float(sample.get('sampled_at', 0.0)))
                status['telemetry_age_seconds'] = age
                status['telemetry_stale'] = status.get('connection') != '实时连接' or age > 15.0
            target = HERE.parent / 'v319_continuation/TWO_HOUR_TARGET.json'
            if target.exists():
                status['two_hour_target'] = json.loads(target.read_text())
            if status.get('sample') is not None and 'continuation' in LOCAL_STATUS:
                status['sample'] = dict(status['sample'], continuation=LOCAL_STATUS['continuation'])
            content, mime = json.dumps(status, ensure_ascii=False).encode(), 'application/json'
        elif self.path == '/':
            content, mime = (HERE / 'index.html').read_bytes(), 'text/html; charset=utf-8'
        elif self.path == '/v319_ui.js':
            content, mime = (HERE / 'v319_ui.js').read_bytes(), 'text/javascript; charset=utf-8'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, *_):
        pass


if __name__ == '__main__':
    threading.Thread(target=receive, daemon=True).start()
    threading.Thread(target=refresh_local_status, daemon=True).start()
    ThreadingHTTPServer(('127.0.0.1', 8769), Handler).serve_forever()
