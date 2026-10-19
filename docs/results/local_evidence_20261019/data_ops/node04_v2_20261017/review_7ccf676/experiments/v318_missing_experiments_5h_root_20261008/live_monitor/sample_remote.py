"""Stream read-only resource telemetry and actual five-arm output counts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from physical_throughput import PhysicalThroughput
from mmqa_progress import progress as mmqa_progress

TOKEN_THROUGHPUT = PhysicalThroughput()

RUN = Path('/root/autodl-tmp/v318_controls512_root_20261008/cuda_controls512')
FULL_WEBQA = Path('/root/autodl-tmp/v318_webqa_full4966_root_20261008')
MMQA_ENCODING = Path('/root/autodl-tmp/v318_mmqa_pool_encoding_root_20261008/experiments/v318_missing_experiments_5h_root_20261008/mmqa_full2441_preparation/retrieval_assets/encoding_cuda')
GENERATION_CACHE = {}
GENERATION_ARMS = {'direct', 'always-static', 'trigger-static', 'trigger-replication', 'trigger-eab'}
DUAL = Path('/root/autodl-tmp/v318_resume_20261009/dual_gpu_resume_20261009')


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(path):
    if not path.exists():
        return None
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def objsha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def generation_plan(directory, launch, seed, arms=GENERATION_ARMS):
    names = ('config.json', 'tasks.jsonl', 'physical_tasks.jsonl', 'ready.json')
    signature = tuple(fingerprint(directory / name) for name in names) + (launch['config_sha256'], seed, tuple(sorted(arms)))
    cached = GENERATION_CACHE.get(str(directory))
    if cached and cached['signature'] == signature:
        return cached
    config = read(directory / 'config.json'); config_sha = sha(directory / 'config.json')
    if (config_sha != launch['config_sha256'] or config['generation_seed'] != seed
            or config['questions'] != 4966 or set(config['arms']) != arms):
        raise ValueError('Generation launch/config/seed/cohort differs')
    ready = read(directory / 'ready.json')
    for name in names[:-1]:
        if sha(directory / name) != ready['prepared_sha256'][name]:
            raise ValueError('Generation prepared source SHA differs: ' + name)
    expected, physical_ids = {}, {}
    physical_prefix = 'CE-' if arms == {'ce-static', 'ours-ce'} else 'P1-'
    with (directory / 'physical_tasks.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line); pid = row['physical_task_id']; digest = row['physical_identity_sha256']
            if (pid in physical_ids or pid != physical_prefix + digest or objsha(row['identity']) != digest
                    or row['identity']['generation']['generation_seed'] != seed):
                raise ValueError('Generation physical plan/seed identity differs')
            physical_ids[pid] = digest
    with (directory / 'tasks.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line); key = row['qid'], row['arm']; pid = row['physical_task_id']
            if (key in expected or row['arm'] not in arms or row['order_id'] != 'sorted'
                    or pid not in physical_ids or row['physical_identity_sha256'] != physical_ids[pid]):
                raise ValueError('Generation logical task identity differs')
            expected[key] = pid
    qids = {key[0] for key in expected}
    if len(qids) != 4966 or set(expected) != {(q, a) for q in qids for a in arms}:
        raise ValueError('Generation plan does not contain full4966 with every declared arm')
    cached = dict(signature=signature, config_sha=config_sha, expected=expected, physical_ids=physical_ids,
                  logical_answers=len(expected), max_tokens=config['max_new_tokens'], offset=0, file_stamp=None, answers={}, seen=set(),
                  conflicts=set(), invalid_lines=0, completion_stamp=None)
    GENERATION_CACHE[str(directory)] = cached
    return cached


def generation_rows(directory, cached):
    """Inspect only appended complete lines; a replacement/truncation resets the in-memory count."""
    path = directory / 'predictions.jsonl'; stamp = fingerprint(path); prior = cached['file_stamp']
    if stamp != prior and (stamp is None or prior and (stamp[0] != prior[0] or stamp[1] <= prior[1])):
        cached.update(offset=0, answers={}, seen=set(), conflicts=set(), invalid_lines=0, completion_stamp=None)
    if stamp is not None and stamp != prior:
        with path.open('rb') as stream:
            stream.seek(cached['offset'])
            while line := stream.readline():
                if not line.endswith(b'\n'):
                    break
                cached['offset'] = stream.tell()
                try:
                    row = json.loads(line); key = row['qid'], row['arm']; pid = row['physical_task_id']
                    if key in cached['seen']:
                        cached['conflicts'].add(key); cached['answers'].pop(key, None)
                        continue
                    cached['seen'].add(key)
                    tokens = row['answer_token_ids']
                    if (cached['expected'].get(key) != pid or row['order_id'] != 'sorted'
                            or row['config_sha256'] != cached['config_sha'] or not isinstance(row['response'], str)
                            or not isinstance(tokens, list) or not 0 < len(tokens) <= cached['max_tokens']
                            or not all(type(t) is int for t in tokens) or row['output_tokens'] != len(tokens)):
                        raise ValueError('Generation logical record identity/content differs')
                    cached['answers'][key] = (pid, row['physical_call_attempt_id'], row['physical_result_sha256'], objsha(tokens))
                except (ValueError, KeyError, TypeError):
                    cached['invalid_lines'] += 1
    cached['file_stamp'] = stamp
    return int(stamp is not None and stamp[1] > cached['offset'])


def generation_complete(directory, cached):
    receipt_path = directory / 'completion.json'
    if not receipt_path.exists():
        return False
    loads = sorted(directory.glob('model_load_*.json'))
    stamp = tuple(fingerprint(directory / n) for n in ('completion.json', 'predictions.jsonl', 'physical_calls.jsonl'))
    stamp += tuple((p.name, fingerprint(p)) for p in loads)
    if cached['completion_stamp'] == stamp:
        return True
    done = read(receipt_path)
    for name, field in (('predictions.jsonl', 'predictions_sha256'), ('physical_calls.jsonl', 'physical_calls_sha256'),
                        ('config.json', 'config_sha256')):
        if sha(directory / name) != done[field]:
            raise ValueError('Generation completion SHA differs: ' + name)
    if (done['logical_answers'] != cached['logical_answers'] or len(cached['answers']) != cached['logical_answers']
            or done['physical_calls'] != len(cached['physical_ids'])
            or done['workload']['physical_completed_calls'] != len(cached['physical_ids'])
            or (directory / 'predictions.jsonl').stat().st_size != cached['offset']
            or done['model_load_receipts_sha256'] != {p.name: sha(p) for p in loads}):
        raise ValueError('Generation completion count/load receipts differ')
    completed, attempts = {}, {}
    with (directory / 'physical_calls.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line); pid, aid, status = row['physical_task_id'], row['attempt_id'], row['status']
            if cached['physical_ids'].get(pid) != row['physical_identity_sha256']:
                raise ValueError('Generation ledger has unknown physical identity')
            if status == 'started':
                if aid in attempts:
                    raise ValueError('Generation ledger duplicates an attempt')
                attempts[aid] = (pid, status)
            else:
                prior = attempts.get(aid)
                if (not prior or prior[0] != pid or prior[1] in ('completed', 'failed')
                        or status not in ('generation_started', 'completed', 'failed')
                        or status == 'generation_started' and prior[1] != 'started'
                        or status == 'completed' and (prior[1] != 'generation_started' or pid in completed)):
                    raise ValueError('Generation ledger attempt sequence differs')
                attempts[aid] = (pid, status)
                if status == 'completed':
                    completed[pid] = (pid, aid, objsha(row), objsha(row['answer_token_ids']))
    if set(completed) != set(cached['physical_ids']) or any(status not in ('completed', 'failed') for _, status in attempts.values()):
        raise ValueError('Generation completion has missing/unfinished physical calls')
    if any(link != completed.get(link[0]) for link in cached['answers'].values()):
        raise ValueError('Generation answer does not match its completed physical call')
    cached['completion_stamp'] = stamp
    return True


def generation_status(arms, subdirectory, launch_directory, launch_pattern, module):
    runs = []
    for seed in (17, 23, 29):
        directory = FULL_WEBQA / subdirectory.format(seed=seed)
        state = dict(seed=seed, completed_questions=0, total_questions=4966, completed_answers=0, total_answers=4966*len(arms),
                     arms=len(arms), pids=[], process_alive=False, complete=False, error=None, status='NOT_STARTED')
        try:
            launch = read(launch_directory / launch_pattern.format(seed=seed))
            if arms == {'ce-static', 'ours-ce'} and seed == 29 and (DUAL / 'ce_seed29.LAUNCH.json').exists():
                launch = read(DUAL / 'ce_seed29.LAUNCH.json')
            if launch:
                if launch.get('seed',seed) != seed:
                    raise ValueError('Generation launch seed differs')
                pid = launch['pid']; state['pids'] = [pid]
                state['process_alive'] = process_matches(pid, module, 'generate', str(directory))
                cached = generation_plan(directory, launch, seed, arms)
                state['incomplete_lines'] = generation_rows(directory, cached)
                counts = {}
                for qid, arm in cached['answers']:
                    counts[qid] = counts.get(qid, 0) + 1
                state.update(completed_answers=len(cached['answers']), completed_questions=sum(n == len(arms) for n in counts.values()),
                             conflicting_answers=len(cached['conflicts']), invalid_lines=cached['invalid_lines'])
                if cached['conflicts'] or cached['invalid_lines']:
                    raise ValueError('Generation has duplicate, malformed or unbound answer records')
                state['complete'] = generation_complete(directory, cached)
                state['status'] = 'COMPLETE' if state['complete'] else 'RUNNING' if state['process_alive'] else 'EXITED_NEEDS_CHECK'
        except (OSError, ValueError, KeyError, TypeError) as error:
            state.update(status='CHECK_REQUIRED', complete=False, error=str(error))
        runs.append(state)
    return dict(runs=runs, completed_questions=sum(r['completed_questions'] for r in runs), total_questions=14898,
                completed_answers=sum(r['completed_answers'] for r in runs), total_answers=14898*len(arms),
                pids=[pid for r in runs if r['process_alive'] for pid in r['pids']],
                process_alive=any(r['process_alive'] for r in runs), complete=all(r['complete'] for r in runs),
                error='; '.join(f"seed{r['seed']}: {r['error']}" for r in runs if r['error']) or None)



def full4966_generation():
    if not any((FULL_WEBQA/f'GENERATION_SEED{s}_LAUNCH.json').exists() for s in (17,23,29)):
        return None
    return generation_status(GENERATION_ARMS,'work_cuda/seed{seed}',FULL_WEBQA,
                             'GENERATION_SEED{seed}_LAUNCH.json','webqa_full4966_execution.execute')


def ce_generation():
    root = Path('/root/autodl-tmp/v318_resume_20261009/ce_queue')
    if not (root/'QUEUE_PLAN.json').exists():
        return None
    result = generation_status({'ce-static','ours-ce'},'work_cuda/ce_seed{seed}',root,
                               'QUEUE_SEED{seed}_LAUNCH.json','ce_full4966_execution.execute')
    queue = read(root/'QUEUE_STATE.json', {})
    if queue:
        if queue.get('plan_sha256') != sha(root/'QUEUE_PLAN.json'):
            result['error'] = 'CE queue plan SHA differs'
        queue['process_alive'] = process_matches(queue.get('pid'),'queue_after_mmqa.py','--run')
        hold = read(root.parent/'CE_ADMISSION_HOLD.json', {})
        if hold.get('status') == 'CE_ADMISSION_HELD_FOR_DRAGIN_TRAIN8':
            queue['phase'] = 'WAIT_DRAGIN_TRAIN8'
    result['queue'] = queue
    if (DUAL / 'STATUS.json').exists():
        current = dual_state()
        result['queue'] = dict(current, process_alive=process_matches(current['pid'], str(DUAL / 'resume.py')),
                               phase='ALL_JOBS_LAUNCHED' if current['jobs']['ce_seed29']['status'] == 'RUNNING' else 'WAITING')
    return result


def process_matches(pid, *parts):
    try:
        command = Path(f'/proc/{int(pid)}/cmdline').read_bytes().decode().replace('\0', ' ')
        return all(part in command for part in parts)
    except (OSError, TypeError, ValueError, UnicodeDecodeError):
        return False


def initial_observer(directory, launch_path, total, cohort_total, module, receipt_name, cohort_field, cohort_file, cost_field):
    if not (directory / 'plan.json').exists():
        return None
    state = dict(completed=0, total=total, calibration_completed=0, calibration_total=500,
                 cohort_completed=0, cohort_total=cohort_total, pid=None, process_alive=False, complete=False, error=None)
    try:
        launch = read(launch_path, {})
        state['pid'] = launch.get('pid')
        state['process_alive'] = process_matches(state['pid'], module, 'run', str(directory))
        plan = read(directory / 'plan.json'); plan_sha = sha(directory / 'plan.json')
        if plan['questions'] != total or sha(directory / 'tasks.jsonl') != plan['tasks_sha256']:
            raise ValueError('Initial observation plan/tasks mismatch')
        tasks = [json.loads(line) for line in (directory / 'tasks.jsonl').read_text().splitlines()]
        expected = {t['task_id']: t for t in tasks}
        if len(tasks) != len(expected) or len(tasks) != total:
            raise ValueError('Initial observation task count/duplicates')
        receipt = read(directory / receipt_name)
        if receipt:
            if (receipt['plan_sha256'] != plan_sha or receipt['calibration_questions'] != 500
                    or receipt[cohort_field] != cohort_total or receipt['cost'][cost_field] != total):
                raise ValueError('Initial observation completion binding/count differs')
            for name, digest in receipt['files_sha256'].items():
                if sha(directory / name) != digest:
                    raise ValueError('Initial observation output SHA differs')
            records = [json.loads(line) for name in ('calibration_observations.jsonl', cohort_file)
                       for line in (directory / name).read_text().splitlines()]
        else:
            records = [read(path) for path in directory.glob('attempts/*/attempt_*/result.json')]
        seen = set()
        for row in records:
            identity = row['identity']; task_id = identity['task_id']; task = expected[task_id]
            if (task_id in seen or row['status'] != 'completed' or row['config_sha256'] != plan_sha
                    or identity != dict(task_id=task_id, qid=task['qid'], group=task['group'], plan_sha256=plan_sha)
                    or row['qid'] != task['qid'] or row['actual_device'].split(':')[0] != 'cuda'):
                raise ValueError('Initial observation record identity/device/duplicate differs')
            seen.add(task_id)
            state['calibration_completed' if task['group'] == 'calibration' else 'cohort_completed'] += 1
        state['completed'] = len(seen)
        state['complete'] = bool(receipt and state['completed'] == total
                                 and state['calibration_completed'] == 500 and state['cohort_completed'] == cohort_total)
        if receipt and not state['complete']:
            raise ValueError('Initial observation completion does not cover the whole cohort')
    except (OSError, ValueError, KeyError, TypeError) as error:
        state.update(complete=False, error=str(error))
    return state



def full4966_observer():
    return initial_observer(FULL_WEBQA/'work_cuda/observer', FULL_WEBQA/'OBSERVER_RUN_LAUNCH.json',
                            5466, 4966, 'observe_initial', 'completion.json',
                            'cohort_questions', 'cohort_observations.jsonl', 'completed_prefills')


def mmqa_observer():
    root = Path('/root/autodl-tmp/v318_mmqa_full2441_root_20261008')
    if (root/'OBSERVER_DECODER_RECOVERY_LAUNCH.json').exists():
        state = initial_observer(root/'work_cuda/observer_decoder_recovery', root/'OBSERVER_DECODER_RECOVERY_LAUNCH.json',
                                 2941, 2441, 'mmqa_decoder_recovery.observer', 'COMPLETE.json',
                                 'validation_questions', 'validation_observations.jsonl', 'completed')
        if state is not None:
            prior = read(root/'work_cuda/observer_decoder_recovery/PRIOR_ATTEMPTS.json')
            state['prior_completed'] = prior['cost']['completed']
            state['prior_failed'] = prior['cost']['failed']
            state['recovery_full_rerun'] = True
        return state
    return initial_observer(root/'work_cuda/observer', root/'OBSERVER_RUN_LAUNCH.json',
                            2941, 2441, 'mmqa_full2441_execution.observe', 'COMPLETE.json',
                            'validation_questions', 'validation_observations.jsonl', 'completed')


def mmqa_decoder_check():
    root = Path('/root/autodl-tmp/v318_mmqa_full2441_root_20261008/decoder_recovery_cpu_v2')
    if not (root / 'worker_0_LAUNCH.json').exists():
        return None
    state = dict(completed=0, total=2941, pids=[], complete=False, error=None)
    try:
        for worker in range(8):
            launch = read(root / f'worker_{worker}_LAUNCH.json')
            directory = root / f'worker_{worker}'
            expected = len(range(worker, 2941, 8))
            alive = process_matches(launch['pid'], 'mmqa_decoder_recovery.cpu_regression',
                                    str(root), '--worker ' + str(worker))
            if alive:
                state['pids'].append(launch['pid'])
            progress = read(directory / 'PROGRESS.json', {})
            if progress and (progress['expected'] != expected or not 0 <= progress['completed'] <= expected):
                raise ValueError('CPU input check shard denominator differs')
            state['completed'] += progress.get('completed', 0)
            done = read(directory / 'COMPLETE.json')
            if done:
                if (done['status'] != 'CPU_FULL_PROCESSOR_SHARD_PASS' or done['questions'] != expected
                        or done['qa_forward_calls'] != 0 or done['cuda_initialized']
                        or sha(directory / 'results.jsonl') != done['results_sha256']):
                    raise ValueError('CPU input check shard receipt differs')
            elif not alive:
                state['error'] = 'CPU 输入检查进程已退出，等待检查日志'
        merged = read(root / 'COMPLETE.json')
        if merged:
            if (merged['status'] != 'ALL_2941_ACTUAL_CPU_PROCESSOR_PASS' or merged['questions'] != 2941
                    or merged['qa_forward_calls'] != 0
                    or sha(root / 'ALL_RESULTS.jsonl') != merged['all_results_sha256']):
                raise ValueError('CPU input check completion receipt differs')
            state['complete'] = state['completed'] == 2941 and state['error'] is None
    except json.JSONDecodeError:
        state['updating'] = True
    except (OSError, ValueError, KeyError, TypeError) as error:
        state['error'] = str(error)
    return state


def mmqa_gpu_queue_alive(queue, state):
    version_sha = state.get('admission_version_sha256')
    if not version_sha:
        launch = read(queue / 'LAUNCH.json', {})
        return bool(launch and process_matches(launch['pid'], 'mmqa_decoder_recovery.gpu_continue', str(queue)))
    owners = [path.parent for path in queue.parent.glob('mmqa_admission_*/ADMISSION_VERSION.json')
              if sha(path) == version_sha]
    if len(owners) != 1:
        raise ValueError('MMQA admission identity is missing or ambiguous')
    owner = owners[0]
    version = read(owner / 'ADMISSION_VERSION.json')
    launch = read(owner / 'LAUNCH.json'); started = read(owner / 'STARTED.json')
    plan = read(queue / 'PLAN.json')
    if not (state['pid'] == started['pid'] == launch['pid']
            and state['plan_sha256'] == started['plan_sha256'] == version['queue_plan_sha256'] == sha(queue / 'PLAN.json')
            and started['admission_version_sha256'] == version_sha):
        raise ValueError('MMQA versioned owner identity differs')
    command = launch['command']; source = Path(command[4])
    if source.parent != owner or sha(source) != version['source_sha256']:
        raise ValueError('MMQA versioned scheduler source differs')
    if command != [plan['python'], '-B', '-P', '-u', str(source), '--queue', str(queue)]:
        raise ValueError('MMQA versioned launch command differs')
    try:
        actual = Path(f"/proc/{state['pid']}/cmdline").read_bytes().decode().rstrip('\0').split('\0')
    except (OSError, UnicodeDecodeError):
        return False
    return actual == command


def mmqa_continuation():
    if (DUAL / 'STATUS.json').exists():
        return dual_mmqa()
    root = Path('/root/autodl-tmp/v318_mmqa_full2441_root_20261008')
    resumed = Path('/root/autodl-tmp/v318_resume_20261009')
    cpu = resumed / 'mmqa_cpu_queue'; gpu = resumed / 'mmqa_gpu_queue'
    if not (cpu / 'PLAN.json').exists():
        return None
    state = dict(cpu_ready=0, total_jobs=6, gpu_running=0, gpu_complete=0,
                 logical_trials_planned=43938, jobs=[], error=None)
    try:
        plan = read(cpu / 'PLAN.json'); state['cpu'] = read(cpu / 'STATUS.json', {})
        launch = read(cpu / 'STARTED.json', {})
        state['cpu_alive'] = bool(launch and process_matches(launch['pid'], 'mmqa_continue.py', 'run-cpu', str(cpu)))
        for job in plan['jobs']:
            label = f'seed{job["seed"]}_{job["kind"]}'
            ready = read(cpu / (label + '_READY.json'))
            if ready:
                if (ready['seed'], ready['kind'], ready['output'], ready['questions']) != (job['seed'], job['kind'], job['output'], 2441):
                    raise ValueError('MMQA CPU ready scope differs')
                if sha(Path(job['output']) / 'config.json') != ready['config_sha256']:
                    raise ValueError('MMQA CPU ready config SHA differs')
                state['cpu_ready'] += 1
        if (gpu / 'PLAN.json').exists():
            queued = read(gpu / 'STATUS.json', {}); state['gpu'] = queued
            state['gpu_queue_alive'] = mmqa_gpu_queue_alive(gpu, queued)
            if queued and queued['plan_sha256'] != sha(gpu / 'PLAN.json'):
                raise ValueError('MMQA GPU queue identity differs')
            for job in read(gpu / 'PLAN.json')['jobs']:
                row = queued.get('jobs', {}).get(job['label'], {'status': 'PENDING'})
                alive = row['status'] == 'RUNNING' and process_matches(row['pid'], 'mmqa_decoder_recovery.generation', job['output'])
                state['gpu_running'] += bool(alive)
                if row['status'] == 'COMPLETE':
                    if sha(Path(job['output']) / 'COMPLETE.json') != row['complete_sha256']:
                        raise ValueError('MMQA generation completion SHA differs')
                    state['gpu_complete'] += 1
                ready = read(cpu / (job['label'] + '_READY.json'), {})
                detail = dict(seed=job['seed'], kind=job['kind'], status=row['status'], process_alive=bool(alive),
                              pid=row.get('pid'), model_loaded=row.get('model_loaded', False),
                              completed_physical=0, total_physical=ready.get('physical_calls'))
                if row['status'] != 'PENDING':
                    detail.update(mmqa_progress(job['output'], row['config_sha256']))
                state['jobs'].append(detail)
            if queued.get('status') == 'FAILED':
                state['error'] = queued.get('error', 'MMQA GPU queue failed')
            elif queued.get('status') == 'RUNNING' and not state['gpu_queue_alive']:
                state['error'] = 'MMQA GPU 接续进程已退出，需要检查'
        if (cpu / 'FAILED.json').exists():
            state['error'] = read(cpu / 'FAILED.json').get('error', 'MMQA CPU preparation failed')
    except (OSError, ValueError, KeyError, TypeError) as error:
        state['error'] = str(error)
    return state


def dual_state():
    state = read(DUAL / 'STATUS.json')
    plan = read(DUAL / 'PLAN.json')
    if (state['plan_sha256'] != sha(DUAL / 'PLAN.json') or sha(DUAL / 'resume.py') != plan['controller_sha256']
            or state['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip()):
        raise ValueError('Dual GPU resume identity differs')
    fields = ('status','pid','seed','kind','gpu_index','gpu_uuid','output','config_sha256',
              'model_loaded','first_generation_complete','completion_sha256','error','started_at')
    state['jobs'] = {key:{name:row[name] for name in fields if name in row} for key,row in state['jobs'].items()}
    return state


def dual_mmqa():
    current = dual_state()
    state = dict(cpu_ready=6, total_jobs=6, gpu_running=0, gpu_complete=0,
                 logical_trials_planned=43938, jobs=[], error=None,
                 gpu_queue_alive=process_matches(current['pid'], str(DUAL / 'resume.py')),
                 gpu=dict(status=current['status'], pid=current['pid'], phase='ALL_MM_QA_JOBS_ADMITTED'))
    for label, row in current['jobs'].items():
        if row['kind'] == 'ce':
            continue
        alive = row['status'] == 'RUNNING' and process_matches(row.get('pid'), 'mmqa_decoder_recovery.generation', row['output'])
        detail = dict(row, process_alive=alive)
        detail.update(mmqa_progress(row['output'], row['config_sha256']))
        if row['status'] == 'COMPLETE':
            if sha(Path(row['output']) / 'COMPLETE.json') != row['completion_sha256']:
                raise ValueError('MMQA completed receipt changed')
            state['gpu_complete'] += 1
        state['gpu_running'] += bool(alive)
        state['jobs'].append(detail)
        if row['status'] == 'FAILED':
            state['error'] = label + ': ' + row.get('error', 'worker failed')
    if not state['gpu_queue_alive'] and current['status'] == 'RUNNING':
        state['error'] = '双 GPU 接续进程已退出，需要检查'
    return state


def parse_gpus(text):
    numeric = lambda x: float(x) if x not in ('[N/A]', 'N/A', '[Not Supported]') else None
    gpus = []
    for line in text.strip().splitlines():
        index, uuid, name, utilization, used, total, temperature, power, limit = [p.strip() for p in line.split(',')]
        gpus.append(dict(index=int(index), uuid=uuid, name=name, utilization=numeric(utilization),
                         used_mib=numeric(used), total_mib=numeric(total), temperature=numeric(temperature),
                         power=numeric(power), power_limit=numeric(limit)))
    if not gpus:
        raise ValueError('No GPU telemetry returned')
    def combine(key, operation):
        values = [g[key] for g in gpus]
        return operation(values) if all(v is not None for v in values) else None
    aggregate = dict(name=f'{len(gpus)} GPUs', utilization=combine('utilization', lambda a: sum(a)/len(a)),
                     used_mib=combine('used_mib', sum), total_mib=combine('total_mib', sum),
                     temperature=combine('temperature', max), power=combine('power', sum), power_limit=combine('power_limit', sum))
    return gpus, aggregate


def additional_gpu_jobs(gpus):
    jobs = []
    work = FULL_WEBQA / 'dragin_iterative_formal_20261009/work_cuda'
    for path in work.glob('seed*/model_load_*.json'):
        row = read(path)
        if row['status'] != 'completed' or not process_matches(row['pid'], 'dragin_full_preparation.gpu_worker', str(path.parent)):
            continue
        try:
            env = dict(item.split('=',1) for item in Path(f"/proc/{row['pid']}/environ").read_bytes().decode().split('\0') if '=' in item)
        except OSError:
            continue
        gpu = next((g for g in gpus if g['uuid'] == env.get('CUDA_VISIBLE_DEVICES')), None)
        if gpu is not None:
            jobs.append(dict(kind='dragin', seed=row['seed'], pid=row['pid'], gpu_index=gpu['index'], model_loaded=True))
    return jobs


def mmqa_encoding():
    if not (MMQA_ENCODING / 'PLAN.json').exists():
        return None
    result = {}
    for kind, total in (('bge', 210139), ('clip', 44551)):
        state = dict(completed=None, total=total, pid=None, process_alive=False, complete=False, error=None)
        try:
            live = read(MMQA_ENCODING / f'{kind}_LIVE_STATUS.json', {})
            progress = read(MMQA_ENCODING / kind / 'PROGRESS.json', {})
            plan_sha = sha(MMQA_ENCODING / 'PLAN.json')
            if any(record and record['plan_sha256'] != plan_sha for record in (live, progress)):
                raise ValueError('MMQA encoder status belongs to another plan')
            if read(MMQA_ENCODING / 'PLAN.json')['missing_counts'][kind] != total:
                raise ValueError('MMQA encoder cohort count changed')
            state.update(pid=live.get('pid'), controller_pid=live.get('controller_pid'), status=live.get('status'),
                         updated_utc=progress.get('updated_utc', live.get('updated_utc')))
            state['process_alive'] = process_matches(state['pid'], 'encode_missing_cuda.py', '--kind ' + kind)
            state['completed'] = progress.get('completed_missing')
            if state['completed'] is not None and not 0 <= state['completed'] <= total:
                raise ValueError('MMQA encoder completed count outside cohort')
            state['complete'] = (live.get('status') == progress.get('status') == 'COMPLETE'
                                 and state['completed'] == total)
            if live.get('status') in ('FAILED', 'NUMERICAL_GATE_FAILED'):
                state['error'] = live['status'] + (f" (exit {live['exit_code']})" if 'exit_code' in live else '')
            if kind == 'clip' and live.get('status') == 'PREFIX_COMPLETE':
                assets = read(MMQA_ENCODING / 'IMAGE_ASSET_STATUS.json', {})
                state['waiting_for_images'] = assets.get('status') not in ('FAILED', 'CLIP_FULL_LAUNCHED')
                state['image_asset_status'] = assets.get('status', 'WAITING')
                if assets.get('status') == 'FAILED':
                    state['error'] = 'Image asset preparation failed'
        except (OSError, ValueError, KeyError, TypeError) as error:
            state.update(complete=False, error=str(error))
        result[kind] = state
    return result


def order_experiment():
    directory = RUN.parent / 'e1_cuda_extension'
    if not (directory / 'PLAN.json').exists():
        return None
    plan = read(directory / 'PLAN.json')
    progress = [read(directory / f'worker_{i}/PROGRESS.json', {}) for i in range(5)]
    queue = read(directory / 'QUEUE_STATUS.json', {})
    cal_rows = directory / 'CALIBRATION_ROWS.jsonl'
    return {'completed': sum(row.get('completed', 0) for row in progress),
            'total': plan['logical_answers'], 'questions': plan['samples'],
            'calibration': len(cal_rows.read_text().splitlines()) if cal_rows.exists() else 0,
            'calibration_total': plan['calibration_observations'], 'queue': queue,
            'complete': (directory / 'merged/COMPLETE.json').exists(),
            'running': [i for i, pid in queue.get('running', {}).items()
                        if str(directory / 'run_parallel_worker.py').encode() in
                        (Path(f'/proc/{pid}/cmdline').read_bytes() if Path(f'/proc/{pid}/cmdline').exists() else b'')]}


def paper_experiment():
    directory = RUN.parent / 'phase1_paper_cuda'
    launch = directory / 'OBSERVER_LAUNCH.json'
    if not launch.exists():
        return None
    observer = read(directory / 'work_cuda/observer/completion.json', {})
    prepared = directory / 'work_cuda/sorted/prepared'
    predictions = prepared / 'predictions.jsonl'
    answers, invalid = {}, set()
    if predictions.exists():
        for line in predictions.read_text().splitlines():
            try:
                row = json.loads(line)
                key = (row['qid'], row['arm'], row['order_id'])
                if row['order_id'] != 'sorted' or not isinstance(row['response'], str) or not row['answer_token_ids']:
                    invalid.add(key)
                if key in answers and answers[key] != (row['response'], row['answer_token_ids']):
                    invalid.add(key)
                answers[key] = (row['response'], row['answer_token_ids'])
            except (ValueError, KeyError):
                continue
    return {'completed': len(set(answers) - invalid), 'total': 768,
            'observation_completed': observer.get('cost', {}).get('completed_prefills', 0),
            'observation_total': 756, 'conflicts': len(invalid),
            'complete': (prepared / 'completion.json').exists(),
            'sequence': read(directory / 'SEQUENCE_STATUS.json', {})}


def completed_output_throughput(result):
    sources = []
    for group, template in (('full4966_generation', 'work_cuda/seed{seed}'),
                            ('ce_generation', 'work_cuda/ce_seed{seed}')):
        for row in (result.get(group) or {}).get('runs', []):
            if row.get('error'):
                return dict(status='CHECK_REQUIRED', error=row['error'], output_tokens_per_second=None)
            directory = FULL_WEBQA / template.format(seed=row['seed'])
            if row.get('status') == 'NOT_STARTED' and not (directory/'physical_calls.jsonl').exists():
                continue
            cached = GENERATION_CACHE.get(str(directory))
            if cached:
                sources.append(dict(label=directory.name, path=directory/'physical_calls.jsonl',
                                    config_sha256=cached['config_sha'], physical_ids=cached['physical_ids'],
                                    max_tokens=cached['max_tokens']))
    return TOKEN_THROUGHPUT.sample(sources, time.time())


def sample(previous):
    global DUAL
    active = read(DUAL.parent / 'ACTIVE_DUAL_RESUME.json')
    if active:
        candidate = Path(active['directory'])
        if candidate.parent != DUAL.parent or sha(candidate / 'PLAN.json') != active['plan_sha256']:
            raise ValueError('Active resume binding differs')
        if (candidate / 'STATUS.json').exists():
            DUAL = candidate
    now = time.time()
    plan = read(RUN / 'CONTROLS_PLAN.json')
    expected = set(plan['qids'])
    answers, incomplete, conflicts = {}, 0, set()
    files = [RUN / 'execution_cuda/predictions.jsonl', *RUN.glob('parallel/**/predictions.jsonl')]
    for path in files:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                incomplete += 1
                continue
            qid = row['qid']
            if qid not in expected or set(row['predictions']) != set(plan['arms']) or not all(isinstance(v, str) for v in row['predictions'].values()) or set(row.get('tokens', {})) != set(plan['arms']) or not all(isinstance(v, list) and all(isinstance(t, int) for t in v) for v in row['tokens'].values()):
                conflicts.add(qid)
                continue
            if qid in answers and (answers[qid]['predictions'] != row['predictions'] or answers[qid]['tokens'] != row['tokens']):
                conflicts.add(qid)
            answers[qid] = row
    for qid in conflicts:
        answers.pop(qid, None)
    gpu_text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit',
                                        '--format=csv,noheader,nounits'], text=True, timeout=5)
    gpus, gpu = parse_gpus(gpu_text)
    root = Path('/sys/fs/cgroup')
    stat = dict(line.split() for line in (root / 'cpu.stat').read_text().splitlines())
    usage = int(stat['usage_usec']) / 1e6
    quota, period = (root / 'cpu.max').read_text().split()
    cores = float(quota) / float(period) if quota != 'max' else len(os.sched_getaffinity(0))
    cpu = (usage - previous[1]) / (now - previous[0]) / cores * 100 if previous else None
    memory_limit = (root / 'memory.max').read_text().strip()
    memory_used = int((root / 'memory.current').read_text())
    workers, other_workers = [], []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            command = (process / 'cmdline').read_bytes().decode().split('\0')
        except (OSError, UnicodeDecodeError):
            continue
        if any(str(RUN) in arg and (arg.endswith('run_cuda_controls.py') or arg.endswith('run_worker.py') or arg.endswith('run_parallel_worker.py')) for arg in command):
            workers.append(int(process.name))
        elif any(arg.endswith(("run_refined.py", "run_mmqa_verify.py", "run_mmqa_cells.py", "run_video_dense.py", "run_worker.py")) for arg in command):
            other_workers.append({"pid": int(process.name), "dataset": command[command.index("--dataset")+1] if "--dataset" in command else "unknown"})
    cal_path = RUN / 'execution_cuda/CALIBRATION_ROWS.jsonl'
    calibration = sum(bool(line.strip()) for line in cal_path.read_text().splitlines()) if cal_path.exists() else 0
    result = {'sampled_at': now, 'run_path': str(RUN), 'total': len(expected), 'completed': len(answers),
              'arms': list(plan['arms']), 'logical_predictions': len(answers) * len(plan['arms']),
              'subsets': {s: sum(r['subset'] == s for r in answers.values()) for s in ('img', 'text')},
              'subset_totals': plan['subset_counts'], 'calibration': calibration, 'calibration_total': 64,
              'gpu': gpu, 'gpus': gpus, 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
              'cpu_pct': cpu, 'cpu_cores': cores, 'ram_used_bytes': memory_used,
              'ram_total_bytes': int(memory_limit) if memory_limit != 'max' else None,
              'workers': workers, 'incomplete_json_lines': incomplete, 'conflicting_qids': len(conflicts),
              'complete_receipt': (RUN / 'execution_cuda/COMPLETE.json').exists() or (RUN / 'parallel/merged/COMPLETE.json').exists(),
              'queue': read(RUN / 'parallel/QUEUE_STATUS.json', {}), 'other_model_workers': other_workers,
              'order_experiment': order_experiment(),
              'paper_experiment': paper_experiment(),
              'full4966_observer': full4966_observer(),
              'mmqa_observer': mmqa_observer(),
              'mmqa_decoder_check': mmqa_decoder_check(),
              'mmqa_continuation': mmqa_continuation(),
              'mmqa_encoding': mmqa_encoding(),
              'full4966_generation': full4966_generation(),
              'ce_generation': ce_generation(),
              'launch': read(RUN / 'ACTIVE_LAUNCH.json', read(RUN / 'LAUNCH.json', {})),
              'log': '\n'.join((RUN / 'RUN.log').read_text().splitlines()[-7:])}
    result['physical_output_throughput'] = completed_output_throughput(result)
    result['official_scoring'] = read(Path('/root/autodl-tmp/v318_resume_20261009/official_scoring/STATUS.json'), {})
    scoring = result['official_scoring']
    scoring['process_alive'] = process_matches(scoring.get('pid'), '/official_scoring/watch_server.py')
    if (DUAL / 'STATUS.json').exists():
        result['dual_resume'] = dual_state()
        for job in scoring.get('jobs', {}).values():
            if job.get('status') in ('RUNNING', 'SCORING', 'CHECKING', 'VALIDATING') and not scoring['process_alive']:
                job['status'] = 'INTERRUPTED_BY_MACHINE_MOVE'
    result['additional_gpu_jobs'] = additional_gpu_jobs(gpus)
    result['sample_duration_seconds'] = time.time() - now
    return result, (now, usage)


if __name__ == '__main__':
    previous = None
    while True:
        try:
            row, previous = sample(previous)
            import base64, zlib
            print('z1:' + base64.b64encode(zlib.compress(json.dumps(row, ensure_ascii=False).encode(), 3)).decode(), flush=True)
        except Exception as error:
            print(json.dumps({'sampled_at': time.time(), 'error': str(error)}), flush=True)
        time.sleep(3)
