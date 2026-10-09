"""Continue the fixed full-corpus CPU pipeline after the live encoders finish.

Two independent lanes replace, rather than overlap, their upstream encoder.
No signals, downloads, QA generation, MPS or CUDA work. Failed or partial upstream
outputs stop that lane; the other lane may still finish its independent work.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from .encode_global_text import HERE, ROOT, sha, save
from . import build_dense_index as B

REPO = ROOT.parents[1]
OUTPUT = HERE/'cpu_continuation_v2'
PY = ROOT/'cache/runtime/bin/python'
DENSE_PY = ROOT/'cache/runtime_dense_20261006/bin/python'
PREFIX = 'experiments.eab_first_token.runs.phase1_dense_retrieval_codex_20261006.'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def processes():
    result = {}
    text = subprocess.check_output(['ps','-axo','pid=,ppid=,lstart=,state=,command='], text=True)
    for line in text.splitlines():
        fields = line.split(None, 8)
        if len(fields) == 9:
            pid, parent = int(fields[0]), int(fields[1])
            result[pid] = dict(pid=pid, parent=parent, started=' '.join(fields[2:7]),
                               state=fields[7], command=fields[8])
    return result


def identity(process):
    return {k:process[k] for k in ('pid','started','command')}


def command(module, *args, encoder=False):
    return [str(PY if encoder else DENSE_PY), '-u', '-B', '-m', PREFIX+module, *map(str,args)]


def stages():
    lane, indices, searches = {}, {}, {}
    for kind in ('bge','clip'):
        indices[kind] = HERE/f'faiss_{kind}_full'
        encodings = ([HERE/'bge_text_cpu'] if kind == 'bge' else
                     [HERE/name for name in ('clip_images_cpu_wave0','clip_images_cpu_wave1',
                                             'clip_images_cpu_official_jpeg3','clip_images_cpu_original_30172755')])
        lane[kind] = []
        if kind == 'clip':
            lane[kind].append(('clip_wave1',command('encode_clip_assets','--output',encodings[1],encoder=True)))
        lane[kind].append((kind+'_index',command('build_dense_index','--kind',kind,'--encodings',*encodings,
                                               '--output',indices[kind])))
        for name, queries in (('calibration500','webqa_calibration500_queries_cpu'),('train256','webqa_train256_queries_cpu')):
            searches[kind,name] = HERE/'dense_top5_full'/f'{kind}_{name}'
            lane[kind].append((kind+'_'+name,command('search_dense_queries','--index',indices[kind],
                '--queries',HERE/queries,'--output',searches[kind,name])))
    cal, selected = HERE/'dense_relevance_full/calibration', HERE/'dense_relevance_full/train256'
    final = [
        ('calibrate',command('calibrate_dense_relevance','fit','--bge',searches['bge','calibration500'],
                            '--clip',searches['clip','calibration500'],'--output',cal)),
        ('select',command('calibrate_dense_relevance','merge','--bge',searches['bge','train256'],
                         '--clip',searches['clip','train256'],'--calibration',cal,'--output',selected)),
        ('sources',command('join_dense_source_metadata','--selection',selected,'--output',HERE/'dense_sources_full/train256'))]
    return lane, final


def prepare(output=OUTPUT):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    lane, final = stages()
    owners = {}
    current = processes()
    for kind, suffix, folder, count, complete in (
        ('bge', '-m '+PREFIX+'encode_global_text', 'bge_text_cpu', 544489, 'ENCODING_COMPLETE'),
        ('clip', '-m '+PREFIX+'encode_clip_assets --output experiments/eab_first_token/runs/phase1_dense_retrieval_codex_20261006/clip_images_cpu_wave0',
         'clip_images_cpu_wave0', 309793, 'SNAPSHOT_ENCODING_COMPLETE')):
        matched = [p for p in current.values() if p['command'].endswith(suffix)]
        if len(matched) != 1:
            raise ValueError(f'Need exactly one current {kind} encoder before booking continuation: {matched}')
        owners[kind] = dict(directory=str(HERE/folder), expected=count, completed_status=complete,
                            plan_sha256=sha(HERE/folder/'plan.json'), process=identity(matched[0]))
    files = [Path(__file__), HERE/'encode_global_text.py', ROOT/'score_visual_candidates.py']
    files += [HERE/(name+'.py') for name in ('encode_clip_assets','build_dense_index','search_dense_queries',
              'calibrate_dense_relevance','join_dense_source_metadata','encode_webqa_queries')]
    files += [HERE/name for name in ('GLOBAL_POOL_MANIFEST.json','DENSE_ECDF_PROTOCOL.md','SOURCE_METADATA_PROTOCOL.md')]
    for folder in ('clip_images_cpu_wave0','clip_images_cpu_wave1','clip_images_cpu_official_jpeg3','clip_images_cpu_original_30172755'):
        files += [HERE/folder/name for name in ('manifest.json','assets.jsonl')]
    for folder in ('webqa_calibration500_queries_cpu','webqa_train256_queries_cpu'):
        files += [HERE/folder/name for name in ('plan.json','completion.json','bge.json','bge.npz','clip.json','clip.npz')]
    plan = dict(created_utc=now(), upstream=owners, lanes=lane, final=final,
                files_sha256={str(p):sha(p) for p in files}, threads_per_lane=2,
                max_parallel_lanes=2, new_MPS_calls=0, new_QA_calls=0, network_required=False,
                final_scope='Full-corpus selected source metadata. Actual initial U and QA generation remain separate.')
    if (output/'plan.json').exists():
        old = read(output/'plan.json'); plan['created_utc'] = old['created_utc']
        if json.loads(json.dumps(plan)) != old:
            raise ValueError('Existing continuation binding differs; preserve it')
    else:
        save(output/'plan.json',plan)
    return plan


def wait_upstream(kind, bound, output):
    directory = Path(bound['directory']); expected = bound['process']
    save(output/(kind+'_status.json'),dict(status='WAITING_LIVE_ENCODER',observed_utc=now(),process=expected))
    while True:
        observed = processes().get(expected['pid'])
        # T is a scheduler/user pause, not failure. A reused PID never receives any action.
        if observed is not None and identity(observed) == expected and not observed['state'].startswith('Z'):
            time.sleep(30)
            continue
        with (directory/'worker.lock').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
            complete = read(directory/'completion.json')
            if (sha(directory/'plan.json') != bound['plan_sha256']
                    or complete['plan_sha256'] != bound['plan_sha256']
                    or complete['status'] != bound['completed_status']
                    or complete['completed'] != complete['expected'] or complete['completed'] != bound['expected']):
                raise ValueError(f'{kind} exited without the complete frozen corpus')
        save(output/(kind+'_upstream_complete.json'),dict(status='VERIFIED_TERMINAL_COMPLETE',checked_utc=now(),
            original_process=expected,current_pid_observation=observed,
            completion_sha256=sha(directory/'completion.json')))
        return


def run_stage(name, args, output, plan):
    for path, expected in plan['files_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'Continuation input changed: {path}')
    receipt = output/(name+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    event = dict(stage=name,command=args,started_utc=now(),status='started')
    save(receipt.with_suffix('.json'),event)
    env = dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
               TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    tick = time.perf_counter()
    with receipt.with_suffix('.log').open('w') as log:
        process = subprocess.Popen(args,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        event.update(pid=process.pid)
        save(receipt.with_suffix('.json'),event)
        code = process.wait()
    event.update(status='completed' if code == 0 else 'failed',returncode=code,
                 wall_seconds=time.perf_counter()-tick,completed_utc=now(),log_sha256=sha(receipt.with_suffix('.log')))
    save(receipt.with_suffix('.json'),event)
    print(json.dumps(event),flush=True)
    if code:
        raise subprocess.CalledProcessError(code,args)


def run_lane(kind, plan, output):
    wait_upstream(kind,plan['upstream'][kind],output)
    for name, args in plan['lanes'][kind]:
        if name == 'clip_wave1':
            directory = Path(args[args.index('--output')+1])
            if (directory/'completion.json').exists():
                _, _, ids, _ = B.inspect_inputs('clip',[directory])
                save(output/'clip_wave1_reused.json',dict(status='VERIFIED_ENCODING_REUSED',
                    checked_utc=now(),directory=str(directory),verified_rows=len(ids),
                    completion_sha256=sha(directory/'completion.json'),new_encoder_calls=0))
                continue
        save(output/(kind+'_status.json'),dict(status='RUNNING_STAGE',stage=name,updated_utc=now()))
        run_stage(name,args,output,plan)
    save(output/(kind+'_status.json'),dict(status='LANE_COMPLETE',completed_utc=now()))


def run(output=OUTPUT):
    output = Path(output)
    with (output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read(output/'plan.json')
        expected_lanes, expected_final = stages()
        if plan['lanes'] != json.loads(json.dumps(expected_lanes)) or plan['final'] != json.loads(json.dumps(expected_final)):
            raise ValueError('Continuation commands differ from the fixed full-corpus workflow')
        save(output/'owner.json',dict(pid=os.getpid(),started_utc=now(),plan_sha256=sha(output/'plan.json')))
        errors = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(run_lane,k,plan,output):k for k in ('bge','clip')}
            for future in as_completed(futures):
                kind = futures[future]
                try:
                    future.result()
                except Exception as error:
                    errors[kind] = dict(type=type(error).__name__,message=str(error))
                    save(output/(kind+'_status.json'),dict(status='FAILED',updated_utc=now(),error=errors[kind]))
                    print(json.dumps(dict(lane=kind,error=errors[kind])),flush=True)
        if errors:
            save(output/'failed.json',dict(failed_utc=now(),errors=errors))
            raise RuntimeError(f'Upstream lane failure: {errors}')
        for name, args in plan['final']:
            run_stage(name,args,output,plan)
        result = dict(status='CPU_FULL_SOURCE_PIPELINE_COMPLETE_U_PENDING',completed_utc=now(),
                      plan_sha256=sha(output/'plan.json'),labels_read=False,new_QA_calls=0,new_MPS_calls=0,
                      source_completion_sha256=sha(HERE/'dense_sources_full/train256/completion.json'))
        save(output/'completion.json',result)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    args = parser.parse_args()
    print(json.dumps(prepare() if args.action == 'prepare' else run()),flush=True)
