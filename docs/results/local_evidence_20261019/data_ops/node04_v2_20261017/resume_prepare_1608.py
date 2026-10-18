"""Authorized preparation recovery; original experiment source and cache remain untouched."""
import csv, datetime, hashlib, json, os, pathlib, subprocess, sys, traceback
import numpy as np

B = pathlib.Path('/home/user/Futaoran')
A = B / 'omniaware/assets'
R = B / 'omniaware_v2_exec'
L = A / 'manifests/v2_run_20261017'
V = B / 'omniaware_v2_assets_20261017_resume'
C = B / 'omniaware_v2_cache_20261017_resume'
record = {'start': datetime.datetime.now().astimezone().isoformat(), 'command': [sys.executable, str(pathlib.Path(__file__).resolve())], 'asset_view': str(V), 'new_cache': str(C)}

def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

try:
    if V.exists() or C.exists():
        raise RuntimeError('Recovery destination already exists; inspect before resuming')
    source = A / 'downloads/okvqa_gs/okvqa_train_corpus.csv'
    assert sha(source) == 'f6646ad30d05166cfd32c281566a0032656035761e7d33acaf3267bfbb9d4997'
    with source.open(newline='', encoding='utf8') as f:
        texts = [' '.join((row['text'] or '').split()) for row in csv.DictReader(f)]
    texts = [t for t in texts if t]
    assert len(texts) == 112724, len(texts)
    old = A.parent / 'cache/okvqa_gs112k'
    # The existing embedding is reusable only if its complete indexed prefix equals the official CSV.
    old_offsets = []
    with (old / 'corpus.jsonl').open('rb') as f:
        for i, text in enumerate(texts):
            old_offsets.append(f.tell())
            row = json.loads(f.readline())
            assert row == {'id': i, 'contents': text}, 'old corpus prefix differs at ' + str(i)
        prefix_bytes = f.tell()
        tail = f.read()
    assert np.array_equal(np.load(old / 'offsets.npy'), np.array(old_offsets, dtype=np.int64)), 'offsets differ'
    assert (old / 'emb_112724.f16').stat().st_size == 230858752
    assert (old / 'emb_112724.done').read_text().strip() == '112724'
    qids = []
    with (old / 'candidates_full.jsonl').open() as f:
        for line in f:
            row = json.loads(line)
            qids.append(str(row['question_id']))
            assert len(row['candidates']) == 20
            assert [c['rank'] for c in row['candidates']] == list(range(1, 21))
            for c in row['candidates']:
                assert c['text'] == texts[int(c['doc_id'])], 'candidate/corpus mismatch'
    assert len(qids) == len(set(qids)) == 5046
    new = C / 'okvqa_gs112k'
    new.mkdir(parents=True)
    offsets = []
    with (new / 'corpus.jsonl').open('xb') as f:
        for i, text in enumerate(texts):
            offsets.append(f.tell())
            f.write((json.dumps({'id': i, 'contents': text}, ensure_ascii=False) + '\n').encode('utf8'))
    with (new / 'offsets.npy').open('xb') as f:
        np.save(f, np.array(offsets, dtype=np.int64))
    for name in ['emb_112724.f16', 'emb_112724.done', 'candidates_full.jsonl']:
        os.link(old / name, new / name)
    record['gs'] = {'official_csv_sha256': sha(source), 'rows': len(texts), 'old_indexed_prefix_matches_csv': True, 'old_offsets_match_prefix': True, 'candidate_texts_match_csv': True, 'candidate_rows': len(qids), 'old_corpus_sha256': sha(old / 'corpus.jsonl'), 'new_corpus_sha256': sha(new / 'corpus.jsonl'), 'prefix_bytes': prefix_bytes, 'extra_tail_bytes': len(tail), 'original_cache_changed': False, 'reused': ['emb_112724.f16', 'emb_112724.done', 'candidates_full.jsonl']}
    V.mkdir()
    for child in A.iterdir():
        if child.name != 'cache':
            (V / child.name).symlink_to(child, target_is_directory=child.is_dir())
    (V / 'cache').mkdir()
    for child in (A / 'cache').iterdir():
        target = R / 'external/M2RAG' if child.name == 'official_m2rag' else child
        (V / 'cache' / child.name).symlink_to(target, target_is_directory=target.is_dir())
    pinned = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=V / 'cache/official_m2rag', text=True).strip()
    assert pinned == 'ea6b9977f5a153215b38cd9a6aa3e4702ffbda22'
    record['m2rag_commit'] = pinned
    env_lines = ['export PATH=/home/user/Futaoran/omniaware_v2_python311/bin:$PATH', 'export OMNIAWARE_ASSETS=' + str(V), 'export OMNIAWARE_CACHE=' + str(C), 'export OMNIAWARE_WEBQA_OPEN_DIR=' + str(A / 'webqa_open'), 'export NO_PROXY=localhost,127.0.0.1,::1', 'export no_proxy=localhost,127.0.0.1,::1', 'export HF_ENDPOINT=https://hf-mirror.com']
    with (L / 'env_resume_1608.sh').open('x') as f: f.write('\n'.join(env_lines) + '\n')
    record.update(ok=True, exit_code=0)
except Exception:
    record.update(ok=False, exit_code=1, traceback=traceback.format_exc())
record['end'] = datetime.datetime.now().astimezone().isoformat()
with (L / 'resume_prepare_1608.json').open('x') as f: json.dump(record, f, indent=2)
print(json.dumps(record), flush=True)
sys.exit(record['exit_code'])
