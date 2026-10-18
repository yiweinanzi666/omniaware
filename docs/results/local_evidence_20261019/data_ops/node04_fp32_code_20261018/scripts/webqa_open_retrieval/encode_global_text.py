"""Resume CPU BGE encoding of the complete public WebQA text pool, without Q/A labels."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import itertools
import json
from pathlib import Path
import time

from omniaware import paths

HERE = paths.WEBQA_OPEN
ROOT = paths.ASSET_ROOT
CORPUS = HERE / 'text_documents.jsonl'
MODEL = ROOT / 'cache/bge-large-en-v1.5'
OUTPUT = HERE / 'bge_text_cpu'
COUNT, SHARD, BATCH = 544489, 128, 4
CORPUS_SHA = '57b2b79d24c0fbc7b36d065bef287642f3f8c6c2c048f29027d29c721053bdc8'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def chunks():
    with CORPUS.open() as stream:
        start = 0
        while lines := list(itertools.islice(stream, SHARD)):
            rows = [json.loads(line) for line in lines]
            assert [r['row_index'] for r in rows] == list(range(start, start + len(rows)))
            for row in rows:
                assert row['document_id'] == 'text-sha256:' + hashlib.sha256(row['text'].encode()).hexdigest()
            yield start, rows
            start += len(rows)
        assert start == COUNT


def main(max_new_shards=None):
    import numpy as np
    import torch
    import transformers
    from omniaware.rerank import MODEL_FILES, MODEL_REVISION, QUERY_INSTRUCTION, encode_bge

    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert sha(CORPUS) == CORPUS_SHA
        plan = dict(corpus=str(CORPUS), corpus_sha256=CORPUS_SHA, documents=COUNT,
            model=str(MODEL), model_revision=MODEL_REVISION,
            model_files_sha256={name: sha(MODEL/name) for name in MODEL_FILES},
            source_sha256={str(p): sha(p) for p in (Path(__file__), ROOT/'multimodal_retrieval.py')},
            torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
            device='cpu', dtype='float32', threads=2, batch_size=BATCH, shard_size=SHARD,
            pooling='CLS, L2 normalized', max_length=512, document_input='exact text body; no title or instruction',
            query_instruction=QUERY_INSTRUCTION, attention='sdpa',
            protocol='New reproducible global-public-pool protocol; not recovered historical paper settings',
            cost_boundary='Encoding seconds include tokenization/forward/normalization; loading and disk I/O separate',
            labels_read=False, MPS_calls=0)
        plan_path = OUTPUT/'plan.json'
        if plan_path.exists():
            assert json.loads(plan_path.read_text()) == plan, 'Frozen encoder plan changed'
        else:
            save(plan_path, plan)
        plan_sha = sha(plan_path)
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        model = tokenizer = None
        completed = new = 0
        load_seconds = 0.
        encoding_seconds = 0.
        for start, rows in chunks():
            array_path = OUTPUT/f'{start:06d}.npz'
            receipt_path = array_path.with_suffix('.json')
            identity = dict(start=start, stop=start+len(rows), plan_sha256=plan_sha,
                document_ids_sha256=hashlib.sha256('\n'.join(r['document_id'] for r in rows).encode()).hexdigest())
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                assert receipt['identity'] == identity and receipt['array_sha256'] == sha(array_path)
                completed += len(rows)
                continue
            if max_new_shards is not None and new >= max_new_shards:
                break
            if model is None:
                tick = time.perf_counter()
                tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
                model = transformers.AutoModel.from_pretrained(MODEL, local_files_only=True,
                    dtype=torch.float32, attn_implementation='sdpa').to('cpu').eval()
                load_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            vectors, lengths = encode_bge([r['text'] for r in rows], tokenizer, model, BATCH)
            elapsed = time.perf_counter() - tick
            array = vectors.numpy()
            assert array.shape == (len(rows), 1024) and np.isfinite(array).all()
            assert np.allclose(np.linalg.norm(array, axis=1), 1., atol=2e-6)
            temporary = array_path.with_suffix('.npz.tmp')
            with temporary.open('wb') as stream:
                np.savez(stream, vectors=array, input_lengths=np.asarray(lengths, dtype=np.int32))
            temporary.replace(array_path)
            save(receipt_path, dict(identity=identity, array_sha256=sha(array_path),
                completed_utc=datetime.now(timezone.utc).isoformat(), encoding_seconds=elapsed,
                encoder_batches=(len(rows)+BATCH-1)//BATCH, documents=len(rows)))
            completed += len(rows)
            new += 1
            encoding_seconds += elapsed
            progress = dict(status='ENCODING', completed=completed, expected=COUNT,
                new_shards_this_session=new, plan_sha256=plan_sha,
                last_shard_seconds=elapsed, load_seconds=load_seconds, MPS_calls=0)
            save(OUTPUT/'progress.json', progress)
            print(json.dumps(progress), flush=True)
        result = dict(status='ENCODING_COMPLETE' if completed == COUNT else 'PARTIAL_CHECKPOINT',
            completed=completed, expected=COUNT, plan_sha256=plan_sha, new_shards_this_session=new,
            encoding_seconds_this_session=encoding_seconds, load_seconds=load_seconds,
            ended_utc=datetime.now(timezone.utc).isoformat(), MPS_calls=0, labels_read=False)
        save(OUTPUT/'progress.json', result)
        if completed == COUNT:
            save(OUTPUT/'completion.json', result)
        print(json.dumps(result), flush=True)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-new-shards', type=int)
    args = parser.parse_args()
    main(args.max_new_shards)
