"""Freeze BGE/CLIP text-query vectors for a WebQA question cohort on CPU (default: legacy train256).

python -m scripts.webqa_open_retrieval.encode_webqa_queries --split val --output <dir>
writes the cohort (qid, question; no answers or facts) and its query vectors for search_dense_queries.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import time

from .encode_global_text import HERE, ROOT, sha, save

COHORT = ROOT/'runs/protocol_alignment_phase0_codex_20261006/PHASE1_WEBQA_TRAIN256_IDS.jsonl'
COHORT_SHA = '0066999cee7d20ff9b6aa6493c0bf97a55da50cde4aa4b5ae13440c99d87ccee'
OUTPUT = HERE/'webqa_train256_queries_cpu'


def write_cohort(split, path):
    """All questions of a WebQA split as {qid, question}; answers and facts are never written."""
    from omniaware import paths
    from omniaware.data import _clean
    raw = json.loads(paths.WEBQA_JSON.read_text())
    rows = [dict(qid=guid, question=_clean(v["Q"])) for guid, v in raw.items() if v["split"] == split]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return path


def main(cohort=COHORT, cohort_sha=COHORT_SHA, output=OUTPUT):
    import numpy as np
    from PIL import Image
    import torch
    import transformers
    from omniaware.rerank import QUERY_INSTRUCTION, encode_bge, normalize_features

    output.mkdir(exist_ok=True)
    with (output/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert cohort_sha is None or sha(cohort) == cohort_sha
        rows = [json.loads(line) for line in cohort.read_text().splitlines()]
        n = len(rows)
        assert n and len({r['qid'] for r in rows}) == n
        assert not any({'answer', 'answers', 'A', 'references', 'gold'} & set(r) for r in rows)
        questions = [r['question'] for r in rows]
        ids = np.asarray([r['qid'] for r in rows])
        assert all(isinstance(q, str) and q.strip() for q in questions)
        corpus_plans = {name: HERE/path for name, path in
                        (('bge', 'bge_text_cpu/plan.json'), ('clip', 'clip_images_cpu_wave0/plan.json'))}
        models = {name: ROOT/'cache'/folder for name, folder in
                  (('bge', 'bge-large-en-v1.5'), ('clip', 'clip-vit-large-patch14'))}
        expected = {name: json.loads(path.read_text())['model_files_sha256'] for name, path in corpus_plans.items()}
        for name, assets in expected.items():
            assert {asset: sha(models[name]/asset) for asset in assets} == assets, f'{name} assets changed'
        fixture = json.loads((HERE/'clip_images_cpu_wave0/assets.jsonl').open().readline())
        plan = dict(cohort_sha256=sha(cohort), questions=n, ordered_qids=ids.tolist(),
                    corpus_plan_sha256={name: sha(path) for name, path in corpus_plans.items()},
                    model_files_sha256=expected,
                    bge_input=QUERY_INSTRUCTION+'<exact question, including original quotes>', bge_max_length=512,
                    clip_input='Exact question, no added template', clip_max_length=77,
                    device='cpu', dtype='float32', attention='sdpa', threads=2, batch_size=4,
                    pooling=dict(bge='CLS L2 normalized', clip='text projection L2 normalized'),
                    source_sha256={str(p): sha(p) for p in (Path(__file__), HERE/'encode_global_text.py',
                                  ROOT/'multimodal_retrieval.py', ROOT/'prepare_webqa.py', ROOT/'score_visual_candidates.py')},
                    native_clip_fixture=dict(image_id=fixture['image_id'], sha256=fixture['sha256']),
                    torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
                    labels_read=False, MPS_calls=0, cohort_finalized=False,
                    scope='Query preparation only; final consumption check and cross-modality retrieval protocol still required')
        if (output/'plan.json').exists():
            assert json.loads((output/'plan.json').read_text()) == plan, 'Query encoding plan changed'
        else:
            save(output/'plan.json', plan)
        plan_sha = sha(output/'plan.json')
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        receipts = {}
        for name, dimension, max_length in (('bge', 1024, 512), ('clip', 768, 77)):
            path = output/f'{name}.npz'
            receipt_path = output/f'{name}.json'
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                assert receipt['plan_sha256'] == plan_sha and receipt['array_sha256'] == sha(path)
                with np.load(path) as saved:
                    assert np.array_equal(saved['qids'], ids) and saved['vectors'].shape == (n, dimension)
                receipts[name] = receipt
                continue
            assert not path.exists(), 'Unreceipted query array requires inspection'
            started = time.perf_counter()
            tokenizer = transformers.AutoTokenizer.from_pretrained(models[name], local_files_only=True)
            cls = transformers.AutoModel if name == 'bge' else transformers.CLIPModel
            model = cls.from_pretrained(models[name], local_files_only=True, dtype=torch.float32,
                                        attn_implementation='sdpa').eval()
            loaded = time.perf_counter()
            inputs = [QUERY_INSTRUCTION+q for q in questions] if name == 'bge' else questions
            raw_lengths = [len(tokenizer(q, truncation=False)['input_ids']) for q in inputs]
            if name == 'bge':
                vectors, lengths = encode_bge(inputs, tokenizer, model, 4)
                batch = tokenizer(inputs[:1], truncation=True, max_length=512, return_tensors='pt')
                with torch.inference_mode():
                    single = torch.nn.functional.normalize(model(**batch).last_hidden_state[:, 0].float(), dim=1)
                check_error = float((vectors[0]-single[0]).abs().max())
                assert check_error < 1e-5
                check = dict(check='first query independently encoded without batch padding', max_absolute_error=check_error)
            else:
                vectors, lengths = [], []
                with torch.inference_mode():
                    for start in range(0, len(inputs), 4):
                        batch = tokenizer(inputs[start:start+4], padding=True, truncation=True,
                                          max_length=77, return_tensors='pt')
                        lengths.extend(batch['attention_mask'].sum(1).tolist())
                        vectors.append(normalize_features(model.get_text_features(**batch).pooler_output))
                    vectors = torch.cat(vectors)
                    assert sha(fixture['path']) == fixture['sha256']
                    processor = transformers.CLIPProcessor.from_pretrained(models[name], local_files_only=True)
                    with Image.open(fixture['path']) as im:
                        batch = processor(text=inputs[:1], images=[im.convert('RGB')],
                                          truncation=True, max_length=77, return_tensors='pt')
                    native = model(**batch)
                    native_cosine = float(native.logits_per_text[0, 0]/model.logit_scale.exp())
                    cached_cosine = float(vectors[0] @ native.image_embeds[0])
                check_error = abs(native_cosine-cached_cosine)
                assert check_error < 1e-5
                check = dict(check='first query cosine against native full CLIP forward',
                             native_cosine=native_cosine, cached_cosine=cached_cosine, absolute_error=check_error)
            array = vectors.numpy()
            assert array.shape == (n, dimension) and np.isfinite(array).all()
            norm_error = float(np.abs(np.linalg.norm(array, axis=1)-1).max())
            assert norm_error < 2e-6
            temporary = path.with_suffix('.npz.tmp')
            with temporary.open('wb') as stream:
                np.savez(stream, vectors=array, qids=ids, input_lengths=np.asarray(lengths, dtype=np.int32),
                         untruncated_lengths=np.asarray(raw_lengths, dtype=np.int32))
            temporary.replace(path)
            receipt = dict(status='QUERY_ENCODING_COMPLETE', plan_sha256=plan_sha, array_sha256=sha(path),
                           queries=n, dimension=dimension, batches=-(-n // 4), model_load_seconds=loaded-started,
                           encoding_validation_write_seconds=time.perf_counter()-loaded,
                           truncated_queries=sum(n > max_length for n in raw_lengths),
                           max_norm_error=norm_error, numerical_validation=check,
                           labels_read=False, MPS_calls=0, completed_utc=datetime.now(timezone.utc).isoformat())
            save(receipt_path, receipt)
            receipts[name] = receipt
            del model, tokenizer, vectors
        record = dict(status='QUERY_VECTORS_COMPLETE', plan_sha256=plan_sha,
                      encoders={name: dict(receipt_sha256=sha(output/f'{name}.json'), array_sha256=r['array_sha256'])
                                for name, r in receipts.items()}, labels_read=False, MPS_calls=0,
                      scope='Resource preparation; no retrieval or answer accuracy result')
        if (output/'completion.json').exists():
            assert json.loads((output/'completion.json').read_text()) == record
        else:
            save(output/'completion.json', record)
        print(json.dumps(record), flush=True)
        return record



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split', choices=['train', 'val'], help='encode every question of this WebQA split')
    parser.add_argument('--cohort', type=Path, help='jsonl of {qid, question}; default: legacy train256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.split:
        output = args.output or HERE/f'webqa_{args.split}_queries_cpu'
        main(write_cohort(args.split, output/'cohort.jsonl'), None, output)
    elif args.cohort:
        main(args.cohort, None, args.output or HERE/f'{args.cohort.stem}_queries_cpu')
    else:
        main()
