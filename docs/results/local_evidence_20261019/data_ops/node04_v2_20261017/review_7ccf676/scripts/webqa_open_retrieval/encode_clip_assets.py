"""Resume fixed-snapshot CLIP image encoding on CPU; no questions, labels, or MPS work."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import itertools
import json
from pathlib import Path
import time

from .encode_global_text import HERE, ROOT, sha, save

MODEL = ROOT/'cache/clip-vit-large-patch14'
REVISION = '32bd64288804d66eefd0ccbe215aa642df71cc41'
BATCH, SHARD = 4, 128


def main(output, max_new_shards=None):
    import numpy as np
    import PIL
    from PIL import Image
    import torch
    import transformers
    from omniaware.rerank import normalize_features

    output = Path(output)
    with (output/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads((output/'manifest.json').read_text())
        assert sha(output/'assets.jsonl') == manifest['assets_sha256']
        model_manifest = json.loads((MODEL/'download_manifest.json').read_text())
        assert model_manifest['revision'] == REVISION
        actual = {name: sha(MODEL/name) for name in model_manifest['assets']}
        assert actual == {name: r['sha256'] for name, r in model_manifest['assets'].items()}, 'CLIP asset identity changed'
        plan = dict(assets_sha256=manifest['assets_sha256'], images=manifest['images'], revision=REVISION,
            model_files_sha256=actual, device='cpu', dtype='float32', attention='sdpa', threads=2,
            batch_size=BATCH, shard_size=SHARD, projection_dimension=768,
            preprocessing='PIL strict decode to RGB; official CLIP processor resize/center crop; L2 normalized projection',
            torch=torch.__version__, transformers=transformers.__version__, pillow=PIL.__version__, numpy=np.__version__,
            source_sha256={str(p): sha(p) for p in (Path(__file__), HERE/'encode_global_text.py', ROOT/'score_visual_candidates.py')},
            labels_read=False, MPS_calls=0, global_pool_complete=False)
        if (output/'plan.json').exists():
            assert json.loads((output/'plan.json').read_text()) == plan, 'Frozen CLIP plan changed'
        else:
            save(output/'plan.json', plan)
        plan_sha = sha(output/'plan.json')
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        model = processor = None
        completed = new = 0
        load_seconds = encoding_seconds = 0.
        started = time.perf_counter()
        with (output/'assets.jsonl').open() as stream:
            while lines := list(itertools.islice(stream, SHARD)):
                rows = [json.loads(line) for line in lines]
                ids = [r['image_id'] for r in rows]
                path = output/f'{completed:06d}.npz'
                receipt_path = path.with_suffix('.json')
                identity = dict(start=completed, stop=completed+len(rows), plan_sha256=plan_sha,
                    assets_rows_sha256=hashlib.sha256(''.join(lines).encode()).hexdigest())
                if receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text())
                    assert receipt['identity'] == identity and receipt['array_sha256'] == sha(path)
                    completed += len(rows)
                    continue
                assert not path.exists(), 'Array without completed receipt; inspect interrupted write before resuming'
                if max_new_shards is not None and new >= max_new_shards:
                    break
                if model is None:
                    tick = time.perf_counter()
                    processor = transformers.CLIPProcessor.from_pretrained(MODEL, local_files_only=True)
                    model = transformers.CLIPModel.from_pretrained(MODEL, local_files_only=True,
                        dtype=torch.float32, attn_implementation='sdpa').eval()
                    load_seconds = time.perf_counter()-tick
                tick = time.perf_counter()
                vectors = []
                with torch.inference_mode():
                    for start in range(0, len(rows), BATCH):
                        images = []
                        for row in rows[start:start+BATCH]:
                            assert sha(row['path']) == row['sha256'], f"Pixels changed: {row['image_id']}"
                            with Image.open(row['path']) as image:
                                assert image.size == (row['width'], row['height'])
                                images.append(image.convert('RGB'))
                        batch = processor(images=images, return_tensors='pt')
                        vectors.append(normalize_features(model.get_image_features(**batch).pooler_output).numpy())
                elapsed = time.perf_counter()-tick
                array = np.concatenate(vectors)
                assert array.shape == (len(rows), 768) and np.isfinite(array).all()
                assert np.allclose(np.linalg.norm(array, axis=1), 1., atol=2e-6)
                # Verify projection/cosine convention once against the native full CLIP forward.
                if not (output/'native_forward_validation.json').exists():
                    with Image.open(rows[0]['path']) as im, torch.inference_mode():
                        native = processor(text=['a photograph'], images=[im.convert('RGB')], return_tensors='pt')
                        answer = model(**native)
                        direct = float(answer.logits_per_text[0, 0]/model.logit_scale.exp())
                        cached = float(torch.from_numpy(array[0]) @ answer.text_embeds[0])
                    assert abs(direct-cached) < 1e-5
                    save(output/'native_forward_validation.json', dict(status='PASS', image_id=ids[0],
                        plan_sha256=plan_sha, native_cosine=direct, cached_cosine=cached, absolute_error=abs(direct-cached),
                        scope='Synthetic text checks embedding convention, not benchmark answer quality'))
                temporary = path.with_suffix('.npz.tmp')
                with temporary.open('wb') as handle:
                    np.savez(handle, vectors=array, image_ids=np.asarray(ids, dtype='U8'))
                temporary.replace(path)
                save(receipt_path, dict(identity=identity, array_sha256=sha(path), images=len(rows),
                    encoder_batches=(len(rows)+BATCH-1)//BATCH, encoding_seconds=elapsed,
                    cost_boundary='image hash/decode, processor and CLIP image forward; loading/native check/disk I/O separate',
                    completed_utc=datetime.now(timezone.utc).isoformat()))
                completed += len(rows)
                new += 1
                encoding_seconds += elapsed
                progress = dict(status='ENCODING', completed=completed, expected=manifest['images'],
                    new_shards_this_session=new, last_shard_seconds=elapsed, plan_sha256=plan_sha, MPS_calls=0)
                save(output/'progress.json', progress)
                print(json.dumps(progress), flush=True)
        result = dict(status='SNAPSHOT_ENCODING_COMPLETE' if completed == manifest['images'] else 'PARTIAL_CHECKPOINT',
            completed=completed, expected=manifest['images'], global_pool_complete=False, new_shards_this_session=new,
            load_seconds=load_seconds, encoding_seconds_this_session=encoding_seconds,
            wall_seconds_this_session=time.perf_counter()-started, plan_sha256=plan_sha,
            ended_utc=datetime.now(timezone.utc).isoformat(), labels_read=False, MPS_calls=0)
        save(output/'progress.json', result)
        if completed == manifest['images']:
            save(output/'completion.json', result)
        print(json.dumps(result), flush=True)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-new-shards', type=int)
    args = parser.parse_args()
    main(args.output, args.max_new_shards)
