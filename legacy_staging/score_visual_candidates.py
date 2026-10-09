"""Encode actual WebQA pixels with CLIP and retain query-image cosine scores."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .multimodal_retrieval import load_inputs, row_hash, save_json
from .prepare_webqa import ROOT, sha256_file, write_jsonl


MODEL = ROOT / 'cache/clip-vit-large-patch14'
REVISION = '32bd64288804d66eefd0ccbe215aa642df71cc41'


def image_assets(manifests, required):
    assets = {}
    for path in manifests:
        manifest = json.loads(path.read_text())
        for row in manifest['results']:
            image_id = str(row['image_id'])
            if image_id not in required:
                continue
            if row['status'] != 'available':
                raise ValueError(f'Image unavailable: {image_id}')
            previous = assets.get(image_id)
            if previous is not None and previous['sha256'] != row['sha256']:
                raise ValueError(f'Conflicting pixels for image {image_id}')
            if json.loads(row['src_content'])['image_id'] != image_id:
                raise ValueError(f'Image source identity differs: {image_id}')
            assets[image_id] = dict(image_id=image_id, path=row['local_path'],
                sha256=row['sha256'], width=row['width'], height=row['height'],
                repository=manifest['repository'], revision=manifest['revision'],
                official_tsv_bytes_verified=row['official_tsv_bytes_verified'])
    if set(assets) != required:
        raise ValueError(f'Missing required images: {sorted(required - set(assets))}')
    return assets


def normalize_features(features):
    norms = features.float().norm(dim=-1, keepdim=True)
    if not torch.isfinite(features).all() or not (norms > 0).all():
        raise ValueError('CLIP produced non-finite or zero features')
    return features.float() / norms


@torch.inference_mode()
def run(args):
    rows = [row for path in args.inputs for row in load_inputs(path)]
    if len({row['qid'] for row in rows}) != len(rows):
        raise ValueError('Input files contain duplicate questions')
    required = {str(c['image_id']) for row in rows for c in row['candidates']
                if c['modality'] == 'image'}
    assets = image_assets(args.images_manifest, required)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config_path = args.output / 'config.json'
    if config_path.exists():
        saved_config = json.loads(config_path.read_text())
        args.threads, args.batch_size = saved_config['threads'], saved_config['batch_size']
    torch.set_num_threads(args.threads)
    torch.manual_seed(17)
    protocol = ROOT / 'VISUAL_RETRIEVAL_PROTOCOL.md'
    config = dict(started_utc=datetime.now(timezone.utc).isoformat(),
        model='openai/clip-vit-large-patch14', revision=REVISION,
        device='cpu', dtype='float32', attention='sdpa', batch_size=args.batch_size,
        threads=args.threads, questions=len(rows), unique_images=len(required),
        query='unmodified question; CLIP tokenizer truncation at 77 tokens',
        images='PIL RGB, official CLIP processor resize and center crop',
        score='cosine of L2-normalized text and actual-image embeddings; no logit-scale multiplication',
        inputs={str(p): sha256_file(p) for p in args.inputs},
        image_manifests={str(p): sha256_file(p) for p in args.images_manifest},
        model_manifest_sha256=sha256_file(MODEL / 'download_manifest.json'),
        code_sha256=sha256_file(__file__), protocol_sha256=sha256_file(protocol), label_access=False,
        scope='raw visual retrieval scores over frozen per-question candidates; no answer scoring or source selection')
    execution = args.output / 'executions' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    (execution / 'source').mkdir(parents=True)
    save_json(execution / 'config.json', config)
    (execution / 'source' / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (execution / 'source' / protocol.name).write_bytes(protocol.read_bytes())
    execution_path = str(execution.relative_to(args.output))
    if not config_path.exists():
        save_json(config_path, config)
    checkpoint = args.output / 'embeddings.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True) if checkpoint.exists() else {}
    if set(saved) - required:
        raise ValueError('Cached image embeddings are outside the supplied candidates')
    dimension = json.loads((MODEL / 'config.json').read_text())['projection_dim']
    for image_id, row in saved.items():
        if row['asset']['sha256'] != assets[image_id]['sha256']:
            raise ValueError(f'Cached embedding uses different pixels: {image_id}')
        vector = row['vector']
        if (vector.shape != (dimension,) or not torch.isfinite(vector).all()
                or abs(float(vector.norm()) - 1) > 1e-5):
            raise ValueError(f'Cached embedding is not a finite unit vector: {image_id}')

    load_started = time.perf_counter()
    processor = CLIPProcessor.from_pretrained(MODEL, local_files_only=True)
    model = CLIPModel.from_pretrained(MODEL, local_files_only=True,
        dtype=torch.float32, attn_implementation='sdpa').eval()
    load_seconds = time.perf_counter() - load_started
    pending = sorted(required - set(saved))
    for offset in range(0, len(pending), args.batch_size):
        image_ids = pending[offset:offset + args.batch_size]
        images = []
        for image_id in image_ids:
            asset = assets[image_id]
            path = Path(asset['path'])
            if sha256_file(path) != asset['sha256']:
                raise ValueError(f'Image content differs from manifest: {image_id}')
            with Image.open(path) as im:
                if im.size != (asset['width'], asset['height']):
                    raise ValueError(f'Image dimensions differ from manifest: {image_id}')
                images.append(im.convert('RGB'))
        tick = time.perf_counter()
        batch = processor(images=images, return_tensors='pt')
        features = normalize_features(model.get_image_features(**batch).pooler_output)
        seconds = time.perf_counter() - tick
        for image_id, feature in zip(image_ids, features, strict=True):
            saved[image_id] = dict(asset=assets[image_id], vector=feature.cpu(),
                batch_seconds=seconds, batch_image_ids=image_ids, execution=execution_path)
        if offset == 0 or len(saved) % 32 == 0 or offset + args.batch_size >= len(pending):
            temporary = checkpoint.with_suffix('.tmp')
            torch.save(saved, temporary)
            temporary.replace(checkpoint)
            print(json.dumps(dict(event='clip_images', complete=len(saved), total=len(required),
                                  elapsed_seconds=time.perf_counter() - started)), flush=True)

    records, queries = [], {}
    for offset in range(0, len(rows), args.batch_size):
        selected = rows[offset:offset + args.batch_size]
        questions = [row['question'] for row in selected]
        untruncated = processor.tokenizer(questions, truncation=False)['input_ids']
        batch = processor.tokenizer(questions, padding=True, truncation=True,
                                   max_length=77, return_tensors='pt')
        tick = time.perf_counter()
        vectors = normalize_features(model.get_text_features(**batch).pooler_output)
        query_seconds = time.perf_counter() - tick
        for row, vector, original_ids, input_ids, mask in zip(selected, vectors, untruncated,
                batch['input_ids'], batch['attention_mask'], strict=True):
            candidates = [c for c in row['candidates'] if c['modality'] == 'image']
            pixels = torch.stack([saved[str(c['image_id'])]['vector'] for c in candidates])
            cosines = (pixels @ vector).tolist()
            if any(not math.isfinite(value) or abs(value) > 1.000001 for value in cosines):
                raise ValueError(f'Invalid query-image cosine: {row["qid"]}')
            scored = [dict(source_id=f"image:{c['fact_id']}", image_id=str(c['image_id']),
                           cosine=cosine) for c, cosine in zip(candidates, cosines, strict=True)]
            ordered = sorted(scored, key=lambda c: (-c['cosine'], c['source_id']))
            records.append(dict(qid=row['qid'], source_row_sha256=row_hash(row),
                image_scores=scored, image_only_order=[c['source_id'] for c in ordered],
                original_query_tokens=len(original_ids), actual_query_tokens=int(mask.sum()),
                query_truncated=len(original_ids) > 77,
                query_batch_seconds=query_seconds, query_batch_ids=[q['qid'] for q in selected]))
            queries[row['qid']] = dict(vector=vector.cpu(), input_ids=input_ids[mask.bool()].tolist())
    write_jsonl(args.output / 'scores.jsonl', records)
    torch.save(queries, args.output / 'query_embeddings.pt')

    # Verify the saved cosine convention against CLIP's native scaled logits.
    first = rows[0]
    candidate = next(c for c in first['candidates'] if c['modality'] == 'image')
    with Image.open(assets[str(candidate['image_id'])]['path']) as im:
        batch = processor(text=[first['question']], images=[im.convert('RGB')],
                          padding=True, truncation=True, max_length=77, return_tensors='pt')
    direct = model(**batch).logits_per_text[0, 0] / model.logit_scale.exp()
    cached = saved[str(candidate['image_id'])]['vector'] @ queries[first['qid']]['vector']
    error = float(abs(direct - cached))
    if not all(math.isfinite(float(x)) for x in (direct, cached, error)) or error > 1e-5:
        raise ValueError(f'Cached CLIP cosine differs from native forward: {error}')
    summary = dict(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
        questions=len(rows), unique_images=len(saved),
        image_candidate_occurrences=sum(len(r['image_scores']) for r in records),
        truncated_questions=sum(r['query_truncated'] for r in records),
        native_forward_check=dict(qid=first['qid'], image_id=str(candidate['image_id']),
            native_cosine=float(direct), saved_cosine=float(cached), absolute_error=error),
        model_load_seconds=load_seconds, invocation_seconds=time.perf_counter() - started,
        execution=execution_path,
        scores_sha256=sha256_file(args.output / 'scores.jsonl'),
        image_embeddings_sha256=sha256_file(checkpoint),
        query_embeddings_sha256=sha256_file(args.output / 'query_embeddings.pt'),
        quality_scores_computed=False, source_selection_changed=False)
    save_json(args.output / 'completion.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--images-manifest', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=4)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2), flush=True)
