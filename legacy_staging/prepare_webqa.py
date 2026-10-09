"""Freeze WebQA train text questions and rank their official candidate facts.

This prepares a fixed-evidence ablation over official per-question candidates.
It is not 390k-corpus retrieval, multimodal WebQA, or the original system.
"""

import argparse
from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time


ROOT = Path(__file__).resolve().parent
SALT = 'webqa-eab-first-token-fixed-candidates-v1\n'
MODEL_ID = 'BAAI/bge-large-en-v1.5'
MODEL_REVISION = 'd4aa6901d3a41ba39fb536a557fa166f842b0e09'
MODEL_FILES = ['README.md', 'config.json', 'model.safetensors',
               'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt']
QUERY_INSTRUCTION = 'Represent this sentence for searching relevant passages: '
DATA_SOURCE_COMMIT = '81cc377aaddb2cf9de98cf082a2795ce53ccc1f4'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def merge_candidates(row):
    """Discard positive/negative labels, deduplicate IDs, and sort by ID."""
    by_id = {}
    for fact in row['txt_posFacts'] + row['txt_negFacts']:
        raw_id = fact.get('snippet_id', fact.get('fact_id'))
        if raw_id is None:
            raise ValueError('text fact is missing its source snippet_id/fact_id')
        fact_id = str(raw_id)
        text = fact['fact']
        if not fact_id or not isinstance(text, str) or not text.strip():
            raise ValueError('text candidates must have a nonempty ID and text')
        if fact_id in by_id and by_id[fact_id] != text:
            raise ValueError(f'conflicting text for fact ID {fact_id}')
        by_id[fact_id] = text
    return [dict(fact_id=fact_id, text=by_id[fact_id]) for fact_id in sorted(by_id)]


def freeze_examples(source, calibration_count=500, development_count=32):
    """Choose train/text IDs by salted hash without inspecting answer values."""
    eligible = {}
    for key, row in source.items():
        if row['split'] != 'train' or row['Qcate'] != 'text':
            continue
        # Eligibility is based only on question/candidate validity, never A.
        question = row['Q']
        if not isinstance(question, str) or not question.strip():
            continue
        candidates = merge_candidates(row)
        if candidates:
            eligible[str(key)] = (row, candidates)
    ordered = sorted(eligible, key=lambda qid: (hashlib.sha256((SALT + qid).encode()).hexdigest(), qid))
    total = calibration_count + development_count
    if len(ordered) < total:
        raise ValueError(f'need {total} valid train/text questions; found {len(ordered)}')
    frozen, global_facts = [], {}
    for index, qid in enumerate(ordered[:total]):
        row, candidates = eligible[qid]
        for candidate in candidates:
            fact_id, text = candidate['fact_id'], candidate['text']
            if fact_id in global_facts and global_facts[fact_id] != text:
                raise ValueError(f'conflicting text across questions for fact ID {fact_id}')
            global_facts[fact_id] = text
        references = row['A']
        frozen.append(dict(qid=qid, question=row['Q'], references=references,
                           category=row['Qcate'], source_split=row['split'],
                           subset='calibration' if index < calibration_count else 'development',
                           candidates=candidates))
    return frozen


def rank_candidates(candidates, scores, top_k=5):
    scored = [dict(candidate, similarity=float(score)) for candidate, score in zip(candidates, scores, strict=True)]
    return sorted(scored, key=lambda row: (-row['similarity'], row['fact_id']))[:top_k]


def ecdf_relevance(scores, calibration_scores):
    """Right-continuous global ECDF fitted exclusively to calibration scores."""
    reference = sorted(float(score) for score in calibration_scores)
    return [bisect_right(reference, float(score)) / len(reference) for score in scores]


def write_jsonl(path, rows):
    content = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(content, encoding='utf-8')
    temporary.replace(path)


def encode_texts(texts, tokenizer, model, device, batch_size, label):
    import torch
    chunks = []
    start_time = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                              max_length=512, return_tensors='pt').to(device)
            cls = model(**batch).last_hidden_state[:, 0]
            chunks.append(torch.nn.functional.normalize(cls.float(), p=2, dim=1).cpu())
            if start == 0 or (start // batch_size + 1) % 10 == 0 or start + batch_size >= len(texts):
                print(f'{label}: {min(start + batch_size, len(texts))}/{len(texts)} '
                      f'elapsed={time.monotonic() - start_time:.1f}s', flush=True)
    return torch.cat(chunks)


def prepare(args):
    source_path, output = Path(args.data).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads(source_path.read_text(encoding='utf-8'))
    rows = freeze_examples(source, args.calibration_count, args.development_count)
    selection_path = output / 'frozen_selection.jsonl'
    write_jsonl(selection_path, rows)
    manifest = dict(
        schema_version=1, status='SELECTION_FROZEN', created_utc=datetime.now(timezone.utc).isoformat(),
        scope='WebQA official per-question text candidate-set fixed-evidence ablation',
        exclusions=['not 390k-corpus retrieval', 'not multimodal WebQA', 'not original-system reproduction'],
        source=dict(path=str(source_path), sha256=sha256_file(source_path),
                    repository='https://github.com/WebQnA/WebQA', revision=DATA_SOURCE_COMMIT,
                    official_archive_drive_id='19f1LGo7SMOa1UOZrkHME9t94zdKYQNZ5'),
        selection=dict(salt=SALT, ranking='ascending sha256(salt + qid), then qid',
                       eligibility='split=train; Qcate=text; nonempty Q and valid text candidates; no answer filtering',
                       calibration_count=args.calibration_count, development_count=args.development_count,
                       sha256=sha256_file(selection_path), candidates='positive and negative pooled, deduplicated by source ID; no gold ordering'),
        retrieval=dict(model_id=MODEL_ID, revision=MODEL_REVISION, query_instruction=QUERY_INSTRUCTION,
                       pooling='last_hidden_state[:,0] (CLS)', normalization='L2', similarity='cosine',
                       embedding_max_tokens=512, top_k=5, tie_break='ascending source fact ID',
                       max_chunk_tokens=args.max_chunk_tokens,
                       chunk_limit_use='runner generation-token budget only; retrieval embeds full source text truncated at 512 BGE tokens'),
        relevance=dict(method='right-continuous global empirical CDF', fit_subset='calibration',
                       fit_values='all per-question candidate cosine similarities; no development values or answer labels'),
        implementation=dict(path=str(Path(__file__).resolve()), sha256=sha256_file(__file__),
                            python=platform.python_version()),
    )
    manifest_path = output / 'manifest.json'
    if args.freeze_only:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps({'status': 'SELECTION_FROZEN', 'counts': dict(Counter(row['subset'] for row in rows)),
                          'selection_sha256': manifest['selection']['sha256']}, indent=2), flush=True)
        return

    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer
    model_dir = Path(args.model_dir).resolve()
    snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=model_dir,
                      allow_patterns=MODEL_FILES, max_workers=2)
    model_hashes = {filename: sha256_file(model_dir / filename) for filename in MODEL_FILES}
    device = args.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True, dtype=torch.float32).to(device).eval()
    unique_facts = {candidate['fact_id']: candidate['text'] for row in rows for candidate in row['candidates']}
    fact_ids = sorted(unique_facts)
    fact_index = {fact_id: index for index, fact_id in enumerate(fact_ids)}
    query_vectors = encode_texts([QUERY_INSTRUCTION + row['question'] for row in rows],
                                tokenizer, model, device, args.batch_size, 'queries')
    fact_vectors = encode_texts([unique_facts[fact_id] for fact_id in fact_ids],
                               tokenizer, model, device, args.batch_size, 'facts')
    calibration_scores = []
    for index, row in enumerate(rows):
        indices = [fact_index[candidate['fact_id']] for candidate in row['candidates']]
        scores = (fact_vectors[indices] @ query_vectors[index]).tolist()
        for candidate, score in zip(row['candidates'], scores):
            candidate['similarity'] = score
        row['evidence'] = rank_candidates(row['candidates'], scores)
        if row['subset'] == 'calibration':
            calibration_scores.extend(scores)
    calibration_scores.sort()
    for row in rows:
        relevance = ecdf_relevance([evidence['similarity'] for evidence in row['evidence']], calibration_scores)
        for evidence, value in zip(row['evidence'], relevance):
            evidence['relevance'] = value
    outputs = {}
    for subset in ('calibration', 'development'):
        path = output / f'{subset}.jsonl'
        subset_rows = [row for row in rows if row['subset'] == subset]
        write_jsonl(path, subset_rows)
        outputs[path.name] = dict(sha256=sha256_file(path), rows=len(subset_rows))
    ecdf_path = output / 'calibration_similarity_ecdf.json'
    ecdf_path.write_text(json.dumps(calibration_scores) + '\n')
    manifest.update(status='PREPARED', outputs=outputs,
                    counts=dict(queries=len(rows), unique_fact_ids=len(fact_ids),
                                calibration_candidate_scores=len(calibration_scores)))
    manifest['retrieval'].update(model_files_sha256=model_hashes, device=device, dtype='float32',
                                 batch_size=args.batch_size)
    manifest['relevance'].update(fitted_distribution_sha256=sha256_file(ecdf_path),
                                 fitted_distribution_file=ecdf_path.name)
    manifest['implementation'].update(torch=torch.__version__, transformers=transformers.__version__)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'status': 'PREPARED', 'output': str(output), 'counts': manifest['counts'],
                      'outputs': outputs}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Official WebQA_train_val.json')
    parser.add_argument('--output', default=str(ROOT / 'data' / 'prepared'))
    parser.add_argument('--model-dir', default=str(ROOT / 'cache' / 'bge-large-en-v1.5'))
    parser.add_argument('--calibration-count', type=int, default=500)
    parser.add_argument('--development-count', type=int, default=32)
    parser.add_argument('--max-chunk-tokens', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'))
    parser.add_argument('--freeze-only', action='store_true', help='Freeze selection without loading or running BGE')
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_chunk_tokens < 1:
        parser.error('batch-size and max-chunk-tokens must be positive')
    prepare(args)


if __name__ == '__main__':
    main()
