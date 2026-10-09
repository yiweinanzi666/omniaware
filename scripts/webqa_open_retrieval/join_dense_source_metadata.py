"""Join frozen top5 IDs to reader text and the exact indexed image assets, without U."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import time

from . import calibrate_dense_relevance as C
from .encode_webqa_queries import COHORT, COHORT_SHA

PROTOCOL = C.HERE/'SOURCE_METADATA_PROTOCOL.md'


def captions(values):
    if any(not isinstance(v, str) for v in values):
        raise ValueError('Image captions must be strings')
    result = sorted({' '.join(v.split()) for v in values} - {''})
    if not result:
        raise ValueError('Selected image has no caption in the public source metadata')
    return result


def write_rows(path, rows):
    payload = ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in rows)
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f'Preserve changed source bundle: {path}')
    else:
        path.write_text(payload)


def join(selection_directory, output, engineering_prefix=False):
    started = time.perf_counter()
    selection_directory, output = Path(selection_directory), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selection = C.read(selection_directory/'plan.json')
        C.merge(selection['calibration_directory'], selection['retrievals']['bge']['directory'],
                selection['retrievals']['clip']['directory'], selection_directory, engineering_prefix)
        selected = [json.loads(line) for line in (selection_directory/'selected.jsonl').read_text().splitlines()]
        indices = {kind: C.read(Path(C.read(Path(selection['retrievals'][kind]['directory'])/'plan.json')
                                ['index_directory'])/'plan.json') for kind in C.KINDS}
        if not engineering_prefix and not all(p['global_pool_complete'] for p in indices.values()):
            raise ValueError('Formal source bundle requires complete text and all389750 image IDs')
        assert C.sha(COHORT) == COHORT_SHA
        cohort = [json.loads(line) for line in COHORT.read_text().splitlines()]
        assert [r['qid'] for r in cohort] == [r['qid'] for r in selected]
        pool = C.read(C.HERE/'GLOBAL_POOL_MANIFEST.json')
        corpus = {kind: C.HERE/name for kind, name in (('bge', 'text_documents.jsonl'), ('clip', 'image_corpus.jsonl'))}
        corpus_hashes = {kind: C.sha(path) for kind, path in corpus.items()}
        assert all(corpus_hashes[k] == pool['files'][p.name]['sha256'] for k, p in corpus.items())
        assert indices['bge']['inputs'][0]['source_sha256'] == corpus_hashes['bge']
        assets = {}
        asset_manifests = {}
        wanted = {c['source_id'] for r in selected for c in r['selected_sources']}
        for encoding in indices['clip']['inputs']:
            path = Path(encoding['directory'])/'assets.jsonl'
            asset_manifests[str(path)] = C.sha(path)
            assert asset_manifests[str(path)] == encoding['source_sha256']
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    source_id = 'image:'+row['image_id']
                    if source_id in wanted:
                        if source_id in assets:
                            raise ValueError('Duplicate image asset across encoding snapshots')
                        assert C.sha(row['path']) == row['sha256']
                        assets[source_id] = dict(row, encoding_manifest=str(path))
        plan = dict(rule='webqa-global-source-metadata-union-v1', engineering_prefix=engineering_prefix,
            selection_plan_sha256=C.sha(selection_directory/'plan.json'),
            selection_completion_sha256=C.sha(selection_directory/'completion.json'),
            selected_sha256=C.sha(selection_directory/'selected.jsonl'), cohort_sha256=COHORT_SHA,
            public_pool_manifest_sha256=C.sha(C.HERE/'GLOBAL_POOL_MANIFEST.json'),
            corpus_sha256=corpus_hashes, asset_manifests_sha256=asset_manifests,
            source_sha256={str(p): C.sha(p) for p in (Path(__file__), PROTOCOL, Path(C.__file__))},
            questions=len(selected), initial_U_included=False, executable_phase1_manifest=False,
            labels_read=False, model_calls=0, MPS_calls=0)
        C.freeze(output/'plan.json', plan)
        if (output/'completion.json').exists():
            complete = C.read(output/'completion.json')
            assert complete['plan_sha256'] == C.sha(output/'plan.json')
            assert all(C.sha(output/name) == sha for name, sha in complete['artifacts_sha256'].items())
            return dict(complete, new_join=False)
        metadata, provenance = {}, {}
        with corpus['bge'].open() as stream:
            for line in stream:
                row = json.loads(line)
                identifier = row['document_id']
                if identifier not in wanted:
                    continue
                assert identifier == 'text-sha256:'+hashlib.sha256(row['text'].encode()).hexdigest()
                assert identifier not in metadata
                metadata[identifier] = dict(source_id=identifier, fact_id=identifier, modality='text', text=row['text'])
                provenance[identifier] = dict(source_id=identifier, corpus_row=row['row_index'],
                    canonical_source_snippet_id=row['canonical_source_snippet_id'], source_snippet_ids=row['source_snippet_ids'])
        caption_histogram, caption_characters = Counter(), Counter()
        raw_multi = 0
        with corpus['clip'].open() as stream:
            for line in stream:
                row = json.loads(line)
                variants = captions(row['captions'])
                caption_histogram[len(variants)] += 1
                caption_characters[len('\n'.join(variants))] += 1
                raw_multi += len(row['captions']) > 1
                identifier = row['fact_id']
                if identifier not in wanted:
                    continue
                assert identifier == 'image:'+row['image_id'] and identifier not in metadata
                asset = assets[identifier]
                metadata[identifier] = dict(source_id=identifier, fact_id=row['image_id'], modality='image',
                    text='\n'.join(variants), image_path=asset['path'], image_sha256=asset['sha256'])
                provenance[identifier] = dict(source_id=identifier, caption_variants=variants,
                    raw_caption_count=len(row['captions']), image_asset=asset)
        if metadata.keys() != wanted:
            raise ValueError('Selected IDs lack public metadata or matching pixel assets')
        rows = []
        for query, retrieval in zip(cohort, selected, strict=True):
            sources = [dict(metadata[c['source_id']], rank=c['rank'], r_j=c['r_j'])
                       for c in retrieval['selected_sources']]
            assert len(sources) == 5 and [s['rank'] for s in sources] == [1, 2, 3, 4, 5]
            rows.append(dict(qid=query['qid'], question=query['question'], sources=sources))
        write_rows(output/'sources.jsonl', rows)
        write_rows(output/'provenance.jsonl', [provenance[k] for k in sorted(provenance)])
        complete = dict(status='SOURCE_METADATA_JOIN_COMPLETE_U_PENDING', plan_sha256=C.sha(output/'plan.json'),
            artifacts_sha256={name: C.sha(output/name) for name in ('sources.jsonl', 'provenance.jsonl')},
            questions=len(rows), selected_sources=sum(len(r['sources']) for r in rows), unique_sources=len(metadata),
            unique_selected_images=len(assets), engineering_prefix=engineering_prefix,
            global_caption_audit=dict(image_ids=sum(caption_histogram.values()), raw_multiple_ids=raw_multi,
                normalized_multiple_ids=sum(n for k, n in caption_histogram.items() if k > 1),
                normalized_variants=sum(k*n for k, n in caption_histogram.items()),
                maximum_variants=max(caption_histogram), maximum_joined_characters=max(caption_characters),
                count_histogram=dict(sorted(caption_histogram.items()))),
            wall_seconds=time.perf_counter()-started, completed_utc=datetime.now(timezone.utc).isoformat(),
            initial_U_included=False, executable_phase1_manifest=False, labels_read=False, model_calls=0, MPS_calls=0,
            scope='Source packing only. Same assets/captions for all arms. No U placeholders, source-utility or QA result.')
        C.save(output/'completion.json', complete)
        return dict(complete, new_join=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--engineering-prefix', action='store_true')
    args = parser.parse_args()
    print(json.dumps(join(args.selection, args.output, args.engineering_prefix)), flush=True)
