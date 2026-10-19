"""Validate real prefix indices against brute-force cosine; no answers or scoring."""
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

from . import build_dense_index as B


def rejection(operation, exception=AssertionError):
    try:
        operation()
    except exception:
        return True
    raise AssertionError('Invalid input was accepted')


def main():
    import faiss
    import numpy as np
    faiss.omp_set_num_threads(2)
    results = {}
    for kind, directory in (('bge', 'bge_text_cpu'), ('clip', 'clip_images_cpu_wave0')):
        source = B.HERE/directory
        output = B.HERE/f'faiss_{kind}_engineering1024'
        complete = B.build(kind, [source], output, 8)
        assert complete['vectors'] == 1024 and not complete['global_pool_complete']
        index = faiss.read_index(str(output/'index.faiss'))
        query_dir = B.HERE/'webqa_train256_queries_cpu'
        query_completion = B.read(query_dir/'completion.json')
        query_receipt = B.read(query_dir/f'{kind}.json')
        query_plan = B.read(query_dir/'plan.json')
        assert query_completion['plan_sha256'] == B.sha(query_dir/'plan.json')
        assert query_completion['encoders'][kind]['receipt_sha256'] == B.sha(query_dir/f'{kind}.json')
        assert query_receipt['array_sha256'] == query_completion['encoders'][kind]['array_sha256'] == B.sha(query_dir/f'{kind}.npz')
        index_plan = B.read(output/'plan.json')
        assert query_plan['model_files_sha256'][kind] == index_plan['model_files_sha256']
        with np.load(query_dir/f'{kind}.npz') as archive:
            queries = archive['vectors']
        ids = (output/'ids.txt').read_text().splitlines()
        inputs, chunks, source_ids, _ = B.inspect_inputs(kind, [source], 8)
        assert ids == source_ids
        # Brute force uses the original encoding shards, not FAISS-reconstructed vectors.
        original = np.concatenate([B.load_vectors(c, kind) for c in chunks])
        tick = time.perf_counter()
        distances, neighbors = index.search(queries, 5)
        search_seconds = time.perf_counter()-tick
        brute = queries @ original.T
        selected = np.take_along_axis(brute, neighbors, axis=1)
        error = float(np.max(np.abs(selected-distances)))
        assert error < 2e-6 and np.all((neighbors >= 0) & (neighbors < len(ids)))
        assert all(len(set(row)) == 5 for row in neighbors)
        cutoffs = np.partition(brute, -5, axis=1)[:, -5: -4]
        assert np.all(selected >= cutoffs-2e-6)
        expected = np.argsort(-brute, axis=1, kind='stable')[:, :5]
        identical_sets = sum(set(a) == set(b) for a, b in zip(neighbors, expected, strict=True))
        before = {p.name: B.sha(p) for p in output.iterdir() if p.name != 'worker.lock'}
        with patch.object(faiss, 'IndexFlatIP', side_effect=AssertionError('completed index rebuilt')):
            resumed = B.build(kind, [source], output, 8)
        assert resumed == complete
        assert before == {p.name: B.sha(p) for p in output.iterdir() if p.name != 'worker.lock'}
        results[kind] = dict(vectors=1024, queries=256, top_k=5, max_cosine_error=error,
                             identical_top5_sets=identical_sets, cutoff_check_pass=True, search_seconds=search_seconds,
                             index_completion_sha256=B.sha(output/'completion.json'),
                             query_array_sha256=B.sha(query_dir/f'{kind}.npz'),
                             reentry_index_builds=0, reentry_bytes_unchanged=True)
    # Corrupt only isolated fixtures, never active encoder outputs.
    invalid = {}
    clip_source = B.HERE/'clip_images_cpu_wave0'
    _, chunks, _, _ = B.inspect_inputs('clip', [clip_source], 1)
    chunk = chunks[0]
    with tempfile.TemporaryDirectory() as folder:
        temp = Path(folder)
        for name in ('plan.json', 'manifest.json', 'assets.jsonl', '000000.npz'):
            (temp/name).symlink_to((clip_source/name).resolve())
        (temp/'000000.json').write_bytes((clip_source/'000000.json').read_bytes())
        invalid['incomplete_encoding_rejected_for_complete_index'] = rejection(
            lambda: B.inspect_inputs('clip', [temp]), FileNotFoundError)
        receipt = B.read(temp/'000000.json')
        receipt['identity']['stop'] += 1
        B.save(temp/'000000.json', receipt)
        invalid['wrong_source_row_receipt_rejected'] = rejection(lambda: B.inspect_inputs('clip', [temp], 1))
        with np.load(chunk['path']) as data:
            vectors, image_ids = data['vectors'], data['image_ids']
        path = temp/'bad_ids.npz'
        np.savez(path, vectors=vectors, image_ids=image_ids[::-1])
        bad = dict(chunk, path=str(path), sha256=B.sha(path))
        invalid['vector_id_reordering_rejected'] = rejection(lambda: B.load_vectors(bad, 'clip'))
        path = temp/'bad_vectors.npz'
        vectors[0, 0] = np.nan
        np.savez(path, vectors=vectors, image_ids=image_ids)
        bad = dict(chunk, path=str(path), sha256=B.sha(path))
        invalid['nonfinite_vector_rejected'] = rejection(lambda: B.load_vectors(bad, 'clip'))
    invalid['duplicate_wave_ids_rejected'] = rejection(lambda: B.inspect_inputs('clip', [clip_source, clip_source], 1))
    record = dict(status='PASS', real_prefix_checks=results, invalid_input_checks=invalid,
                  source_sha256={str(p): B.sha(p) for p in (Path(__file__), Path(B.__file__))},
                  labels_read=False, model_calls=0, MPS_calls=0, full_global_index_built=False,
                  scope='Engineering validation on1024 documents per modality; not benchmark retrieval/accuracy')
    B.save(B.HERE/'FAISS_REAL_PREFIX_VALIDATION.json', record)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
