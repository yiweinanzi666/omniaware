"""Build exact cosine FAISS indices from verified, completed BGE/CLIP shards (CPU only).

Use runtime_dense_20261006/bin/python. --pilot-shards N is an explicitly partial
engineering check; omit it to require complete encoding of every supplied input.
An encoded CLIP snapshot can still be an incomplete global image pool.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import itertools
import json
from pathlib import Path
import time

from .encode_global_text import CORPUS_SHA, HERE, sha, save


def read(path):
    return json.loads(Path(path).read_text())


def inspect_inputs(kind, encodings, pilot_shards=None):
    """Bind each shard to its exact source rows, not just to the vector file name."""
    inputs, chunks, all_ids, first_assets = [], [], [], None
    assert pilot_shards is None or pilot_shards > 0
    assert kind in ('bge', 'clip') and encodings
    if kind == 'bge':
        assert len(encodings) == 1, 'BGE uses the single canonical exact-text corpus'
    for directory in map(Path, encodings):
        plan_path = directory/'plan.json'
        plan, plan_sha = read(plan_path), sha(plan_path)
        expected = plan['documents'] if kind == 'bge' else plan['images']
        asset_identity = plan['model_files_sha256']
        if first_assets is None:
            first_assets = asset_identity
        assert asset_identity == first_assets, 'Encoding waves use different model assets'
        assert plan['device'] == 'cpu' and plan['dtype'] == 'float32'
        completion_path = directory/'completion.json'
        complete = None
        if pilot_shards is None:
            complete = read(completion_path)
            assert complete['plan_sha256'] == plan_sha
            assert complete['completed'] == complete['expected'] == expected
            assert complete['status'] == ('ENCODING_COMPLETE' if kind == 'bge' else 'SNAPSHOT_ENCODING_COMPLETE')
        if kind == 'bge':
            source = Path(plan['corpus'])
            assert sha(source) == plan['corpus_sha256'] == CORPUS_SHA
            source_sha = plan['corpus_sha256']
        else:
            source = directory/'assets.jsonl'
            manifest = read(directory/'manifest.json')
            assert sha(source) == plan['assets_sha256'] == manifest['assets_sha256']
            assert manifest['images'] == expected
            source_sha = manifest['assets_sha256']
        inputs.append(dict(directory=str(directory.resolve()), plan_sha256=plan_sha,
                           source_sha256=source_sha, expected=expected,
                           completion_sha256=sha(completion_path) if complete else None,
                           manifest_sha256=sha(directory/'manifest.json') if kind == 'clip' else None))
        start = count = 0
        with source.open() as stream:
            while lines := list(itertools.islice(stream, plan['shard_size'])):
                if pilot_shards is not None and count >= pilot_shards:
                    break
                rows = [json.loads(line) for line in lines]
                if kind == 'bge':
                    ids = [r['document_id'] for r in rows]
                    assert [r['row_index'] for r in rows] == list(range(start, start+len(rows)))
                    assert all(r['document_id'] == 'text-sha256:'+hashlib.sha256(r['text'].encode()).hexdigest() for r in rows)
                    identity = dict(start=start, stop=start+len(rows), plan_sha256=plan_sha,
                                    document_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest())
                else:
                    ids = ['image:'+r['image_id'] for r in rows]
                    identity = dict(start=start, stop=start+len(rows), plan_sha256=plan_sha,
                                    assets_rows_sha256=hashlib.sha256(''.join(lines).encode()).hexdigest())
                array_path = directory/f'{start:06d}.npz'
                receipt_path = array_path.with_suffix('.json')
                receipt = read(receipt_path)
                assert receipt['identity'] == identity, f'Wrong source rows: {receipt_path}'
                assert receipt['array_sha256'] == sha(array_path), f'Changed vectors: {array_path}'
                chunks.append(dict(path=str(array_path.resolve()), sha256=receipt['array_sha256'],
                                   receipt_sha256=sha(receipt_path), ids=ids))
                all_ids.extend(ids)
                start += len(rows)
                count += 1
        assert start > 0
        if pilot_shards is None:
            assert start == expected
        else:
            assert count == pilot_shards, 'Not enough source shards for requested engineering pilot'
    assert len(all_ids) == len(set(all_ids)), 'Duplicate document/image ID across encoding inputs'
    if kind == 'clip':
        assert all(i.startswith('image:') and i[6:].isdigit() and 30000000 <= int(i[6:]) <= 30389749 for i in all_ids)
    return inputs, chunks, all_ids, first_assets


def load_vectors(chunk, kind):
    import numpy as np
    path = Path(chunk['path'])
    assert sha(path) == chunk['sha256'], f'Vectors changed after input inspection: {path}'
    with np.load(path, allow_pickle=False) as archive:
        vectors = archive['vectors']
        dimension = 1024 if kind == 'bge' else 768
        assert vectors.dtype == np.float32 and vectors.shape == (len(chunk['ids']), dimension)
        assert np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors, axis=1), 1., atol=2e-6, rtol=0)
        if kind == 'clip':
            assert ['image:'+str(i) for i in archive['image_ids']] == chunk['ids'], 'CLIP vector/ID order differs from assets'
    return np.ascontiguousarray(vectors)


def build(kind, encodings, output, pilot_shards=None):
    import faiss
    import numpy as np
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inputs, chunks, ids, assets = inspect_inputs(kind, encodings, pilot_shards)
        dimension = 1024 if kind == 'bge' else 768
        count_expected = 544489 if kind == 'bge' else 389750
        plan = dict(kind=kind, dimension=dimension, metric='cosine as inner product of unit vectors',
                    index_type='IndexFlatIP', faiss=faiss.__version__, numpy=np.__version__, threads=2,
                    vectors=len(ids), inputs=inputs, model_files_sha256=assets,
                    ordered_ids_sha256=hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest(),
                    shards=[{k: v for k, v in c.items() if k != 'ids'} for c in chunks],
                    source_sha256={str(p): sha(p) for p in (Path(__file__), HERE/'encode_global_text.py')},
                    pilot_shards_per_input=pilot_shards, full_encoded_inputs=pilot_shards is None,
                    global_pool_complete=pilot_shards is None and len(ids) == count_expected,
                    scope='Engineering prefix only' if pilot_shards is not None else 'Complete supplied encoding snapshots',
                    labels_read=False, MPS_calls=0)
        if (output/'plan.json').exists():
            assert read(output/'plan.json') == plan, 'Index input/configuration changed'
        else:
            save(output/'plan.json', plan)
        plan_sha = sha(output/'plan.json')
        if (output/'completion.json').exists():
            result = read(output/'completion.json')
            assert result['plan_sha256'] == plan_sha
            assert result['index_sha256'] == sha(output/'index.faiss')
            assert result['ids_sha256'] == sha(output/'ids.txt') == plan['ordered_ids_sha256']
            assert result['vectors'] == len(ids) and result['global_pool_complete'] == plan['global_pool_complete']
            return result
        faiss.omp_set_num_threads(2)
        index = faiss.IndexFlatIP(dimension)
        for chunk in chunks:
            index.add(load_vectors(chunk, kind))
        assert index.ntotal == len(ids)
        temporary = output/'index.faiss.tmp'
        faiss.write_index(index, str(temporary))
        restored = faiss.read_index(str(temporary))
        assert restored.ntotal == len(ids) and restored.d == dimension and restored.metric_type == faiss.METRIC_INNER_PRODUCT
        # Serialized index must preserve every vector and its exact row position.
        offset = 0
        for chunk in chunks:
            expected = load_vectors(chunk, kind)
            assert np.array_equal(restored.reconstruct_n(offset, len(expected)), expected)
            offset += len(expected)
        temporary.replace(output/'index.faiss')
        ids_path = output/'ids.txt'
        ids_path.write_text('\n'.join(ids)+'\n')
        assert sha(ids_path) == plan['ordered_ids_sha256']
        result = dict(status='INDEX_COMPLETE', plan_sha256=plan_sha, index_sha256=sha(output/'index.faiss'),
                      ids_sha256=sha(ids_path), vectors=len(ids), dimension=dimension,
                      full_encoded_inputs=plan['full_encoded_inputs'], global_pool_complete=plan['global_pool_complete'],
                      scope=plan['scope'], wall_seconds=time.perf_counter()-started,
                      completed_utc=datetime.now(timezone.utc).isoformat(),
                      validation='Every serialized vector reconstructed and byte-equal in source-row order',
                      labels_read=False, MPS_calls=0)
        save(output/'completion.json', result)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('bge', 'clip'), required=True)
    parser.add_argument('--encodings', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot-shards', type=int)
    args = parser.parse_args()
    print(json.dumps(build(args.kind, args.encodings, args.output, args.pilot_shards)), flush=True)
