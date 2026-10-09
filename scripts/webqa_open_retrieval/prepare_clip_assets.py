"""Freeze currently verified pixels for incremental global CLIP preparation, without selecting questions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .encode_global_text import HERE, sha, save


def snapshot_rows(path):
    # The downloader appends complete JSON lines. Freeze only the complete prefix read at entry.
    with path.open('rb') as stream:
        data = stream.read(path.stat().st_size)
    data = data[:data.rfind(b'\n') + 1]
    return [json.loads(line) for line in data.splitlines()], dict(
        path=str(path), prefix_bytes=len(data), prefix_sha256=hashlib.sha256(data).hexdigest())


def prepare(output, previous=()):
    output = Path(output)
    assert not output.exists(), 'Snapshot already exists; use the saved manifest for resume'
    validation = json.loads((HERE/'LOCAL_IMAGE_VALIDATION.json').read_text())
    assert validation['receipt_sha256'] == sha(HERE/'verified_local_images.jsonl')
    assets, sources = {}, []
    local, source = snapshot_rows(HERE/'verified_local_images.jsonl')
    sources.append(source)
    for row in local:
        if row['verified']:
            asset = row['verified'][0]
            assets[row['image_id']] = dict(image_id=row['image_id'], path=asset['path'], sha256=asset['sha256'],
                width=asset['width'], height=asset['height'], origin='previously_verified_local',
                source_manifest=asset['manifest'])
    fetched, source = snapshot_rows(HERE/'global_image_assets/images.jsonl')
    sources.append(source)
    for row in fetched:
        assert row['image_id'] not in assets, 'Downloader and verified-local identities overlap'
        assets[row['image_id']] = {k: row[k] for k in ('image_id', 'path', 'sha256', 'width', 'height')}
        assets[row['image_id']]['origin'] = 'pinned_public_mirror'
    repair = HERE/'global_image_assets/tsv_gaps/metadata_repaired/receipt.json'
    for row in json.loads(repair.read_text())['results']:
        assert row['strict_decode'] and row['image_data_chunks_unchanged'] and row['image_id'] not in assets
        assets[row['image_id']] = {k: row[k] for k in ('image_id', 'path', 'sha256', 'width', 'height')}
        assets[row['image_id']]['origin'] = 'TSV_image_chunks_unchanged_metadata_repair'
    sources.append(dict(path=str(repair), sha256=sha(repair)))
    prior = set()
    for path in previous:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            assert row['image_id'] not in prior, 'Earlier snapshot manifests overlap'
            prior.add(row['image_id'])
            if row['image_id'] in assets:
                assert assets[row['image_id']]['sha256'] == row['sha256'], 'Previously prepared pixels changed'
    all_ids = {str(i) for i in range(30000000, 30389750)}
    assert assets.keys() <= all_ids and prior <= all_ids
    rows = [assets[i] for i in sorted(assets.keys() - prior)]
    assert rows, 'No new verified pixels'
    output.mkdir(parents=True)
    manifest = output/'assets.jsonl'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    result = dict(status='PARTIAL_GLOBAL_ASSET_SNAPSHOT', created_utc=datetime.now(timezone.utc).isoformat(),
        images=len(rows), previously_prepared=len(prior), global_ids=len(all_ids),
        not_in_this_or_previous_snapshot=len(all_ids - (assets.keys() | prior)),
        assets_sha256=sha(manifest), source_prefixes=sources,
        previous_manifests_sha256={str(p): sha(p) for p in previous},
        source_sha256=sha(__file__), labels_read=False, model_calls=0,
        rule='Ascending image ID, every verified available asset; future snapshots add missing IDs without duplication',
        protocol='New global-public-pool preparation; image availability is not question selection or historical protocol recovery')
    save(output/'manifest.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.previous), indent=2))
