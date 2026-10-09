"""Fetch missing global-pool pixels from the pinned, ID-verified public mirror, without Q/A access."""
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import time
from experiments.eab_first_token.fetch_webqa_parquet_images import REPOSITORY, REVISION, Shard, checked_image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REUSE = ROOT/'cache/webqa_full4966_images'
OUT = HERE/'global_image_assets'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, data):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    temporary.replace(path)


def main(max_groups=None):
    OUT.mkdir(exist_ok=True)
    (OUT/'images').mkdir(exist_ok=True)
    with (OUT/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads((REUSE/'manifest.json').read_text())
        assert (previous['repository'],previous['revision']) == (REPOSITORY,REVISION)
        validation = json.loads((HERE/'LOCAL_IMAGE_VALIDATION.json').read_text())
        assert validation['receipt_sha256'] == sha(HERE/'verified_local_images.jsonl')
        verified = set()
        with (HERE/'verified_local_images.jsonl').open() as stream:
            for line in stream:
                row=json.loads(line)
                if row['verified']:
                    verified.add(row['image_id'])
        locations=json.loads((REUSE/'locations.json').read_text())
        entries=sorted([r for r in json.loads((REUSE/'catalog.json').read_text()) if r['path'].endswith('.parquet')],key=lambda r:r['path'])
        all_ids=set(str(i) for i in range(30000000,30389750))
        assert len(all_ids)==389750 and verified <= all_ids
        missing=sorted(all_ids-verified)
        unavailable=sorted(set(missing)-locations.keys())
        plan=dict(repository=REPOSITORY,revision=REVISION,global_ids=389750,already_verified=len(verified),
            requested=len(missing),not_in_pinned_mirror=unavailable,device='network/CPU',parallel_groups=1,
            inputs_sha256={str(p):sha(p) for p in (HERE/'verified_local_images.jsonl', REUSE/'locations.json', REUSE/'catalog.json')},
            source_sha256={str(p):sha(p) for p in (Path(__file__),ROOT/'fetch_webqa_parquet_images.py')},
            selection='Every missing ID in the global pool, independent of questions, answers and scores',
            original_tsv_identity=False, labels_read=False, model_calls=0)
        if (OUT/'plan.json').exists():
            assert json.loads((OUT/'plan.json').read_text())==plan
        else:
            save(OUT/'plan.json',plan)
        receipts=OUT/'images.jsonl'
        done={}
        if receipts.exists():
            with receipts.open() as stream:
                for line in stream:
                    r=json.loads(line)
                    assert r['image_id'] not in done and sha(r['path'])==r['sha256']
                    done[r['image_id']]=r
        groups=defaultdict(list)
        for image_id in missing:
            if image_id in locations and image_id not in done:
                loc=locations[image_id]
                groups[loc['shard'],loc['row_group']].append(image_id)
        shards={}; completed_groups=0; started=time.perf_counter()
        with receipts.open('a') as stream:
            for (index,group),image_ids in sorted(groups.items()):
                if max_groups is not None and completed_groups>=max_groups:
                    break
                if index not in shards:
                    shards[index]=Shard(entries[index],OUT,REUSE)
                rows=shards[index].read(group,['src_content','img_path','img'])
                for image_id in image_ids:
                    loc=locations[image_id]
                    data,properties=checked_image(rows[loc['row_in_group']],image_id)
                    suffix='.jpg' if properties['format']=='JPEG' else '.'+properties['format'].lower()
                    target=OUT/'images'/(image_id+suffix)
                    temporary=target.with_suffix(target.suffix+'.tmp')
                    temporary.write_bytes(data); temporary.replace(target)
                    r=dict(loc,**properties,image_id=image_id,path=str(target),shard_path=loc['path'],
                        bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),official_tsv_bytes_verified=False)
                    stream.write(json.dumps(r)+'\n');stream.flush();done[image_id]=r
                completed_groups+=1
                progress=dict(status='FETCHING',new_available=len(done),requested=len(missing),
                    already_verified=len(verified),global_verified=len(verified)+len(done),global_expected=389750,
                    completed_groups_this_session=completed_groups,remaining_groups=len(groups)-completed_groups,
                    seconds_this_session=time.perf_counter()-started,updated_utc=datetime.now(timezone.utc).isoformat())
                save(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)
        if completed_groups==len(groups):
            result=dict(status='PINNED_MIRROR_FETCH_COMPLETE_ORIGINAL_TSV_GAPS_REMAIN' if unavailable else 'ALL_GLOBAL_IMAGE_IDS_AVAILABLE',
                global_verified=len(verified)+len(done),global_expected=389750,not_in_pinned_mirror=unavailable,
                receipts_sha256=sha(receipts),plan_sha256=sha(OUT/'plan.json'),labels_read=False,model_calls=0,
                completed_utc=datetime.now(timezone.utc).isoformat())
            save(OUT/'completion.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-groups',type=int)
    main(parser.parse_args().max_groups)
