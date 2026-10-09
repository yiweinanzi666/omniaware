"""Try original public URLs for every ID absent from the pinned mirror; retain failures."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
from PIL import Image
from experiments.eab_first_token.prepare_okvqa_data import download

HERE=Path(__file__).resolve().parent/'global_image_assets'
OUT=HERE/'original_url_gaps'


def fetch(task):
    image_id,index,url=task
    suffix=Path(urlparse(url).path).suffix or '.image'
    target=OUT/f'{image_id}_{index}{suffix}'
    try:
        http=download(url,target)
        with Image.open(target) as image:
            image.load();rgb=image.convert('RGB')
            props=dict(width=rgb.width,height=rgb.height,rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
        return dict(image_id=image_id,status='available',path=str(target),**http,**props,
            identity='Original URL bound to this image ID in the official public fact pool; not original TSV byte identity')
    except Exception as error:
        return dict(image_id=image_id,status='failed',url=url,error_type=type(error).__name__,error=str(error))


def main():
    OUT.mkdir(exist_ok=True)
    source=HERE/'MIRROR_GAPS.json'
    rows=json.loads(source.read_text())
    tasks=[(r['image_id'],index,url) for r in rows for index,url in enumerate(r['image_urls'])]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(fetch,tasks))
    available={r['image_id'] for r in results if r['status']=='available'}
    receipt=dict(expected_ids=len(rows),available_ids=len(available),results=results,
        missing_ids=sorted({r['image_id'] for r in rows}-available),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        labels_read=False,model_calls=0,source_version_change='Original URL assets replace missing mirror assets only in the new protocol')
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='results'}),flush=True)


if __name__=='__main__':
    main()
