"""Prepare 32 midpoint frames per video using only source media, never labels."""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import time

import av
from PIL import Image

ASSETS = Path(os.environ.get('OMNIAWARE_ASSETS') or os.environ['ERC_ASSET_ROOT']).resolve()
DATA = ASSETS / 'data'
OUTPUT = DATA / 'videomme_short_frames32'   # use with: omniaware.run --video-frames-dir <OUTPUT>


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def sample(video):
    source = DATA / 'videomme_short/videos' / (video + '.mp4')
    original = json.loads((DATA / 'videomme_short_frames' / video / 'frames.json').read_text())
    assert digest(source) == original['source_sha256']
    folder = OUTPUT / video
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / 'frames.json'
    if manifest.exists():
        record = json.loads(manifest.read_text())
        assert len(record['frames']) == 32
        assert all(digest(OUTPUT / r['path']) == r['sha256'] for r in record['frames'])
        return record
    frames = []
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        duration = stream.duration * stream.time_base
        start = (stream.start_time or 0) * stream.time_base
        for index in range(32):
            target = start + duration * Fraction(2 * index + 1, 64)
            container.seek(int(target / stream.time_base), stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                actual = frame.pts * frame.time_base
                if actual >= target:
                    pixels = frame.to_image().convert('RGB')
                    pixels.thumbnail((672, 672), Image.Resampling.BICUBIC)
                    path = folder / f'{index:02d}.jpg'
                    pixels.save(path, quality=95, subsampling=0)
                    frames.append({'index': index, 'actual_seconds': float(actual - start),
                                   'target_seconds': float(target - start), 'pts': frame.pts,
                                   'path': str(path.relative_to(OUTPUT)), 'sha256': digest(path),
                                   'width': pixels.width, 'height': pixels.height})
                    break
            else:
                raise ValueError(f'Missing frame: {video} {index}')
    assert len(frames) == 32 and all(a['pts'] < b['pts'] for a, b in zip(frames, frames[1:]))
    result = {'video': video, 'source_sha256': original['source_sha256'], 'frames': frames}
    manifest.write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    started = time.monotonic()
    videos = sorted(p.name for p in (DATA / 'videomme_short_frames').iterdir() if (p / 'frames.json').exists())
    assert len(videos) == 300, len(videos)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, record in enumerate(pool.map(sample, videos), 1):
            print(i, '/300', record['video'], round(time.monotonic() - started, 1), flush=True)
    (OUTPUT / 'READY.json').write_text(json.dumps({'videos': 300, 'frames': 9600,
        'seconds': time.monotonic() - started, 'sampling': '32 equal temporal-bin midpoints',
        'encoding': 'JPEG quality95 subsampling0, max side672 bicubic, then unchanged model processor',
        'answer_labels_read': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
