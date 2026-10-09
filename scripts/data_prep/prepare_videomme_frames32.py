"""Prepare 32 midpoint frames per video using only source media, never labels."""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

import av
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / 'experiments/eab_first_token/data'
OUTPUT = HERE / 'video_frames32'


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
    payload = json.loads((HERE / 'input_videomme.json').read_text())
    videos = sorted({Path(c['image']).parent.name for e in payload['rows'] for c in e['candidates'] if c.get('image')})
    assert len(videos) == 300
    records = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, record in enumerate(pool.map(sample, videos), 1):
            records[record['video']] = record
            print(i, '/300', record['video'], round(time.monotonic() - started, 1), flush=True)
    for e in payload['rows']:
        video = Path(next(c['image'] for c in e['candidates'] if c.get('image'))).parent.name
        candidates = []
        for frame in records[video]['frames']:
            sec = frame['actual_seconds']
            candidates.append({'cid': f"f{frame['index']}", 'modality': 'image',
                               'title': f'Frame at {int(sec)//60:02d}:{sec%60:05.2f}', 'text': '',
                               'image': '/root/autodl-tmp/emnlp_three_official_20261008/video_frames32/' + frame['path']})
        candidates.extend(c for c in e['candidates'] if c['modality'] != 'image')
        e['candidates'] = candidates
    (HERE / 'input_videomme32.json').write_text(json.dumps(payload, ensure_ascii=False) + '\n')
    (HERE / 'VIDEO32_READY.json').write_text(json.dumps({'videos': 300, 'questions': 900, 'frames': 9600,
        'seconds': time.monotonic() - started, 'sampling': '32 equal temporal-bin midpoints',
        'encoding': 'JPEG quality95 subsampling0, max side672 bicubic, then unchanged model processor',
        'answer_labels_read': False, 'input_sha256': digest(HERE / 'input_videomme32.json')}, indent=2) + '\n')


if __name__ == '__main__':
    main()
