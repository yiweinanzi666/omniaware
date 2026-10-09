"""Decode eight fixed temporal-bin midpoint frames from every Short video."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import time

import av
from PIL import Image


ROOT = Path(os.environ.get('OMNIAWARE_ASSETS') or os.environ['ERC_ASSET_ROOT']).resolve()  # legacy experiments/eab_first_token
DATA = ROOT / 'data/videomme_short'
OUTPUT = ROOT / 'data/videomme_short_frames'
FRAME_COUNT = 8


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def sample(row):
    tick = time.perf_counter()
    source = Path(row['path'])
    target = OUTPUT / row['video_id']
    target.mkdir(parents=True, exist_ok=True)
    source_hash = digest(source)
    if source_hash != row['sha256']:
        raise ValueError('Downloaded video differs from recorded identity: ' + str(source))
    result_path = target / 'frames.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['source_sha256'] != source_hash:
            raise ValueError('Cached frame source differs: ' + str(source))
        for record in result['frames']:
            frame_path = OUTPUT / record['path']
            if digest(frame_path) != record['sha256']:
                raise ValueError('Cached frame bytes differ: ' + str(frame_path))
            with Image.open(frame_path) as image:
                if hashlib.sha256(image.convert('RGB').tobytes()).hexdigest() != record['rgb_sha256']:
                    raise ValueError('Cached frame pixels differ: ' + str(frame_path))
        return dict(result, reused=True)
    try:
        frames = []
        with av.open(str(source)) as container:
            stream = container.streams.video[0]
            stream.thread_count = 1
            duration = stream.duration * stream.time_base
            start = stream.start_time * stream.time_base
            if duration <= 0:
                raise ValueError('Nonpositive video duration: ' + str(source))
            for index in range(FRAME_COUNT):
                relative_target = duration * Fraction(2 * index + 1, 2 * FRAME_COUNT)
                target_seconds = start + relative_target
                container.seek(int(target_seconds / stream.time_base), stream=stream,
                               backward=True, any_frame=False)
                for frame in container.decode(stream):
                    seconds = frame.pts * frame.time_base
                    if seconds >= target_seconds:
                        image = frame.to_image().convert('RGB')
                        path = target / f'{index:02d}.png'
                        image.save(path, compress_level=1)
                        frames.append(dict(index=index, pts=frame.pts,
                            frame_time_base=str(frame.time_base), target_seconds=float(relative_target),
                            actual_seconds=float(seconds-start), delay_seconds=float(seconds-target_seconds),
                            width=image.width, height=image.height,
                            path=str(path.relative_to(OUTPUT)), sha256=digest(path),
                            rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
                        break
                else:
                    raise ValueError(f'No decoded frame at target {relative_target}: {source}')
        if len(frames) != FRAME_COUNT or any(a['pts'] >= b['pts'] for a, b in zip(frames, frames[1:])):
            raise ValueError('Expected eight distinct frames in time order: ' + str(source))
        result = dict(video_id=row['video_id'], source=str(source), source_sha256=source_hash,
            stream_start_seconds=float(start), stream_duration_seconds=float(duration),
            frames=frames, reused=False, seconds=time.perf_counter()-tick)
        save_json(result_path, result)
        return result
    except Exception as error:
        save_json(target / 'failure.json', dict(video_id=row['video_id'], error=repr(error),
            elapsed_seconds=time.perf_counter()-tick,
            failed_utc=datetime.now(timezone.utc).isoformat()))
        raise


def main():
    tick = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    downloads = [json.loads(line) for line in (DATA / 'downloads.jsonl').read_text().splitlines()]
    rows = sorted(downloads, key=lambda row: row['video_id'])
    if len(rows) != 300 or len({row['video_id'] for row in rows}) != 300:
        raise ValueError('Expected the complete 300-video Short download')
    save_json(OUTPUT / 'config.json', dict(frame_count=FRAME_COUNT, selection='all 300 Short videos',
        sampling='equal temporal-bin midpoint, first frame at or after target',
        cpu_workers=2, decoder_threads_per_video=1, av_version=av.__version__,
        source_manifest_sha256=digest(DATA / 'manifest.json'),
        downloads_sha256=digest(DATA / 'downloads.jsonl'), code_sha256=digest(Path(__file__)),
        protocol_sha256=digest(ROOT / 'VIDEOMME_FRAME_PROTOCOL.md'),
        started_utc=datetime.now(timezone.utc).isoformat(), answer_column_read=False))
    completed = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(sample, rows):
            completed.append(result)
            print(json.dumps(dict(event='video_frames', count=len(completed), total=300,
                video_id=result['video_id'], reused=result['reused'], seconds=result['seconds'])), flush=True)
    save_json(OUTPUT / 'completion.json', dict(status='completed', videos=completed,
        video_count=300, frame_count=sum(len(row['frames']) for row in completed),
        maximum_frame_delay_seconds=max(f['delay_seconds'] for row in completed for f in row['frames']),
        elapsed_seconds=time.perf_counter()-tick,
        completed_utc=datetime.now(timezone.utc).isoformat(), model_calls=0, answer_column_read=False))


if __name__ == '__main__':
    main()
