"""Prepare a reproducible 10% OK-VQA val subset with only its COCO images.

The COCO val2014 distribution is a large official ZIP.  The script reads its
ZIP64 central directory through an HTTPS Range request and fetches only the
selected JPEG members.  It never disables TLS verification and never writes a
partial JPEG as a final file.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time
import urllib.request
import zlib
import zipfile


COCO_URL = "https://s3.amazonaws.com/images.cocodataset.org/zips/val2014.zip"
COCO_BYTES = 6_645_013_297
SEED = 20261013


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def http_range(start: int, end: int) -> bytes:
    req = urllib.request.Request(COCO_URL, headers={"User-Agent": "Mozilla/5.0", "Accept-Encoding": "identity", "Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=240) as response:
        if response.status != 206:
            raise RuntimeError(f"COCO range {start}-{end} returned HTTP {response.status}")
        data = response.read()
        cr = response.headers.get("Content-Range", "")
        if len(data) != end - start + 1 or not cr.startswith(f"bytes {start}-{end}/"):
            raise RuntimeError(f"COCO range mismatch: got {len(data)} bytes, Content-Range={cr!r}")
        return data


def parse_zip64_directory() -> dict[str, dict[str, int | str]]:
    tail_start = COCO_BYTES - 8 * 1024 * 1024
    tail = http_range(tail_start, COCO_BYTES - 1)
    eocd_pos = tail.rfind(b"PK\x05\x06")
    zip64_pos = tail.rfind(b"PK\x06\x06")
    if eocd_pos < 0 or zip64_pos < 0:
        raise RuntimeError("COCO ZIP64 end records were not found")
    _, record_size, _, _, _, _, _, _, _, _ = struct.unpack_from("<4sQ2H2L4Q", tail, zip64_pos)
    fields = struct.unpack_from("<4sQ2H2L4Q", tail, zip64_pos)
    total_entries = fields[7]
    directory_size = fields[8]
    directory_offset = fields[9]
    directory = http_range(directory_offset, directory_offset + directory_size - 1)
    records: dict[str, dict[str, int | str]] = {}
    pos = 0
    for _ in range(total_entries):
        if directory[pos:pos + 4] != b"PK\x01\x02":
            raise RuntimeError(f"bad central-directory signature at offset {pos}")
        fixed = struct.unpack_from("<4s6H3L5H2L", directory, pos)
        (_, _, _, flags, compression, _, _, crc, comp_size, raw_size, name_len, extra_len, comment_len, _, _, _, local_offset) = fixed
        name_raw = directory[pos + 46:pos + 46 + name_len]
        extra = directory[pos + 46 + name_len:pos + 46 + name_len + extra_len]
        name = name_raw.decode("utf-8" if flags & 0x800 else "cp437")
        if comp_size == 0xFFFFFFFF or raw_size == 0xFFFFFFFF or local_offset == 0xFFFFFFFF:
            ep = 0
            values: list[int] = []
            while ep + 4 <= len(extra):
                tag, length = struct.unpack_from("<HH", extra, ep)
                payload = extra[ep + 4:ep + 4 + length]
                if tag == 0x0001:
                    for p in range(0, len(payload), 8):
                        values.append(struct.unpack_from("<Q", payload, p)[0])
                    break
                ep += 4 + length
            vi = 0
            if raw_size == 0xFFFFFFFF:
                raw_size = values[vi]; vi += 1
            if comp_size == 0xFFFFFFFF:
                comp_size = values[vi]; vi += 1
            if local_offset == 0xFFFFFFFF:
                local_offset = values[vi]
        records[name] = {"flags": flags, "compression": compression, "crc32": crc, "compressed": comp_size, "size": raw_size, "offset": local_offset}
        pos += 46 + name_len + extra_len + comment_len
    if len(records) != total_entries:
        raise RuntimeError(f"central directory entries mismatch: {len(records)} vs {total_entries}")
    return records


def fetch_member(name: str, record: dict[str, int | str], output: Path) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size == int(record["size"]):
        data = output.read_bytes()
    else:
        offset = int(record["offset"])
        header = http_range(offset, offset + 30 - 1)
        if header[:4] != b"PK\x03\x04":
            raise RuntimeError(f"bad local header for {name}")
        _, _, _, _, _, _, _, _, _, name_len, extra_len = struct.unpack_from("<4s5H3L2H", header)
        end = offset + 30 + name_len + extra_len + int(record["compressed"]) - 1
        payload = http_range(offset, end)[30 + name_len + extra_len:]
        if int(record["compression"]) == 0:
            data = payload
        elif int(record["compression"]) == 8:
            data = zlib.decompress(payload, -15)
        else:
            raise RuntimeError(f"unsupported ZIP compression {record['compression']} for {name}")
        if len(data) != int(record["size"]) or (zlib.crc32(data) & 0xFFFFFFFF) != int(record["crc32"]):
            raise RuntimeError(f"integrity mismatch for {name}")
        tmp = output.with_name(output.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, output)
    if not data.startswith(b"\xff\xd8"):
        raise RuntimeError(f"{name} is not a JPEG")
    return {"member": name, "path": str(output), "source_compressed_bytes": int(record["compressed"]), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def load_json_zip(path: Path, key: str) -> object:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        if not names:
            raise RuntimeError(f"no JSON in {path}")
        data = json.loads(zf.read(names[0]))
    return data[key] if key else data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(r"D:\WWW2027\data_ops\subsets\OKVQA_10pct"))
    ap.add_argument("--questions-zip", type=Path, default=Path(r"D:\WWW2027\data_ops\subsets\OKVQA\raw\OpenEnded_mscoco_val2014_questions.json.zip"))
    ap.add_argument("--annotations-zip", type=Path, default=Path(r"D:\WWW2027\data_ops\subsets\OKVQA\raw\mscoco_val2014_annotations.json.zip"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    questions = load_json_zip(args.questions_zip, "questions")
    annotations = load_json_zip(args.annotations_zip, "annotations")
    if len(questions) != 5046 or len(annotations) != 5046:
        raise RuntimeError(f"official OK-VQA val count mismatch: questions={len(questions)} annotations={len(annotations)}")
    ann_by_id = {int(a["question_id"]): a for a in annotations}
    ranked = sorted(questions, key=lambda q: hashlib.sha256(f"{SEED}:{int(q['question_id'])}".encode()).hexdigest())
    n = math.ceil(len(questions) * 0.10)
    selected = sorted(ranked[:n], key=lambda q: int(q["question_id"]))
    selected_ids = {int(q["question_id"]) for q in selected}
    selected_annotations = [ann_by_id[int(q["question_id"])] for q in selected]
    image_ids = sorted({int(q["image_id"]) for q in selected})
    (root / "selected_ids.jsonl").write_text("".join(json.dumps({"question_id": int(q["question_id"]), "image_id": int(q["image_id"]), "split": "val2014"}) + "\n" for q in selected), encoding="utf-8")
    (root / "questions.json").write_text(json.dumps({"info": {"source": "official OK-VQA val questions", "seed": SEED}, "questions": selected}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "annotations.json").write_text(json.dumps({"info": {"source": "official OK-VQA val annotations", "seed": SEED}, "annotations": selected_annotations}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    records = parse_zip64_directory()
    members = {}
    for image_id in image_ids:
        name = f"val2014/COCO_val2014_{image_id:012d}.jpg"
        if name not in records:
            raise RuntimeError(f"COCO member missing: {name}")
        members[name] = records[name]
    image_dir = root / "images"
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {pool.submit(fetch_member, name, rec, image_dir / Path(name).name): name for name, rec in members.items()}
        downloads = [f.result() for f in concurrent.futures.as_completed(futs)]
    downloads.sort(key=lambda r: str(r["member"]))
    (root / "resource_downloads.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in downloads), encoding="utf-8")
    manifest = {
        "dataset": "OK-VQA",
        "scope": "official val2014 question set, deterministic 10% preparatory subset",
        "seed": SEED,
        "algorithm": "rank sha256(f'{seed}:{question_id}') ascending; ceil(0.10*N); sort selected IDs",
        "question_count": len(selected),
        "question_denominator": len(questions),
        "image_count": len(image_ids),
        "source_questions_zip": {"path": str(args.questions_zip), "bytes": args.questions_zip.stat().st_size, "sha256": sha256(args.questions_zip)},
        "source_annotations_zip": {"path": str(args.annotations_zip), "bytes": args.annotations_zip.stat().st_size, "sha256": sha256(args.annotations_zip)},
        "source_coco_object": {"url": COCO_URL, "logical_bytes": COCO_BYTES, "selection_method": "ZIP64 central directory + bounded HTTPS ranges", "range_bytes_received": sum(int(r["source_compressed_bytes"]) + 30 for r in downloads)},
        "resources": downloads,
    }
    (root / "subset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"root": str(root), "questions": len(selected), "denominator": len(questions), "images": len(image_ids), "image_bytes": sum(int(r["bytes"]) for r in downloads)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
