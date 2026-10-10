"""Run-directory and cache plumbing shared by run.py, rerank.py, relevance.py, merge_shards.py,
scripts/launch.py and eval/score_official.py.

- atomic_write_text: write a temporary file and os.replace it, so readers never see a partial file;
- fingerprint / encoder_identity: content keys for cache entries. A cached entry is reused only when its
  fingerprint (question, candidate texts and image paths, encoder paths and the settings that change its raw
  values) matches; otherwise it is recomputed;
- load_partial_jsonl: resume reading that cuts the incomplete last line an interrupted run leaves behind;
- dumps_strict: JSON with NaN and infinities written as null, so every output is strict JSON;
- completion marker <dataset>.merged.json: written when a run's answers are complete (single process or merged
  shards). It records the number of questions and the sha256 of the configuration and of the output.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from . import paths

CANDIDATE_FIELDS = ("cid", "modality", "title", "text", "image", "frames")


def atomic_write_text(path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf8")
    os.replace(tmp, path)


def json_safe(value):
    """NaN and infinities become None (JSON null); containers are converted recursively."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def dumps_strict(value, **kwargs) -> str:
    """json.dumps that never writes NaN or Infinity."""
    return json.dumps(json_safe(value), allow_nan=False, **kwargs)


def digest(value) -> str:
    """sha256 of a JSON-serialisable value with sorted keys."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def qid_set_sha256(qids) -> str:
    """sha256 of the sorted question ids, one per line: identifies an evaluation set regardless of order."""
    return hashlib.sha256("\n".join(sorted(str(q) for q in qids)).encode("utf8")).hexdigest()


def encoder_identity() -> dict:
    """Location and directory name of the retrieval encoders, part of every relevance-related fingerprint."""
    return {name: {"path": str(path), "dir": Path(path).name}
            for name, path in (("bge", paths.BGE), ("clip", paths.CLIP))}


def example_digest(example: dict) -> str:
    """Content every raw score of a question depends on: question and options, query images, and the cid,
    modality, title, text, image path and frame paths of each candidate."""
    return digest({"qid": example["qid"], "question": example["question"], "options": example.get("options"),
                   "query_images": example.get("query_images"),
                   "candidates": [{k: c.get(k) for k in CANDIDATE_FIELDS} for c in example["candidates"]]})


def fingerprint(example: dict, settings: dict) -> str:
    return digest({"example": example_digest(example), "settings": settings})


def load_partial_jsonl(path) -> list[dict]:
    """Records of a JSONL file that an interrupted run may have left half-written.

    A last line that is unterminated or not valid JSON is cut off the file, so that the run can append after
    the last complete record (the question of the cut line is answered again). A damaged line before the end
    is an error: that is corruption, not an interrupted write.
    """
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    rows, kept = [], 0
    for i, line in enumerate(lines):
        if not line.strip():
            kept += len(line)
            continue
        try:
            if not line.endswith(b"\n"):
                raise ValueError("line is not terminated")
            rows.append(json.loads(line.decode("utf8")))
        except ValueError as exc:   # JSONDecodeError and UnicodeDecodeError are both ValueErrors
            if i != len(lines) - 1:
                raise ValueError(f"{path}: line {i + 1} is damaged ({exc}); only an incomplete last line is repaired")
            with path.open("r+b") as fh:
                fh.truncate(kept)
            print(f"[resume] removed the incomplete last line of {path}", flush=True)
            break
        kept += len(line)
    return rows


def config_path_for(out_dir, dataset: str) -> Path:
    return Path(out_dir) / f"{dataset}_config.json"


def completion_marker(out_dir, dataset: str) -> Path:
    return Path(out_dir) / f"{dataset}.merged.json"


def has_outputs(out_dir, dataset: str) -> bool:
    """Whether any answers (single-process, shard or merged) for this dataset already exist in out_dir."""
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return False
    return ((out_dir / f"{dataset}.jsonl").exists() or completion_marker(out_dir, dataset).exists()
            or (out_dir / f"{dataset}.shards").exists() or any(out_dir.glob(f"{dataset}.shard*-of-*.jsonl")))


def write_completion_marker(out_dir, dataset: str, questions: int, output, config) -> Path:
    """Mark <dataset>.jsonl as complete: question count, and sha256 of the configuration and of the output."""
    output, config = Path(output), Path(config)
    record = {"dataset": dataset, "questions": questions, "output": output.name,
              "output_sha256": sha256_file(output), "config": config.name,
              "config_sha256": sha256_file(config) if config.exists() else None,
              "written_utc": datetime.now(timezone.utc).isoformat()}
    marker = completion_marker(out_dir, dataset)
    atomic_write_text(marker, json.dumps(record, indent=1) + "\n")
    return marker


def verify_completion(out_dir, dataset: str) -> bool:
    """True if the completion marker matches the configuration and the output (same hashes, and as many answers
    as the configuration's evaluation_questions); False if there is no marker; ValueError if it is stale."""
    out_dir = Path(out_dir)
    marker = completion_marker(out_dir, dataset)
    if not marker.exists():
        return False
    record = json.loads(marker.read_text(encoding="utf8"))
    config, output = config_path_for(out_dir, dataset), out_dir / f"{dataset}.jsonl"
    if not config.exists() or not output.exists():
        raise ValueError(f"{marker.name} is stale: its configuration or output file is missing")
    expected = json.loads(config.read_text(encoding="utf8")).get("evaluation_questions")
    answers = sum(1 for line in output.read_bytes().splitlines() if line.strip())
    if (record.get("config_sha256") != sha256_file(config) or record.get("output_sha256") != sha256_file(output)
            or record.get("questions") != expected or answers != expected):
        raise ValueError(f"{marker.name} is stale: {output.name} has {answers} answers, the configuration expects "
                         f"{expected}, or the files changed after the marker was written; remove the marker and "
                         f"the output, or use a new tag")
    return True
