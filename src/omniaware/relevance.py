"""Retriever relevance: BGE for text/table chunks, CLIP (+ caption BGE) for images.

Raw scores are mapped to [0, 1] with per-channel percentile (ECDF) calibration frozen on
the calibration split, as in the paper (Sec. 3.4).  Ranking and EAB both use the
calibrated score r_j.
"""

from __future__ import annotations

import bisect
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import paths

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def pick_device() -> str:
    if os.environ.get("ERC_REL_DEVICE"):
        return os.environ["ERC_REL_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BGE:
    def __init__(self, device: str | None = None):
        from transformers import AutoModel, AutoTokenizer

        self.device = device or pick_device()
        self.tok = AutoTokenizer.from_pretrained(paths.BGE)
        self.model = AutoModel.from_pretrained(paths.BGE).to(self.device).eval()

    @torch.no_grad()
    def encode(self, texts: list[str], query: bool = False, batch: int = 32) -> np.ndarray:
        if query:
            texts = [BGE_QUERY_PREFIX + t for t in texts]
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok(texts[i:i + batch], padding=True, truncation=True, max_length=512,
                           return_tensors="pt").to(self.device)
            cls = self.model(**enc).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(cls.float(), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1024), dtype=np.float32)


class CLIP:
    def __init__(self, device: str | None = None):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or pick_device()
        self.proc = CLIPProcessor.from_pretrained(paths.CLIP)
        self.model = CLIPModel.from_pretrained(paths.CLIP).to(self.device).eval()

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> np.ndarray:
        enc = self.proc.tokenizer(texts, padding=True, truncation=True, max_length=77,
                                  return_tensors="pt").to(self.device)
        f = self.model.get_text_features(**enc)
        f = f if isinstance(f, torch.Tensor) else f.pooler_output
        return torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()

    @torch.no_grad()
    def encode_images(self, image_paths: list[str], batch: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(image_paths), batch):
            ims = [Image.open(p).convert("RGB") for p in image_paths[i:i + batch]]
            enc = self.proc(images=ims, return_tensors="pt").to(self.device)
            f = self.model.get_image_features(**enc)
            f = f if isinstance(f, torch.Tensor) else f.pooler_output
            out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 768), dtype=np.float32)


def query_text(e: dict) -> str:
    if e.get("options"):
        return e["question"] + " " + " ".join(o[3:] for o in e["options"])
    return e["question"]


def raw_scores(examples: list[dict], bge: BGE, clip: CLIP | None) -> dict[str, dict[str, dict[str, float]]]:
    """qid -> cid -> {channel: raw score}. Channels: text, table, image_clip, image_cap."""
    result = {}
    for e in examples:
        q = query_text(e)
        qv = bge.encode([q], query=True)[0]
        scores = {c["cid"]: {} for c in e["candidates"]}
        txt = [c for c in e["candidates"] if c["modality"] in ("text", "table")]
        if txt:
            mats = bge.encode([f"{c['title']}: {c['text']}" if c["modality"] == "text" else c["text"] for c in txt])
            for c, v in zip(txt, mats):
                scores[c["cid"]][c["modality"]] = float(v @ qv)
        vids = [c for c in e["candidates"] if c["modality"] == "video"]
        if vids:
            with_text = [c for c in vids if c["text"]]
            if with_text:
                for c, v in zip(with_text, bge.encode([c["text"] for c in with_text])):
                    scores[c["cid"]]["video_text"] = float(v @ qv)
            if clip is not None:
                tv = clip.encode_text([q])[0]
                for c in vids:
                    fv = clip.encode_images(c["frames"])
                    m = fv.mean(0)
                    scores[c["cid"]]["video_clip"] = float(m @ tv / (np.linalg.norm(m) + 1e-12))
        imgs = [c for c in e["candidates"] if c["modality"] == "image"]
        if imgs:
            caps = [c for c in imgs if (c["title"] or c["text"]) and not c["cid"].startswith("f")]
            if caps:
                mats = bge.encode([f"{c['title']}: {c['text']}" for c in caps])
                for c, v in zip(caps, mats):
                    scores[c["cid"]]["image_cap"] = float(v @ qv)
            if clip is not None:
                tv = clip.encode_text([q])[0]
                iv = clip.encode_images([c["image"] for c in imgs])
                for c, v in zip(imgs, iv):
                    scores[c["cid"]]["image_clip"] = float(v @ tv)
        result[e["qid"]] = scores
    return result


class Calibrator:
    """Per-channel ECDF frozen on calibration scores."""

    def __init__(self, sorted_scores: dict[str, list[float]]):
        self.sorted = {k: sorted(v) for k, v in sorted_scores.items()}

    @classmethod
    def fit(cls, raw: dict, qids: list[str]) -> "Calibrator":
        pools: dict[str, list[float]] = {}
        for qid in qids:
            for ch in raw[qid].values():
                for k, v in ch.items():
                    pools.setdefault(k, []).append(v)
        return cls(pools)

    def ecdf(self, channel: str, value: float) -> float:
        arr = self.sorted.get(channel)
        if not arr:
            return 0.5
        return bisect.bisect_right(arr, value) / len(arr)

    def relevance(self, channels: dict[str, float], image_mode: str = "mix") -> float:
        video = [self.ecdf(k, channels[k]) for k in ("video_text", "video_clip") if k in channels]
        if video:
            return float(np.mean(video))
        if "text" in channels:
            return self.ecdf("text", channels["text"])
        if "table" in channels:
            return self.ecdf("table", channels["table"])
        parts = []
        if image_mode in ("clip", "mix") and "image_clip" in channels:
            parts.append(self.ecdf("image_clip", channels["image_clip"]))
        if image_mode in ("caption", "mix") and "image_cap" in channels:
            parts.append(self.ecdf("image_cap", channels["image_cap"]))
        if not parts and "image_clip" in channels:
            parts.append(self.ecdf("image_clip", channels["image_clip"]))
        return float(np.mean(parts)) if parts else 0.5

    @staticmethod
    def raw(channels: dict[str, float], image_mode: str = "mix") -> float:
        """The retriever's own score without per-modality calibration (raw-score ablation)."""
        for key in ("text", "table"):
            if key in channels:
                return float(channels[key])
        video = [channels[k] for k in ("video_text", "video_clip") if k in channels]
        if video:
            return float(np.mean(video))
        parts = [channels[k] for k, m in (("image_clip", ("clip", "mix")), ("image_cap", ("caption", "mix")))
                 if k in channels and image_mode in m] or [channels[k] for k in ("image_clip",) if k in channels]
        return float(np.mean(parts)) if parts else 0.0

    def to_json(self) -> dict:
        return {k: v for k, v in self.sorted.items()}


def save_raw(entries: dict, path: Path) -> None:
    """Atomic write of {qid: {"fingerprint": str, "scores": {cid: channels}}}; merges the entries saved
    meanwhile by other workers (an entry for the same qid is replaced)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = {}
    if path.exists():
        try:
            merged = json.loads(path.read_text(encoding="utf8"))
        except json.JSONDecodeError:
            print(f"[relevance] {path} is unreadable; starting it afresh", flush=True)
    merged.update(entries)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(merged), encoding="utf8")
    os.replace(tmp, path)


def load_raw(path: Path) -> dict:
    """{qid: {"fingerprint": str, "scores": {cid: channels}}}, as written by save_raw."""
    return json.loads(path.read_text(encoding="utf8"))
