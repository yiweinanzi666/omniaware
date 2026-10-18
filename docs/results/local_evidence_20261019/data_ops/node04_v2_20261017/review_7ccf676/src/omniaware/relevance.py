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
            mats = bge.encode([f"{c['title']}: {c['text']}" if c["modality"] == "text" and c["title"]
                               and c["title"] != c["text"] else c["text"] for c in txt])
            for c, v in zip(txt, mats):
                scores[c["cid"]][c["modality"]] = float(v @ qv)
        vids = [c for c in e["candidates"] if c["modality"] in ("video", "clip")]
        if vids:
            with_text = [c for c in vids if c["text"]]
            if with_text:
                for c, v in zip(with_text, bge.encode([c["text"] for c in with_text])):
                    scores[c["cid"]][VIDEO_CHANNELS[c["modality"]][0]] = float(v @ qv)
            if clip is not None:
                tv = clip.encode_text([q])[0]
                for c in vids:
                    fv = clip.encode_images(c["frames"])
                    m = fv.mean(0)
                    scores[c["cid"]][VIDEO_CHANNELS[c["modality"]][1]] = float(m @ tv / (np.linalg.norm(m) + 1e-12))
        imgs = [c for c in e["candidates"] if c["modality"] == "image"]
        if imgs:
            caps = [c for c in imgs if c["title"] or c["text"]]
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


# Transcript (BGE) and mean-frame (CLIP) channels of the two video modalities. Clips are a modality of their own
# (D6), so they get their own channels and hence their own per-modality ECDF.
VIDEO_CHANNELS = {"video": ("video_text", "video_clip"), "clip": ("clip_text", "clip_frames")}


def used_channels(channels: dict[str, float], image_mode: str = "mix") -> list[str]:
    """The scored channels that define one candidate's relevance (see Calibrator.relevance)."""
    for pair in VIDEO_CHANNELS.values():
        video = [k for k in pair if k in channels]
        if video:
            return video
    if "text" in channels:
        return ["text"]
    if "table" in channels:
        return ["table"]
    parts = [k for k, modes in (("image_clip", ("clip", "mix")), ("image_cap", ("caption", "mix")))
             if k in channels and image_mode in modes]
    if not parts and "image_clip" in channels:
        parts = ["image_clip"]
    return parts


class Calibrator:
    """Per-channel ECDF frozen on calibration scores.

    A candidate scored on several channels (video text and clip, image clip and caption) has the mean of its
    channel percentiles as the raw combined score. The mean of percentiles is not itself uniform (its spread
    shrinks), so ``combined`` holds a second ECDF per channel set, fitted on the calibration candidates'
    means; relevance is then again a percentile of the calibration distribution.
    """

    def __init__(self, sorted_scores: dict[str, list[float]], combined: dict[tuple, list[float]] | None = None):
        self.sorted = {k: sorted(v) for k, v in sorted_scores.items()}
        self.combined = {k: sorted(v) for k, v in (combined or {}).items()}

    @classmethod
    def fit(cls, raw: dict, qids: list[str], image_mode: str = "mix") -> "Calibrator":
        pools: dict[str, list[float]] = {}
        for qid in qids:
            for ch in raw[qid].values():
                for k, v in ch.items():
                    pools.setdefault(k, []).append(v)
        cal = cls(pools)
        combined: dict[tuple, list[float]] = {}
        for qid in qids:
            for channels in raw[qid].values():
                used = used_channels(channels, image_mode)
                if len(used) > 1:
                    combined.setdefault(tuple(used), []).append(cal._mean_ecdf(used, channels))
        cal.combined = {k: sorted(v) for k, v in combined.items()}
        return cal

    def ecdf(self, channel: str, value: float) -> float:
        arr = self.sorted.get(channel)
        if not arr:
            raise ValueError(f"no calibration scores for channel {channel!r}")
        return bisect.bisect_right(arr, value) / len(arr)

    def _mean_ecdf(self, used: list[str], channels: dict[str, float]) -> float:
        return float(np.mean([self.ecdf(k, channels[k]) for k in used]))

    def relevance(self, channels: dict[str, float], image_mode: str = "mix") -> float:
        """r_j in [0, 1]: the channel's percentile, or for several channels the percentile of their mean
        percentile within the calibration candidates scored on the same channels. Missing calibration data
        is an error (an absent channel never silently maps to a constant)."""
        used = used_channels(channels, image_mode)
        if not used:
            raise ValueError(f"no relevance channel among {sorted(channels)} (image_mode={image_mode!r})")
        if len(used) == 1:
            return self.ecdf(used[0], channels[used[0]])
        pool = self.combined.get(tuple(used))
        if not pool:
            raise ValueError(f"no calibration candidates scored on the channels {used}")
        return bisect.bisect_right(pool, self._mean_ecdf(used, channels)) / len(pool)

    @staticmethod
    def raw(channels: dict[str, float], image_mode: str = "mix") -> float:
        """The retriever's own score without per-modality calibration (raw-score ablation)."""
        for key in ("text", "table"):
            if key in channels:
                return float(channels[key])
        for pair in VIDEO_CHANNELS.values():
            video = [channels[k] for k in pair if k in channels]
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
