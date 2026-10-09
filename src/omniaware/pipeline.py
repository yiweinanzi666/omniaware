"""Prompt construction, controller (U_omni, gates) and EAB variant definitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import scoring
from .qwen_eab import BiasSpec, Segment, chunk_mass, grounding_entropy, predictive_entropy

INSTRUCTION = {
    "okvqa": "Answer the question using a single word or phrase.",
    "mmqa_table": "Answer the question using a single word or phrase.",
    "webqa": "Answer the question in one short sentence.",
    "videomme": "Answer with the option's letter from the given choices directly.",
}
# Match saved baseline budgets; a short-answer instruction must still allow complete lists.
MAX_NEW_TOKENS = {"okvqa": 30, "mmqa_table": 256, "webqa": 384, "videomme": 4}
MAX_NEW_TOKENS["okvqa_dense"] = MAX_NEW_TOKENS["okvqa"]
MAX_DOC_WORDS = 160


def _trim(text: str, n: int = MAX_DOC_WORDS) -> str:
    words = text.split()
    return text if len(words) <= n else " ".join(words[:n]) + " ..."


SHORT_INSTRUCTION = "Answer the question using a single word or phrase."
TABLE_INSTRUCTION = (
    "Use the provided tables to answer the question. Apply every condition to the matching rows, "
    "then compute any requested count, average, minimum, maximum, or most recent value. "
    "For a question that first looks up a value, use that value to complete the remaining lookup. "
    "Return only the final answer; include every requested item."
)


TOKEN_PIXELS = 32 * 32  # one Qwen3-VL visual token after 2x2 merging of 16x16 patches


def focus_image_pixels(chunks: list[dict], rel: dict[str, float], base_tokens: int = 256,
                       low_tokens: int = 128) -> dict[str, int]:
    """Relevance-weighted visual budget at a fixed total: with n evidence images, the most
    relevant one may use base + (n-1)*(base-low) tokens and every other image `low` tokens,
    so the cap sums to n*base, the uniform budget."""
    imgs = sorted((c for c in chunks if c["modality"] == "image"), key=lambda c: -rel[c["cid"]])
    if len(imgs) <= 1:
        return {}
    top = base_tokens + (len(imgs) - 1) * (base_tokens - low_tokens)
    caps = {imgs[0]["cid"]: top * TOKEN_PIXELS}
    caps.update({c["cid"]: low_tokens * TOKEN_PIXELS for c in imgs[1:]})
    return caps


def build_segments(e: dict, chunks: list[dict] | None, answer_style: str = "default",
                   image_pixels: dict[str, int] | None = None) -> tuple[list[Segment], int]:
    """User turn = [query images] [evidence chunks] [question suffix]; returns (segments, split index)."""
    segs = [Segment(text="<|im_start|>user\n")]
    for img in e["query_images"]:
        segs += [Segment(image=img, chunk="__query__"), Segment(text="\n")]
    query_chunks = e.get("query_chunks", [])
    for c in query_chunks:
        segs.append(Segment(text=f"Provided table:\n{c.get('prompt_text', c['text'])}\n\n", chunk=c["cid"]))
    if query_chunks and chunks:
        # Initial table rows stay visible in both branches and are never duplicated.
        initial_ids = {c["cid"] for c in query_chunks}
        chunks = [c for c in chunks if c["cid"] not in initial_ids]
    if chunks:
        segs.append(Segment(text="Based on the following evidence, answer the question.\n\n"))
        for j, c in enumerate(chunks, 1):
            cid = c["cid"]
            if c["modality"] == "image":
                head = f"Document {j} (image): {c['title']}\n" if c["title"] else f"Document {j} (image):\n"
                tail = f"\nCaption: {_trim(c['text'], 60)}\n\n" if c["text"] and c["text"] != c["title"] else "\n\n"
                cap = (image_pixels or {}).get(cid)
                segs += [Segment(text=head, chunk=cid), Segment(image=c["image"], chunk=cid, max_pixels=cap),
                         Segment(text=tail, chunk=cid)]
            elif c["modality"] == "table":
                segs.append(Segment(text=f"Document {j} (table):\n{c.get('prompt_text', c['text'])}\n\n", chunk=cid))
            else:
                body = f"{c['title']}: {c['text']}" if c["title"] and c["title"] != c["text"] else c["text"]
                segs.append(Segment(text=f"Document {j} (text): {_trim(body)}\n\n", chunk=cid))
    split = len(segs)
    q = f"Question: {e['question']}\n"
    if e.get("options"):
        q += "\n".join(e["options"]) + "\n"
    instruction = SHORT_INSTRUCTION if answer_style == "short" and not e.get("options") else INSTRUCTION[e["dataset"]]
    if answer_style == "table" and e["dataset"] == "mmqa_table":
        instruction = TABLE_INSTRUCTION
    q += instruction + "<|im_end|>\n<|im_start|>assistant\n"
    segs.append(Segment(text=q))
    return segs, split


def score(e: dict, pred: str) -> dict:
    ds = e["dataset"]
    if ds == "webqa":
        return scoring.score_webqa(pred, e["golds"][0], e["question"], e["qcate"])
    if ds == "videomme":
        return scoring.score_choice(pred, e["gold_letter"], e["options"])
    return scoring.score_short_answer(pred, e["golds"])


# ----------------------------------------------------------------- control --
@dataclass
class Norm:
    median: float
    iqr: float

    def __call__(self, s: float) -> float:
        return float(np.clip((s - self.median) / (1.5 * max(self.iqr, 1e-6)) + 0.5, 0.0, 1.0))

    @classmethod
    def fit(cls, values: list[float]) -> "Norm":
        v = np.asarray(values, dtype=float)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        return cls(float(med), float(q3 - q1))


def raw_uncertainty(attn, enc, evidence_ids: list[str], logits, beta: float) -> dict:
    tokens = [t for c in evidence_ids if c in enc.spans for t in range(*enc.spans[c])]
    h_grd = grounding_entropy(attn, tokens)
    h_sem = predictive_entropy(logits)
    return {"h_grd": h_grd, "h_sem": h_sem, "s": h_grd + beta * h_sem}


@dataclass
class Variant:
    """EAB family.  rows: 'ans2' paper (a >= q_ans), 'ans1' includes the first answer
    token, 'q' starts at the question.  form: 'pos' gamma*U*r, 'ctr' gamma*U*(2r-1),
    'fixed' gamma (PASTA-style uniform), 'pos_nou' gamma*r, 'sup' gamma*U*(r-1)
    (suppress low-relevance chunks only)."""

    name: str
    gamma: float
    rows: str = "ans2"
    form: str = "pos"
    gate: bool = True
    layers: str = "all"
    relevance: str = "retriever"   # 'retriever': r_j; 'fuse<a>': rank fusion with latent grounding

    def spec(self, enc, rel: dict[str, float], u: float, tau_bias: float, n_layers: int) -> BiasSpec | None:
        if self.gate and u <= tau_bias:
            return None
        if self.form.endswith("k"):
            # contrast inside the selected evidence: min-max rescale r over the k chunks
            lo, hi = min(rel.values()), max(rel.values())
            rel = {c: (r - lo) / (hi - lo) if hi > lo else 1.0 for c, r in rel.items()}
        if self.form == "pos":
            vals = {c: self.gamma * u * r for c, r in rel.items()}
        elif self.form == "ctr":
            vals = {c: self.gamma * u * (2 * r - 1) for c, r in rel.items()}
        elif self.form == "fixed":
            vals = {c: self.gamma for c in rel}
        elif self.form == "pos_nou":
            vals = {c: self.gamma * r for c, r in rel.items()}
        elif self.form == "ctr_nou":
            vals = {c: self.gamma * (2 * r - 1) for c, r in rel.items()}
        elif self.form == "ctrk":
            vals = {c: self.gamma * u * (2 * r - 1) for c, r in rel.items()}
        elif self.form == "supk":
            vals = {c: self.gamma * u * (r - 1) for c, r in rel.items()}
        elif self.form == "ctrk_nouk" or self.form == "ctr_nouk":
            vals = {c: self.gamma * (2 * r - 1) for c, r in rel.items()}
        elif self.form == "sup_nouk":
            vals = {c: self.gamma * (r - 1) for c, r in rel.items()}
        elif self.form == "sup":
            vals = {c: self.gamma * u * (r - 1) for c, r in rel.items()}
        elif self.form == "sup_nou":
            vals = {c: self.gamma * (r - 1) for c, r in rel.items()}
        else:
            raise ValueError(self.form)
        q_start = {"ans2": enc.length, "ans1": enc.length - 1, "q": enc.split}[self.rows]
        layers = None
        if self.layers == "mid":
            layers = set(range(n_layers // 4, 3 * n_layers // 4))
        elif self.layers == "late":
            layers = set(range(n_layers // 2, n_layers))
        return BiasSpec(values=vals, q_start=q_start, layers=layers)


def parse_variant(name: str) -> Variant:
    """e.g. 'eab:g1.2:ans2:pos:gate:all' or 'pasta:g1.2:ans2'."""
    parts = name.split(":")
    kind = parts[0]
    gamma = float(parts[1][1:]) if len(parts) > 1 else 1.2
    rows = parts[2] if len(parts) > 2 else "ans2"
    if kind == "pasta":
        return Variant(name, gamma, rows, "fixed", gate=False)
    form = parts[3] if len(parts) > 3 else "pos"
    gate = (parts[4] if len(parts) > 4 else "gate") == "gate"
    layers = parts[5] if len(parts) > 5 else "all"
    relevance = parts[6] if len(parts) > 6 else "retriever"
    return Variant(name, gamma, rows, form, gate, layers, relevance)


def _rank01(values: dict[str, float]) -> dict[str, float]:
    order = sorted(values, key=lambda k: -values[k])
    return {k: 1 - i / max(len(order) - 1, 1) for i, k in enumerate(order)}


def fused_relevance(rel: dict[str, float], mass: dict[str, float], alpha: float) -> dict[str, float]:
    """Rank fusion of retriever relevance r_j with latent grounding (attention mass of the
    last prompt token on chunk j in the unbiased prefill, the same pass that gives H_grd).
    Returns scores in [0, 1]; ties keep the retriever order."""
    rr = _rank01(rel)
    mr = _rank01({c: mass.get(c, 0.0) for c in rel})
    fused = {c: alpha * rr[c] + (1 - alpha) * mr[c] + 1e-6 * rel[c] for c in rel}
    return _rank01(fused)


def variant_relevance(v: "Variant", rel: dict[str, float], mass: dict[str, float]) -> dict[str, float]:
    if v.relevance == "retriever":
        return rel
    if v.relevance.startswith("fuse"):
        return fused_relevance(rel, mass, float(v.relevance[4:]))
    raise ValueError(v.relevance)


def chunk_masses(attn, enc) -> dict[str, float]:
    return chunk_mass(attn, enc.spans)
