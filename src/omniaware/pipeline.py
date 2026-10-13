"""Prompt construction, controller (U_omni, gates) and EAB variant definitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from. import prompts
from .qwen_eab import BiasSpec, Segment, chunk_mass, grounding_entropy, predictive_entropy

# Frozen per-family strings live in prompts.INSTRUCTIONS. This alias keeps older callers working.
INSTRUCTION = prompts.INSTRUCTIONS
MAX_NEW_TOKENS = {"okvqa": 30, "webqa": 384, "hybridqa": 32, "videorag": 128}
MAX_NEW_TOKENS["okvqa_dense"] = MAX_NEW_TOKENS["okvqa"]
MAX_DOC_WORDS = 160
MAX_CAPTION_WORDS = 60


def _trim(text: str, n: int = MAX_DOC_WORDS) -> str:
    words = text.split()
    return text if len(words) <= n else " ".join(words[:n]) + " ..."


def chunk_text(c: dict) -> str:
    """The text one evidence chunk contributes to the prompt after the word budget: the title and text of a
    passage (160 words), the caption of an image (60 words, empty when it only repeats the title), the transcript
    of a video (160 words), or a table in full. Every arm that shows a chunk as text uses this string."""
    if c["modality"] == "image":
        return _trim(c["text"], MAX_CAPTION_WORDS) if c["text"] and c["text"] != c["title"] else ""
    if c["modality"] in ("video", "clip"):
        return _trim(c["text"]) if c["text"] else ""
    if c["modality"] == "table":
        return c["text"]
    body = f"{c['title']}: {c['text']}" if c["title"] and c["title"] != c["text"] else c["text"]
    return _trim(body)


SHORT_INSTRUCTION = "Answer the question using a single word or phrase."
# Citation output format of the CiteRec protocol (docs/citation_support_protocol.md).
CITATION_INSTRUCTION = (
    'Answer the question using the supplied evidence and images. '
    'Write each answer as a complete factual statement on a separate line. '
    'Cite the supplied evidence that supports each statement using its exact [E<number>] identifier. '
    'Use only identifiers present in the supplied evidence. Do not include explanations or a reference list.'
)


def build_segments(e: dict, chunks: list[dict] | None, answer_style: str = "default") -> tuple[list[Segment], int]:
    """User turn = [query images] [evidence chunks] [question suffix]; returns (segments, split index).
    Every evidence chunk is labelled [E<j>] (the paper's chunk identifier), whatever the answer style."""
    segs = [Segment(text="<|im_start|>user\n")]
    for img in e["query_images"]:
        segs += [Segment(image=img, chunk="__query__"), Segment(text="\n")]
    if chunks:
        segs.append(Segment(text="Based on the following evidence, answer the question.\n\n"))
        for j, c in enumerate(chunks, 1):
            cid = c["cid"]
            label = f"[E{j}]"
            if c["modality"] == "image":
                head = f"{label} (image): {c['title']}\n" if c["title"] else f"{label} (image):\n"
                caption = chunk_text(c)
                tail = f"\nCaption: {caption}\n\n" if caption else "\n\n"
                segs += [Segment(text=head, chunk=cid), Segment(image=c["image"], chunk=cid),
                         Segment(text=tail, chunk=cid)]
            elif c["modality"] in ("video", "clip"):      # UniversalRAG clips are video chunks too
                segs.append(Segment(text=f"{label} (video):\n", chunk=cid))
                for frame in c["frames"]:
                    segs.append(Segment(image=frame, chunk=cid))
                transcript = chunk_text(c)
                tail = f"\nTranscript: {transcript}\n\n" if transcript else "\n\n"
                segs.append(Segment(text=tail, chunk=cid))
            elif c["modality"] == "table":
                segs.append(Segment(text=f"{label} (table):\n{chunk_text(c)}\n\n", chunk=cid))
            else:
                segs.append(Segment(text=f"{label} (text): {chunk_text(c)}\n\n", chunk=cid))
    split = len(segs)
    q = f"Question: {e['question']}\n"
    if answer_style == "cite":
        instruction = CITATION_INSTRUCTION
    elif answer_style == "short":
        instruction = SHORT_INSTRUCTION
    else:
        instruction = prompts.instruction_text(e["dataset"])
    q += instruction + "<|im_end|>\n<|im_start|>assistant\n"
    segs.append(Segment(text=q))
    return segs, split


# ----------------------------------------------------------------- control --
IQR_FLOOR = 1e-6


def robust_scale(iqr: float) -> float:
    """Scale of a calibrated quantity: its IQR, or 1 when the calibration values are (almost) constant.
    Norm and the zsum terms share this fallback. Dividing by a near-zero IQR would turn the map into a step
    function (the gate becomes binary) and zsum would silently drop the term, so both fall back to a unit scale
    instead: a median-centred linear map. This is a documented choice, not a tuned value."""
    return iqr if iqr >= IQR_FLOOR else 1.0


@dataclass
class Norm:
    median: float
    iqr: float

    def __call__(self, s: float) -> float:
        return float(np.clip((s - self.median) / (1.5 * robust_scale(self.iqr)) + 0.5, 0.0, 1.0))

    @classmethod
    def fit(cls, values: list[float]) -> "Norm":
        v = np.asarray(values, dtype=float)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        return cls(float(med), float(q3 - q1))


@dataclass
class Combiner:
    """s = H_grd + beta * H_pred ('sum', paper Eq. uomni) or, with 'zsum', the same after standardising each
    term on the calibration queries, (H - median) / scale, so that beta weighs the two terms on one scale.
    A term that is constant on the calibration queries (H_grd without evidence) is centred on its constant
    value and scaled by the robust_scale fallback."""
    beta: float
    mode: str = "sum"
    grd: Norm | None = None
    pred: Norm | None = None

    def __call__(self, raw: dict) -> float:
        if self.mode == "sum":
            return raw["h_grd"] + self.beta * raw["h_sem"]

        def z(x, n):
            return (x - n.median) / robust_scale(n.iqr)
        return z(raw["h_grd"], self.grd) + self.beta * z(raw["h_sem"], self.pred)

    @classmethod
    def fit(cls, raws: list[dict], beta: float, mode: str = "sum") -> "Combiner":
        if mode == "sum":
            return cls(beta, mode)
        if mode != "zsum":
            raise ValueError(mode)
        return cls(beta, mode, Norm.fit([r["h_grd"] for r in raws]), Norm.fit([r["h_sem"] for r in raws]))

    def to_json(self) -> dict:
        return {"beta": self.beta, "mode": self.mode, "grd": self.grd and vars(self.grd),
                "pred": self.pred and vars(self.pred)}

    @classmethod
    def from_json(cls, d: dict) -> "Combiner":
        return cls(d["beta"], d["mode"], d["grd"] and Norm(**d["grd"]), d["pred"] and Norm(**d["pred"]))


def context_ids(e: dict) -> list[str]:
    """Supplied-context chunk ids of a question: its query image (none for text-only questions)."""
    return ["__query__"] if e["query_images"] else []


def raw_uncertainty(attn, enc, evidence_ids: list[str], logits, beta: float, vocab_size: int) -> dict:
    """Raw H_grd (over the evidence spans) and H_pred (over the vocabulary). An evidence id without a span in
    the encoded prompt is an error, not a silent drop: it would make H_grd describe different evidence."""
    missing = [c for c in evidence_ids if c not in enc.spans]
    if missing:
        raise KeyError(f"evidence ids without a span in the encoded prompt: {missing}")
    tokens = [t for c in evidence_ids for t in range(*enc.spans[c])]
    h_grd = grounding_entropy(attn, tokens)
    h_sem = predictive_entropy(logits, vocab_size)
    return {"h_grd": h_grd, "h_sem": h_sem, "s": h_grd + beta * h_sem}


@dataclass
class Variant:
    """EAB family.  rows: 'ans1' biases every row that predicts an answer
    token, starting at t_end, the row that predicts the first one; 'ans2' is a decode-only variant
    (a >= t_ans, cannot change the first answer token; ablation); 'q' starts at the question.  form: 'pos'
    gamma*U*r, 'ctr' gamma*U*(2r-1), 'fixed' gamma (PASTA-style uniform bias),
    'pos_nou' gamma*r (relevance bias without U scaling, paper ablation).
    relevance: 'retriever' (calibrated r_j) or 'random' (uniform r_j per question, ablation);
    context: also bias the supplied context (query image / provided tables) with r = 1 (ablation)."""

    name: str
    gamma: float
    rows: str = "ans2"
    form: str = "pos"
    gate: bool = True
    layers: str = "all"
    relevance: str = "retriever"
    context: bool = False

    def spec(self, enc, rel: dict[str, float], u: float, tau_bias: float, n_layers: int) -> BiasSpec | None:
        if self.gate and u <= tau_bias:
            return None
        if self.form == "pos":
            vals = {c: self.gamma * u * r for c, r in rel.items()}
        elif self.form == "ctr":
            vals = {c: self.gamma * u * (2 * r - 1) for c, r in rel.items()}
        elif self.form == "fixed":
            vals = {c: self.gamma for c in rel}
        elif self.form == "pos_nou":
            vals = {c: self.gamma * r for c, r in rel.items()}
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
    if relevance not in ("retriever", "random"):
        raise ValueError(relevance)
    context = len(parts) > 7 and parts[7] == "ctx"
    return Variant(name, gamma, rows, form, gate, layers, relevance, context)


def chunk_masses(attn, enc) -> dict[str, float]:
    return chunk_mass(attn, enc.spans)


def citation_sources(segments: list[Segment], chunks: list[dict]) -> list[dict]:
    """CiteRec evidence records for a cite-style prompt: [E<j>] -> rendered chunk text (premise for NLI)."""
    out = []
    for j, c in enumerate(chunks, 1):
        text = "".join(s.text for s in segments if s.chunk == c["cid"] and s.text)
        prefix = f"[E{j}] ({c['modality']}):"
        assert text.startswith(prefix), (c["cid"], text[:40])
        out.append({"citation_id": f"E{j}", "source_id": c["cid"], "modality": c["modality"],
                    "rendered_text": text[len(prefix):].strip()})
    return out
