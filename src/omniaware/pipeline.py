"""Prompt construction, controller (U_omni, gates) and EAB variant definitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
# Answer formats from the Pro6000 runs (pro6000_three_official_20261008/run_refined.py).
JSON_LIST_INSTRUCTION = (
    "Use the provided tables to answer the question. Identify the column being asked for, "
    "then find every row satisfying all conditions. Return values from the requested column, "
    "not from the columns used as conditions. Keep each answer cell as one string, including "
    "names or date ranges within that cell; do not return a whole row. For calculations, return "
    "the calculated value. For yes/no questions use [\"Yes\"] or [\"No\"]. "
    "Return only a valid JSON array of answer strings, with double quotes inside strings escaped. "
    "Do not include explanations."
)
STRICT_LETTER_INSTRUCTION = (
    "These frames appear in chronological order and the subtitles contain spoken content. "
    "Use the visual and subtitle evidence to select the option that best answers the question. "
    "Return exactly one uppercase letter: A, B, C, or D. Do not include explanations."
)
# Citation output format of the CiteRec protocol (docs/citation_support_protocol.md).
CITATION_INSTRUCTION = (
    'Answer the question using the supplied evidence and images. '
    'Write each answer as a complete factual statement on a separate line. '
    'Cite the supplied evidence that supports each statement using its exact [E<number>] identifier. '
    'Use only identifiers present in the supplied evidence. Do not include explanations or a reference list.'
)
TABLE_INSTRUCTION = (
    "Use the provided tables to answer the question. Apply every condition to the matching rows, "
    "then compute any requested count, average, minimum, maximum, or most recent value. "
    "For a question that first looks up a value, use that value to complete the remaining lookup. "
    "Return only the final answer; include every requested item."
)


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
            label = f"[E{j}]" if answer_style == "cite" else f"Document {j}"
            if c["modality"] == "image":
                head = f"{label} (image): {c['title']}\n" if c["title"] else f"{label} (image):\n"
                tail = f"\nCaption: {_trim(c['text'], 60)}\n\n" if c["text"] and c["text"] != c["title"] else "\n\n"
                cap = (image_pixels or {}).get(cid)
                segs += [Segment(text=head, chunk=cid), Segment(image=c["image"], chunk=cid, max_pixels=cap),
                         Segment(text=tail, chunk=cid)]
            elif c["modality"] == "table":
                segs.append(Segment(text=f"{label} (table):\n{c.get('prompt_text', c['text'])}\n\n", chunk=cid))
            else:
                body = f"{c['title']}: {c['text']}" if c["title"] and c["title"] != c["text"] else c["text"]
                segs.append(Segment(text=f"{label} (text): {_trim(body)}\n\n", chunk=cid))
    split = len(segs)
    q = f"Question: {e['question']}\n"
    if e.get("options"):
        q += "\n".join(e["options"]) + "\n"
    instruction = SHORT_INSTRUCTION if answer_style == "short" and not e.get("options") else INSTRUCTION[e["dataset"]]
    if answer_style == "table" and e["dataset"] == "mmqa_table":
        instruction = TABLE_INSTRUCTION
    if answer_style == "json_list":
        if e["dataset"] != "mmqa_table":
            raise ValueError("answer style json_list is defined for mmqa_table")
        instruction = JSON_LIST_INSTRUCTION
    if answer_style == "cite":
        instruction = CITATION_INSTRUCTION
    if answer_style == "strict_letter":
        if e["dataset"] != "videomme":
            raise ValueError("answer style strict_letter is defined for videomme")
        instruction = STRICT_LETTER_INSTRUCTION
    q += instruction + "<|im_end|>\n<|im_start|>assistant\n"
    segs.append(Segment(text=q))
    return segs, split


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
    """EAB family.  rows: 'ans2' paper decode-only (a >= t_ans), 'ans1' EAB-P (also the
    row that predicts the first answer token), 'q' starts at the question.  form: 'pos'
    gamma*U*r (paper), 'ctr' gamma*U*(2r-1), 'fixed' gamma (PASTA-style uniform bias),
    'pos_nou' gamma*r (relevance bias without U scaling, paper ablation)."""

    name: str
    gamma: float
    rows: str = "ans2"
    form: str = "pos"
    gate: bool = True
    layers: str = "all"

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
    return Variant(name, gamma, rows, form, gate, layers)


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
