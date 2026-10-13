"""Secondary metric of v2: semantic accuracy (``semantic_acc``), pre-registered in docs/DEV_DECISIONS.md.

The official metrics stay primary: OK-VQA VQA accuracy and WebQA acc_approx decide every selection, gate and check.
``semantic_acc`` is reported beside them for every arm in every table, whatever its value; it credits a correct
answer that the official string rules miss (a paraphrase, a synonym) and gives no credit for a wrong one.

Definition. A prediction is correct (100) when the pinned NLI checker gives an entailment probability of at least
ENTAILMENT_THRESHOLD for

    premise    "Question: <question>\\nAnswer: <prediction>"
    hypothesis "Question: <question>\\nAnswer: <reference>"

for at least one reference, and 0 otherwise. An empty prediction scores 0 without a call. References: WebQA, the one
reference sentence of the question; OK-VQA, the distinct answers (lower-cased, stripped) given by at least
MIN_ANNOTATORS of the 10 annotators, the count at which VQA accuracy gives full credit, or, when no answer reaches it,
the answers with the highest count. A rare answer of one annotator is therefore never a reference, unlike the
withdrawn loose scorer ("any annotator substring").

The checker is the citation-support checker of eval/citation (MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
at the manifest revision), verified file by file against checker_asset_manifest.json, in float32, without
truncation: a pair longer than the model's window is an error. Its agreement with human labels is measured on a
blind, stratified sample of development answers (scripts/analysis/semantic_audit.py) and reported; the audit does not
change which metric is primary or whether this one is reported.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .. import paths

NAME = "semantic_acc"
ENTAILMENT_THRESHOLD = 0.5
MIN_ANNOTATORS = 3
BATCH_SIZE = 16
FAMILIES = ("webqa", "okvqa")
MANIFEST = Path(__file__).resolve().parent / "citation" / "checker_asset_manifest.json"


def statement(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer}"


def okvqa_references(annotation: dict) -> list[str]:
    """Distinct annotator answers given by at least MIN_ANNOTATORS annotators, else those with the highest count."""
    counts = Counter(str(a["answer"]).strip().lower() for a in annotation["answers"] if str(a["answer"]).strip())
    if not counts:
        raise ValueError(f"question {annotation.get('question_id')}: no annotator answer")
    strong = sorted(answer for answer, n in counts.items() if n >= MIN_ANNOTATORS)
    if strong:
        return strong
    top = max(counts.values())
    return sorted(answer for answer, n in counts.items() if n == top)


def checker_identity() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"model": manifest["model"], "revision": manifest["revision"],
            "asset_manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            "threshold": ENTAILMENT_THRESHOLD, "min_annotators": MIN_ANNOTATORS, "dtype": "float32",
            "truncation": False, "batch_size": BATCH_SIZE}


class NliChecker:
    """Entailment probabilities from the pinned checker. Loads lazily; every file is checked against the manifest."""

    def __init__(self, model_dir: Path | None = None, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model_dir = Path(model_dir or paths.NLI_CHECKER)
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for record in manifest["files"]:
            with (model_dir / record["file"]).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]:
                    raise ValueError(f"{model_dir / record['file']} differs from the checker manifest")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), local_files_only=True, dtype=torch.float32).to(self.device).eval()
        if self.model.config.id2label != {0: "entailment", 1: "neutral", 2: "contradiction"}:
            raise ValueError(f"unexpected checker labels {self.model.config.id2label}")
        self.window = int(self.tokenizer.model_max_length)

    def entailment(self, pairs: list[tuple[str, str]]) -> list[float]:
        import torch
        out = []
        for start in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[start:start + BATCH_SIZE]
            enc = self.tokenizer([p for p, _ in batch], [h for _, h in batch], padding=True, truncation=False,
                                 return_tensors="pt")
            if enc["input_ids"].shape[1] > self.window:
                raise ValueError(f"an NLI pair has {enc['input_ids'].shape[1]} tokens, more than the checker's "
                                 f"{self.window}; it is not truncated")
            with torch.no_grad():
                logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits.float()
            out.extend(torch.softmax(logits, dim=-1)[:, 0].cpu().tolist())
        return out


def load_checker(device: str | None = None) -> NliChecker:
    return NliChecker(device=device)


def score_semantic(family: str, preds: dict[str, dict[str, str]], qids: list[str], questions: dict[str, str],
                   references: dict[str, list[str]], checker) -> dict[str, dict[str, float]]:
    """``{arm: {qid: 0.0 or 100.0}}``. ``preds`` are the parsed answers that the official metric scored;
    ``references`` the reference answers of each question (webqa: the reference sentence; okvqa: okvqa_references).
    Identical (premise, hypothesis) pairs are scored once."""
    if family not in FAMILIES:
        raise ValueError(f"{NAME} is defined for {FAMILIES}, not {family!r}")
    pairs, index = [], {}
    for arm_answers in preds.values():
        for q in qids:
            answer = (arm_answers[q] or "").strip()
            if not answer:
                continue
            if not references.get(q):
                raise ValueError(f"{q}: no reference answer for {NAME}")
            for ref in references[q]:
                pair = (statement(questions[q], answer), statement(questions[q], ref))
                if pair not in index:
                    index[pair] = len(pairs)
                    pairs.append(pair)
    probs = checker.entailment(pairs) if pairs else []
    if len(probs) != len(pairs):
        raise ValueError("the checker returned a different number of probabilities than pairs")
    out = {}
    for arm, arm_answers in preds.items():
        cells = {}
        for q in qids:
            answer = (arm_answers[q] or "").strip()
            hit = bool(answer) and any(
                probs[index[(statement(questions[q], answer), statement(questions[q], ref))]] >= ENTAILMENT_THRESHOLD
                for ref in references[q])
            cells[q] = 100.0 if hit else 0.0
        out[arm] = cells
    return out
