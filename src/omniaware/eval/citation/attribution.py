"""Paper citation metrics (Appendix `Citation support`): Faith, attribution precision (AP), invalid citations.

    python -m omniaware.eval.citation.attribution runs/<tag>/<dataset>.jsonl [--arms static eab:...]
        [--intervention distractors] [--device cuda] [--output runs/<tag>/<dataset>.attribution.json]

For answer y with valid cited chunks C(x) and binary support S(y, e):
    Supp(x) = sum_{e in C(x)} S(y, e);  Faith = 1[Supp > 0] (all answers);  AP = Supp / |C| (answers citing >= 1).
Missing, malformed and out-of-prompt citations get no credit. Inv = invalid markers / all markers (malformed or
out-of-prompt [E<j>]); `answers_with_invalid` is the share of answers containing at least one.
S(y, e) = 1 when alias matching succeeds -- a gold alias occurs in the answer statement and in the chunk -- else
sentence-level NLI: some answer sentence citing e is entailed by the chunk text (DeBERTa-v3-large MNLI checker of
checker_asset_manifest.json, argmax decision). Only arms decoded on the cited [E<j>] input are scored (static and
the bias variants; with --intervention, those arms on the perturbed input).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ...interventions import normalize

VALID = re.compile(r"\[E(\d+)\]")
# anything that looks like an attempt at an evidence identifier but is not exactly [E<j>]
LOOSE = re.compile(r"\[\s*[Ee]?\s*\d+\s*\]|\(\s*[Ee]\s*\d+\s*\)|(?<![\[\w])[Ee]\d+(?![\]\w])|\[\s*[Ee]\s+\d+\s*\]"
                   r"|\[e\d+\]")
SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def parse_citations(answer: str, n_sources: int) -> dict:
    valid, invalid = [], 0
    for m in VALID.finditer(answer):
        j = int(m.group(1))
        if 1 <= j <= n_sources:
            valid.append(j)
        else:
            invalid += 1
    stripped = VALID.sub(" ", answer)
    invalid += len(LOOSE.findall(stripped))
    return {"valid": valid, "invalid": invalid, "markers": len(valid) + invalid}


def statements(answer: str) -> list[tuple[str, set[int]]]:
    """Answer sentences with the [E<j>] identifiers each one carries (citation-free text for NLI)."""
    out = []
    for sent in SENTENCE.split(answer):
        cites = {int(j) for j in VALID.findall(sent)}
        text = " ".join(LOOSE.sub(" ", VALID.sub(" ", sent)).split())
        if text:
            out.append((text, cites))
    return out


def alias_support(answer_text: str, chunk_text: str, golds: list[str]) -> bool:
    a, c = normalize(answer_text), normalize(chunk_text)
    for g in golds:
        g = normalize(g or "")
        pattern = rf"(?<!\w){re.escape(g)}(?!\w)"
        if g and re.search(pattern, a) and re.search(pattern, c):
            return True
    return False


def score_answer(answer: str, sources: list[dict], golds: list[str], entails) -> dict:
    cites = parse_citations(answer, len(sources))
    cited = sorted(set(cites["valid"]))
    sents = statements(answer)
    whole = " ".join(s for s, _ in sents)
    support = {}
    for j in cited:
        premise = sources[j - 1]["rendered_text"]
        hyps = [s for s, cs in sents if j in cs] or [whole]
        support[j] = bool(alias_support(" ".join(hyps), premise, golds)
                          or any(entails(premise, h) for h in hyps if h))
    supp = sum(support.values())
    return {"cited": cited, "support": {f"E{j}": v for j, v in support.items()}, "faith": float(supp > 0),
            "ap": supp / len(cited) if cited else None, "invalid": cites["invalid"], "markers": cites["markers"]}


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    cited = [r for r in rows if r["ap"] is not None]
    markers = sum(r["markers"] for r in rows)
    return {"answers": n, "Faith": 100 * sum(r["faith"] for r in rows) / n if n else None,
            "AP": 100 * sum(r["ap"] for r in cited) / len(cited) if cited else None,
            "coverage": 100 * len(cited) / n if n else None,
            "Inv": 100 * sum(r["invalid"] for r in rows) / markers if markers else 0.0,
            "answers_with_invalid": 100 * sum(r["invalid"] > 0 for r in rows) / n if n else None}


def score_records(records: list[dict], arms: list[str], entails, intervention: str | None = None) -> dict:
    per_arm = {arm: [] for arm in arms}
    for r in records:
        if intervention:
            entry = (r.get("interventions") or {}).get(intervention)
            if not entry or "citation_sources" not in entry:
                continue
            sources, preds = entry["citation_sources"], entry["preds"]
        else:
            sources, preds = r["citation_sources"], r["preds"]
        for arm in arms:
            row = score_answer(preds[arm] or "", sources, r["golds"], entails)
            per_arm[arm].append({"qid": r["qid"], **row})
    return {"summary": {arm: aggregate(rows) for arm, rows in per_arm.items()}, "details": per_arm}


class DebertaChecker:
    """The pinned DeBERTa-v3-large MNLI checker; entailment = argmax label 0 (manifest id2label)."""

    def __init__(self, device: str = "cpu", verify: bool = True):
        import hashlib
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from .run_nli import MODEL
        manifest = json.loads(Path(__file__).with_name("checker_asset_manifest.json").read_text())
        if verify:
            for rec in manifest["files"]:
                with (MODEL / rec["file"]).open("rb") as fh:
                    assert hashlib.file_digest(fh, "sha256").hexdigest() == rec["sha256"], rec["file"]
        self.entail_id = int([k for k, v in manifest["id2label"].items() if v.lower() == "entailment"][0])
        self.torch, self.device = torch, device
        self.tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(MODEL), local_files_only=True).to(device).eval()
        self.cache = {}

    def __call__(self, premise: str, hypothesis: str) -> bool:
        key = (premise, hypothesis)
        if key not in self.cache:
            with self.torch.no_grad():
                enc = self.tok(premise, hypothesis, return_tensors="pt", truncation="only_first", max_length=512)
                logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits[0]
            self.cache[key] = int(logits.argmax()) == self.entail_id
        return self.cache[key]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", type=Path)
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--intervention", default=None, help="score the arms on this intervention's perturbed input")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    config = json.loads(args.predictions.with_name(args.predictions.stem + "_config.json").read_text())
    if config.get("answer_style") != "cite":
        raise SystemExit("attribution metrics need a run with --answer-style cite")
    shared = ["static"] + [v["name"] for v in config["variants"]]
    arms = args.arms or shared
    if not set(arms) <= set(shared):
        raise SystemExit(f"only arms decoded on the cited input can be scored: {shared}")
    records = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    result = score_records(records, arms, DebertaChecker(args.device), args.intervention)
    out = args.output or args.predictions.with_suffix(
        f".attribution{'.' + args.intervention if args.intervention else ''}.json")
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps(result["summary"], indent=1))


if __name__ == "__main__":
    main()
