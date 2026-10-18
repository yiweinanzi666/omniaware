"""Paper citation metrics (Appendix `Citation support`): Faith, attribution precision (AP), invalid citations.

    python -m omniaware.eval.citation.attribution runs/<tag>/<dataset>.jsonl [--arms static gated:static ...]
        [--intervention distractors] [--device cuda] [--output runs/<tag>/<dataset>.attribution.json]

For answer y with valid cited chunks C(x) and binary support S(y, e):
    Supp(x) = sum_{e in C(x)} S(y, e);  Faith = 1[Supp > 0] (all answers);  AP = Supp / |C| (answers citing >= 1).
Missing, malformed and out-of-prompt citations get no credit. Inv = invalid markers / all markers (malformed or
out-of-prompt bracketed [E<j>]; a bare "E5" in ordinary text is not a marker); `answers_with_invalid` is the share
of answers containing at least one.

S(y, e) = S_alias or S_nli (the union, the headline numbers Faith / AP):
  S_alias  a gold alias occurs in the answer statements citing e and in the chunk text;
  S_nli    some answer statement citing e is entailed by the chunk text (DeBERTa-v3-large MNLI checker of
           checker_asset_manifest.json; argmax decision; premises longer than the window are split, see
           DebertaChecker). Both are also reported alone (Faith_alias, AP_alias, Faith_nli, AP_nli).

Arms: static, the bias variants, replication (when decoded in the run) and gated:<arm> (the arm's answer when
retrieval fired; questions without retrieval are not applicable: they leave the denominators, and the
share of applicable questions is reported as gate_coverage). With --intervention, the arms on the perturbed input.

Sentences are split by one splitter for Faith and CiteRec (`sentence_spans`): NLTK Punkt when NLTK and its
English Punkt data load, otherwise the regex SENTENCE. The splitter name is written into every output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

from ... import paths
from ...interventions import normalize

VALID = re.compile(r"\[E(\d+)\]")
# bracketed attempts at an evidence identifier that are not exactly [E<j>] ("[3]", "[e2]", "[E 2]", "[E0x]");
# a bare "E5" in ordinary text is not a citation marker
LOOSE = re.compile(r"\[\s*[Ee]?\s*\d+\s*\]")
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


@lru_cache(maxsize=1)
def _punkt():
    """The pinned English Punkt tokenizer (NLTK 3.9.2 and the frozen punkt_tab tables installed by
    ``scripts/setup/download_models.py --only nltk`` under <assets>/cache/e5_citation_support), or None when those
    assets are absent; the regex fallback is then named in every output (``splitter_name``).  Assets that are
    present but differ from their manifest raise instead of silently changing the splitter."""
    cache = paths.CACHE / "e5_citation_support"
    manifest = cache / "sentence_splitter_assets.json"
    if not manifest.exists():
        return None
    record = json.loads(manifest.read_text(encoding="utf8"))
    for item in record["files"]:
        if hashlib.sha256((paths.ASSET_ROOT / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"{item['path']} differs from the sha256 frozen in {manifest}")
    sys.path.insert(0, str(cache / "nltk_runtime"))
    import nltk
    if nltk.__version__ != record["nltk_version"]:
        raise RuntimeError(f"nltk {nltk.__version__} loaded, the citation protocol pins {record['nltk_version']}")
    nltk.data.path = [str(cache / "nltk_data")]
    from nltk.tokenize import PunktTokenizer
    return PunktTokenizer("english")


def splitter_name() -> str:
    """Name of the sentence splitter in use, recorded in every output that splits sentences."""
    return "nltk-punkt-english" if _punkt() is not None else "regex(?<=[.!?])\\s+|\\n+"


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """[begin, end) of each sentence: the one splitter shared by Faith/AP (statements) and CiteRec (intake)."""
    tok = _punkt()
    if tok is not None:
        return list(tok.span_tokenize(text))
    spans, start = [], 0
    for m in SENTENCE.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return spans


def statements(answer: str) -> list[tuple[str, set[int]]]:
    """Answer sentences with the [E<j>] identifiers each one carries (citation-free text for NLI)."""
    out = []
    for begin, end in sentence_spans(answer):
        sent = answer[begin:end]
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


def _faith_ap(support: dict[int, bool], cited: list[int]) -> tuple[float, float | None]:
    supp = sum(support.values())
    return float(supp > 0), (supp / len(cited) if cited else None)


def score_answer(answer: str, sources: list[dict], golds: list[str], entails) -> dict:
    cites = parse_citations(answer, len(sources))
    cited = sorted(set(cites["valid"]))
    sents = statements(answer)
    whole = " ".join(s for s, _ in sents)
    alias, nli = {}, {}
    for j in cited:
        premise = sources[j - 1]["rendered_text"]
        hyps = [s for s, cs in sents if j in cs] or [whole]
        alias[j] = alias_support(" ".join(hyps), premise, golds)
        nli[j] = any(entails(premise, h) for h in hyps if h)
    union = {j: alias[j] or nli[j] for j in cited}
    faith, ap = _faith_ap(union, cited)
    faith_a, ap_a = _faith_ap(alias, cited)
    faith_n, ap_n = _faith_ap(nli, cited)
    return {"cited": cited, "support": {f"E{j}": v for j, v in union.items()},
            "support_alias": {f"E{j}": v for j, v in alias.items()},
            "support_nli": {f"E{j}": v for j, v in nli.items()},
            "faith": faith, "ap": ap, "faith_alias": faith_a, "ap_alias": ap_a, "faith_nli": faith_n, "ap_nli": ap_n,
            "invalid": cites["invalid"], "markers": cites["markers"]}


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    cited = [r for r in rows if r["ap"] is not None]
    markers = sum(r["markers"] for r in rows)

    def mean_pct(values):
        return 100 * sum(values) / len(values) if values else None

    out = {"answers": n, "Faith": 100 * sum(r["faith"] for r in rows) / n if n else None,
           "AP": mean_pct([r["ap"] for r in cited]),
           "coverage": 100 * len(cited) / n if n else None,
           "Inv": 100 * sum(r["invalid"] for r in rows) / markers if markers else 0.0,
           "answers_with_invalid": 100 * sum(r["invalid"] > 0 for r in rows) / n if n else None}
    for source in ("alias", "nli"):     # each support source alone; Faith / AP above use their union
        out[f"Faith_{source}"] = 100 * sum(r[f"faith_{source}"] for r in rows) / n if n else None
        out[f"AP_{source}"] = mean_pct([r[f"ap_{source}"] for r in cited])
    return out


def arm_base(arm: str) -> str:
    """The decoded arm behind a scored arm name: gated:<arm> answers with <arm> when retrieval fired."""
    return arm[len("gated:"):] if arm.startswith("gated:") else arm


def score_records(records: list[dict], arms: list[str], entails, intervention: str | None = None) -> dict:
    per_arm = {arm: [] for arm in arms}
    considered = 0
    for r in records:
        if intervention:
            owner = (r.get("interventions") or {}).get(intervention)
            if not owner or "citation_sources" not in owner:    # ineligible or failed perturbation
                continue
        else:
            owner = r
        considered += 1
        for arm in arms:
            if arm.startswith("gated:") and not r["retrieve"]:  # no retrieval: the arm has no evidence to cite
                continue
            base = arm_base(arm)
            sources = owner.get("replication_citation_sources") if base == "replication" else owner.get("citation_sources")
            if sources is None:
                raise ValueError(f"{r['qid']}: no citation sources recorded for {arm}")
            row = score_answer(owner["preds"][base] or "", sources, r["golds"], entails)
            per_arm[arm].append({"qid": r["qid"], **row})
    summary = {}
    for arm, rows in per_arm.items():
        summary[arm] = dict(aggregate(rows), questions=considered, applicable=len(rows),
                            gate_coverage=100 * len(rows) / considered if considered else None)
    return {"summary": summary, "details": per_arm, "sentence_splitter": splitter_name()}


def window_decision(windows: list[tuple[float, bool]]) -> bool:
    """Premise windows -> one decision: the window with the highest entailment probability decides, entailed
    when that window's argmax is the entailment label. A premise that fits one window reduces to argmax."""
    return max(windows, key=lambda w: w[0])[1]


def premise_windows(premise_ids: list[int], hypothesis_ids: list[int], max_length: int = 512,
                    special: int = 3) -> list[list[int]]:
    """Overlapping windows of premise tokens (stride half a window) so that [CLS] premise [SEP] hypothesis [SEP]
    fits ``max_length``. Every premise token lies in some window; a premise that fits is a single window."""
    budget = max_length - len(hypothesis_ids) - special
    if budget < 1:
        raise ValueError("hypothesis too long for one DeBERTa window")
    if len(premise_ids) <= budget:
        return [list(premise_ids)]
    stride = max(1, budget // 2)
    windows, start = [], 0
    while True:
        windows.append(premise_ids[start:start + budget])
        if start + budget >= len(premise_ids):
            return windows
        start += stride


class DebertaChecker:
    """The pinned DeBERTa-v3-large MNLI checker; entailment = argmax label 0 (manifest id2label).

    No truncation: a premise longer than the 512-token window is split into overlapping windows
    (premise_windows), each scored with the hypothesis, and window_decision picks the answer."""

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
            p_ids = self.tok.encode(premise, add_special_tokens=False)
            h_ids = self.tok.encode(hypothesis, add_special_tokens=False)
            windows = []
            with self.torch.no_grad():
                for win in premise_windows(p_ids, h_ids):
                    ids = self.tok.build_inputs_with_special_tokens(win, h_ids)
                    types = self.tok.create_token_type_ids_from_sequences(win, h_ids)
                    logits = self.model(
                        input_ids=self.torch.tensor([ids], device=self.device),
                        token_type_ids=self.torch.tensor([types], device=self.device),
                        attention_mask=self.torch.ones(1, len(ids), dtype=self.torch.long, device=self.device),
                    ).logits[0].float()
                    probs = logits.softmax(-1)
                    windows.append((float(probs[self.entail_id]), int(logits.argmax()) == self.entail_id))
            self.cache[key] = window_decision(windows)
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
    decoded = shared + [f"gated:{arm}" for arm in shared]
    if config.get("replication_gamma") is not None:
        decoded += ["replication", "gated:replication"]
    # Default: every decoded arm, so the gated rows the ablation table reports (S-20) are always scored.
    arms = args.arms or decoded
    if not set(arms) <= set(decoded):
        raise SystemExit(f"only arms decoded in this run can be scored: {decoded}")
    records = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    result = score_records(records, arms, DebertaChecker(args.device), args.intervention)
    out = args.output or args.predictions.with_suffix(
        f".attribution{'.' + args.intervention if args.intervention else ''}.json")
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps(result["summary"], indent=1))


if __name__ == "__main__":
    main()
