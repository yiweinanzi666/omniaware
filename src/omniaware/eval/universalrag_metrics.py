"""Metrics of the UniversalRAG suite (Yeo et al., Table 1), which its repository does not ship.

    HybridQA   EM and F1 (SQuAD answer normalisation, max over answer + aliases)
    VideoRAG   ROUGE-L F-measure (rouge_score's default tokenisation, no stemming) and, with
               bertscore=True, BERTScore F1 with the fixed model BERTSCORE_MODEL (max over the golds)

All values are per question on a 0-100 scale, like the other official scorers. VideoRAG-Synth has several
questions per source video, so its bootstrap is clustered by video: the record's ``cluster_id`` (written by the
data loader from the query index videorag_synth_<video>_<n>).
"""
from __future__ import annotations

import collections
import re
import string
import unicodedata

# bert_score's default English model and layer, pinned here; the model name is written to the output.
BERTSCORE_MODEL = "roberta-large"
BERTSCORE_LAYERS = 17

# ASCII punctuation plus the common full-width / CJK marks that NFKC does not map to ASCII.
_PUNCT = set(string.punctuation) | set("，。、；：？！“”‘’（）【】《》〈〉「」『』—–…·～")


def normalize_answer(s: str) -> str:
    """SQuAD v1.1 evaluation normalisation, after NFKC (full-width letters and digits become ASCII) and with
    the full-width / CJK punctuation of _PUNCT removed, so that "Paris。" matches "Paris"."""
    s = unicodedata.normalize("NFKC", s).lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if not p or not g:
        return float(p == g)
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def _rouge_tokens(text: str) -> list[str]:
    # rouge_score.tokenize: lowercase, non-alphanumerics -> spaces
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).split()


def rouge_l(pred: str, gold: str) -> float:
    a, b = _rouge_tokens(gold), _rouge_tokens(pred)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    recall, precision = lcs / len(a), lcs / len(b)
    return 2 * precision * recall / (precision + recall)


def _golds(family: str, records: dict, qids: list[str]) -> dict[str, list[str]]:
    """Non-empty golds per question. A question without a gold, or whose gold normalises to nothing (which would
    let an empty answer score as correct), is an error rather than a silent zero or a silent match."""
    out = {}
    for q in qids:
        golds = [g for g in records[q]["golds"] if g]
        if not golds:
            raise ValueError(f"{q}: no gold answers")
        for g in golds:
            empty = not _rouge_tokens(g) if family == "videorag" else not normalize_answer(g)
            if empty:
                raise ValueError(f"{q}: gold {g!r} is empty after normalisation")
        out[q] = golds
    return out


def _cluster(q: str, record: dict) -> str:
    if record.get("cluster_id") is not None:
        return str(record["cluster_id"])
    m = re.fullmatch(r"videorag_synth_(.+)_(\d+)", str(q))
    if m is None:
        raise ValueError(f"{q}: VideoRAG query id is not videorag_synth_<video>_<n>, and the record has no cluster_id")
    return m.group(1)


def score(family: str, preds: dict[str, dict[str, str]], qids: list[str], records: dict, bertscore: bool = False):
    """preds: arm -> qid -> answer; records: qid -> run record (golds). Returns (scores, metrics, extra)."""
    golds = _golds(family, records, qids)
    scores = {arm: {} for arm in preds}
    extra = None
    if family == "hybridqa":
        metrics = ["em", "f1"]
        for arm, answers in preds.items():
            for q in qids:
                scores[arm][q] = {"em": 100 * max(exact_match(answers[q], g) for g in golds[q]),
                                  "f1": 100 * max(f1(answers[q], g) for g in golds[q])}
    elif family == "videorag":
        metrics = ["rouge_l"] + (["bertscore_f1"] if bertscore else [])
        for arm, answers in preds.items():
            for q in qids:
                scores[arm][q] = {"rouge_l": 100 * max(rouge_l(answers[q], g) for g in golds[q])}
        if bertscore:
            from bert_score import score as bert_score
            # every (question, gold) pair in one call; each question keeps the best-matching gold
            pairs = [(q, g) for q in qids for g in golds[q]]
            for arm, answers in preds.items():
                cands = [answers[q] or "" for q, _ in pairs]
                refs = [g for _, g in pairs]
                _, _, f = bert_score(cands, refs, model_type=BERTSCORE_MODEL, num_layers=BERTSCORE_LAYERS,
                                     verbose=False)
                best = {}
                for (q, _), v in zip(pairs, f.tolist()):
                    best[q] = max(best.get(q, float("-inf")), v)
                for q in qids:
                    scores[arm][q]["bertscore_f1"] = 100 * best[q]
            extra = {"bertscore": {"model_type": BERTSCORE_MODEL, "num_layers": BERTSCORE_LAYERS,
                                   "references": "max over the gold answers of each question"}}
    else:
        raise ValueError(family)
    if family == "videorag":
        extra = {**(extra or {}), "clusters": [_cluster(q, records[q]) for q in qids]}
    return scores, metrics, extra
