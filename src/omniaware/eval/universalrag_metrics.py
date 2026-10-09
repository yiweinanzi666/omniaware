"""Metrics of the UniversalRAG suite (Yeo et al., Table 1), which its repository does not ship.

    HybridQA   EM and F1 (SQuAD answer normalisation, max over answer + aliases)
    InfoSeek   Acc: normalised exact match against answer + aliases
    VideoRAG   ROUGE-L F-measure (rouge_score's default tokenisation, no stemming) and, with
               bertscore=True, BERTScore F1 (bert_score package, its default English model)

All values are per question on a 0-100 scale, like the other official scorers.
"""
from __future__ import annotations

import collections
import re
import string


def normalize_answer(s: str) -> str:
    """SQuAD v1.1 evaluation normalisation."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
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


def score(family: str, preds: dict[str, dict[str, str]], qids: list[str], records: dict, bertscore: bool = False):
    """preds: arm -> qid -> answer; records: qid -> run record (golds). Returns (scores, metrics, extra)."""
    golds = {q: [g for g in records[q]["golds"] if g] for q in qids}
    scores = {arm: {} for arm in preds}
    if family == "hybridqa":
        metrics = ["em", "f1"]
        for arm, answers in preds.items():
            for q in qids:
                scores[arm][q] = {"em": 100 * max(exact_match(answers[q], g) for g in golds[q]),
                                  "f1": 100 * max(f1(answers[q], g) for g in golds[q])}
    elif family == "infoseek":
        metrics = ["accuracy"]
        for arm, answers in preds.items():
            for q in qids:
                scores[arm][q] = {"accuracy": 100 * max(exact_match(answers[q], g) for g in golds[q])}
    elif family == "videorag":
        metrics = ["rouge_l"] + (["bertscore_f1"] if bertscore else [])
        for arm, answers in preds.items():
            for q in qids:
                scores[arm][q] = {"rouge_l": 100 * max(rouge_l(answers[q], g) for g in golds[q])}
        if bertscore:
            from bert_score import score as bert_score
            for arm, answers in preds.items():
                cands = [answers[q] or "" for q in qids]
                refs = [golds[q][0] for q in qids]
                _, _, f = bert_score(cands, refs, lang="en", verbose=False)
                for q, v in zip(qids, f.tolist()):
                    scores[arm][q]["bertscore_f1"] = 100 * v
    else:
        raise ValueError(family)
    return scores, metrics, None
