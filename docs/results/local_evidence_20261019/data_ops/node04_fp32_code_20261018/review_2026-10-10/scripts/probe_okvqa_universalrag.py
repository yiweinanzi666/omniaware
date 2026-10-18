"""Adversarial probes for OK-VQA (official vqaEval_py3.py, unmodified) and UniversalRAG metrics.

Run with: python -B probe_okvqa_universalrag.py   (repo is read only; no files written there)
"""
import importlib.util
import sys
from types import SimpleNamespace
from contextlib import redirect_stdout
from io import StringIO
from copy import deepcopy

REPO = r"C:\Users\22688\Desktop\www_code\omniaware"
sys.path.insert(0, REPO + r"\src")

spec = importlib.util.spec_from_file_location("official_vqa", REPO + r"\third_party\vqa_eval\vqaEval_py3.py")
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)


def vqa(pred, golds):
    """Same construction as score_official.score_okvqa for a single question."""
    qid = "1"
    truth_qa = {qid: {"question_type": "other", "answer_type": "other",
                      "answers": [{"answer": g, "answer_id": i + 1} for i, g in enumerate(golds)]}}
    truth = SimpleNamespace(qa=deepcopy(truth_qa), getQuesIds=lambda: [qid])
    prediction = SimpleNamespace(qa={qid: {"answer": pred}})
    ev = official.VQAEval(truth, prediction, n=10)
    with redirect_stdout(StringIO()):
        ev.evaluate([qid])
    return ev.evalQA[qid]


print("== OK-VQA official evaluator probes (score in 0-100) ==")
cases = [
    ("unanimous gold 'Dog', pred 'dog'", "dog", ["Dog"] * 10),
    ("unanimous gold 'dog', pred 'dog.'", "dog.", ["dog"] * 10),
    ("unanimous gold '2', pred 'two'", "two", ["2"] * 10),
    ("unanimous gold '2', pred '2'", "2", ["2"] * 10),
    ("9/10 gold 'dog' + 'dogs', pred 'dog.'", "dog.", ["dog"] * 9 + ["dogs"]),
    ("9/10 gold '2' + 'two', pred 'two'", "two", ["2"] * 9 + ["two"]),
    ("gold 'cat' x10, pred 'not a cat'", "not a cat", ["cat"] * 10),
    ("gold 'cat' x10, pred 'the cat is black'", "the cat is black", ["cat"] * 10),
    ("gold 'cat' x10, pred 'cat' ", "cat", ["cat"] * 10),
    ("gold 'cat' x10, pred ''", "", ["cat"] * 10),
    ("gold 'cat' x10, pred None", None, ["cat"] * 10),
    ("gold 'New York' x10, pred 'new york city'", "new york city", ["New York"] * 10),
    ("9/10 gold 'New York', pred 'new york'", "new york", ["New York"] * 9 + ["NYC"]),
]
for label, pred, golds in cases:
    try:
        print(f"{label:45s} -> {vqa(pred, golds)}")
    except Exception as exc:  # noqa: BLE001
        print(f"{label:45s} -> EXC {type(exc).__name__}: {exc}")

print()
print("== UniversalRAG metric probes ==")
from omniaware.eval import universalrag_metrics as U  # noqa: E402

print("em('', 'the')            =", U.exact_match("", "the"))
print("f1('', 'a')              =", U.f1("", "a"))
print("em('In 1960', '1960')    =", U.exact_match("In 1960", "1960"), " f1 =", round(U.f1("In 1960", "1960"), 4))
print("rouge_l('', 'x')         =", U.rouge_l("", "x"))
print("rouge_l('x', '')         =", U.rouge_l("x", ""))
print("rouge_l(long-with-gold)  =", round(U.rouge_l("the answer is 1960 and many other words here", "1960"), 4))
for name, pred, golds in [("yes-no hedge", "yes and no", ["no"])]:
    pass
try:
    U.score("hybridqa", {"s": {"q": ""}}, ["q"], {"q": {"golds": []}})
    print("empty golds: no error")
except Exception as exc:  # noqa: BLE001
    print("empty golds -> EXC", type(exc).__name__, exc)
