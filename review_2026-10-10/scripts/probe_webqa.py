"""Probe the WebQA acc_approx metric built exactly as webqa_acc.load_official_metric does,
except that spaCy and word2number are NOT installed in this environment:
  - nlp(text) is replaced by an identity lemmatizer (token.lemma_ == token.text), so
    lemma-dependent matches (e.g. 'cats' vs 'cat') are NOT reproduced here;
  - w2n.word_to_num is replaced by a small 0-20 / tens table.
Numbers/colors/yes-no/recall paths that do not depend on lemmas are exercised as published.
"""
import re
import string
import sys
from collections import Counter
from types import SimpleNamespace
from pathlib import Path

REPO = Path(r"C:\Users\22688\Desktop\www_code\omniaware")
sys.path.insert(0, str(REPO / "src"))
from omniaware.eval import webqa_acc as W  # noqa: E402

SMALL = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}


class _W2N:
    @staticmethod
    def word_to_num(word):
        if word in SMALL:
            return SMALL[word]
        raise ValueError(word)


class _Tok:
    def __init__(self, t):
        self.text = t
        self.lemma_ = t


namespace = dict(re=re, string=string, Counter=Counter, w2n=_W2N,
                 nlp=lambda text: [_Tok(t) for t in text.split()] if text else [])
src = {f: (REPO / "third_party" / "webqa" / f).read_text(encoding="utf-8") for f in W.OFFICIAL_FILES}
tree = W._selected_ast(src["eval_1022.py"], function_names=("detectNum", "toNum", "normalize_text"),
                       constant_names=("color_set", "shape_set", "yesno_set"))
exec(compile(tree, "eval_1022:selected", "exec"), namespace)
blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", src["eval_webqa.md"], re.DOTALL)
code = next(b for b in blocks if "def webqa_metrics_approx" in b)
tree = W._selected_ast(code, function_names=("_webqa_acc_approx", "webqa_metrics_approx"))
exec(compile(tree, "eval_webqa.md:published", "exec"), namespace)
metric = namespace["webqa_metrics_approx"]

cases = [
    ("Paris, Berlin, Tokyo, Rome, Lima and Paris", "Paris", "text"),
    ("It is not Paris", "Paris", "text"),
    ("The capital of France is a city with two million people in Europe and Paris", "Paris", "text"),
    ("I do not know", "Paris", "text"),
    ("the cat", "cat", "text"),
    ("", "", "text"),
    ("Paris", "", "text"),
    ("two", "2", "number"),
    ("2", "two", "number"),
    ("3 or 2", "2", "number"),
    ("yes", "yes", "YesNo"),
    ("no", "yes", "YesNo"),
    ("yes and no", "no", "YesNo"),
    ("No, not yes", "no", "YesNo"),
    ("red and blue", "blue", "color"),
    ("red", "blue", "color"),
    ("blue, red, green, black, white, orange, pink", "blue", "color"),
    ("", "red", "color"),
]
print("prediction | reference | category -> acc_approx")
for pred, ref, cat in cases:
    print(f"{pred!r:70s} | {ref!r:10s} | {cat:7s} -> {metric(pred, ref, Qcate=cat)['acc_approx']:.4f}")

# Verbose-answer inflation under recall: gold 'Paris' inside a 60-word answer.
filler = " ".join(["word"] * 60)
print("\nverbose 60-word answer containing gold, text category:",
      metric(f"Paris {filler}", "Paris", Qcate="text")["acc_approx"])
print("same answer under a 'color' F1 (not recall):",
      round(metric(f"blue red green {filler}", "blue", Qcate="color")["acc_approx"], 4))
