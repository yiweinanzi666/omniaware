import sys, importlib.util
from types import SimpleNamespace
from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
sys.path.insert(0, r"C:\Users\22688\Desktop\www_code\omniaware\src")
from omniaware.eval import universalrag_metrics as U
spec = importlib.util.spec_from_file_location("ov", r"C:\Users\22688\Desktop\www_code\omniaware\third_party\vqa_eval\vqaEval_py3.py")
ov = importlib.util.module_from_spec(spec); spec.loader.exec_module(ov)
def vqa(pred, golds):
    q = "1"
    truth = SimpleNamespace(qa={q: {"question_type": "o", "answer_type": "o", "answers": [{"answer": g, "answer_id": i+1} for i, g in enumerate(golds)]}}, getQuesIds=lambda: [q])
    ev = ov.VQAEval(SimpleNamespace(qa=deepcopy(truth.qa), getQuesIds=lambda: [q]), SimpleNamespace(qa={q: {"answer": pred}}), n=10)
    with redirect_stdout(StringIO()): ev.evaluate([q])
    return ev.evalQA[q]
for p, g in [("Paris。", "Paris"), ("Paris", "Paris"), ("café", "cafe"), ("Paris.", "Paris"), ("Paris’s", "Paris's")]:
    print(f"EM {p!r} vs {g!r}: hybrid_em={U.exact_match(p, g)}  hybrid_f1={round(U.f1(p, g),3)}  rougeL={round(U.rouge_l(p, g),3)}")
print("VQA multi-gold 'Paris。' vs ['paris']*9+['Paris']:", vqa("Paris。", ["paris"]*9+["Paris"]))
print("VQA multi-gold 'Paris.' vs ['paris']*9+['Paris']:", vqa("Paris.", ["paris"]*9+["Paris"]))
print("VQA multi-gold 'café' vs ['cafe']*9+['café']:", vqa("café", ["cafe"]*9+["café"]))
