import sys, re, importlib.util
from types import SimpleNamespace
from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
spec = importlib.util.spec_from_file_location("ov", r"C:\Users\22688\Desktop\www_code\omniaware\third_party\vqa_eval\vqaEval_py3.py")
ov = importlib.util.module_from_spec(spec); spec.loader.exec_module(ov)
ev = ov.VQAEval(SimpleNamespace(qa={}, getQuesIds=lambda: []), SimpleNamespace(qa={}), n=10)
text = "a." * 40 + "x"
out = ev.periodStrip.sub("", text, re.UNICODE)
print("periodStrip.sub(..., re.UNICODE) removed", text.count(".") - out.count("."), "of", text.count("."), "periods (count arg = re.UNICODE =", re.UNICODE, ")")
def vqa(pred, answers):
    q = "1"
    truth = SimpleNamespace(qa={q: {"question_type": "o", "answer_type": "o", "answers": answers}}, getQuesIds=lambda: [q])
    e = ov.VQAEval(truth, SimpleNamespace(qa={q: {"answer": pred}}), n=10)
    with redirect_stdout(StringIO()): e.evaluate([q])
    return e.evalQA[q]
# two annotators with byte-identical dicts (no answer_id) are both excluded from the 10-choose-9 average
same = [{"answer": "car"} for _ in range(2)] + [{"answer": "truck", "answer_id": i} for i in range(8)]
distinct = [{"answer": "car", "answer_id": i} for i in range(2)] + [{"answer": "truck", "answer_id": i} for i in range(8, 16)]
print("identical-dict annotators, pred 'car':", vqa("car", same), " vs distinct ids:", vqa("car", distinct))
