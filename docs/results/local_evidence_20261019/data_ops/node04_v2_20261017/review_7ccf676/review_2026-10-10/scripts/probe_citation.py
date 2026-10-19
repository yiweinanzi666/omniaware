"""Probe citation-support code (Faith / AP / Inv in attribution.py; CiteRec premises in requests.py).
The NLI model is NOT loaded: `entails` is a stub, so only the alias shortcut and parsing are exercised.
"""
import sys
from pathlib import Path

REPO = Path(r"C:\Users\22688\Desktop\www_code\omniaware")
sys.path.insert(0, str(REPO / "src"))
from omniaware.eval.citation import attribution as at  # noqa: E402
from omniaware.eval.citation import requests as rq  # noqa: E402

never = lambda premise, hyp: False  # the NLI checker says "not entailed" for every pair

print("== 1. alias shortcut credits a contradicted citation (entails always False) ==")
src = [{"rendered_text": "Paris is a city in France."}]
r = at.score_answer("Paris is in Germany [E1].", src, ["Paris"], never)
print("answer 'Paris is in Germany [E1].' gold ['Paris'] ->", r["support"], "faith", r["faith"])

print("\n== 2. short gold 'no' matches an unrelated chunk ==")
src = [{"rendered_text": "There is no evidence for this claim."}]
r = at.score_answer("No, the claim is false [E1].", src, ["no"], never)
print("answer 'No, the claim is false [E1].' gold ['no'] ->", r["support"], "faith", r["faith"])

print("\n== 3. multiple golds (OK-VQA stores all 10 annotator strings) widen the shortcut ==")
src = [{"rendered_text": "The sign on the bus was white and red."}]
golds = ["white", "black", "gray", "silver", "blue", "green", "red", "yellow", "brown", "orange"]
r = at.score_answer("The bus is red [E1].", src, golds, never)
print("answer 'The bus is red [E1].' ->", r["support"], "faith", r["faith"])

print("\n== 4. invalid-marker regex counts ordinary text as invalid citations ==")
for text in ["The E5 model improved accuracy [E1].", "Figure E2 shows the result [E1].", "Using e5 embeddings [E1]."]:
    c = at.parse_citations(text, 1)
    print(f"{text!r:45s} -> valid={c['valid']} invalid={c['invalid']} markers={c['markers']}")

print("\n== 5. sentence splitter (regex, used by Faith) vs NLTK Punkt (used by CiteRec) ==")
print("regex statements('Dr. Smith said X [E1]. Y [E2].') ->", at.statements("Dr. Smith said X [E1]. Y [E2]."))

print("\n== 6. CiteRec premise is the union of ALL cited chunks for every statement ==")
row = dict(qid="q", variant="eab", raw_answer="Paris is in France [E1]. Berlin is in Germany [E2].",
           generation_identity={}, raw_token_ids=[], statements=[], evidence=[
               {"citation_id": "E1", "source_id": "s1", "rendered_text": "Paris is in France."},
               {"citation_id": "E2", "source_id": "s2", "rendered_text": "Berlin is in Germany."}])
row["statements"] = [{"text": "Paris is in France.", "citation_ids": ["E1"]},
                     {"text": "Berlin is in Germany.", "citation_ids": ["E2"]}]
for req in rq.requests_for_answer(row):
    print(f"statement {req['statement_index']} cites {req['hypothesis']!r:28s} premise sent to NLI = {req['premise']!r}")
