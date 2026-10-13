import sys, tempfile, json
from pathlib import Path
sys.path.insert(0, "C:/Users/22688/Desktop/www_code/omniaware")
sys.path.insert(0, "C:/Users/22688/Desktop/www_code/omniaware/src")
from omniaware import data, paths, run
import tests.test_run_smoke as tr

examples = tr._examples(374)            # VideoRAG-Synth size per docs/PROTOCOL.md
data.LOADERS["okvqa"] = lambda **kw: examples   # stand-in loader; the split logic is identical
tmp = Path(tempfile.mkdtemp())
paths.RUNS = tmp / "runs"; paths.LOCAL_CACHE = tmp / "cache"
run.QwenEAB = tr.FakeQwen
run.load_relevance = lambda exs, path: {e["qid"]: {c["cid"]: {"text": 1.0 / int(c["cid"][1:])} for c in e["candidates"]} for e in exs}
# exactly the paper_runs.py call for videorag_synth: heldout calibration, split test, no --n-calib
run.main(["--dataset", "okvqa", "--split", "test", "--calib-source", "heldout", "--tag", "vr",
          "--variants", "eab:g1.2:ans1:pos:gate"])
out = paths.RUNS / "vr" / "okvqa.jsonl"
print("rows written:", len(out.read_text().splitlines()) if out.exists() else "no file")
cfg = json.loads((paths.RUNS / "vr" / "okvqa_config.json").read_text())
print("config evaluation_questions:", cfg["evaluation_questions"], "| u_retrieval_rate_calibration:", cfg["u_retrieval_rate_calibration"])
