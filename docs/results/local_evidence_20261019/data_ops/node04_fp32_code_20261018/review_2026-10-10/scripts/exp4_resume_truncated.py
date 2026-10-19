import sys, tempfile, json
from pathlib import Path
sys.path.insert(0, "C:/Users/22688/Desktop/www_code/omniaware")
sys.path.insert(0, "C:/Users/22688/Desktop/www_code/omniaware/src")
import pytest
from omniaware import data, paths, run, merge_shards
import tests.test_run_smoke as tr

examples = tr._examples(20)
data.LOADERS["okvqa"] = lambda **kw: examples
tmp = Path(tempfile.mkdtemp())
paths.RUNS = tmp / "runs"; paths.LOCAL_CACHE = tmp / "cache"
run.QwenEAB = tr.FakeQwen
run.load_relevance = lambda exs, path: {e["qid"]: {c["cid"]: {"text": 1.0 / int(c["cid"][1:])} for c in e["candidates"]} for e in exs}
args = ["--dataset", "okvqa", "--split", "test", "--n-calib", "6", "--calib-source", "heldout", "--n-dev", "4",
        "--tag", "resume", "--variants", "eab:g1.2:ans2:pos:gate", "--n-eval", "5"]
run.main(args)
out = paths.RUNS / "resume" / "okvqa.jsonl"
lines = out.read_text().splitlines()
print("full run wrote", len(lines), "rows")
# simulate SIGKILL in the middle of the last write: keep half of the final line, no newline
out.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][: len(lines[-1]) // 2])
try:
    run.main(args)
    print("resume succeeded")
except Exception as err:
    print("RESUME CRASH:", type(err).__name__, str(err)[:80])
print("file still has partial line:", not out.read_text().endswith("\n"))

# merge: an incomplete shard set passes when --expected is omitted (CLI default)
stem = tmp / "m" / "okvqa"; stem.parent.mkdir()
(stem.parent / "okvqa.shard00-of-02.jsonl").write_text("".join(json.dumps({"qid": f"a{i}"}) + "\n" for i in range(3)))
(stem.parent / "okvqa.shard01-of-02.jsonl").write_text("".join(json.dumps({"qid": f"b{i}"}) + "\n" for i in range(2)))
merge_shards.main([str(stem)])
print("merge without --expected accepted 5 rows (questions in split unknown)")
