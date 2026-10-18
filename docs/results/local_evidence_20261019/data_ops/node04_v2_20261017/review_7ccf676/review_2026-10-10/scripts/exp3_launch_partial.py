import sys, tempfile, subprocess
from pathlib import Path
sys.path.insert(0, "C:/Users/22688/Desktop/www_code/omniaware")
sys.path.insert(0, "C:/Users/22688/Desktop/www_code/omniaware/src")
from scripts import launch

calls = []
def fake_run(cmd, **kw):
    calls.append(cmd); return subprocess.CompletedProcess(cmd, 0)
launch.subprocess.run = fake_run
launch.subprocess.Popen = None

root = Path(tempfile.mkdtemp())
tag_dir = root / "v321_main_8b"
tag_dir.mkdir()
# an earlier single-process run (same tag) was killed after 3 of 1000 answers; it wrote the merged-looking file
(tag_dir / "okvqa_gs.jsonl").write_text('{"qid": "q1"}\n' * 3)
run_args = ["--dataset", "okvqa_gs", "--split", "test", "--tag", "v321_main_8b", "--model", "8b"]
out = launch.launch(run_args, gpus=["0", "1"], runs_dir=root)
print("returned:", out)
print("subprocess calls made:", len(calls), "(no shard was started; no completeness check)")
print("lines in file:", len((tag_dir / 'okvqa_gs.jsonl').read_text().splitlines()))
