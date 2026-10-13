"""Parser, empty-answer tags, and the local half split. No C:\\rc fixture and no arbitration import."""
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.analysis import offline_v2 as off

ROOT = Path(__file__).resolve().parents[1]


def test_help_and_import_do_not_load_arbitration():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    imported = subprocess.run(
        [sys.executable, "-c",
         "import sys, scripts.analysis.offline_v2; "
         "sys.exit(0 if 'omniaware.arbitration' not in sys.modules else 1)"],
        cwd=ROOT, env=env, capture_output=True, text=True, check=False,
    )
    assert imported.returncode == 0, imported.stderr
    help_run = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "analysis" / "offline_v2.py"), "--help"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert help_run.returncode == 0, help_run.stderr
    assert "--seed" in help_run.stdout
    assert "--out" in help_run.stdout
    assert "arbitration.py is not in this worktree yet" not in help_run.stdout + help_run.stderr


def test_parser_accepts_the_reproduce_command():
    args = off.build_parser().parse_args(["--seed", "20261013", "--out", "runs/offline_v2"])
    assert args.seed == 20261013
    assert args.out == Path("runs/offline_v2")
    defaults = off.build_parser().parse_args([])
    assert defaults.seed == 20261013
    assert defaults.out == Path("runs/offline_v2")


def test_empty_answer_flags_on_temp_jsonl(tmp_path):
    rows = [
        {"qid": "blank", "prediction": "", "parsed": {"answer": ""}, "raw_response": ""},
        {"qid": "sources-none", "prediction": "Sources: none",
         "parsed": {"answer": "Sources: none"}, "raw_response": "Sources: none"},
        {"qid": "sources-evidence", "prediction": "Sources: [Evidence 1] [Evidence 4]",
         "parsed": {"answer": "Sources: [Evidence 1] [Evidence 4]"},
         "raw_response": "Sources: [Evidence 1] [Evidence 4]"},
        {"qid": "answer-plus-sources", "prediction": "Yes", "parsed": {"answer": "Yes"},
         "raw_response": "Yes\n\nSources: [Evidence 1]"},
        {"qid": "blank-parsed", "prediction": "A fountain.", "parsed": {"answer": ""},
         "raw_response": "\nSources: none"},
        {"qid": "sentence", "prediction": "A fountain is sitting in front of the tower.",
         "parsed": {"answer": "A fountain is sitting in front of the tower."},
         "raw_response": "A fountain is sitting in front of the tower.\n\nSources: none"},
    ]
    path = tmp_path / "tiny.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n\n", encoding="utf-8")
    loaded = off.read_jsonl(path)
    assert [row["qid"] for row in loaded] == [row["qid"] for row in rows]
    assert [off.is_empty_answer(row) for row in loaded] == [True, True, True, False, True, False]
    reasons = {row["qid"]: off.empty_answer_reasons(row) for row in loaded}
    assert "sources_only" in reasons["sources-none"]
    assert "blank_parsed_answer" in reasons["blank-parsed"]
    assert "empty_answer_with_sources" in reasons["blank-parsed"]
    assert reasons["answer-plus-sources"] == []
    assert reasons["sentence"] == []


def test_local_split_half_is_disjoint_and_deterministic():
    qids = [f"q{i:02d}" for i in range(11)]
    calib, ev = off.local_split_half(qids, 20261013)
    assert set(calib).isdisjoint(ev)
    assert set(calib) | set(ev) == set(qids)
    assert len(calib) + len(ev) == len(qids)
    assert abs(len(calib) - len(ev)) <= 1
    assert off.local_split_half(list(reversed(qids)), 20261013) == (calib, ev)
    assert off.local_split_half(qids, 1) != (calib, ev)


def test_proxy_margin_from_u_initial_decreases():
    assert off.proxy_margin_from_u(0.0) == 1.0
    assert off.proxy_margin_from_u(1.0) == 0.0
    assert off.proxy_margin_from_u(0.25) > off.proxy_margin_from_u(0.9)
    assert off.proxy_margin_from_u(1.5) == -1.5
