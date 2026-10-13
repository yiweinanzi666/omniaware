"""S2 go/no-go: PASS, FAIL, and a tie on a zero baseline."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "analysis" / "v2_gate.py"
PASS_FIXTURE = ROOT / "tests" / "fixtures" / "v2_gate_example.json"
FAIL_FIXTURE = ROOT / "tests" / "fixtures" / "v2_gate_fail.json"


def _gate(path: Path):
    return subprocess.run(
        [sys.executable, str(GATE), "--scores", str(path)],
        cwd=ROOT, capture_output=True, text=True,
    )


def test_pass_fixture_exits_zero_and_reports_the_gap():
    result = _gate(PASS_FIXTURE)
    lines = result.stdout.splitlines()
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert lines == [
        "PASS",
        "oracle_mean=0.625000 best_single=static_text best_single_mean=0.500000 relative_gap=0.250000",
    ]


def test_fail_fixture_exits_one():
    result = _gate(FAIL_FIXTURE)
    lines = result.stdout.splitlines()
    assert result.returncode == 1, result.stderr
    assert lines == [
        "FAIL",
        "oracle_mean=0.531250 best_single=direct best_single_mean=0.500000 relative_gap=0.062500",
    ]


def test_tie_on_zero_baseline_uses_difference_not_ratio(tmp_path):
    # Both means are 0, so the gap is oracle - best, not a ratio. static_text wins the
    # tie over rerank_static because it is earlier in the v2 action order.
    tied = {
        "questions": [
            {"qid": "a", "scores": {"rerank_static": 0.0, "static_text": 0.5}},
            {"qid": "b", "scores": {"rerank_static": 0.0, "static_text": -0.5}},
        ]
    }
    path = tmp_path / "tie_zero.json"
    path.write_text(json.dumps(tied), encoding="utf-8")
    result = _gate(path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "PASS",
        "oracle_mean=0.250000 best_single=static_text best_single_mean=0.000000 relative_gap=0.250000",
    ]

    zeros = {
        "questions": [
            {"qid": "a", "scores": {"rerank_static": 0.0, "static_text": 0.0}},
            {"qid": "b", "scores": {"rerank_static": 0.0, "static_text": 0.0}},
        ]
    }
    path.write_text(json.dumps(zeros), encoding="utf-8")
    result = _gate(path)
    assert result.returncode == 1
    assert result.stdout.splitlines() == [
        "FAIL",
        "oracle_mean=0.000000 best_single=static_text best_single_mean=0.000000 relative_gap=0.000000",
    ]


def test_bad_input_exits_two(tmp_path):
    missing = _gate(tmp_path / "nope.json")
    assert missing.returncode == 2
    assert missing.stdout == ""
    assert "not found" in missing.stderr

    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    parsed = _gate(bad)
    assert parsed.returncode == 2
    assert parsed.stdout == ""

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"questions": []}), encoding="utf-8")
    assert _gate(empty).returncode == 2
