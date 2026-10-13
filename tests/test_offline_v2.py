"""offline_v2: loaders map the legacy dumps onto action rows, the naive baseline and the split behave, and the
whole script runs end to end on a few synthetic questions (the real dumps under C:\\rc are never read here)."""
import json

import numpy as np
import pytest

from omniaware.actions import DIRECT
from scripts.analysis import offline_v2 as off


def _webqa_line(qid, variant, *, correct, u, subset="text", wrong="Sources: none"):
    return {"qid": qid, "variant": variant, "subset": subset, "u_initial": u, "acc_approx": 1.0 if correct else 0.0,
            "prediction": "Paris" if correct else "Rome", "raw_response": "Paris\n\nSources: none" if correct else wrong,
            "references": ["Paris"]}


def _write_webqa(path, qids, *, empty_wrong=True):
    """empty_wrong=False writes the wrong answers as "Rome", so both answers of every question are non-empty."""
    rows = []
    wrong = "Sources: none" if empty_wrong else "Rome\n\nSources: none"
    for i, qid in enumerate(qids):
        direct_right = i % 2 == 0
        rows.append(_webqa_line(qid, "direct", correct=direct_right, u=0.1 if direct_right else 0.9,
                                subset="img" if i % 3 == 0 else "text", wrong=wrong))
        rows.append(_webqa_line(qid, "always-static", correct=not direct_right, u=0.1 if direct_right else 0.9,
                                wrong=wrong))
        rows.append(_webqa_line(qid, "trigger-eab", correct=True, u=0.5))      # not a v2 variant: ignored
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _write_okvqa(directory, qids):
    directory.mkdir(parents=True, exist_ok=True)
    preds, official = [], []
    for i, qid in enumerate(qids):
        direct_right = i % 2 == 0
        preds.append({"qid": qid, "preds": {"direct": "gold" if direct_right else "wrong", "static": "x"},
                      "raw_pre": {"h_sem": 0.1 if direct_right else 0.9},
                      "raw_post": {"h_sem": 0.3}, "rel": {"k2": 0.2, "k5": 0.7}})
        official.append({"qid": qid, "scores": {"direct": {"vqa_accuracy": 100.0 if direct_right else 0.0},
                                                "static": {"vqa_accuracy": 0.0 if direct_right else 100.0}}})
    (directory / "okvqa_gs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in preds), encoding="utf-8")
    (directory / "okvqa_gs.official_scores.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in official), encoding="utf-8")
    return directory


def test_webqa_loader_keeps_the_two_variants_with_one_margin_proxy(tmp_path):
    path = _write_webqa(tmp_path / "scores.jsonl", ["q1", "q2"])
    rows = off.load_webqa(path)
    assert [r["qid"] for r in rows] == ["q1", "q2"]
    first = rows[0]
    assert set(first["actions"]) == {DIRECT, off.STATIC}
    assert first["actions"][DIRECT]["first_token"]["margin"] == pytest.approx(0.9)      # 1 - u_initial
    assert first["actions"][off.STATIC]["first_token"]["margin"] == pytest.approx(0.9)  # the same proxy
    assert first["actions"][DIRECT]["kl_to_direct"] is None and first["actions"][DIRECT]["retriever"]["top1_score"] is None
    assert first["scores"] == {DIRECT: 100.0, off.STATIC: 0.0} and first["metric"] == "acc_approx"
    assert first["subset"] == "image_question" and rows[1]["subset"] == "text_question"
    assert rows[1]["actions"][off.STATIC]["answer_empty"] is False           # "Paris"
    assert rows[0]["actions"][off.STATIC]["answer_empty"] is True           # "Sources: none" alone is empty


def test_okvqa_loader_uses_each_actions_own_entropy_and_the_retriever_top1(tmp_path):
    directory = _write_okvqa(tmp_path / "okvqa", ["1", "2"])
    rows = off.load_okvqa(directory)
    first = rows[0]
    assert first["actions"][DIRECT]["first_token"]["margin"] == pytest.approx(-0.1)     # -h_sem, pre
    assert first["actions"][off.STATIC]["first_token"]["margin"] == pytest.approx(-0.3)  # -h_sem, post
    assert first["actions"][DIRECT]["retriever"]["top1_score"] is None
    # the static action's top-1 is the largest relevance of the evidence it received (the rel field)
    assert first["actions"][off.STATIC]["retriever"]["top1_score"] == pytest.approx(0.7)
    assert first["scores"] == {DIRECT: 100.0, off.STATIC: 0.0} and first["metric"] == "vqa_accuracy"
    assert first["subset"] is None


def test_split_halves_are_disjoint_and_cover_every_question(tmp_path):
    rows = [{"qid": f"q{i:02d}"} for i in range(11)]
    calib, evaluation = off.split_halves(rows, seed=20261013)
    ids_calib = [r["qid"] for r in calib]
    ids_eval = [r["qid"] for r in evaluation]
    assert len(ids_calib) == 5 and len(ids_eval) == 6
    assert set(ids_calib).isdisjoint(ids_eval)
    assert sorted(ids_calib + ids_eval) == sorted(r["qid"] for r in rows)
    # Same seed, same split; calibration rows keep the sorted order inside the half.
    assert [r["qid"] for r in off.split_halves(rows, seed=20261013)[0]] == ids_calib
    assert ids_calib == sorted(ids_calib)


def test_naive_baseline_takes_the_best_calibration_threshold():
    def row(qid, margin_direct, direct_score, static_score):
        return {"qid": qid, "scores": {DIRECT: direct_score, off.STATIC: static_score},
                "actions": {DIRECT: {"first_token": {"margin": margin_direct}, "answer_empty": False},
                            off.STATIC: {"first_token": {"margin": 0.0}, "answer_empty": False}}}
    # Direct is right exactly when its margin is high; static is right otherwise.
    calib = [row(f"c{i}", 0.9, 100.0, 0.0) if i % 2 == 0 else row(f"c{i}", 0.1, 0.0, 100.0) for i in range(10)]
    naive = off.naive_threshold(calib)
    assert naive["calib_mean"] == pytest.approx(100.0)
    assert 0.1 < naive["tau"] <= 0.9
    evaluation = [row("e1", 0.9, 100.0, 0.0), row("e2", 0.1, 0.0, 100.0)]
    choices = off.naive_choices(naive["tau"], evaluation)
    assert choices == [DIRECT, off.STATIC]


def test_direct_audit_counts_static_empty_answers_and_length_classes():
    def row(direct_answer, static_answer, static_empty):
        return {"actions": {DIRECT: {"answer": direct_answer, "answer_empty": False},
                            off.STATIC: {"answer": static_answer, "answer_empty": static_empty}}}
    evaluation = [row("Paris", "", True),                                  # direct, static empty: same class
                  row("one two three four", "Paris", False),               # direct, long against short
                  row("Paris", "one two three four", False)]               # static chosen: not counted as direct
    audit = off.direct_audit(evaluation, [DIRECT, DIRECT, off.STATIC])
    assert audit == {"rule_direct": 2, "rule_direct_static_empty": 1, "rule_direct_length_differs": 1,
                     "evaluation": 3, "evaluation_length_differs": 2}


def test_script_runs_end_to_end_on_synthetic_dumps(tmp_path, capsys):
    webqa = _write_webqa(tmp_path / "scores.jsonl", [f"q{i:02d}" for i in range(40)], empty_wrong=False)
    okvqa = _write_okvqa(tmp_path / "okvqa", [str(i) for i in range(40)])
    out = tmp_path / "out"
    assert off.main(["--seed", "7", "--out", str(out), "--webqa", str(webqa), "--okvqa-dir", str(okvqa)]) == 0
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    report = (out / "REPORT.md").read_text(encoding="utf-8")
    assert results["exploratory"] is True and results["paper_table"] is False
    assert set(results["datasets"]) == {"webqa", "okvqa"} and set(results["both_nonempty"]) == {"webqa", "okvqa"}
    names = [b["name"] for b in results["datasets"]["webqa"]["blocks"]]
    assert "朴素基线" in names and "v2-Rule ε=0" in names
    assert any(n.startswith("v2-Cal ε=1") and "对内部拆分种子敏感，见下方种子范围" in n for n in names)
    assert sorted(p.name for p in out.iterdir()) == ["REPORT.md", "results.json"]
    assert report.startswith("# v2 离线探索（旧管线）") and report.count("探索性数字，不进论文主表") == 1
    assert "只在两路答案都非空的题上重算" in report
    # The oracle is a ceiling: no strategy scores above it on the evaluation half.
    oracle = next(b for b in results["datasets"]["webqa"]["blocks"] if b["name"] == "逐题上限")["mean"]
    assert all(b["mean"] <= oracle + 1e-9 for b in results["datasets"]["webqa"]["blocks"])
    assert np.isfinite(results["datasets"]["webqa"]["cal_seed_spread"]["median"])
    assert results["both_nonempty"]["webqa"]["dropped"] == 0
    assert results["datasets"]["webqa"]["direct_audit"]["evaluation"] == results["datasets"]["webqa"]["eval"]


def test_both_nonempty_table_refuses_when_too_few_questions_survive(tmp_path):
    webqa = _write_webqa(tmp_path / "scores.jsonl", [f"q{i:02d}" for i in range(40)])   # empty wrong answers
    okvqa = _write_okvqa(tmp_path / "okvqa", [str(i) for i in range(40)])
    with pytest.raises(ValueError, match="webqa: 0 questions have both answers non-empty"):
        off.run(webqa, okvqa, seed=3)


def test_too_few_nonempty_questions_exit_two(tmp_path, capsys):
    webqa = _write_webqa(tmp_path / "scores.jsonl", [f"q{i:02d}" for i in range(40)])
    okvqa = _write_okvqa(tmp_path / "okvqa", [str(i) for i in range(40)])
    assert off.main(["--out", str(tmp_path / "out"), "--webqa", str(webqa), "--okvqa-dir", str(okvqa)]) == 2
    assert "both answers non-empty" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
