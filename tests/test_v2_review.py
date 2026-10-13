"""A4, A5, A7: EAB is opt-in, image diagnostic fixture, M2-RAG image, SeaKR question set."""
import json

import pytest

from omniaware import run
from omniaware.baselines.m2rag_webqa import direct_answer_document, extractor_messages, official_query
from omniaware.qwen_eab import require_sdpa
import scripts.analysis.image_evidence_check as image_evidence
import scripts.paper_runs as pr


def test_v2_flags_need_the_reranker_and_the_relevance_selector(tmp_path):
    parser = run.build_parser()
    decisions = tmp_path / "dev_decisions.json"
    decisions.write_text('{"v2": {"name": "frozen"}}', encoding="utf-8")
    base = ["--dataset", "webqa_full", "--split", "calib", "--tag", "t", "--model", "8b", "--decisions", str(decisions)]
    args = parser.parse_args(base + ["--v2", "--v2-reranker"])
    run.check_decisions_file(parser, args)
    run.check_v2_args(parser, args)
    assert args.split == "calib" and args.v2 and args.v2_reranker and args.decisions_sha256
    assert not hasattr(args, "save_signals") and not hasattr(args, "v2_actions")
    for extra in (["--v2"], ["--v2-reranker"], ["--v2", "--v2-reranker", "--selector", "ce"]):
        bad = parser.parse_args(base + extra)
        with pytest.raises(SystemExit):
            run.check_v2_args(parser, bad)
    removed = ["--save-signals", "--empty-answer-accounting", "--shared-prompt-registry", "--v2-actions", "direct"]
    for flag in removed:
        with pytest.raises(SystemExit):
            parser.parse_args(base + [flag])


def test_missing_or_invalid_decisions_file_is_rejected(tmp_path):
    parser = run.build_parser()
    missing = parser.parse_args(["--dataset", "okvqa", "--split", "test", "--tag", "t",
                                 "--decisions", str(tmp_path / "absent.json")])
    with pytest.raises(SystemExit):
        run.check_decisions_file(parser, missing)
    listing = tmp_path / "list.json"
    listing.write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit):
        run.check_decisions_file(parser, parser.parse_args(["--dataset", "okvqa", "--split", "test", "--tag", "t",
                                                            "--decisions", str(listing)]))


def test_default_variants_have_no_eab_and_analysis_eab_adds_ans1():
    parser = run.build_parser()
    args = parser.parse_args(["--dataset", "okvqa", "--split", "dev", "--tag", "t"])
    assert args.variants == "" and args.analysis_eab is False
    assert "eab:" not in args.variants
    args = parser.parse_args(["--dataset", "okvqa", "--split", "dev", "--tag", "t", "--analysis-eab"])
    run.apply_analysis_eab(args)
    assert args.variants == "eab:g1.2:ans1:pos:gate"


def test_require_sdpa_raises_and_does_not_fall_back():
    from types import SimpleNamespace
    model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(_attn_implementation="eager")))
    with pytest.raises(RuntimeError, match="sdpa"):
        require_sdpa(model)


def test_main_baselines_share_the_main_methods_probe_and_route():
    main = pr.stages("8b")["main"]["commands"]
    runs = [c for c in main if c[:3] == ["python", "-m", "omniaware.run"]]
    seakr = [c for c in runs if "--seakr-samples" in c]
    others = [c for c in runs if "--seakr-samples" not in c]
    assert seakr and others

    def n_eval(cmd):
        return cmd[cmd.index("--n-eval") + 1] if "--n-eval" in cmd else None

    # SeaKR runs on the full test split, like the other main arms
    assert {n_eval(c) for c in seakr} == {n_eval(c) for c in others} == {None}
    # The baselines read the joint input, which no probe or route changes. They are run with the same probe and
    # route flags as the main method of their dataset (S-4 was a wrong fix: the main method lost its probe).
    for c in runs:
        dataset = c[c.index("--dataset") + 1]
        cold = dataset in pr.COLD
        assert ("--probe-uncertainty" in c) == cold and ("--first-round-routing" in c) == cold, dataset
    assert all(not any(str(part).startswith("eab:") for part in c) for c in runs)


def test_m2rag_inputs_contain_the_question_image():
    query = official_query("q1", "What is the man holding?", [], query_images=["/synthetic/question.jpg"])
    assert [img["cached_image_url"] for img in query["aux_images"]] == ["/synthetic/question.jpg"]
    messages = extractor_messages("What is the man holding?", direct_answer_document("a glove"),
                                  query_images=["/synthetic/question.jpg"])
    urls = [part.get("image_url", {}).get("url") for part in messages if part.get("type") == "image_url"]
    assert "/synthetic/question.jpg" in urls


def test_image_evidence_fixture_runs_without_assets(tmp_path):
    out = tmp_path / "report.json"
    assert image_evidence.main(["--fixture", "--output", str(out), "--gpu-probe"]) == 0
    report = json.loads(out.read_text(encoding="utf8"))
    assert report["gpu_probe_executed"] is False
    assert report["gpu_probe_command"].startswith("python -m scripts.analysis.image_evidence_check")
    assert report["max_pixels"] == 256 * 32 * 32
    assert report["n"] >= 1
    assert report["budget_looks_too_small"] is True
    assert any("EMPTY" not in line for line in report["findings"])
    assert any("too small" in line for line in report["findings"])
    assert any("path mismatch" in line for line in report["findings"])
    assert any("title/image order" in line for line in report["findings"])
    images = [im for q in report["questions"] for im in q["images"]]
    read = [im for im in images if im["read"]]
    unread = [im for im in images if not im["read"]]
    assert read and unread
    for im in read:
        assert im["pixels"] == im["width"] * im["height"]
        assert im["visual_tokens"] > 0
    for im in unread:
        assert im["width"] is None and im["visual_tokens"] is None
    golds = [im for im in images if im["gold"]]
    assert golds and golds[0]["rank"] == 1
    assert report["questions"][0]["gold_in_context"] is True


def test_image_evidence_real_run_errors_when_annotations_are_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("omniaware.paths.WEBQA_JSON", tmp_path / "missing.json")
    with pytest.raises(SystemExit, match="not found"):
        image_evidence.main(["--n", "50", "--output", str(tmp_path / "nope.json")])
