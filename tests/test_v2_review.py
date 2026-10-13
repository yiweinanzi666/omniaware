"""A4, A5, A7: EAB is opt-in, image diagnostic fixture, M2-RAG image, SeaKR question set."""
import json

import pytest

from omniaware import run
from omniaware.baselines.m2rag_webqa import direct_answer_document, extractor_messages, official_query
from omniaware.qwen_eab import require_sdpa
import scripts.analysis.image_evidence_check as image_evidence
import scripts.paper_runs as pr


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


def test_main_seakr_command_does_not_use_a_smaller_n_eval_or_the_probe_route():
    main = pr.stages("8b")["main"]["commands"]
    runs = [c for c in main if c[:3] == ["python", "-m", "omniaware.run"]]
    seakr = [c for c in runs if "--seakr-samples" in c]
    others = [c for c in runs if "--seakr-samples" not in c]
    assert seakr and others

    def n_eval(cmd):
        return cmd[cmd.index("--n-eval") + 1] if "--n-eval" in cmd else None

    seakr_n = {n_eval(c) for c in seakr}
    other_n = {n_eval(c) for c in others}
    assert seakr_n == other_n
    assert None in seakr_n
    assert all("--probe-uncertainty" not in c and "--first-round-routing" not in c for c in seakr)
    assert all(not any(str(part).startswith("eab:") for part in c) for c in runs)
    # Order experiments may subsample, and the stage text says so.
    order = pr.stages("8b")["order"]
    assert "1,000" in order["fills"] or "1000" in order["fills"]
    assert all(c[c.index("--n-eval") + 1] == "1000" for c in order["commands"] if "--order" in c)


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
