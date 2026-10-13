"""EAB opt-in, the WebQA image-input check, the M2-RAG query image and the v1 baseline commands."""
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
    # The baselines read the joint input, which no probe or route changes. They run with the same probe and route
    # flags as the main method of their dataset.
    for c in runs:
        dataset = c[c.index("--dataset") + 1]
        cold = dataset in pr.COLD
        assert ("--probe-uncertainty" in c) == cold and ("--first-round-routing" in c) == cold, dataset
    # The main, CE and SeaKR runs of the v1 pipeline all decode the EAB arm (the column analyses read gated:<EAB>).
    eab = pr.PLACEHOLDER["eab_variant"]
    for c in runs:
        assert c[c.index("--variants") + 1] == eab, c


def test_m2rag_inputs_contain_the_question_image():
    query = official_query("q1", "What is the man holding?", [], query_images=["/synthetic/question.jpg"])
    assert [img["cached_image_url"] for img in query["aux_images"]] == ["/synthetic/question.jpg"]
    messages = extractor_messages("What is the man holding?", direct_answer_document("a glove"),
                                  query_images=["/synthetic/question.jpg"])
    urls = [part.get("image_url", {}).get("url") for part in messages if part.get("type") == "image_url"]
    assert "/synthetic/question.jpg" in urls


def test_image_evidence_fixture_checks_the_inputs_run_py_would_build(tmp_path):
    out = tmp_path / "report.json"
    assert image_evidence.main(["--fixture", "--output", str(out), "--gpu-probe"]) == 0
    report = json.loads(out.read_text(encoding="utf8"))
    assert report["gpu_probe_executed"] is False and "Describe the image" in report["gpu_probe"]
    assert report["max_pixels"] == 256 * 32 * 32 and report["n"] == 1
    actions = report["questions"][0]["actions"]
    assert set(actions) == {"static_image", "static_mixed"}
    # static_image: the image candidates by relevance; the gold image ranks second among them (0.5 vs 0.7 for ib).
    image_input = actions["static_image"]
    assert image_input["selected"] == ["ib", "ig", "iswap", "imiss"]
    assert image_input["gold_image_positions"] == [2]
    # static_mixed: the joint top-5 (text first by relevance), so the gold image is third.
    assert actions["static_mixed"]["selected"] == ["tp", "ib", "ig", "iswap", "imiss"]
    assert actions["static_mixed"]["gold_image_positions"] == [3]
    summary = report["summary"]
    assert summary["static_image"]["unreadable_images"] == 1          # the missing file is sent, and reported
    assert summary["static_image"]["images_over_budget"] == 1         # the 800x800 image
    assert summary["static_image"]["images_with_id_mismatch"] == 1    # iswap reuses gold.png under another id
    assert summary["static_mixed"]["images_with_pairing_error"] == 0  # every image has its own title and caption
    big = next(im for im in actions["static_image"]["images"] if im["cid"] == "ib")
    assert big["pixels"] == 800 * 800 and big["resized_pixels"] <= 256 * 32 * 32
    assert big["visual_tokens"] == big["resized_pixels"] // (28 * 28)     # both sides are multiples of 28
    assert any("cannot be opened" in line for line in report["findings"])
    assert not any("already checked" in line.lower() for line in report["findings"])


def test_image_evidence_real_run_names_the_inputs_it_needs(tmp_path, monkeypatch):
    monkeypatch.setattr("omniaware.paths.WEBQA_JSON", tmp_path / "missing.json")
    with pytest.raises(SystemExit, match="needs real WebQA inputs"):
        image_evidence.main(["--n", "50", "--output", str(tmp_path / "nope.json")])
