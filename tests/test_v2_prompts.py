"""A2: one frozen instruction, one sha256, for every arm unless registered."""
from omniaware import pipeline, prompts

ARMS = ["direct", "static", "static_text", "static_image", "static_mixed", "rerank_static",
        "gated", "seakr", "fitm", "m2rag", "universalrag"]


def test_default_exceptions_are_empty():
    assert prompts.EXCEPTIONS == {}


def test_one_run_shares_one_sha256_unless_registered():
    for dataset in ("webqa", "okvqa", "hybridqa", "videorag"):
        record = prompts.instruction_for_style(
            dataset, "default", cite_instruction=pipeline.CITATION_INSTRUCTION,
            short_instruction=pipeline.SHORT_INSTRUCTION)
        shas = prompts.arm_sha256s(dataset, ARMS, instruction=record["instruction"])
        assert set(shas) == set(ARMS)          # nothing is registered as an exception
        assert len(set(shas.values())) == 1
        assert record["instruction_sha256"] == next(iter(shas.values()))
        assert record["instruction_sha256"] == prompts.instruction_sha256(record["instruction"])


def test_webqa_and_okvqa_wording_is_the_frozen_default():
    webqa = prompts.instruction_text("webqa")
    okvqa = prompts.instruction_text("okvqa")
    assert "complete sentence" in webqa and "key object" in webqa
    assert "3 words" in okvqa
    assert "complete sentence" in prompts.instruction_text("videorag")
    assert prompts.instruction_text("hybridqa")


def test_build_segments_uses_the_registry_for_every_default_arm():
    question = {"dataset": "webqa", "question": "Where is the tower?", "query_images": []}
    text = pipeline.build_segments(question, None)[0][-1].text
    assert prompts.instruction_text("webqa") in text
    cite = pipeline.build_segments(question, None, "cite")[0][-1].text
    assert pipeline.CITATION_INSTRUCTION in cite
    assert prompts.instruction_text("webqa") not in cite


def test_overrides_are_not_chosen_by_this_test_as_the_default():
    custom = prompts.instruction_text("okvqa", overrides={"okvqa": "CUSTOM"})
    assert custom == "CUSTOM"
    assert prompts.instruction_text("okvqa") != "CUSTOM"
