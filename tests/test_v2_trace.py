"""What a v2 run leaves behind for tracing a result back through every stage: the environment of each launch,
the log, the candidate ranking, each action's input and decode, sampled prompts, and the selector's test trace."""
import json
from dataclasses import dataclass

import pytest
import torch

from omniaware import arbitration, run, runio, signals
from omniaware.actions import DIRECT, V2_ACTIONS
from tests import test_v2_prompts

fake_run = test_v2_prompts.fake_run          # the fake model and assets of the v2 run tests (a fixture)

pytestmark = pytest.mark.usefixtures("tiny_reserves")

IMAGE, START, END = 7, 5, 6


@dataclass
class _Enc:
    length: int
    input_ids: torch.Tensor | None


def test_input_trace_counts_the_visual_tokens_of_each_image_in_order():
    ids = [1, START] + [IMAGE] * 4 + [END, 2, 3, START] + [IMAGE] * 9 + [END, 4]
    trace = signals.input_trace(_Enc(len(ids), torch.tensor([ids])), IMAGE)
    assert trace["prompt_tokens"] == len(ids)
    assert trace["n_images"] == 2 and trace["image_tokens"] == [4, 9]
    other = signals.input_trace(_Enc(len(ids), torch.tensor([ids[:-1] + [9]])), IMAGE)
    assert other["prompt_sha256"] != trace["prompt_sha256"]          # the hash follows the token ids


def test_input_trace_reports_unknown_fields_as_null_not_as_zero():
    assert signals.input_trace(_Enc(5, torch.tensor([[1, 2, 3, 4, 5]])), None)["n_images"] is None
    blank = signals.input_trace(_Enc(5, None), IMAGE)
    assert blank["prompt_sha256"] is None and blank["image_tokens"] is None and blank["prompt_tokens"] == 5


def _v2_run(tmp_path, tag="trace", extra=()):
    run.main(["--dataset", "okvqa", "--split", "test", "--n-calib", "6", "--calib-source", "heldout", "--n-dev", "4", "--image-mode", "clip",
              "--tag", tag, "--v2", "--v2-reranker", *extra])
    out = tmp_path / "runs" / tag
    return out, [json.loads(line) for line in (out / "okvqa.jsonl").read_text(encoding="utf8").splitlines()]


def test_every_available_action_records_its_input_and_its_decode(fake_run):
    _, rows = _v2_run(fake_run)
    assert rows
    for row in rows:
        for name in V2_ACTIONS:
            action = row["actions"][name]
            if not action["available"]:
                assert "decode" not in action
                continue
            assert action["input"]["prompt_tokens"] > 0 and len(action["input"]["prompt_sha256"]) == 64
            decode = action["decode"]
            assert len(decode["tokens"]) == action["gen_len"] == len(decode["token_logprobs"])
            assert decode["hit_token_cap"] is False and decode["seconds"] is not None
            assert sum(decode["token_logprobs"]) == pytest.approx(action["seq_logprob"], abs=1e-3)
        # A different prompt has a different hash; the direct prompt holds no evidence.
        assert row["actions"][DIRECT]["input"]["prompt_sha256"] != row["actions"]["static_mixed"]["input"]["prompt_sha256"]


def test_the_candidate_ranking_is_stored_whole_and_explains_each_selection(fake_run):
    _, rows = _v2_run(fake_run)
    for row in rows:
        ranked = row["candidate_scores"]
        assert len(ranked) == 7                                       # every candidate, not only the selected five
        assert [c["relevance"] for c in ranked] == sorted((c["relevance"] for c in ranked), reverse=True)
        assert all(c["ce_logit"] is not None for c in ranked)
        cut = ranked[4]["relevance"]                                  # ties at the cut may fall either way
        mixed = set(row["actions"]["static_mixed"]["chunks"])
        assert {c["cid"] for c in ranked if c["relevance"] > cut} <= mixed <= {c["cid"] for c in ranked
                                                                              if c["relevance"] >= cut}
        by_ce = sorted(ranked, key=lambda c: -c["ce_logit"])[:5]
        assert set(row["actions"]["rerank_static"]["chunks"]) == {c["cid"] for c in by_ce}


def test_prompt_samples_hold_the_text_the_model_read_for_the_first_questions_only(fake_run):
    out, rows = _v2_run(fake_run, extra=("--trace-prompts", "2"))
    samples = [json.loads(line) for line in (out / "okvqa.prompts.ndjson").read_text(encoding="utf8").splitlines()]
    assert len(samples) == 2 and len(rows) > 2
    for sample in samples:
        assert sample["question"] in sample["actions"][DIRECT]["prompt"]
        mixed = sample["actions"]["static_mixed"]["prompt"]
        assert "passage 1" in mixed and "passage 1" not in sample["actions"][DIRECT]["prompt"]
        assert "<image:img1.jpg>" in sample["actions"]["static_image"]["prompt"]
        assert sample["actions"]["static_image"]["images"] == ["img1.jpg", "img2.jpg"]


def test_prompt_sampling_can_be_switched_off(fake_run):
    out, _ = _v2_run(fake_run, extra=("--trace-prompts", "0"))
    assert not (out / "okvqa.prompts.ndjson").exists()


def test_each_launch_appends_its_environment_and_the_log_survives_a_rerun(fake_run):
    out, _ = _v2_run(fake_run)
    env = [json.loads(line) for line in (out / "okvqa.env.ndjson").read_text(encoding="utf8").splitlines()]
    assert len(env) == 1
    for key in ("git_commit", "git_dirty", "torch", "transformers", "python", "platform", "gpus", "argv", "model"):
        assert key in env[0]
    assert env[0]["torch"] == torch.__version__
    log = (out / "okvqa.log").read_text(encoding="utf8")
    assert "[env] commit" in log and "[done] okvqa" in log
    run.main(["--dataset", "okvqa", "--split", "test", "--n-calib", "6", "--calib-source", "heldout", "--n-dev", "4", "--image-mode", "clip",
              "--tag", "trace", "--v2", "--v2-reranker"])                                   # resumes: nothing to decode
    assert len((out / "okvqa.env.ndjson").read_text(encoding="utf8").splitlines()) == 2
    assert (out / "okvqa.log").read_text(encoding="utf8").startswith(log[:200])            # appended, not replaced


def test_the_log_keeps_the_traceback_of_a_failed_run(tmp_path):
    path = tmp_path / "x.log"
    with pytest.raises(ValueError):
        with runio.tee_output(path, "header line"):
            print("before the failure")
            raise ValueError("decode exploded")
    text = path.read_text(encoding="utf8")
    assert "header line" in text and "before the failure" in text
    assert "[failed] ValueError: decode exploded" in text and "Traceback" in text


def test_trace_files_are_not_taken_for_shard_outputs(fake_run):
    base = ["--dataset", "okvqa", "--split", "test", "--n-calib", "6", "--calib-source", "heldout", "--n-dev", "4",
            "--image-mode", "clip", "--tag", "sh", "--v2", "--v2-reranker"]
    run.main([*base, "--calibrate-only"])
    for shard in range(2):
        run.main([*base, "--num-shards", "2", "--shard-id", str(shard)])
    out = fake_run / "runs" / "sh"
    assert (out / "okvqa.shard00-of-02.env.ndjson").exists() and (out / "okvqa.shard01-of-02.log").exists()
    from omniaware import merge_shards
    merge_shards.main([str(out / "okvqa")])
    assert (out / "okvqa.jsonl").exists()


def _row(i, direct_right):
    scores = {DIRECT: 100.0 if direct_right else 0.0, "static_mixed": 0.0 if direct_right else 100.0}
    actions = {}
    for name in scores:
        margin = (5.0 if direct_right else 0.1) if name == DIRECT else 1.0
        actions[name] = {"answer": f"{name}{i}", "available": True, "kl_to_direct": 0.1,
                         "first_token": {"margin": margin, "entropy": 0.5, "entropy_norm": 0.5},
                         "answer_token": {"margin": margin, "entropy_norm": 0.5, "u_raw": -margin},
                         "retriever": {"top1_score": 0.3}}
    return {"qid": f"q{i}", "actions": actions, "scores": scores}


def test_the_decision_stores_every_tested_candidate_with_its_p_value():
    decision = arbitration.select_rule([_row(i, i % 2 == 0) for i in range(200)], delta=0.1, epsilon=1.0)
    trace = decision["trace"]
    assert not decision["fallback"] and trace
    assert {t["family"] for t in trace} <= {"rule:best", "rule:confident"}
    for family in {t["family"] for t in trace}:
        steps = [t for t in trace if t["family"] == family]
        assert [t["step"] for t in steps] == list(range(len(steps)))
        assert all(t["accepted"] for t in steps[:-1])                # only the last tested one may fail
    assert sum(t["accepted"] for t in trace) == decision["n_accepted"]
    chosen = [t for t in trace if t["accepted"] and t["mean_gap"] == pytest.approx(decision["calib_utility"])]
    assert chosen and all(0.0 <= t["p_value"] <= 1.0 and t["n"] == 200 for t in trace)
    json.dumps(decision)                                             # strict JSON, thresholds included


def test_a_fallback_says_why_and_still_carries_the_trace():
    rows = [_row(i, True) for i in range(100)]
    for row in rows:                                                 # the signal carries no information
        row["actions"][DIRECT]["first_token"]["margin"] = 1.0
        row["actions"][DIRECT]["answer_token"].update(margin=1.0, u_raw=-1.0)
    decision = arbitration.select_rule(rows, delta=0.1, epsilon=1.0)
    assert decision["fallback"] and decision["fallback_reason"]
    assert isinstance(decision["trace"], list)



def _rule(tau, choice, confidence="margin"):
    ranking = [DIRECT, "static_mixed", "rerank_static", "static_text", "static_image"]
    return {"kind": "rule", "fallback": False, "confidence": confidence, "tau": tau, "evidence_choice": choice,
            "best_single": DIRECT, "action_ranking": ranking, "evidence_ranking": ranking[1:], "delta": 0.1,
            "epsilon": 1.0}


def _state(margins, unavailable=()):
    return {name: {"available": name not in unavailable, "answer": None,
                   **({"first_token": {"margin": margins[name], "entropy": 0.1}} if name in margins else {})}
            for name in V2_ACTIONS}


def test_signals_needed_asks_only_for_what_the_decision_reads():
    assert arbitration.signals_needed({"kind": "single"}, _state({})) == []
    assert arbitration.signals_needed(_rule(1.0, "best"), _state({})) == [DIRECT]
    assert arbitration.signals_needed(_rule(1.0, "best"), _state({DIRECT: 5.0})) == []      # direct is kept
    assert arbitration.signals_needed(_rule(1.0, "best"), _state({DIRECT: 0.1})) == []      # best needs no signal
    need = arbitration.signals_needed(_rule(1.0, "confident"), _state({DIRECT: 0.1}, unavailable=("static_image",)))
    assert need == ["static_mixed", "rerank_static", "static_text"]
    assert arbitration.signals_needed(_rule(1.0, "confident"), _state({DIRECT: 5.0})) == []
    with pytest.raises(ValueError):
        arbitration.signals_needed({"kind": "cal"}, _state({}))


def test_signals_needed_asks_for_the_answer_token_under_u_omni_and_for_static_mixed_under_route():
    state = _state({DIRECT: 0.1})                          # first_token only: not enough for u_omni
    assert arbitration.signals_needed(_rule(1.0, "best", "u_omni"), state) == [DIRECT]
    state[DIRECT]["answer_token"] = {"u_raw": 0.9, "margin": 0.1, "entropy_norm": 0.5}
    assert arbitration.signals_needed(_rule(-0.5, "route", "u_omni"), state) == ["static_mixed"]   # -0.9 < -0.5
    state["static_mixed"]["route"] = {"mode": "attention", "modalities": ["image"], "statistics": {}}
    assert arbitration.signals_needed(_rule(-0.5, "route", "u_omni"), state) == []
    assert arbitration.signals_needed(_rule(-1.0, "route", "u_omni"), state) == []                 # direct is kept


@pytest.mark.parametrize("confidence", ["margin", "u_omni"])
@pytest.mark.parametrize("choice", ["best", "confident", "route"])
@pytest.mark.parametrize("tau", [-1e9, 1.0, 1e9])
def test_the_deployment_path_agrees_with_the_full_decode_and_decodes_once(
        fake_run, monkeypatch, choice, tau, confidence):
    # A real model gives the same logits for the same prompt; the stand-in draws them at random, so seed by prompt.
    original = run.QwenEAB.run

    def same_prompt_same_logits(self, enc, make_variants, max_new_tokens, decode_static=True):
        torch.manual_seed(enc.length)
        return original(self, enc, make_variants, max_new_tokens, decode_static)

    monkeypatch.setattr(run.QwenEAB, "run", same_prompt_same_logits)
    decisions = fake_run / "decisions.json"
    rule = _rule(tau, choice, confidence)
    decisions.write_text(json.dumps({"v2": {"rule": rule, "cal": {**rule, "kind": "single", "fallback": True,
                                                                  "action": DIRECT}, "primary": "rule"}}))
    out, rows = _v2_run(fake_run, tag=f"deploy_{confidence}_{choice}_{tau}", extra=("--decisions", str(decisions)))
    assert rows
    for row in rows:
        deployed = row["v2_deploy"]
        assert deployed["matches_full_decode"] is True and deployed["decode_calls"] == 1
        assert deployed["action"] == arbitration.apply_decision(rule, row["actions"])["action"]
        available = [name for name in V2_ACTIONS if row["actions"][name]["available"]]
        if deployed["action"] == DIRECT:
            assert deployed["prefilled"] == [DIRECT]                 # kept on its own signal: one prefill
        elif choice == "best":
            assert deployed["prefilled"] == [DIRECT]                 # the best evidence action needs no signal
        elif choice == "route":
            assert deployed["prefilled"] == [DIRECT, "static_mixed"]  # the route is read on static_mixed alone
        else:
            assert set(deployed["prefilled"]) == set(available)
        assert deployed["prefill_calls"] <= len(available)


def test_the_deployment_path_parses_the_answer_first_format_like_the_full_decode():
    """WebQA answers in two lines. The deployed answer is the sentence line, as in the full decode, so the two agree."""
    import dataclasses

    from omniaware import pipeline
    from tests.test_v2_prompts import FakeQwen, _examples

    class TwoLineQwen(FakeQwen):
        def run(self, enc, make_variants, max_new_tokens, decode_static=True):
            torch.manual_seed(enc.length)                            # same prompt, same logits and attention
            results, attn, logits = super().run(enc, make_variants, max_new_tokens, decode_static)
            if "static" in results:
                results["static"] = dataclasses.replace(results["static"],
                                                        text=f"answer{enc.length}\nThe answer is {enc.length}.")
            return results, attn, logits

    e = {**_examples(1)[0], "dataset": "webqa_full"}
    cands = {c["cid"]: c for c in e["candidates"]}
    rel = {cid: 1.0 / int(cid[1:]) for cid in cands}
    chunks = {DIRECT: None, "static_text": [cands[k] for k in ("k1", "k2", "k3")],
              "static_image": [cands["i1"], cands["i2"]], "static_mixed": [cands["k1"], cands["i1"], cands["k2"]],
              "rerank_static": [cands[k] for k in ("k2", "k3", "k4")]}
    qwen = TwoLineQwen("unused")
    enc_d = qwen.encode(*pipeline.build_segments(e, None, "default"))
    full = run.decode_v2_actions(qwen, e, rel, chunks, (qwen.run(enc_d, None, 64)[0]["static"], enc_d), {}, 64,
                                 "default", 0.3)
    assert full[DIRECT]["answer"].startswith("The answer is") and full[DIRECT]["answer_first_ok"] is True
    for tau in (-1e9, 1e9):
        for choice in ("best", "route"):
            rule = _rule(tau, choice, "u_omni")
            deployed = run.deploy_v2(qwen, e, rel, chunks, rule, 64, "default", 0.3)
            expected = arbitration.apply_decision(rule, full)
            assert deployed["action"] == expected["action"] and deployed["answer"] == expected["answer"]
            assert deployed["answer"].startswith("The answer is")
