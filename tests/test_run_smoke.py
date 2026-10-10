"""End-to-end run.main() with a fake model and synthetic OK-VQA-like data."""
import json

import pytest
from dataclasses import dataclass, field

import torch

from omniaware import data, paths, run
from omniaware.qwen_eab import Result


@pytest.fixture(autouse=True)
def tiny_reserves(monkeypatch):
    # The synthetic universes hold a few dozen questions: shrink the calibration reserves to the --n-calib
    # used below, so that calibration, dev and test are all non-empty.
    for family, reserve in {"okvqa": 6, "webqa": 4, "videorag_synth": 4}.items():
        monkeypatch.setitem(data.CALIB_RESERVE, family, reserve)
    monkeypatch.setitem(data.TRAIN_RESERVE, "webqa", 4)


def _examples(n):
    out = []
    for i in range(n):
        cands = [{"cid": f"k{j}", "modality": "text", "title": f"T{j}", "text": f"passage {j} of q{i}",
                  "image": None, "gold": None} for j in range(1, 8)]
        out.append({"dataset": "okvqa", "qid": f"q{i:03d}", "question": f"question {i}?", "qcate": "okvqa",
                    "query_images": [], "candidates": cands, "fixed_evidence": False, "golds": ["x"]})
    return out


@dataclass
class FakeEncoded:
    length: int
    split: int
    spans: dict = field(default_factory=dict)
    input_ids: torch.Tensor | None = None


class FakeQwen:
    """Mimics QwenEAB.encode/run; answers encode which arm produced them."""

    calls = []

    def __init__(self, model_path, **kw):
        self.device, self.n_layers = "cpu", 4
        self.last_prefill_seconds = 0.0
        self.tok = type("Tok", (), {"pad_token_id": 0, "eos_token_id": 2})()

    def encode(self, segments, split_marker):
        spans, pos = {}, 0
        for s in segments:
            n = len(s.text or "")
            if s.chunk:
                a, _ = spans.get(s.chunk, (pos, pos))
                spans[s.chunk] = (a, pos + n)
            pos += n
        return FakeEncoded(length=pos, split=sum(len(s.text or "") for s in segments[:split_marker]), spans=spans,
                           input_ids=torch.ones(1, pos, dtype=torch.long))

    def run(self, enc, make_variants, max_new_tokens, decode_static=True):
        attn = torch.rand(enc.length) + 1e-3
        logits = torch.randn(32) * (enc.length % 7 + 1)
        results = {}
        decode = decode_static(attn, logits) if callable(decode_static) else decode_static
        if decode:
            results["static"] = Result(f"static@{enc.length}", logits, [1], 0.01, attn)
        variants = make_variants(attn, logits) if make_variants else {}
        for name, spec in variants.items():
            results[name] = Result(f"{name}@{enc.length}" if spec else f"static@{enc.length}", logits, [1], 0.01, attn)
        FakeQwen.calls.append((enc.length, sorted(results)))
        return results, attn, logits


def test_run_writes_every_arm_from_one_input(tmp_path, monkeypatch):
    examples = _examples(20)
    monkeypatch.setitem(data.LOADERS, "okvqa", lambda **kw: examples)
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(paths, "LOCAL_CACHE", tmp_path / "cache")
    monkeypatch.setattr(run, "QwenEAB", FakeQwen)
    monkeypatch.setattr(run, "load_relevance", lambda exs, path: {
        e["qid"]: {c["cid"]: {"text": 1.0 / int(c["cid"][1:])} for c in e["candidates"]} for e in exs})

    run.main(["--dataset", "okvqa", "--split", "test", "--n-calib", "6", "--calib-source", "heldout",
              "--n-dev", "4", "--tag", "smoke", "--tau", "0.35", "--variants", "eab:g1.2:ans2:pos:gate"])

    out = tmp_path / "runs" / "smoke"
    rows = [json.loads(line) for line in (out / "okvqa.jsonl").read_text().splitlines()]
    assert len(rows) == 20 - 6 - 4
    config = json.loads((out / "okvqa_config.json").read_text())
    assert config["split"] == "test" and config["eab_uncertainty"] == "pre"
    assert config["split_sizes"] == {"universe": 20, "calib": 6, "dev": 4, "test": 10}
    calib_ids = {e["qid"] for e in data.split_examples(examples, 6, 4, reserve=6)["calib"]}
    for r in rows:
        assert r["qid"] not in calib_ids
        assert set(r["preds"]) == {"direct", "static", "eab:g1.2:ans2:pos:gate", "gated:static",
                                   "gated:eab:g1.2:ans2:pos:gate", "nte_trigger:static", "nte_shared:static"}
        assert len(r["chunks"]) == 5 and r["chunks"] == ["k1", "k2", "k3", "k4", "k5"]
        expected = r["preds"]["static"] if r["retrieve"] else r["preds"]["direct"]
        assert r["preds"]["gated:static"] == expected
        assert r["u_eab"] == r["u_pre"]
        assert "scores" not in r
        assert set(r["arm_seconds"]) == {"direct", "static", "eab:g1.2:ans2:pos:gate"}
        assert set(r["prefill_seconds"]) == {"direct", "evidence"}
        assert set(r["ard"]) == {"static", "eab:g1.2:ans2:pos:gate"} and all(0 <= v <= 0.7 for v in r["ard"].values())
        assert set(r["span_tokens"]) == set(r["chunks"]) and len(r["chunk_gold"]) == 5
        assert len(r["chunk_modalities"]) == 5 and r["chunk_has_answer"] == [False] * 5   # gold "x" in no passage


def test_fixed_permutation_and_replication_arm(tmp_path, monkeypatch):
    from omniaware import order
    examples = _examples(14)
    monkeypatch.setitem(data.LOADERS, "okvqa", lambda **kw: examples)
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(paths, "LOCAL_CACHE", tmp_path / "cache")
    monkeypatch.setattr(run, "QwenEAB", FakeQwen)
    monkeypatch.setattr(run, "load_relevance", lambda exs, path: {
        e["qid"]: {c["cid"]: {"text": 1.0 / int(c["cid"][1:])} for c in e["candidates"]} for e in exs})
    schedule = {o["order_id"]: o["permutation"] for o in order.permutation_schedule()}

    run.main(["--dataset", "okvqa", "--split", "test", "--n-calib", "6", "--calib-source", "heldout",
              "--n-dev", "0", "--tag", "perm", "--order", "perm:s17-p1", "--replication-gamma", "1.2",
              "--variants", "eab:g1.2:ans2:pos:gate"])

    rows = [json.loads(line) for line in (tmp_path / "runs" / "perm" / "okvqa.jsonl").read_text().splitlines()]
    ranked = ["k1", "k2", "k3", "k4", "k5"]
    for r in rows:
        assert r["chunks"] == [ranked[i] for i in schedule["s17-p1"]]
        assert {"replication", "gated:replication"} <= set(r["preds"])
        counts = r["replication_counts"]
        gate = r["u_eab"] > 0.40
        assert counts == order.replication_counts([r["rel"][c] for c in r["chunks"]], r["u_eab"], gate, 1.2)
        if all(n == 1 for n in counts):
            assert r["preds"]["replication"] == r["preds"]["static"]
        else:
            assert r["preds"]["replication"] != r["preds"]["static"]


def test_cross_encoder_selectors(tmp_path, monkeypatch):
    from omniaware import rerank
    examples = _examples(14)
    monkeypatch.setitem(data.LOADERS, "okvqa", lambda **kw: examples)
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(paths, "LOCAL_CACHE", tmp_path / "cache")
    monkeypatch.setattr(run, "QwenEAB", FakeQwen)
    monkeypatch.setattr(run, "load_relevance", lambda exs, path: {
        e["qid"]: {c["cid"]: {"text": 1.0 / int(c["cid"][1:])} for c in e["candidates"]} for e in exs})
    # The cross-encoder prefers the passages the retriever ranks last (k3..k7).
    monkeypatch.setattr(rerank, "ce_logits", lambda exs, path, model: {
        e["qid"]: {c["cid"]: float(c["cid"][1:]) for c in e["candidates"]} for e in exs})
    for selector in ("ce", "ce_calibrated"):
        run.main(["--dataset", "okvqa", "--split", "test", "--n-calib", "6", "--calib-source", "heldout",
                  "--n-dev", "0", "--tag", selector, "--selector", selector, "--variants", "eab:g1.2:ans2:pos:gate"])
        rows = [json.loads(line) for line in (tmp_path / "runs" / selector / "okvqa.jsonl").read_text().splitlines()]
        for r in rows:
            # CE top-5, presented in retriever-relevance order.
            assert r["chunks"] == ["k3", "k4", "k5", "k6", "k7"]
            # EAB relevance is the per-modality CE percentile, increasing with the CE logit.
            assert [r["rel"][c] for c in r["chunks"]] == sorted(r["rel"][c] for c in r["chunks"])


def test_cite_style_run_exports_citerec_inputs(tmp_path, monkeypatch):
    from omniaware.eval.citation.from_run import export
    from omniaware.eval.citation.requests import requests_for_answer
    examples = _examples(12)
    monkeypatch.setitem(data.LOADERS, "okvqa", lambda **kw: examples)
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(paths, "LOCAL_CACHE", tmp_path / "cache")
    monkeypatch.setattr(run, "QwenEAB", FakeQwen)
    monkeypatch.setattr(run, "load_relevance", lambda exs, path: {
        e["qid"]: {c["cid"]: {"text": 1.0 / int(c["cid"][1:])} for c in e["candidates"]} for e in exs})
    run.main(["--dataset", "okvqa", "--split", "test", "--n-calib", "6", "--calib-source", "heldout",
              "--n-dev", "0", "--tag", "cite", "--answer-style", "cite", "--variants", "eab:g1.2:ans2:pos:gate"])
    pred = tmp_path / "runs" / "cite" / "okvqa.jsonl"
    rows = [json.loads(line) for line in pred.read_text().splitlines()]
    sources = rows[0]["citation_sources"]
    assert [s["citation_id"] for s in sources] == ["E1", "E2", "E3", "E4", "E5"]
    assert sources[0]["source_id"] == "k1" and sources[0]["rendered_text"].startswith("T1: passage 1")
    summary = export(pred, tmp_path / "citerec")
    assert summary["arms"] == ["static", "eab:g1.2:ans2:pos:gate"] and summary["answers"] == 2 * len(rows)
    answer = json.loads((tmp_path / "citerec" / "answers.jsonl").read_text().splitlines()[0])
    source = json.loads((tmp_path / "citerec" / "sources.jsonl").read_text().splitlines()[0])
    # A cited statement turns into an NLI request whose premise is the cited chunk text.
    row = dict(answer, raw_answer="It is passage one [E1] [E9].", evidence=source["evidence"],
               statements=[{"text": "It is passage one.", "citation_ids": ["E1", "E9"]}])
    (request,) = requests_for_answer(row)
    assert request["valid_citation_ids"] == ["E1"] and request["invalid_citation_ids"] == ["E9"]
    assert request["premise"] == sources[0]["rendered_text"]


def _webqa_examples(n):
    out = []
    for i in range(n):
        cands = [{"cid": f"t{j}", "modality": "text", "title": "", "text": f"fact {j} of q{i}", "image": None,
                  "gold": None} for j in range(1, 6)]
        cands += [{"cid": f"i{j}", "modality": "image", "title": f"cap {j}", "text": f"cap {j}", "image": f"img{j}.jpg",
                   "gold": None} for j in range(1, 6)]
        out.append({"dataset": "webqa", "qid": f"w{i:03d}", "question": f"q {i}?", "qcate": "text",
                    "query_images": [], "candidates": cands, "fixed_evidence": False, "golds": ["x"],
                    "gold_modalities": ["image"] if i % 2 else ["text"]})
    return out


@pytest.mark.parametrize("rule", ["mh", "mass", "entropy", "retriever", "all", "oracle"])
def test_route_rules_end_to_end(tmp_path, monkeypatch, rule):
    examples = _webqa_examples(12)
    monkeypatch.setitem(data.LOADERS, "webqa_open_full", lambda **kw: examples)
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(paths, "LOCAL_CACHE", tmp_path / "cache")
    monkeypatch.setattr(run, "QwenEAB", FakeQwen)
    # images score higher for odd questions, text for even ones
    monkeypatch.setattr(run, "load_relevance", lambda exs, path: {
        e["qid"]: {c["cid"]: {("image_clip" if c["modality"] == "image" else "text"):
                              (0.9 if (c["modality"] == "image") == (int(e["qid"][1:]) % 2 == 1) else 0.1)
                              - 0.01 * int(c["cid"][1:])} for c in e["candidates"]} for e in exs})
    extra = ["--first-round-routing"] if rule in ("mh", "mass", "entropy") else []
    run.main(["--dataset", "webqa_open_full", "--split", "test", "--n-calib", "4", "--calib-source", "heldout",
              "--n-dev", "0", "--tag", rule, "--route-rule", rule, "--image-mode", "clip",
              "--variants", "eab:g1.2:ans2:pos:gate"] + extra)
    rows = [json.loads(line) for line in (tmp_path / "runs" / rule / "webqa_open_full.jsonl").read_text().splitlines()]
    for r in rows:
        mods = r["route"]["modalities"]
        chosen = {c[0] for c in r["chunks"]}
        if rule == "all":
            assert mods == [] and len(r["chunks"]) == 5
        elif rule == "oracle":
            assert mods == r["gold_modalities"] and chosen == {r["gold_modalities"][0][0]}
        elif rule == "retriever":
            # rule logic is unit-tested; here: per-modality calibrated maxima are recorded and used
            assert set(r["route"]["statistics"]) == {"text", "image"} and mods and set(mods) <= {"text", "image"}
            assert chosen <= {m[0] for m in mods}
        else:
            assert r["route"]["rule"] == rule and set(mods) <= {"text", "image"}
            assert chosen <= {m[0] for m in mods} or not mods


def test_universalrag_query_routes(tmp_path, monkeypatch):
    from omniaware.baselines import universalrag
    examples = _webqa_examples(12)
    monkeypatch.setitem(data.LOADERS, "webqa_open_full", lambda **kw: examples)
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(paths, "LOCAL_CACHE", tmp_path / "cache")
    monkeypatch.setattr(run, "QwenEAB", FakeQwen)
    monkeypatch.setattr(run, "load_relevance", lambda exs, path: {
        e["qid"]: {c["cid"]: {("image_clip" if c["modality"] == "image" else "text"): 1 - 0.01 * int(c["cid"][1:])}
                   for c in e["candidates"]} for e in exs})
    labels = {0: ["image"], 1: ["paragraph"], 2: ["no"], 3: ["paragraph", "image"]}
    routes = [dict(q, retrieval=labels[i % 4]) for i, q in enumerate(universalrag.export_queries(examples))]
    assert all("answer" not in q and "golds" not in q for q in routes)
    (tmp_path / "routes.json").write_text(json.dumps(routes))
    run.main(["--dataset", "webqa_open_full", "--split", "test", "--n-calib", "4", "--calib-source", "heldout",
              "--n-dev", "0", "--tag", "uq", "--route-rule", "query", "--query-routes", str(tmp_path / "routes.json"),
              "--image-mode", "clip", "--variants", "eab:g1.2:ans2:pos:gate"])
    rows = [json.loads(line) for line in (tmp_path / "runs" / "uq" / "webqa_open_full.jsonl").read_text().splitlines()]
    for r in rows:
        lab = labels[int(r["qid"][1:]) % 4]
        assert r["route"]["labels"] == lab
        if lab == ["no"]:
            assert r["preds"]["universalrag"] == r["preds"]["direct"]
        else:
            assert r["preds"]["universalrag"] == r["preds"]["static"]
            assert {c[0] for c in r["chunks"]} == {{"image": "i", "text": "t"}[m] for m in r["route"]["modalities"]}


def test_seakr_arms(tmp_path, monkeypatch):
    from omniaware.baselines import seakr
    examples = _examples(14)
    monkeypatch.setitem(data.LOADERS, "okvqa", lambda **kw: examples)
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(paths, "LOCAL_CACHE", tmp_path / "cache")
    monkeypatch.setattr(run, "QwenEAB", FakeQwen)
    monkeypatch.setattr(run, "load_relevance", lambda exs, path: {
        e["qid"]: {c["cid"]: {"text": 1.0 / int(c["cid"][1:])} for c in e["candidates"]} for e in exs})
    # eigen score = prompt length (question-only prompts differ by question number; k3 is the most confident doc)
    def fake_score(qwen, enc, n, m, layer=None, seed=42):
        return float(enc.length) if "k3" not in enc.spans else -1e9
    monkeypatch.setattr(seakr, "score_input", fake_score)
    run.main(["--dataset", "okvqa", "--split", "test", "--n-calib", "6", "--calib-source", "heldout",
              "--n-dev", "0", "--tag", "sk", "--seakr-samples", "4", "--seakr-threshold", "0",
              "--variants", "eab:g1.2:ans2:pos:gate"])
    rows = [json.loads(line) for line in (tmp_path / "runs" / "sk" / "okvqa.jsonl").read_text().splitlines()]
    for r in rows:
        assert r["seakr"]["retrieve"]                     # every eigen score > 0
        assert r["seakr"]["chosen"] == "k3"
        assert r["preds"]["seakr"] != r["preds"]["direct"]
        assert r["preds"]["seakr_trigger:static"] == r["preds"]["static"]


@pytest.mark.parametrize("extra", [["--relevance-norm", "raw"], ["--relevance-norm", "minmax"], ["--no-calibration"]])
def test_controller_ablation_options(tmp_path, monkeypatch, extra):
    examples = _examples(14)
    for e in examples:
        e["query_images"] = ["q.jpg"]
    monkeypatch.setitem(data.LOADERS, "okvqa", lambda **kw: examples)
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(paths, "LOCAL_CACHE", tmp_path / "cache")
    monkeypatch.setattr(run, "QwenEAB", FakeQwen)
    monkeypatch.setattr(run, "load_relevance", lambda exs, path: {
        e["qid"]: {c["cid"]: {"text": 3.0 / int(c["cid"][1:])} for c in e["candidates"]} for e in exs})
    variants = "eab:g1.2:ans2:pos:gate,eab:g1.2:ans2:pos:gate:all:random,eab:g1.2:ans2:pos:gate:all:retriever:ctx"
    run.main(["--dataset", "okvqa", "--split", "test", "--n-calib", "6", "--calib-source", "heldout", "--n-dev", "0",
              "--tag", "abl", "--variants", variants] + extra)
    rows = [json.loads(line) for line in (tmp_path / "runs" / "abl" / "okvqa.jsonl").read_text().splitlines()]
    config = json.loads((tmp_path / "runs" / "abl" / "okvqa_config.json").read_text())
    for r in rows:
        assert r["chunks"] == ["k1", "k2", "k3", "k4", "k5"]
        if "--relevance-norm" in extra and "raw" in extra:
            assert r["rel"]["k1"] == 3.0
        else:
            assert r["rel"]["k1"] == 1.0 and 0 <= r["rel"]["k5"] < 1
        assert set(r["eab_active"]) == set(variants.split(","))
    if "--no-calibration" in extra:
        assert config["norm_pre"] == {"median": 0.0, "iqr": 1.0} and config["relevance_norm"] == "minmax"


def test_interventions_and_relevance_noise(tmp_path, monkeypatch):
    cities = ["Paris", "Rome", "Berlin", "Madrid", "Vienna", "Prague", "Oslo", "Lisbon"]
    examples = _examples(16)
    for i, e in enumerate(examples):
        e["golds"] = [cities[i % len(cities)]]
        e["candidates"][0]["text"] = f"The capital is {e['golds'][0]} and more words follow here."
    monkeypatch.setitem(data.LOADERS, "okvqa", lambda **kw: examples)
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(paths, "LOCAL_CACHE", tmp_path / "cache")
    monkeypatch.setattr(run, "QwenEAB", FakeQwen)
    monkeypatch.setattr(run, "load_relevance", lambda exs, path: {
        e["qid"]: {c["cid"]: {"text": 1.0 / int(c["cid"][1:])} for c in e["candidates"]} for e in exs})
    kinds = ["mask", "cf", "random_edit", "rerun", "distractors", "inject_answer", "inject_instruction"]
    run.main(["--dataset", "okvqa", "--split", "test", "--n-calib", "6", "--calib-source", "heldout",
              "--n-dev", "2", "--tag", "iv", "--tau=-1e9", "--tau-bias=-1e9",
              "--variants", "eab:g1.2:ans1:pos:gate", "--relevance-noise", "0.1", "--intervention", *kinds])
    rows = [json.loads(line) for line in (tmp_path / "runs" / "iv" / "okvqa.jsonl").read_text().splitlines()]
    assert rows
    for r in rows:
        iv = r["interventions"]
        assert iv["eligible"] and iv["target"] == "k1" and iv["gold_in_target"] == r["golds"][0]
        assert iv["entity"] in cities and iv["entity"] != r["golds"][0]
        for kind in kinds:
            assert set(iv[kind]["preds"]) == {"static", "eab:g1.2:ans1:pos:gate"}
        assert "__inject__" in iv["inject_answer"]["chunks"] and iv["inject_answer"]["chunks"][0] == "__inject__"
        assert len(set(iv["distractors"]["distractors"]) & set(r["chunks"])) == 0
        assert iv["distractors"]["chunks"][r["chunks"].index("k1")] == "k1"
        assert all(0.0 <= v <= 1.0 for v in r["rel"].values())


def test_videorag_run_with_video_chunks_and_m2rag_guard(tmp_path, monkeypatch):
    examples = []
    for i in range(12):
        cands = [{"cid": f"v{j}", "modality": "video", "title": "", "text": f"transcript {j} of q{i}",
                  "image": None, "frames": [], "gold": j == 1, "source_id": f"vid{j}"} for j in range(1, 6)]
        cands += [{"cid": f"c{j}", "modality": "video", "title": "", "text": f"clip {j}", "image": None,
                   "frames": [], "gold": False, "source_id": f"clip{j}"} for j in range(1, 6)]
        examples.append({"dataset": "videorag", "qid": f"videorag_synth_vid{i % 4}_{i}", "question": f"step {i}?",
                         "qcate": "videorag_synth", "query_images": [], "candidates": cands,
                         "fixed_evidence": False, "golds": ["Detach the caliper."], "gold_modalities": ["video"]})
    monkeypatch.setitem(data.LOADERS, "videorag_synth", lambda **kw: examples)
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(paths, "LOCAL_CACHE", tmp_path / "cache")
    monkeypatch.setattr(run, "QwenEAB", FakeQwen)
    monkeypatch.setattr(run, "load_relevance", lambda exs, path: {
        e["qid"]: {c["cid"]: {"video_text": 1.0 / int(c["cid"][1:])} for c in e["candidates"]} for e in exs})
    base = ["--dataset", "videorag_synth", "--split", "test", "--n-calib", "4", "--calib-source", "heldout",
            "--n-dev", "2", "--variants", "eab:g1.2:ans2:pos:gate"]
    with pytest.raises(SystemExit):
        run.main(base + ["--tag", "vr_m2rag", "--m2rag"])
    run.main(base + ["--tag", "vr"])
    rows = [json.loads(line) for line in (tmp_path / "runs" / "vr" / "videorag_synth.jsonl").read_text().splitlines()]
    assert len(rows) == 6 and all(len(r["chunks"]) == 5 for r in rows)
    assert all(m == "video" for r in rows for m, _ in r["span_tokens"].values())


def test_cold_start_probe_round_gates_and_routes(tmp_path, monkeypatch):
    examples = _examples(16)
    for e in examples:   # two modalities so that the probe route has a choice
        for c in e["candidates"][4:]:
            c.update(modality="image", image=f"/img/{c['cid']}.jpg")
    monkeypatch.setitem(data.LOADERS, "okvqa", lambda **kw: examples)
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(paths, "LOCAL_CACHE", tmp_path / "cache")
    monkeypatch.setattr(run, "QwenEAB", FakeQwen)
    monkeypatch.setattr(run, "load_relevance", lambda exs, path: {
        e["qid"]: {c["cid"]: ({"text": 1.0 / int(c["cid"][1:])} if c["modality"] == "text"
                              else {"image_clip": 0.5 / int(c["cid"][1:])}) for c in e["candidates"]} for e in exs})
    run.main(["--dataset", "okvqa", "--split", "test", "--n-calib", "6", "--calib-source", "heldout",
              "--n-dev", "2", "--tag", "probe", "--probe-uncertainty", "--first-round-routing", "--u-combine", "zsum",
              "--attn-layer", "mid"])
    out = tmp_path / "runs" / "probe"
    config = json.loads((out / "okvqa_config.json").read_text())
    assert config["norm_probe"] and config["comb_probe"]["mode"] == "zsum"
    for r in (json.loads(line) for line in (out / "okvqa.jsonl").read_text().splitlines()):
        assert r["u_gate"] == r["u_probe"] and r["retrieve"] == (r["u_probe"] > 0.35)
        assert len(r["probe_chunks"]) == 5 and r["route"]["first_round"] == r["probe_chunks"]
        assert "probe" in r["prefill_seconds"]
        expected = r["preds"]["static"] if r["retrieve"] else r["preds"]["direct"]
        assert r["preds"]["gated:eab:g1.2:ans1:pos:gate"] in (expected, r["preds"]["eab:g1.2:ans1:pos:gate"])


def test_sharded_run_covers_the_split_once(tmp_path, monkeypatch):
    from omniaware import merge_shards
    from scripts import launch
    examples = _examples(23)
    monkeypatch.setitem(data.LOADERS, "okvqa", lambda **kw: examples)
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(paths, "LOCAL_CACHE", tmp_path / "cache")
    monkeypatch.setattr(run, "QwenEAB", FakeQwen)
    monkeypatch.setattr(run, "load_relevance", lambda exs, path: {
        e["qid"]: {c["cid"]: {"text": 1.0 / int(c["cid"][1:])} for c in e["candidates"]} for e in exs})
    base = ["--dataset", "okvqa", "--split", "test", "--n-calib", "6", "--calib-source", "heldout",
            "--n-dev", "2", "--tag", "sh"]
    run.main(base + ["--calibrate-only"])
    out = tmp_path / "runs" / "sh"
    assert not (out / "okvqa.jsonl").exists()
    config = json.loads((out / "okvqa_config.json").read_text())
    assert config["evaluation_questions"] == 15 and "num_shards" not in config
    for i in range(3):
        run.main(base + ["--num-shards", "3", "--shard-id", str(i)])
    with pytest.raises(SystemExit):
        merge_shards.merge(out / "okvqa", expected_questions=16)       # count mismatch is refused
    merge_shards.merge(out / "okvqa", expected_questions=15)
    qids = [json.loads(line)["qid"] for line in (out / "okvqa.jsonl").read_text().splitlines()]
    expected = {e["qid"] for e in data.split_examples(examples, 6, 2, reserve=6)["test"]}
    assert len(qids) == len(set(qids)) == 15 and set(qids) == expected
    assert len(list((out / "okvqa.shards").glob("*.jsonl"))) == 3
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")
    assert launch.detect_gpus() == ["2", "5"]


def test_split_protocol_violations_stop_the_run_before_the_model_loads(tmp_path, monkeypatch):
    # F-1 / D1: an empty evaluation split and a calibration size beyond the reserve are errors, not silent runs.
    monkeypatch.setitem(data.LOADERS, "okvqa", lambda **kw: _examples(20))
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(paths, "LOCAL_CACHE", tmp_path / "cache")
    monkeypatch.setattr(run, "QwenEAB", lambda *a, **k: pytest.fail("the model must not load"))
    base = ["--dataset", "okvqa", "--split", "test", "--calib-source", "heldout", "--tag", "bad"]
    with pytest.raises(ValueError, match="empty"):
        run.main(base + ["--n-calib", "6", "--n-dev", "14"])          # 20 - 6 reserve - 14 dev = 0 test questions
    with pytest.raises(ValueError, match="reserve"):
        run.main(base + ["--n-calib", "7", "--n-dev", "2"])
