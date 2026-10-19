"""Cache fingerprints, resumable outputs, shard cache rules, completion markers and merge checks."""
import copy
import json

import pytest

from omniaware import data, merge_shards, paths, relevance, rerank, runio
from omniaware import run
from scripts import launch
from test_run_smoke import FakeQwen, _examples, _webqa_examples

pytestmark = pytest.mark.usefixtures("tiny_reserves")


# ------------------------------------------------------------------ runio --

def test_partial_last_line_is_cut_and_its_question_is_answered_again(tmp_path):
    path = tmp_path / "okvqa.jsonl"
    path.write_bytes(b'{"qid": "a"}\n{"qid": "b"}\n{"qid": "c", "pr')
    assert [r["qid"] for r in runio.load_partial_jsonl(path)] == ["a", "b"]
    assert path.read_bytes() == b'{"qid": "a"}\n{"qid": "b"}\n'
    # a complete record without its newline is also an interrupted write: it is cut and recomputed
    path.write_bytes(b'{"qid": "a"}\n{"qid": "b"}')
    assert [r["qid"] for r in runio.load_partial_jsonl(path)] == ["a"]
    assert path.read_bytes() == b'{"qid": "a"}\n'


def test_damaged_line_before_the_end_is_corruption_not_resume(tmp_path):
    path = tmp_path / "okvqa.jsonl"
    path.write_bytes(b'{"qid": "a"\n{"qid": "b"}\n')
    with pytest.raises(ValueError, match="line 1 is damaged"):
        runio.load_partial_jsonl(path)


def test_strict_json_writes_nan_and_infinity_as_null():
    text = runio.dumps_strict({"a": float("nan"), "b": [1.0, float("inf")], "c": 2.5, "d": {"e": float("-inf")}},
                              ensure_ascii=False)
    assert json.loads(text) == {"a": None, "b": [1.0, None], "c": 2.5, "d": {"e": None}}
    assert "NaN" not in text and "Infinity" not in text


def test_qid_set_digest_ignores_order_and_detects_changes():
    assert runio.qid_set_sha256(["b", "a", "c"]) == runio.qid_set_sha256(["a", "c", "b"])
    assert runio.qid_set_sha256(["a", "b"]) != runio.qid_set_sha256(["a", "c"])


def test_fingerprint_follows_every_input_of_the_raw_scores(tmp_path, monkeypatch):
    e = _examples(1)[0]
    base = runio.fingerprint(e, {"x": 1})
    assert runio.fingerprint(copy.deepcopy(e), {"x": 1}) == base
    for edit in (lambda x: x["candidates"][0].update(text="other"),
                 lambda x: x["candidates"][0].update(title="other"),
                 lambda x: x["candidates"][0].update(image="/img/a.jpg"),
                 lambda x: x.update(question="another?")):
        changed = copy.deepcopy(e)
        edit(changed)
        assert runio.fingerprint(changed, {"x": 1}) != base
    assert runio.fingerprint(e, {"x": 2}) != base          # settings are part of the key
    gold = copy.deepcopy(e)
    gold["golds"] = ["something else"]                    # gold answers do not change any raw score
    assert runio.fingerprint(gold, {"x": 1}) == base
    monkeypatch.setattr(paths, "BGE", tmp_path / "another-bge")
    assert runio.encoder_identity()["bge"]["path"] == str(tmp_path / "another-bge")
    assert runio.encoder_identity()["bge"]["dir"] == "another-bge"


def test_completion_marker_is_trusted_only_while_config_and_output_match(tmp_path):
    config = tmp_path / "okvqa_config.json"
    config.write_text(json.dumps({"evaluation_questions": 2}))
    out = tmp_path / "okvqa.jsonl"
    out.write_text('{"qid": "a"}\n{"qid": "b"}\n')
    assert runio.verify_completion(tmp_path, "okvqa") is False
    runio.write_completion_marker(tmp_path, "okvqa", 2, out, config)
    assert runio.verify_completion(tmp_path, "okvqa") is True
    out.write_text('{"qid": "a"}\n')                      # a partial file after the marker
    with pytest.raises(ValueError, match="stale"):
        runio.verify_completion(tmp_path, "okvqa")
    out.write_text('{"qid": "a"}\n{"qid": "c"}\n')        # right count, other content
    with pytest.raises(ValueError, match="stale"):
        runio.verify_completion(tmp_path, "okvqa")


# --------------------------------------------------- relevance and CE caches --

def test_relevance_cache_reuses_identical_questions_and_recomputes_the_rest(tmp_path, monkeypatch):
    calls = []

    def fake_raw_scores(examples, bge, clip):
        calls.append([e["qid"] for e in examples])
        return {e["qid"]: {c["cid"]: {"text": float(len(c["text"]))} for c in e["candidates"]} for e in examples}

    monkeypatch.setattr(relevance, "BGE", lambda: None)
    monkeypatch.setattr(relevance, "raw_scores", fake_raw_scores)
    examples = _examples(3)
    cache = tmp_path / "cache" / "relevance_okvqa.json"
    first = run.load_relevance(examples, cache)
    assert calls == [["q000", "q001", "q002"]]
    assert json.loads(cache.read_text())["q000"]["fingerprint"] == run.relevance_fingerprint(examples[0])
    assert run.load_relevance(examples, cache) == first and len(calls) == 1      # all hits: nothing computed

    edited = copy.deepcopy(examples)
    edited[1]["candidates"][0]["title"] = "retitled"                              # same text, other title
    run.load_relevance(edited, cache)
    assert calls[-1] == ["q001"]

    monkeypatch.setattr(paths, "BGE", tmp_path / "another-bge")                   # another encoder
    run.load_relevance(examples, cache)
    assert calls[-1] == ["q000", "q001", "q002"]

    monkeypatch.setattr(paths, "BGE", tmp_path / "third-bge")
    with pytest.raises(SystemExit, match="run --calibrate-only first"):           # sharded workers may not recompute
        run.load_relevance(examples, cache, write=False)


def test_cross_encoder_cache_checks_fingerprints(tmp_path, monkeypatch):
    import transformers

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

    class FakeAuto:
        @staticmethod
        def from_pretrained(path):
            return FakeModel()

    scored = []

    def fake_score_ce(question, texts, tokenizer, model, batch_size):
        scored.append(question)
        return [float(len(t["text"])) for t in texts], [1] * len(texts)

    monkeypatch.setattr(transformers, "AutoTokenizer", FakeAuto)
    monkeypatch.setattr(transformers, "AutoModelForSequenceClassification", FakeAuto)
    monkeypatch.setattr(rerank, "score_ce", fake_score_ce)
    examples = _examples(3)
    cache, model_dir = tmp_path / "ce.json", tmp_path / "reranker"
    first = rerank.ce_logits(examples, cache, model_dir)
    assert scored == ["question 0?", "question 1?", "question 2?"]
    assert rerank.ce_logits(examples, cache, model_dir) == first and len(scored) == 3

    edited = copy.deepcopy(examples)
    edited[2]["candidates"][0]["text"] = "a different passage"
    second = rerank.ce_logits(edited, cache, model_dir)
    assert scored[-1] == "question 2?" and len(scored) == 4
    assert second["q002"]["k1"] == float(len("T1: a different passage"))   # title: text, as scored
    assert second["q000"] == first["q000"]

    with pytest.raises(SystemExit, match="run --calibrate-only first"):
        rerank.ce_logits(examples, cache, tmp_path / "other-reranker", write=False)


# --------------------------------------------------------- run-level behaviour --

def _setup(tmp_path, monkeypatch, n=14, text_suffix=""):
    examples = _examples(n)
    for e in examples:
        for c in e["candidates"]:
            c["text"] += text_suffix
    monkeypatch.setitem(data.LOADERS, "okvqa", lambda **kw: examples)
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(paths, "LOCAL_CACHE", tmp_path / "cache")
    monkeypatch.setattr(run, "QwenEAB", FakeQwen)
    monkeypatch.setattr(run, "load_relevance", lambda exs, path, **kw: {
        e["qid"]: {c["cid"]: {"text": 1.0 / int(c["cid"][1:])} for c in e["candidates"]} for e in exs})
    return examples


BASE = ["--dataset", "okvqa", "--split", "test", "--n-calib", "6", "--calib-source", "heldout", "--n-dev", "0",
        "--variants", "eab:g1.2:ans2:pos:gate"]


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_interrupted_run_resumes_after_a_torn_line_and_writes_its_marker(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    run.main(BASE + ["--tag", "rs"])
    out_dir = tmp_path / "runs" / "rs"
    expected = {r["qid"] for r in _rows(out_dir / "okvqa.jsonl")}
    assert len(expected) == 8 and runio.verify_completion(out_dir, "okvqa") is True
    lines = (out_dir / "okvqa.jsonl").read_bytes().splitlines(keepends=True)
    (out_dir / "okvqa.jsonl").write_bytes(b"".join(lines[:3]) + lines[3][:40])   # killed mid-write
    runio.completion_marker(out_dir, "okvqa").unlink()
    run.main(BASE + ["--tag", "rs"])
    rows = _rows(out_dir / "okvqa.jsonl")
    assert len(rows) == 8 and {r["qid"] for r in rows} == expected
    assert runio.verify_completion(out_dir, "okvqa") is True


def test_saved_answers_refuse_changed_calibration_and_cache_fingerprints(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    run.main(BASE + ["--tag", "cf"])
    _setup(tmp_path, monkeypatch, text_suffix=" (revised)")                  # every candidate text changed
    with pytest.raises(SystemExit, match="cache_fingerprints"):
        run.main(BASE + ["--tag", "cf"])


def test_config_records_the_evaluation_set_and_cache_fingerprints(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    run.main(BASE + ["--tag", "cfg"])
    config = json.loads((tmp_path / "runs" / "cfg" / "okvqa_config.json").read_text())
    evals = [r["qid"] for r in _rows(tmp_path / "runs" / "cfg" / "okvqa.jsonl")]
    assert config["evaluation_questions"] == len(evals) == 8
    assert config["evaluation_qids_sha256"] == runio.qid_set_sha256(evals)
    assert set(config["cache_fingerprints"]) == {"relevance", "ce", "ucal", "seakr"}
    assert config["cache_fingerprints"]["ce"] is None and config["cache_fingerprints"]["seakr"] is None
    assert all(v is None or len(v) == 64 for v in config["cache_fingerprints"].values())


def test_single_process_marker_is_written_and_nans_are_null(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    run.main(BASE + ["--tag", "nan"])
    out_dir = tmp_path / "runs" / "nan"
    marker = json.loads((out_dir / "okvqa.merged.json").read_text())
    assert marker["questions"] == 8 and marker["config_sha256"] is not None
    text = (out_dir / "okvqa.jsonl").read_text()
    assert "NaN" not in text and "Infinity" not in text
    config_text = (out_dir / "okvqa_config.json").read_text()
    assert "NaN" not in config_text and "Infinity" not in config_text


def test_sharded_workers_read_the_caches_and_never_write_them(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    base = BASE + ["--tag", "sh"]
    with pytest.raises(SystemExit):                                          # no configuration yet
        run.main(base + ["--num-shards", "2", "--shard-id", "0"])
    run.main(base + ["--calibrate-only"])
    cache = tmp_path / "cache"
    snapshot = {p.name: p.read_bytes() for p in cache.iterdir()}
    assert any(name.startswith("ucal_") for name in snapshot)
    for i in range(2):
        run.main(base + ["--num-shards", "2", "--shard-id", str(i)])
    assert {p.name: p.read_bytes() for p in cache.iterdir()} == snapshot


def test_shards_refuse_missing_calibration_entries(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    base = BASE + ["--tag", "sm"]
    run.main(base + ["--calibrate-only"])
    (ucal,) = (tmp_path / "cache").glob("ucal_*.json")
    ucal.unlink()
    with pytest.raises(SystemExit, match="run --calibrate-only first"):
        run.main(base + ["--num-shards", "2", "--shard-id", "0"])


def test_query_routes_file_is_part_of_the_calibration_key(tmp_path, monkeypatch):
    from omniaware.baselines import universalrag
    examples = _webqa_examples(12)
    for e in examples:                      # the UniversalRAG arm orders candidates by the suite's own scores
        for c in e["candidates"]:
            c["suite_score"] = 1 - 0.01 * int(c["cid"][1:])
    monkeypatch.setitem(data.LOADERS, "webqa_open_full", lambda **kw: examples)
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(paths, "LOCAL_CACHE", tmp_path / "cache")
    monkeypatch.setattr(run, "QwenEAB", FakeQwen)
    monkeypatch.setattr(run, "load_relevance", lambda exs, path, **kw: {
        e["qid"]: {c["cid"]: {("image_clip" if c["modality"] == "image" else "text"): 1 - 0.01 * int(c["cid"][1:])}
                   for c in e["candidates"]} for e in exs})
    labels = {0: ["image"], 1: ["paragraph"], 2: ["no"], 3: ["paragraph", "image"]}
    routes = [dict(q, retrieval=labels[i % 4]) for i, q in enumerate(universalrag.export_queries(examples))]
    routes_path = tmp_path / "routes.json"
    routes_path.write_text(json.dumps(routes))
    base = ["--dataset", "webqa_open_full", "--split", "test", "--n-calib", "4", "--calib-source", "heldout",
            "--n-dev", "0", "--tag", "uq", "--route-rule", "query", "--query-routes", str(routes_path),
            "--image-mode", "clip", "--variants", "eab:g1.2:ans2:pos:gate"]
    run.main(base)
    routes_path.write_text(json.dumps([dict(r, note="edited") for r in routes]))   # same routes, other file
    with pytest.raises(SystemExit, match="cache_fingerprints"):
        run.main(base)


def test_answer_instruction_is_part_of_the_calibration_key(tmp_path, monkeypatch):
    from omniaware import prompts
    from omniaware.baselines import seakr
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(seakr, "score_input", lambda qwen, enc, n, m, layer=None, seed=42: float(enc.length))
    base = BASE + ["--seakr-samples", "4"]
    cache = tmp_path / "cache"
    run.main(base + ["--tag", "ia"])
    first = {}
    for kind in ("ucal", "seakr"):
        (first[kind],) = cache.glob(f"{kind}_*.json")
    saved = {kind: path.read_bytes() for kind, path in first.items()}
    monkeypatch.setitem(prompts.INSTRUCTIONS, "okvqa", "Answer with one word.")    # same answer style, other text
    with pytest.raises(SystemExit, match="cache_fingerprints"):
        run.main(base + ["--tag", "ia"])
    run.main(base + ["--tag", "ib"])
    for kind, path in first.items():
        # The other instruction gets a file of its own; the first instruction's file is left as it was.
        (other,) = set(cache.glob(f"{kind}_*.json")) - {path}
        assert path.read_bytes() == saved[kind]
        assert path.name.rsplit("_ins", 1)[0] == other.name.rsplit("_ins", 1)[0]
        before, after = ({q: v["fingerprint"] for q, v in json.loads(p.read_text()).items()} for p in (path, other))
        assert before.keys() == after.keys() and all(before[q] != after[q] for q in before)


# ------------------------------------------------------------ launch and merge --

class _NoSubprocess:
    @staticmethod
    def run(*a, **k):
        raise AssertionError("launch must not start processes here")

    Popen = run


def test_launch_trusts_only_the_completion_marker(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    run_args = BASE + ["--tag", "lm"]
    run.main(run_args)
    runs = tmp_path / "runs"
    monkeypatch.setattr(launch, "subprocess", _NoSubprocess)
    assert launch.launch(run_args, ["0", "1"], runs_dir=runs) == runs / "lm" / "okvqa.jsonl"

    out = runs / "lm" / "okvqa.jsonl"
    out.write_text(out.read_text() + "\n")                                    # edited after the marker
    with pytest.raises(SystemExit, match="stale"):
        launch.launch(run_args, ["0", "1"], runs_dir=runs)


def test_launch_refuses_a_partial_single_process_output(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    run_args = BASE + ["--tag", "lp"]
    partial = tmp_path / "runs" / "lp"
    partial.mkdir(parents=True)
    (partial / "okvqa.jsonl").write_text('{"qid": "q000"}\n')
    monkeypatch.setattr(launch, "subprocess", _NoSubprocess)
    with pytest.raises(SystemExit, match="partial single-process run"):
        launch.launch(run_args, ["0", "1"], runs_dir=tmp_path / "runs")


def test_merge_checks_the_configured_question_set_and_writes_the_marker(tmp_path):
    stem = tmp_path / "runs" / "m" / "okvqa"
    stem.parent.mkdir(parents=True)
    qids = ["q1", "q2", "q3", "q4"]
    (stem.parent / "okvqa_config.json").write_text(json.dumps(
        {"evaluation_questions": 4, "evaluation_qids_sha256": runio.qid_set_sha256(qids)}))
    for i, part in enumerate((qids[0::2], qids[1::2])):
        (stem.parent / f"okvqa.shard{i:02d}-of-02.jsonl").write_text("".join(json.dumps({"qid": q}) + "\n" for q in part))
    with pytest.raises(SystemExit, match="not the configuration's evaluation set"):
        merge_shards.merge(stem, 4, runio.qid_set_sha256(["q1", "q2", "q3", "q9"]))
    assert not (stem.parent / "okvqa.jsonl").exists()                         # nothing moved on failure
    with pytest.raises(SystemExit, match="expected 5"):
        merge_shards.merge(stem, 5)
    merge_shards.main([str(stem)])                                            # default: the configuration's count
    assert runio.verify_completion(stem.parent, "okvqa") is True
    assert len(list((stem.parent / "okvqa.shards").glob("*.jsonl"))) == 2


def test_merge_cli_refuses_to_guess_without_a_configuration(tmp_path):
    stem = tmp_path / "okvqa"
    (tmp_path / "okvqa.shard00-of-01.jsonl").write_text('{"qid": "q1"}\n')
    with pytest.raises(SystemExit):
        merge_shards.main([str(stem)])
    merge_shards.main([str(stem), "--expected", "1"])
    assert (tmp_path / "okvqa.jsonl").exists()
