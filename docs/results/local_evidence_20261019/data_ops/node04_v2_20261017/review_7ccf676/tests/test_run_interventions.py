"""Interventions in the run loop: the replication arm is decoded on the perturbed input, and every kind has one entry
per eligible question (predictions or a failure reason)."""
import json

import pytest

from omniaware import data, paths, run

from test_run_smoke import FakeQwen, _examples

pytestmark = pytest.mark.usefixtures("tiny_reserves")


def test_interventions_decode_replication_on_the_perturbed_input(tmp_path, monkeypatch):
    cities = ["Paris", "Rome", "Berlin", "Madrid", "Vienna", "Prague", "Oslo", "Lisbon"]
    examples = _examples(16)
    for i, e in enumerate(examples):
        e["golds"] = [cities[i % len(cities)]]
        e["candidates"][0]["text"] = f"The capital is {e['golds'][0]} and more words follow here."
    monkeypatch.setitem(data.LOADERS, "okvqa", lambda **kw: examples)
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(paths, "LOCAL_CACHE", tmp_path / "cache")
    monkeypatch.setattr(run, "QwenEAB", FakeQwen)
    monkeypatch.setattr(run, "load_relevance", lambda exs, path, **kw: {
        e["qid"]: {c["cid"]: {"text": 1.0 / int(c["cid"][1:])} for c in e["candidates"]} for e in exs})
    kinds = ["mask", "cf", "random_edit", "rerun", "inject_answer"]
    run.main(["--dataset", "okvqa", "--split", "test", "--n-calib", "6", "--calib-source", "heldout",
              "--n-dev", "2", "--tag", "ivrep", "--tau=-1e9", "--tau-bias=-1e9", "--replication-gamma", "1.2",
              "--variants", "eab:g1.2:ans1:pos:gate", "--intervention", *kinds])
    rows = [json.loads(line) for line in (tmp_path / "runs" / "ivrep" / "okvqa.jsonl").read_text().splitlines()]
    assert rows
    arms = {"static", "eab:g1.2:ans1:pos:gate", "replication"}
    for r in rows:
        iv = r["interventions"]
        assert iv["eligible"] and iv["eligible_by_arm"] == {arm: True for arm in arms}
        assert iv["attempted"] == kinds
        for kind in kinds:
            entry = iv[kind]
            assert "failed_reason" in entry or set(entry["preds"]) == arms
