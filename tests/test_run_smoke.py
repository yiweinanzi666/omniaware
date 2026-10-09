"""End-to-end run.main() with a fake model and synthetic OK-VQA-like data."""
import json
from dataclasses import dataclass, field

import torch

from omniaware import data, paths, run
from omniaware.qwen_eab import Result


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


class FakeQwen:
    """Mimics QwenEAB.encode/run; answers encode which arm produced them."""

    calls = []

    def __init__(self, model_path, **kw):
        self.device, self.n_layers = "cpu", 4

    def encode(self, segments, split_marker):
        spans, pos = {}, 0
        for s in segments:
            n = len(s.text or "")
            if s.chunk:
                a, _ = spans.get(s.chunk, (pos, pos))
                spans[s.chunk] = (a, pos + n)
            pos += n
        return FakeEncoded(length=pos, split=sum(len(s.text or "") for s in segments[:split_marker]), spans=spans)

    def run(self, enc, make_variants, max_new_tokens, decode_static=True):
        attn = torch.rand(enc.length) + 1e-3
        logits = torch.randn(32) * (enc.length % 7 + 1)
        results = {}
        decode = decode_static(attn, logits) if callable(decode_static) else decode_static
        if decode:
            results["static"] = Result(f"static@{enc.length}", logits, [1])
        variants = make_variants(attn, logits) if make_variants else {}
        for name, spec in variants.items():
            results[name] = Result(f"{name}@{enc.length}" if spec else f"static@{enc.length}", logits, [1])
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
    calib_ids = {e["qid"] for e in data.split_examples(examples, 6, 4)["calib"]}
    for r in rows:
        assert r["qid"] not in calib_ids
        assert set(r["preds"]) == {"direct", "static", "eab:g1.2:ans2:pos:gate",
                                   "gated:static", "gated:eab:g1.2:ans2:pos:gate"}
        assert len(r["chunks"]) == 5 and r["chunks"] == ["k1", "k2", "k3", "k4", "k5"]
        expected = r["preds"]["static"] if r["retrieve"] else r["preds"]["direct"]
        assert r["preds"]["gated:static"] == expected
        assert r["u_eab"] == r["u_pre"]
        assert "scores" not in r


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
