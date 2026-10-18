from types import SimpleNamespace

import torch

from omniaware.eval.citation import attribution as at

SOURCES = [{"rendered_text": "Paris is the capital of France."}, {"rendered_text": "Berlin is in Germany."},
           {"rendered_text": "Unrelated text."}]


def test_parse_counts_malformed_and_out_of_prompt():
    c = at.parse_citations("Paris [E1][E9] (E2) [3] E2.", 3)
    assert c["valid"] == [1] and c["invalid"] == 2 and c["markers"] == 3   # [E9] and [3]; "(E2)" and "E2" are text


def test_bare_e_identifiers_in_ordinary_text_are_not_invalid_citations():
    c = at.parse_citations("The E5 model improved accuracy [E1].", 3)
    assert c["valid"] == [1] and c["invalid"] == 0 and c["markers"] == 1
    assert at.parse_citations("Figure E2 shows it [e3].", 3)["invalid"] == 1   # bracketed attempt is counted


def test_faith_ap_and_alias_and_nli_reported_separately():
    calls = []

    def entails(p, h):
        calls.append((p, h))
        return "Germany" in p and "Berlin" in h

    r = at.score_answer("The capital is Paris [E1][E3].\nBerlin is German [E2].", SOURCES, ["Paris"], entails)
    assert r["support"] == {"E1": True, "E2": True, "E3": False}           # union: the headline definition
    assert r["support_alias"] == {"E1": True, "E2": False, "E3": False}
    assert r["support_nli"] == {"E1": False, "E2": True, "E3": False}
    assert r["faith"] == 1.0 and abs(r["ap"] - 2 / 3) < 1e-9
    assert r["faith_alias"] == 1.0 and abs(r["ap_alias"] - 1 / 3) < 1e-9
    assert r["faith_nli"] == 1.0 and abs(r["ap_nli"] - 1 / 3) < 1e-9
    assert any(p == SOURCES[0]["rendered_text"] for p, _ in calls)        # NLI is run even when alias matches
    none = at.score_answer("Paris.", SOURCES, ["Paris"], entails)
    assert none["faith"] == 0.0 and none["ap"] is None


def test_aggregate_ap_over_citing_answers_only():
    rows = [dict(faith=1.0, ap=0.5, invalid=1, markers=2, faith_alias=1.0, ap_alias=0.5, faith_nli=0.0, ap_nli=0.0),
            dict(faith=0.0, ap=None, invalid=0, markers=0, faith_alias=0.0, ap_alias=None, faith_nli=0.0, ap_nli=None)]
    s = at.aggregate(rows)
    assert s["Faith"] == 50 and s["AP"] == 50 and s["coverage"] == 50 and s["Inv"] == 50
    assert s["Faith_alias"] == 50 and s["AP_alias"] == 50 and s["Faith_nli"] == 0 and s["AP_nli"] == 0
    assert s["coverage"] * s["AP"] / 100 <= s["Faith"]


def test_statements_use_the_shared_splitter_and_strip_markers():
    st = at.statements("Paris is big [E1]. Berlin is old [E2]!\nNo citation here.")
    assert [text for text, _ in st] == ["Paris is big .", "Berlin is old !", "No citation here."]
    assert [cites for _, cites in st] == [{1}, {2}, set()]


def test_premise_windows_cover_every_token_and_fit_the_window():
    premise = list(range(1000))
    hyp = [7, 8]
    windows = at.premise_windows(premise, hyp)
    budget = 512 - len(hyp) - 3
    assert all(len(w) <= budget for w in windows) and len(windows) > 1
    assert set().union(*map(set, windows)) == set(premise)
    assert windows[0][0] == 0 and windows[-1][-1] == 999
    assert at.premise_windows(list(range(10)), hyp) == [list(range(10))]   # a short premise is one window


def test_window_decision_follows_the_most_entailed_window():
    assert at.window_decision([(0.2, False), (0.9, True), (0.1, False)]) is True
    assert at.window_decision([(0.2, False), (0.6, False), (0.5, True)]) is False


class _FakeTok:
    def encode(self, text, add_special_tokens=False):
        return [999 if w == "MARK" else 1 for w in text.split()]

    def build_inputs_with_special_tokens(self, a, b):
        return [101] + list(a) + [102] + list(b) + [102]

    def create_token_type_ids_from_sequences(self, a, b):
        return [0] * (len(a) + 2) + [1] * (len(b) + 1)


class _FakeModel:
    """Entailed (label 0) exactly when the window contains the marker token."""

    def __call__(self, input_ids, token_type_ids, attention_mask):
        has = bool((input_ids == 999).any())
        return SimpleNamespace(logits=torch.tensor([[5.0, 0.0, 0.0]] if has else [[0.0, 0.0, 5.0]]))


def _checker():
    chk = object.__new__(at.DebertaChecker)
    chk.torch, chk.device, chk.tok, chk.model, chk.entail_id, chk.cache = torch, "cpu", _FakeTok(), _FakeModel(), 0, {}
    return chk


def test_checker_sees_evidence_beyond_the_512_token_window():
    chk = _checker()
    long_premise = " ".join(["x"] * 1000 + ["MARK"] + ["y"] * 50)     # the marker lies beyond the first window
    assert chk(long_premise, "a claim") is True
    assert _checker()(" ".join(["x"] * 1100), "a claim") is False


def test_score_records_gated_arm_not_applicable_without_retrieval_and_replication_uses_own_sources():
    def entails(p, h):
        return False

    retrieved = dict(qid="q1", golds=["Berlin"], retrieve=True,
                     citation_sources=[{"rendered_text": "Paris is in France."}],
                     replication_citation_sources=[{"rendered_text": "Berlin is in Germany."}],
                     preds={"static": "Paris [E1].", "replication": "Berlin [E1]."})
    direct = dict(qid="q2", golds=["Berlin"], retrieve=False,
                  citation_sources=[{"rendered_text": "Paris is in France."}],
                  replication_citation_sources=[{"rendered_text": "Paris is in France."}],
                  preds={"static": "Paris [E1].", "replication": "Paris [E1]."})
    arms = ["static", "gated:static", "replication"]
    result = at.score_records([retrieved, direct], arms, entails)
    s = result["summary"]
    assert s["static"]["answers"] == 2 and s["static"]["questions"] == 2
    assert s["gated:static"]["applicable"] == 1 and s["gated:static"]["gate_coverage"] == 50
    assert s["gated:static"]["answers"] == 1                       # the direct question leaves the denominator
    assert s["replication"]["Faith_alias"] == 50 and s["static"]["Faith_alias"] == 0
    assert result["sentence_splitter"] in ("regex(?<=[.!?])\\s+|\\n+", "nltk-punkt-english")


def test_score_records_refuses_an_arm_without_its_citation_sources():
    rec = dict(qid="q", golds=["x"], retrieve=True, citation_sources=[{"rendered_text": "x"}],
               preds={"static": "x [E1].", "replication": "x [E1]."})
    try:
        at.score_records([rec], ["replication"], lambda p, h: False)
    except ValueError as err:
        assert "replication" in str(err)
    else:
        raise AssertionError("replication scored without its own citation sources")
