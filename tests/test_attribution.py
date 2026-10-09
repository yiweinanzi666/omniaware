from omniaware.eval.citation import attribution as at

SOURCES = [{"rendered_text": "Paris is the capital of France."}, {"rendered_text": "Berlin is in Germany."},
           {"rendered_text": "Unrelated text."}]


def test_parse_counts_malformed_and_out_of_prompt():
    c = at.parse_citations("Paris [E1][E9] (E2) [3] E2.", 3)
    assert c["valid"] == [1] and c["invalid"] == 4 and c["markers"] == 5


def test_faith_ap_and_alias_then_nli():
    calls = []

    def entails(p, h):
        calls.append((p, h))
        return "Germany" in p and "Berlin" in h

    r = at.score_answer("The capital is Paris [E1][E3].\nBerlin is German [E2].", SOURCES, ["Paris"], entails)
    assert r["support"] == {"E1": True, "E2": True, "E3": False}
    assert r["faith"] == 1.0 and abs(r["ap"] - 2 / 3) < 1e-9
    assert not any(p == SOURCES[0]["rendered_text"] for p, _ in calls)   # alias match short-circuits NLI
    none = at.score_answer("Paris.", SOURCES, ["Paris"], entails)
    assert none["faith"] == 0.0 and none["ap"] is None


def test_aggregate_ap_over_citing_answers_only():
    rows = [{"faith": 1.0, "ap": 0.5, "invalid": 1, "markers": 2}, {"faith": 0.0, "ap": None, "invalid": 0, "markers": 0}]
    s = at.aggregate(rows)
    assert s["Faith"] == 50 and s["AP"] == 50 and s["coverage"] == 50 and s["Inv"] == 50
    assert s["coverage"] * s["AP"] / 100 <= s["Faith"]
