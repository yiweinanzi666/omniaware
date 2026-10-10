import math

from omniaware import analysis


def test_ard_zero_when_attention_matches_relevance_softmax():
    rel = {"a": 0.9, "b": 0.5, "c": 0.1}
    target = analysis._softmax(list(rel.values()), analysis.ARD_TEMPERATURE)
    mass = dict(zip(rel, target * 7.0))      # unnormalised mass is normalised inside
    assert analysis.ard(mass, rel) < 1e-9
    assert analysis.ard({"a": 0.0, "b": 0.0, "c": 1.0}, rel) > 0.3
    assert math.isnan(analysis.ard({}, rel))


def test_jsd_bounded_by_ln2():
    assert abs(analysis.jsd([1, 0], [0, 1]) - math.log(2)) < 1e-6


def test_aurc_and_spearman():
    # perfect ranking: errors only at the highest uncertainty
    assert analysis.aurc([0.1, 0.2, 0.9], [0, 0, 1]) < analysis.aurc([0.9, 0.2, 0.1], [0, 0, 1])
    assert abs(analysis.spearman([1, 2, 3], [2, 4, 9]) - 1) < 1e-9
    assert abs(analysis.spearman([1, 2, 3], [3, 2, 1]) + 1) < 1e-9


def test_combiner_sum_and_zsum():
    from omniaware.pipeline import Combiner
    raws = [{"h_grd": g, "h_sem": p} for g, p in [(0.7, 0.01), (0.8, 0.02), (0.9, 0.03), (0.75, 0.05)]]
    s = Combiner.fit(raws, 0.3, "sum")
    assert abs(s(raws[0]) - (0.7 + 0.3 * 0.01)) < 1e-12
    z = Combiner.fit(raws, 0.3, "zsum")
    # standardised terms: the entropy term is no longer negligible next to H_grd
    assert abs(z({"h_grd": z.grd.median, "h_sem": z.pred.median + z.pred.iqr}) - 0.3) < 1e-9
    cold = Combiner.fit([{"h_grd": 0.0, "h_sem": p} for p in (0.1, 0.2, 0.3)], 1.0, "zsum")
    assert cold({"h_grd": 0.0, "h_sem": 0.2}) == 0.0          # constant H_grd contributes 0
    assert Combiner.from_json(z.to_json()) == z
