"""Two-sided paired bootstrap p-values of omniaware.eval.score_official.paired (M-10, D18)."""
import numpy as np

from omniaware.eval.score_official import paired


def test_identical_arms_have_p_one():
    x = np.arange(20, dtype=float)
    assert paired(x, x)["p_two_sided"] == 1.0


def test_a_clear_difference_is_significant_in_both_directions():
    rng = np.random.default_rng(3)
    a = rng.normal(50, 20, 200)
    up = paired(a, a + 10 + rng.normal(0, 5, 200))
    down = paired(a, a - 10 + rng.normal(0, 5, 200))
    assert up["p_two_sided"] < 0.001 and up["delta_pp"] > 0
    assert down["p_two_sided"] < 0.001 and down["delta_pp"] < 0


def test_null_rejection_rate_is_near_the_nominal_five_percent():
    rng = np.random.default_rng(7)
    sims, rejections = 200, 0
    for _ in range(sims):
        a = rng.normal(50, 20, 120)
        b = a + rng.normal(0, 10, 120)                 # zero true mean difference, paired noise
        rejections += paired(a, b)["p_two_sided"] < 0.05
    assert rejections / sims <= 0.12
