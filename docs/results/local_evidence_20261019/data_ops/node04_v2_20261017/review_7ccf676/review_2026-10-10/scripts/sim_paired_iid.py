import sys
sys.path.insert(0, "C:/Users/22688/Desktop/www_code/omniaware/src")
import numpy as np
from omniaware.eval.score_official import paired

def coverage_iid(sims=400, n=256, seed=1, p=0.5, flip=0.2):
    # symmetric null: P(0->100) == P(100->0), so the true mean difference is exactly 0
    rng = np.random.default_rng(seed)
    hit = 0
    for _ in range(sims):
        a = 100.0 * (rng.random(n) < p)
        b = a.copy()
        f = rng.random(n) < flip
        b[f] = 100.0 - b[f]
        ci = paired(a, b)["paired_ci95_pp"]
        hit += ci[0] <= 0 <= ci[1]
    return hit / sims

print("true delta = 0 (symmetric), n=256, iid; coverage of 0 by repo paired CI:", coverage_iid())
print("n=1000:", coverage_iid(n=1000, sims=200))
print("small-n check n=64:", coverage_iid(n=64, sims=400))
