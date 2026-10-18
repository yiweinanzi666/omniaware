"""Coverage check of the repo's paired bootstrap (omniaware.eval.score_official.paired) under the null (true diff 0).
Non-clustered: i.i.d. questions. Clustered: VideoRAG-like, correlated deltas within source video.
Also: a naive (unclustered) bootstrap on clustered data, to show why clustering matters."""
import sys
sys.path.insert(0, "C:/Users/22688/Desktop/www_code/omniaware/src")
import numpy as np
from omniaware.eval.score_official import paired

def coverage_iid(sims=300, n=256, seed=1):
    rng = np.random.default_rng(seed)
    hit = 0
    for _ in range(sims):
        base = rng.uniform(0, 100, n) * (rng.random(n) < 0.5)   # 0/100-ish scores
        other = base.copy(); flip = rng.random(n) < 0.15
        other[flip] = 100 - other[flip]                          # symmetric disagreement -> null
        ci = paired(base, other)["paired_ci95_pp"]
        hit += ci[0] <= 0 <= ci[1]
    return hit / sims

def coverage_clustered(sims=300, n_clusters=120, per=3, seed=2, naive=False):
    rng = np.random.default_rng(seed)
    hit = 0
    for _ in range(sims):
        cl = np.repeat(np.arange(n_clusters), per)
        cluster_effect = rng.normal(0, 20, n_clusters)[cl]        # shared within video
        a = rng.normal(50, 20, len(cl))
        b = a + cluster_effect * 0 + rng.normal(0, 3, len(cl))     # null true difference on average
        b = a + (cluster_effect * 0.5)  # correlated delta within video, mean 0
        ids = None if naive else [str(c) for c in cl]
        ci = paired(a, b, ids)["paired_ci95_pp"]
        hit += ci[0] <= 0 <= ci[1]
    return hit / sims

print("iid paired bootstrap, nominal 0.95, coverage of 0:", coverage_iid())
print("clustered (repo cluster bootstrap), coverage of 0:", coverage_clustered())
print("clustered but resampling questions naively, coverage of 0:", coverage_clustered(naive=True))
