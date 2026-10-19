"""Null simulation of dev_select's EAB rule: 6 candidate forms, zero true gain over static, 4 datasets, 256 dev q.
Uses the repo's pooled_gain (stratified bootstrap) to decide 'eab_effective' (CI lower bound > 0)."""
import sys
sys.path.insert(0, "C:/Users/22688/Desktop/www_code/omniaware/src")
sys.path.insert(0, "C:/Users/22688/Desktop/www_code/omniaware")
import numpy as np
from scripts.analysis.dev_select import pooled_gain

def run(rho, sims=400, n=256, n_ds=4, n_forms=6, sd=25.0, seed=0):
    rng = np.random.default_rng(seed)
    eff_sel, eff_fixed = 0, 0
    for _ in range(sims):
        results = {}
        for f in range(n_forms):
            per = {}
            for d in range(n_ds):
                # shared component across forms (rho) + form-specific noise, paired to static
                common = rng.normal(0, sd, n)
                own = rng.normal(0, sd, n)
                per[f"d{d}"] = np.sqrt(rho) * common + np.sqrt(1 - rho) * own
            results[f] = per
        gains = {f: pooled_gain(per, resamples=1000, seed=int(rng.integers(1 << 30))) for f, per in results.items()}
        best = max(gains, key=lambda f: gains[f]["gain"])
        eff_sel += gains[best]["ci95"][0] > 0
        eff_fixed += gains[0]["ci95"][0] > 0          # a pre-specified single form
    return eff_sel / sims, eff_fixed / sims

for rho in (0.0, 0.5, 0.8):
    sel, fixed = run(rho)
    print(f"rho={rho}: P(eab_effective | null) with best-of-6 selection = {sel:.3f}; single pre-specified form = {fixed:.3f} (nominal 0.025)")
