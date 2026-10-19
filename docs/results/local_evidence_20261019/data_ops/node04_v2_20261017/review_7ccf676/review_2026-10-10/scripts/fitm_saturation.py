import sys
sys.path.insert(0, r"C:\Users\22688\Desktop\www_code\omniaware\src")
import torch
from omniaware.baselines.fitm_attention import calibrated_alpha, redistribute_document_attention, TEMPERATURE

torch.manual_seed(0)
K = 600                      # visible keys in one row (about 5 chunks of ~100 tokens plus question)
docs = [list(range(0, 100)), list(range(100, 200)), list(range(200, 300)), list(range(300, 400)), list(range(400, 500))]
# realistic row: probabilities ~1/K; original doc means differ from null-reference means by ~1e-3
orig_prob = torch.full((1, 1, K), 1.0 / K, dtype=torch.float64)
orig_prob[..., 100:200] *= 2.0           # chunk 1 receives 2x attention than uniform
orig_prob = orig_prob / orig_prob.sum(-1, keepdim=True)
log_orig = orig_prob.log()
ref_prob = torch.full((1, 1, K), 1.0 / K, dtype=torch.float64)
ref_prob = ref_prob / ref_prob.sum(-1, keepdim=True)
def means(lp):
    p = lp.exp()
    return torch.stack([p[..., d].mean(-1) for d in docs], -1)
orig_m, ref_m = means(log_orig), means(ref_prob.log())
print("original doc means", orig_m.flatten().tolist())
print("reference doc means", ref_m.flatten().tolist())
alpha = calibrated_alpha(orig_m, ref_m)
print("alpha (T=5e-5)", alpha.flatten().tolist())
new, stats = redistribute_document_attention(log_orig, alpha, docs)
mass_after = stats["document_mass_after"].flatten().tolist()
print("document mass after", mass_after)
print("row total after", new.sum().item())
print("non-target docs zeroed:", [float(new[..., d].sum()) == 0.0 for d in docs])
# sensitivity: same row, but difference of 1e-4 (more typical for a mild effect)
orig_m2 = ref_m.clone(); orig_m2[..., 1] += 1e-4
alpha2 = calibrated_alpha(orig_m2, ref_m)
print("alpha for 1e-4 gap", alpha2.flatten().tolist())
