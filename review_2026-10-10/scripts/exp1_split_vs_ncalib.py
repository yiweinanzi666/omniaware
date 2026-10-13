import sys
sys.path.insert(0, "C:/Users/22688/Desktop/www_code/omniaware/src")
from omniaware import data

# OK-VQA val has 5046 questions; heldout calibration, dev 256, default split seed 2027.
ex = [{"qid": f"q{i:05d}"} for i in range(5046)]
S = {}
for n in (100, 500, 1000):
    sp = data.split_examples(ex, n_calib=n, n_dev=256, seed=2027, calib_pool=None)
    S[n] = {k: {e["qid"] for e in sp[k]} for k in ("calib", "dev", "test")}
    print(f"n_calib={n}: calib={len(S[n]['calib'])} dev={len(S[n]['dev'])} test={len(S[n]['test'])}")

print("main (n_calib=500) dev vs n_calib=100 test overlap:", len(S[500]["dev"] & S[100]["test"]))
print("main dev vs n_calib=1000 test overlap:", len(S[500]["dev"] & S[1000]["test"]))
print("main test vs n_calib=100 test overlap:", len(S[500]["test"] & S[100]["test"]), "of", len(S[500]["test"]))
print("main test vs n_calib=1000 test overlap:", len(S[500]["test"] & S[1000]["test"]), "of", len(S[500]["test"]))
print("main test questions NOT evaluated by n_calib=1000 run:", len(S[500]["test"] - S[1000]["test"]))
print("n_calib=100 test that were main-run dev (tuning) questions:", len(S[100]["test"] & S[500]["dev"]))
print("dev stage (n_calib=500) dev == dev of any other n_calib?", S[500]["dev"] == S[100]["dev"], S[500]["dev"] == S[1000]["dev"])
