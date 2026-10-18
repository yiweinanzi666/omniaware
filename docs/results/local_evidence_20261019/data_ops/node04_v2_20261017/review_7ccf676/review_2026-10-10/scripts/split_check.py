import sys
sys.path.insert(0, "C:/Users/22688/Desktop/www_code/omniaware/src")
from omniaware import data

def make(n, prefix="q"):
    return [{"qid": f"{prefix}{i:05d}", "dataset": "x"} for i in range(n)]

def report(label, n, n_calib, n_dev=256, pool=None):
    ex = make(n)
    sp = data.split_examples(ex, n_calib, n_dev, 2027, pool)
    dev = {e["qid"] for e in sp["dev"]}
    test = {e["qid"] for e in sp["test"]}
    calib = {e["qid"] for e in sp["calib"]}
    print(f"{label:55s} n={n:5d} n_calib={n_calib:5d} calib={len(calib):5d} dev={len(dev):4d} test={len(test):5d} "
          f"test/n={len(test)/n:.3f} overlap(dev,test)={len(dev&test)} overlap(calib,test)={len(calib&test)}")
    return sp

report("WebQA full val, calib from train (500)", 4966, 500, pool=make(1000, "tr"))
report("OK-VQA val, heldout calib (500)", 5046, 500)
report("HybridQA, heldout calib (500)", 2000, 500)
report("VideoRAG-Synth, heldout calib (500)", 374, 500)
report("VideoRAG-Synth, heldout calib (100)", 374, 100)
report("OK-VQA main split (500) for calib100 run", 5046, 100)
report("OK-VQA calib1000 run", 5046, 1000)
sp_main = data.split_examples(make(5046), 500, 256, 2027)
sp_c100 = data.split_examples(make(5046), 100, 256, 2027)
dev_main = {e["qid"] for e in sp_main["dev"]}
print("dev(main, n_calib=500) questions appearing in test of n_calib=100 run:",
      len(dev_main & {e["qid"] for e in sp_c100["test"]}))
sp_c1000 = data.split_examples(make(5046), 1000, 256, 2027)
print("dev(main) appearing in test of n_calib=1000 run:", len(dev_main & {e["qid"] for e in sp_c1000["test"]}))
