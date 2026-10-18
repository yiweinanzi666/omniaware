import sys
sys.path.insert(0, "C:/Users/22688/Desktop/www_code/omniaware/src")
from omniaware import data
for total in (374, 300):
    ex = [{"qid": f"v{i:03d}"} for i in range(total)]
    for n_calib in (500, 100, 150):
        sp = data.split_examples(ex, n_calib=n_calib, n_dev=256, seed=2027)
        print(f"total={total} n_calib={n_calib}: calib={len(sp['calib'])} dev={len(sp['dev'])} test={len(sp['test'])}")
