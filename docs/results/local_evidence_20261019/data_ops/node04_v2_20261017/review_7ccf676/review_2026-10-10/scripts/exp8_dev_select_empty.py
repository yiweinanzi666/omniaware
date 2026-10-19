import sys
sys.path.insert(0, "C:/Users/22688/Desktop/www_code/omniaware")
sys.path.insert(0, "C:/Users/22688/Desktop/www_code/omniaware/src")
import numpy as np
from scripts.analysis import dev_select
per = {"webqa_open_full": np.array([0.1, 0.2, -0.1]), "videorag_synth": np.array([])}
try:
    r = dev_select.pooled_gain(per)
    print("pooled gain ok:", r["gain"])
except Exception as err:
    print("pooled_gain with an empty VideoRAG dev array:", type(err).__name__, err)
