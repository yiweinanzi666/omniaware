import sys
sys.path.insert(0, "C:/Users/22688/Desktop/www_code/omniaware")
sys.path.insert(0, "C:/Users/22688/Desktop/www_code/omniaware/src")
from pathlib import Path
from scripts import paper_runs
from omniaware import relevance, order

print("dev_decisions.json exists:", (paper_runs.ROOT / "runs" / "dev_decisions.json").exists())
print("decisions('8b') when file missing:", paper_runs.decisions("8b"))
st = paper_runs.stages("8b", "test")
main_cmd = " ".join(st["main"]["commands"][0])
print("main stage first command uses:", [c for c in st["main"]["commands"][0] if c.startswith("eab:")])

# Calibrator fallback: unseen channel -> 0.5 silently
cal = relevance.Calibrator.fit({"q": {"i1": {"image_cap": 0.9}}}, ["q"])
print("ecdf for unseen channel:", cal.ecdf("ce_video", 0.3))
print("relevance of caption-only image under image_mode=clip:", cal.relevance({"image_cap": 0.9}, "clip"))

print("permutation schedule length:", len(order.permutation_schedule()))
