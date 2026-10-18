set -e
cd /home/user/Futaoran/omniaware_v2_exec
git add -f runs/v2_s1_webqa_full runs/image_evidence_check.json
git commit -m "run: complete WebQA v2 S1 smoke"
git status --porcelain
git log -1 --oneline
