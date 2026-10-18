set -e
cd /home/user/Futaoran/omniaware_v2_exec
git add -f runs/v2_s1_okvqa_gs
git commit -m "run: complete OK-VQA v2 S1 smoke"
git status --porcelain
git log -1 --oneline
