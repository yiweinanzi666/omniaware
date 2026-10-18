set -e
cd /home/user/Futaoran/omniaware_v2_exec
git add third_party/webqa/eval_1022.py third_party/webqa/eval_webqa.md
git commit -m "fix: restore pinned WebQA official sources"
git status --porcelain
git rev-parse HEAD
