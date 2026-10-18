cd /home/user/Futaoran/omniaware_v2_exec
git log -1 --oneline
git rev-parse HEAD
git show HEAD:third_party/webqa/eval_1022.py | sha256sum
sha256sum third_party/webqa/eval_1022.py
git status --porcelain=v2
