env | grep '^GIT' || true
unset GIT_DIR GIT_WORK_TREE
git -C /home/user/Futaoran/omniaware/assets/cache/official_m2rag rev-parse --git-dir
git -C /home/user/Futaoran/omniaware/assets/cache/official_m2rag rev-parse HEAD
git -C /home/user/Futaoran/omniaware/assets/cache/official_m2rag status --porcelain
