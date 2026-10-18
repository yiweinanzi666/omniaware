cd /home/user/Futaoran/omniaware/assets/cache/official_m2rag
printf 'PWD=%s\n' "$PWD"
ls -ld .git
unset GIT_DIR GIT_WORK_TREE
env -u GIT_DIR -u GIT_WORK_TREE git --no-optional-locks rev-parse --git-dir
env -u GIT_DIR -u GIT_WORK_TREE git --no-optional-locks rev-parse HEAD
