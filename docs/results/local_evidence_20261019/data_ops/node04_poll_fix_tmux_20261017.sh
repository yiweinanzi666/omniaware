audit=/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/fix_test
date -Iseconds
tail -30 "$audit/test.log" 2>/dev/null || true
if tmux has-session -t v2_fix_test_20261017 2>/dev/null; then echo RUNNING; else echo FINISHED; fi
