audit=/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/env_fix
date -Iseconds
tail -30 "$audit/numpy_install.log" 2>/dev/null || true
for x in numpy_start.txt numpy_end.txt numpy_exit.txt; do test -f "$audit/$x" && { echo "$x"; cat "$audit/$x"; }; done
if tmux has-session -t v2_env_fix_20261017 2>/dev/null; then echo RUNNING; else echo FINISHED; fi
