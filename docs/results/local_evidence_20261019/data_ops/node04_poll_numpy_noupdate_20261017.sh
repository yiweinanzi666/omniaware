audit=/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/env_fix
date -Iseconds
tail -25 "$audit/noupdate.log" 2>/dev/null || true
for x in noupdate_start.txt noupdate_end.txt noupdate_exit.txt; do test -f "$audit/$x" && { echo "$x"; cat "$audit/$x"; }; done
if tmux has-session -t v2_env_fix_noupdate_20261017 2>/dev/null; then echo RUNNING; else echo FINISHED; fi
