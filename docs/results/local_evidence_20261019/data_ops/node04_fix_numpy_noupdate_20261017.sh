set -eu
audit=/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/env_fix
if tmux has-session -t v2_env_fix_noupdate_20261017 2>/dev/null; then echo ALREADY_RUNNING; exit 0; fi
cat > "$audit/run_numpy_noupdate.sh" <<'SH'
#!/bin/bash
set -u
audit=/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/env_fix
date -Iseconds > "$audit/noupdate_start.txt"
/home/user/anaconda3/bin/conda install -y --offline --no-update-deps numpy=1.26.4=py312hc5e2394_0 > "$audit/noupdate.log" 2>&1
rc=$?
echo "$rc" > "$audit/noupdate_exit.txt"
date -Iseconds > "$audit/noupdate_end.txt"
exit "$rc"
SH
tmux new-session -d -s v2_env_fix_noupdate_20261017 "bash '$audit/run_numpy_noupdate.sh'"
echo STARTED
