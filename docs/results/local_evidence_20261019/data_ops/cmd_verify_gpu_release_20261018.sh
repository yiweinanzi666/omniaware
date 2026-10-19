date -Iseconds
printf 'PROCS\n'; ps -eo pid,ppid,etime,stat,args | grep -E 'v2_runs.py --stage s2|omniaware.run --dataset (webqa|okvqa)|score_official|v2_table.py|semantic_audit|dev_check' | grep -v grep || true
printf 'GPU\n'; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
printf 'DECISIONS\n'; find /home/user/Futaoran/omniaware_v2_exec/runs/v2_decisions -maxdepth 1 -type f -printf '%f %s\n' 2>/dev/null || true
printf 'PAUSE_FILES\n'; find /home/user/Futaoran/omniaware_v2_exec/runs/v2_s2_pause_20261018 -maxdepth 1 -type f -printf '%f %s\n' | sort
