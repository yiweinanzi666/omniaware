date -Iseconds
printf 'PROCS\n'; ps -eo pid,ppid,etime,pcpu,pmem,stat,args | grep -E 'v2_runs.py --stage s[124]|omniaware.run|verify_real_model|pytest|fp32|validation|tmux' | grep -v grep || true
printf 'GPU\n'; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
printf 'OLD_COUNTS\n'; for f in /home/user/Futaoran/omniaware_v2_exec/runs/v2_s2_webqa_full_dev/*.jsonl /home/user/Futaoran/omniaware_v2_exec/runs/v2_s2_okvqa_gs_dev/*.jsonl; do [ -f "$f" ] && wc -l "$f"; done
printf 'OLD_LOG_TAIL\n'; tail -12 /home/user/Futaoran/omniaware_v2_exec/runs/v2_s2_webqa_full_dev/webqa_full.log 2>/dev/null || true
printf 'FP32_ARTIFACTS\n'; find /home/user/Futaoran/omniaware_v2_selfcheck_20261018/runs/fp32_20261018 -maxdepth 2 -type f -printf '%f %s %TY-%Tm-%TdT%TH:%TM:%TS\n' 2>/dev/null | sort | tail -15
printf 'DECISIONS\n'; find /home/user/Futaoran/omniaware_v2_exec/runs/v2_decisions -maxdepth 1 -type f -printf '%f %s\n' 2>/dev/null | sort
