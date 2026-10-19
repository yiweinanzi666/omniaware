date -Iseconds
printf 'PROCS\n'; ps -eo pid,ppid,etime,pcpu,pmem,stat,args | grep -E 'v2_runs.py --stage s[124]|omniaware.run|verify_real_model|pytest|fp32|validation|tmux' | grep -v grep || true
printf 'GPU\n'; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
printf 'OLD_COUNTS\n'; for f in /home/user/Futaoran/omniaware_v2_exec/runs/v2_s2_webqa_full_dev/*.jsonl /home/user/Futaoran/omniaware_v2_exec/runs/v2_s2_okvqa_gs_dev/*.jsonl; do [ -f "$f" ] && wc -l "$f"; done
printf 'FP32_FILES\n'; find /home/user/Futaoran/omniaware_v2_selfcheck_20261018/runs/fp32_20261018 -maxdepth 2 -type f -printf '%p %s %TY-%Tm-%TdT%TH:%TM:%TS\n' 2>/dev/null | sort | tail -60
printf 'FP32_DRIVER\n'; tail -20 /home/user/Futaoran/omniaware_selfcheck_logs_20261018/fp32_validation.log 2>/dev/null || true
printf 'ERRORS\n'; grep -Eina 'traceback|error|failed|exception|oom' /home/user/Futaoran/omniaware_selfcheck_logs_20261018/fp32_validation.log /home/user/Futaoran/omniaware_v2_selfcheck_20261018/runs/fp32_20261018/*/*.log 2>/dev/null | tail -20 || true
