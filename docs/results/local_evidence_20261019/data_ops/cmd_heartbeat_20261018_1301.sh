date -Iseconds
printf 'PROCS\n'; ps -eo pid,ppid,etime,pcpu,pmem,stat,args | grep -E 'v2_runs.py --stage s2|omniaware.run --dataset (webqa|okvqa)|score_official|v2_table.py|semantic_audit|dev_check|verify_real_model|pytest' | grep -v grep || true
printf 'GPU\n'; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
printf 'COUNTS\n'; for f in /home/user/Futaoran/omniaware_v2_exec/runs/v2_s2_webqa_full_dev/*.jsonl /home/user/Futaoran/omniaware_v2_exec/runs/v2_s2_okvqa_gs_dev/*.jsonl; do [ -f "$f" ] && wc -l "$f"; done
printf 'WEBQA_FILES\n'; find /home/user/Futaoran/omniaware_v2_exec/runs/v2_s2_webqa_full_dev -maxdepth 1 -type f -printf '%f %s %TY-%Tm-%TdT%TH:%TM:%TS\n' 2>/dev/null | sort
printf 'DECISIONS\n'; find /home/user/Futaoran/omniaware_v2_exec/runs/v2_decisions -maxdepth 1 -type f -printf '%f %s %TY-%Tm-%TdT%TH:%TM:%TS\n' 2>/dev/null | sort
printf 'WEBQA_LOG\n'; tail -12 /home/user/Futaoran/omniaware_v2_exec/runs/v2_s2_webqa_full_dev/webqa_full.log 2>/dev/null || true
printf 'GIT\n'; git -C /home/user/Futaoran/omniaware_v2_exec status --short; git -C /home/user/Futaoran/omniaware_v2_exec rev-parse HEAD
