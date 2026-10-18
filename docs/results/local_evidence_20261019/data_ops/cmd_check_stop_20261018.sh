date -Iseconds
ps -p 2463331 -o pid,ppid,pgid,sid,etime,stat,cmd || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
printf 'PAUSE_TAIL\n'; tail -30 /home/user/Futaoran/omniaware_v2_exec/runs/v2_s2_pause_20261018/pause.log 2>/dev/null || true
printf 'RESUME_TAIL\n'; tail -20 /home/user/Futaoran/omniaware_v2_exec/runs/v2_s2_pause_20261018/resume_okvqa_20261018.log 2>/dev/null || true
wc -l /home/user/Futaoran/omniaware_v2_exec/runs/v2_s2_okvqa_gs_dev/okvqa_gs.jsonl 2>/dev/null || true
