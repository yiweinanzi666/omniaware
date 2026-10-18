date -Iseconds
ps -p 2419637,2420983 -o pid,ppid,pgid,sid,etime,stat,cmd || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
cat /home/user/Futaoran/omniaware_v2_exec/runs/v2_s2_pause_20261018/pause.log 2>/dev/null | tail -30
