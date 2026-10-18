ps -p 2415464 -o pid,etime,%cpu,%mem,rss,stat,cmd || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
tail -25 /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/s1_okvqa_gs_retry_20261017.log 2>/dev/null || true
