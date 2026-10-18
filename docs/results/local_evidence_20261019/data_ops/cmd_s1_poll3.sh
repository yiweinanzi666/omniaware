ps -p 2402602 -o pid,etime,%cpu,%mem,rss,stat,cmd || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
tail -25 /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/s1_webqa_full_20261017.log 2>/dev/null || true
find /home/user/Futaoran/omniaware/runs/v2_s1_webqa_full -maxdepth 2 -type f 2>/dev/null | head -20
