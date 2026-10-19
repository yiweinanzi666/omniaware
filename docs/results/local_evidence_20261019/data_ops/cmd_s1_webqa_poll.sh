ps -p 2402595 -o pid,etime,stat,cmd || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
tail -25 /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/s1_webqa_full_20261017.log 2>/dev/null || true
