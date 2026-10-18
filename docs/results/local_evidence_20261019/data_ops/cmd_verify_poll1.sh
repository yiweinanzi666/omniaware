ps -p 2398573 -o pid,etime,stat,cmd || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
tail -30 /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/verify_real_model_8b_20261017.log 2>/dev/null || true
