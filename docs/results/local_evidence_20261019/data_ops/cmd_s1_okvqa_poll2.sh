ps -ef | grep -E 'v2_runs.py --stage s1 --dataset okvqa|omniaware.run --dataset okvqa' | grep -v grep || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
tail -25 /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/s1_okvqa_gs_retry_20261017.log 2>/dev/null || true
