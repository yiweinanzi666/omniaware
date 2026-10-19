ps -ef | grep -E 'v2_runs.py --stage s2 --dataset webqa|omniaware.run --dataset webqa' | grep -v grep | tail -10
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
tail -35 /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/s2_webqa_full_20261017.log 2>/dev/null || true
