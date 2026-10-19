ps -ef | grep -E 'v2_runs.py --stage s2|omniaware.run --dataset (webqa|okvqa)' | grep -v grep | tail -20
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo WEBQA:
tail -5 /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/s2_webqa_full_20261017.log
echo OKVQA:
tail -15 /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/s2_okvqa_gs_20261017.log
