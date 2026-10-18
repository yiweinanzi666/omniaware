ps -ef | grep -E 'v2_runs.py --stage s2|omniaware.run --dataset (webqa|okvqa)|score_official|v2_table.py|semantic_audit' | grep -v grep | tail -20
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo WEBQA_FILES; find /home/user/Futaoran/omniaware_v2_exec/runs/v2_s2_webqa_full_dev -maxdepth 1 -type f -printf '%f %s\n' 2>/dev/null
echo OKVQA_FILES; find /home/user/Futaoran/omniaware_v2_exec/runs/v2_s2_okvqa_gs_dev -maxdepth 1 -type f -printf '%f %s\n' 2>/dev/null
echo WEBQA_TAIL; tail -5 /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/s2_webqa_full_20261017.log
echo OKVQA_TAIL; tail -5 /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/s2_okvqa_gs_20261017.log
