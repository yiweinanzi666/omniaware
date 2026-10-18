ps -ef | grep -E 'v2_runs.py --stage s2|omniaware.run --dataset (webqa|okvqa)|score_official|v2_table.py|semantic_audit' | grep -v grep | tail -20
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo WEBQA_LOG; tail -20 /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/s2_webqa_full_20261017.log
echo OKVQA_LOG; tail -20 /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/s2_okvqa_gs_20261017.log
echo FILES; for d in /home/user/Futaoran/omniaware_v2_exec/runs/v2_s2_webqa_full_dev /home/user/Futaoran/omniaware_v2_exec/runs/v2_s2_okvqa_gs_dev; do find "$d" -maxdepth 1 -type f -printf '%f %s\n' 2>/dev/null; done
