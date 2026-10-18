ps -ef | grep -E 'v2_runs.py --stage s2|omniaware.run --dataset (webqa|okvqa)|score_official|v2_table.py' | grep -v grep | tail -20
echo OKVQA:
tail -25 /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/s2_okvqa_gs_20261017.log
