ps -ef | grep -E 'v2_runs.py --stage s2|omniaware.run --dataset (webqa|okvqa)|score_official|v2_table.py|semantic_audit' | grep -v grep | tail -20
echo WEBQA:
tail -6 /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/s2_webqa_full_20261017.log
echo OKVQA:
tail -8 /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/s2_okvqa_gs_20261017.log
