ps -ef | grep -E 'v2_runs.py --stage s2 --dataset webqa|omniaware.run --dataset webqa|score_official|v2_table.py' | grep -v grep | tail -10
tail -12 /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/s2_webqa_full_20261017.log
