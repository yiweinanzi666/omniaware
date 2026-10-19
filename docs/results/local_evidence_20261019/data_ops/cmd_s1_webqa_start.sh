set -e
cd /home/user/Futaoran/omniaware_v2_exec
export CUDA_VISIBLE_DEVICES=0
export OMNIAWARE_ASSETS=/home/user/Futaoran/omniaware/assets
export no_proxy=127.0.0.1,localhost
export NO_PROXY=127.0.0.1,localhost
log=/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/s1_webqa_full_20261017.log
nohup /home/user/Futaoran/omniaware_v2_python311/bin/python scripts/v2_runs.py --stage s1 --dataset webqa_full --n 50 > "$log" 2>&1 &
echo PID=$!
echo LOG=$log
