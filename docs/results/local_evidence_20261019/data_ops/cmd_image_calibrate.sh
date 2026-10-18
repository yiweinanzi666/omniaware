set -e
cd /home/user/Futaoran/omniaware_v2_exec
export CUDA_VISIBLE_DEVICES=0
export OMNIAWARE_ASSETS=/home/user/Futaoran/omniaware/assets
export OMNIAWARE_CACHE=/home/user/Futaoran/omniaware_v2_exec/cache
export no_proxy=127.0.0.1,localhost
export NO_PROXY=127.0.0.1,localhost
log=/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/image_check_calibrate_20261017.log
nohup /home/user/Futaoran/omniaware_v2_python311/bin/python -m omniaware.run \
 --dataset webqa_full --split test --offset 0 --n-eval 50 \
 --model 8b --tag image_check --calib-source train --n-calib 500 --n-dev 256 \
 --split-seed 2027 --v2 --calibrate-only > "$log" 2>&1 &
echo PID=$!
