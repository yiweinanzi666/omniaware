set -e
cd /home/user/Futaoran/omniaware_v2_exec
export CUDA_VISIBLE_DEVICES=0
export OMNIAWARE_ASSETS=/home/user/Futaoran/omniaware/assets
export no_proxy=127.0.0.1,localhost
export NO_PROXY=127.0.0.1,localhost
mkdir -p runs
log=/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/verify_real_model_8b_20261017.log
nohup /home/user/Futaoran/omniaware_v2_python311/bin/python -m scripts.verify_real_model \
  --model /home/user/Futaoran/omniaware/assets/cache/qwen3-vl-8b-instruct \
  --out runs/verify_real_model_8b.json > "$log" 2>&1 &
echo PID=$!
echo LOG=$log
