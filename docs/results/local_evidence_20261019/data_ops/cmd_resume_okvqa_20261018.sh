set -e
repo=/home/user/Futaoran/omniaware_v2_exec
cd "$repo"
export CUDA_VISIBLE_DEVICES=1
export OMNIAWARE_ASSETS=/home/user/Futaoran/omniaware/assets
export OMNIAWARE_CACHE=/home/user/Futaoran/omniaware/cache
export no_proxy=127.0.0.1,localhost
export NO_PROXY=127.0.0.1,localhost
log="$repo/runs/v2_s2_pause_20261018/resume_okvqa_20261018.log"
cmd="/home/user/Futaoran/omniaware_v2_python311/bin/python -m omniaware.run --dataset okvqa_gs --split dev --model 8b --tag v2_s2_okvqa_gs_dev --calib-source heldout --n-calib 500 --n-dev 256 --split-seed 2027 --v2 --v2-reranker --fitm --m2rag --seakr-samples 20 --analysis-eab"
printf 'START %s\n' "$(date -Iseconds)" | tee -a "$log"
printf 'COMMAND %s\n' "$cmd" | tee -a "$log"
nohup $cmd >> "$log" 2>&1 &
echo "PID=$!"
sleep 2
ps -p $! -o pid,ppid,etime,stat,args || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
wc -l "$repo/runs/v2_s2_okvqa_gs_dev/okvqa_gs.jsonl"
