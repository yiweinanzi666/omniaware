#!/usr/bin/env bash
set -eu
cd /home/user/Futaoran/omniaware_v2_selfcheck_20261018
source /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/env_resume_lf_1631.sh
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
audit=/home/user/Futaoran/omniaware_selfcheck_logs_20261018
used=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$used" -gt 1000 ]; then echo "GPU1 occupied: $used MiB"; exit 2; fi
set +e
python "$audit/diagnose_precision.py" > "$audit/precision_diagnostic.log" 2>&1
result=$?
printf '%s\n' "$result" > "$audit/precision_diagnostic.exit"
exit "$result"
