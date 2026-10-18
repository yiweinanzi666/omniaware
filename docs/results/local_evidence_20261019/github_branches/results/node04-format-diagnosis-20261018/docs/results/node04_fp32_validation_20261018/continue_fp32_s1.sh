#!/usr/bin/env bash
set -euo pipefail
cd /home/user/Futaoran/omniaware_v2_selfcheck_20261018
source /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/env_resume_lf_1631.sh
export PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMNIAWARE_RUNS="$PWD/runs/fp32_20261018"
export OMNIAWARE_CACHE=/home/user/Futaoran/omniaware_v2_cache_fp32_20261018
audit=/home/user/Futaoran/omniaware_selfcheck_logs_20261018/fp32_revision
exec 9>"$audit/validation.lock"
flock -n 9 || exit 2
test -z "$(git status --porcelain --untracked-files=no)"
test ! -e "$audit/okvqa_retry.started"
used=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
test "$used" -lt 1000
date -Is > "$audit/okvqa_retry.started"
trap 'rc=$?; printf "%s\n" "$rc" > "$audit/okvqa_retry.exit"' EXIT
printf 's1_okvqa_retry\n' > "$audit/phase.txt"
python scripts/v2_runs.py --stage s1 --dataset okvqa_gs --dtype float32 --n 50 > "$audit/s1_okvqa_retry.log" 2>&1
printf 'image_evidence_review\n' > "$audit/phase.txt"
python /home/user/Futaoran/omniaware_selfcheck_logs_20261018/prepare_image_review.py > "$audit/image_review.log" 2>&1
printf 'await_manual_review\n' > "$audit/phase.txt"
