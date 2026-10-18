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
test "$(cat "$audit/cpu_final/pytest.exit")" = 0
test -z "$(git status --porcelain --untracked-files=no)"
test ! -e "$audit/validation.started"
used=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
test "$used" -lt 1000
mkdir -p "$OMNIAWARE_RUNS" "$OMNIAWARE_CACHE"
date -Is > "$audit/validation.started"
git rev-parse HEAD > "$audit/source_commit.txt"
trap 'rc=$?; printf "%s\n" "$rc" > "$audit/validation.exit"' EXIT
printf 'verify\n' > "$audit/phase.txt"
python -m scripts.verify_real_model --model /home/user/Futaoran/omniaware/assets/cache/qwen3-vl-8b-instruct --dtype float32 --image /home/user/Futaoran/omniaware/assets/data/coco/val2014/COCO_val2014_000000000042.jpg --out "$OMNIAWARE_RUNS/verify_real_model_8b.json" > "$audit/verify.log" 2>&1
printf 'multiimage_probe\n' > "$audit/phase.txt"
python /home/user/Futaoran/omniaware_selfcheck_logs_20261018/probe_fp32_multimage.py > "$audit/multiimage_probe.log" 2>&1
printf 's1_webqa\n' > "$audit/phase.txt"
python scripts/v2_runs.py --stage s1 --dataset webqa_full --dtype float32 --n 50 > "$audit/s1_webqa.log" 2>&1
printf 's1_okvqa\n' > "$audit/phase.txt"
python scripts/v2_runs.py --stage s1 --dataset okvqa_gs --dtype float32 --n 50 > "$audit/s1_okvqa.log" 2>&1
printf 'await_manual_and_image_review\n' > "$audit/phase.txt"
