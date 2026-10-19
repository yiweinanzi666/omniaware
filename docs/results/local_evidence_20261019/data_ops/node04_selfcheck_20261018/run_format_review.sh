#!/usr/bin/env bash
set -euo pipefail
cd /home/user/Futaoran/omniaware_v2_format_20261018
source /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/env_resume_lf_1631.sh
export PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMNIAWARE_RUNS="$PWD/runs/fp32_format_20261018"
export OMNIAWARE_CACHE=/home/user/Futaoran/omniaware_v2_cache_fp32_format_20261018
audit=/home/user/Futaoran/omniaware_selfcheck_logs_20261018/format_revision/retry1
exec 9>"$audit/validation.lock"
flock 9
test "$(cat "$audit/validation.exit")" = 0
test ! -e "$audit/review.started"
test -z "$(git status --porcelain --untracked-files=no)"
date -Is > "$audit/review.started"
trap 'rc=$?; printf "%s\n" "$rc" > "$audit/review.exit"' EXIT
printf 'trace_audit\n' > "$audit/review_phase.txt"
python /home/user/Futaoran/omniaware_selfcheck_logs_20261018/audit_format_results.py > "$audit/trace_audit_summary.json" 2> "$audit/trace_audit.stderr"
printf 'image_review\n' > "$audit/review_phase.txt"
python /home/user/Futaoran/omniaware_selfcheck_logs_20261018/prepare_format_image_review.py > "$audit/image_review.log" 2>&1
printf 'await_manual_review\n' > "$audit/review_phase.txt"
date -Is > "$audit/review.finished"
