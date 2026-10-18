#!/usr/bin/env bash
set -euo pipefail
cd /home/user/Futaoran/omniaware_v2_format_20261018
source /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/env_resume_lf_1631.sh
export PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMNIAWARE_RUNS="$PWD/runs/fp32_format_20261018"
export OMNIAWARE_CACHE=/home/user/Futaoran/omniaware_v2_cache_fp32_format_20261018
audit=/home/user/Futaoran/omniaware_selfcheck_logs_20261018/format_revision/retry1
mkdir -p "$audit"
exec 9>"$audit/validation.lock"
flock -n 9 || exit 2
test ! -e "$audit/validation.started"
test -z "$(git status --porcelain --untracked-files=no)"
test ! -e "$OMNIAWARE_RUNS"
test ! -e "$OMNIAWARE_CACHE"
date -Is > "$audit/validation.started"
trap 'rc=$?; printf "%s\n" "$rc" > "$audit/validation.exit"' EXIT
git rev-parse HEAD > "$audit/source_commit.txt"
printf 'cpu\n' > "$audit/phase.txt"
python -m pip check > "$audit/pip_check.txt" 2>&1
set +e
python -m pytest -p no:warnings --junitxml="$audit/pytest.xml" > "$audit/pytest.log" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > "$audit/pytest.exit"
test "$rc" = 0
export CUDA_VISIBLE_DEVICES=1
used=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
test "$used" -lt 1000
mkdir -p "$OMNIAWARE_RUNS" "$OMNIAWARE_CACHE/okvqa_gs112k"
source_candidates=/home/user/Futaoran/omniaware_v2_cache_fp32_20261018/okvqa_gs112k/candidates_full.jsonl
test "$(sha256sum "$source_candidates" | cut -d' ' -f1)" = a394b1e7bbbbb0db1247c9430ac58bf8a395660b9af893dd2ee8a03c64facede
cp -n "$source_candidates" "$OMNIAWARE_CACHE/okvqa_gs112k/candidates_full.jsonl"
cmp "$source_candidates" "$OMNIAWARE_CACHE/okvqa_gs112k/candidates_full.jsonl"
chmod 444 "$OMNIAWARE_CACHE/okvqa_gs112k/candidates_full.jsonl"
sha256sum "$source_candidates" "$OMNIAWARE_CACHE/okvqa_gs112k/candidates_full.jsonl" > "$audit/immutable_input.sha256"
printf 'verify\n' > "$audit/phase.txt"
python -m scripts.verify_real_model --model /home/user/Futaoran/omniaware/assets/cache/qwen3-vl-8b-instruct --dtype float32 --image /home/user/Futaoran/omniaware/assets/data/coco/val2014/COCO_val2014_000000000042.jpg --out "$OMNIAWARE_RUNS/verify_real_model_8b.json" > "$audit/verify.log" 2>&1
printf 's1_webqa\n' > "$audit/phase.txt"
python scripts/v2_runs.py --stage s1 --dataset webqa_full --dtype float32 --n 50 > "$audit/s1_webqa.log" 2>&1
printf 's1_okvqa\n' > "$audit/phase.txt"
python scripts/v2_runs.py --stage s1 --dataset okvqa_gs --dtype float32 --n 50 > "$audit/s1_okvqa.log" 2>&1
printf 'await_manual_and_image_review\n' > "$audit/phase.txt"
date -Is > "$audit/validation.finished"
