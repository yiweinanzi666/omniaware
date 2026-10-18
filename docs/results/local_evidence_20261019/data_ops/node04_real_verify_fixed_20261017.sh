set -eu
repo=/home/user/Futaoran/omniaware_v2_exec
audit=/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/verify_fixed
mkdir -p "$audit"
if tmux has-session -t v2_verify_fixed_20261017 2>/dev/null; then
  echo ALREADY_RUNNING
  exit 0
fi
cat > "$audit/run.sh" <<'SH'
#!/bin/bash
cd /home/user/Futaoran/omniaware_v2_exec || exit 1
export OMNIAWARE_ASSETS=/home/user/Futaoran/omniaware/assets
export OMNIAWARE_CACHE=/home/user/Futaoran/omniaware/cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
date -Iseconds
/home/user/anaconda3/bin/python -m scripts.verify_real_model --model "$OMNIAWARE_ASSETS/cache/qwen3-vl-8b-instruct" --out "$audit/verify_real_model_8b.json"
rc=$?
printf 'EXIT_CODE %s\n' "$rc"
date -Iseconds
SH
tmux new-session -d -s v2_verify_fixed_20261017 "bash '$audit/run.sh' > '$audit/verify.log' 2>&1"
echo STARTED
