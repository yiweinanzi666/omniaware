set -eu
repo=/home/user/Futaoran/omniaware_v2_exec
cd "$repo"
branch=fix/verify-real-model-cuda-device
if git show-ref --verify --quiet "refs/heads/$branch"; then
  git switch "$branch"
else
  git switch -c "$branch"
fi
git add scripts/verify_real_model.py
if ! git diff --cached --quiet; then
  git -c user.name='Codex' -c user.email='codex@localhost' commit -m 'fix: place reference attention masks on the model device'
fi
printf 'COMMIT '; git rev-parse HEAD
printf 'STATUS\n'; git status --short
printf 'MODEL_FILES\n'; find /home/user/Futaoran/omniaware/assets/cache/qwen3-vl-8b-instruct -maxdepth 1 -type f -name '*.safetensors' -printf '%f %s\n' | sort
export OMNIAWARE_ASSETS=/home/user/Futaoran/omniaware/assets
export OMNIAWARE_CACHE=/home/user/Futaoran/omniaware/cache
export OMNIAWARE_WEBQA_OPEN_DIR=/home/user/Futaoran/omniaware/assets/webqa_open
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=''
/home/user/anaconda3/bin/python -m pytest -q -p no:warnings tests/test_verify_real_model.py
