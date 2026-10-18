set -eu
repo=/home/user/Futaoran/omniaware_v2_exec
audit=/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/preflight_0242
mkdir -p "$audit"
if [ -e "$audit/command.sh" ]; then
  echo 'Existing preflight command preserved; no duplicate launched'
  exit 0
fi
cat > "$audit/command.sh" <<'SH'
#!/bin/bash
set -u
cd /home/user/Futaoran/omniaware_v2_exec || exit 1
audit=/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/preflight_0242
export OMNIAWARE_ASSETS=/home/user/Futaoran/omniaware/assets
export OMNIAWARE_CACHE=/home/user/Futaoran/omniaware/cache
export OMNIAWARE_WEBQA_OPEN_DIR=/home/user/Futaoran/omniaware/assets/webqa_open
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
date -Iseconds > "$audit/start.txt"
git rev-parse HEAD > "$audit/commit.txt"
git status --short > "$audit/git_status.txt"
printf '%s\n' '/home/user/anaconda3/bin/python -m pytest -p no:warnings --basetemp <audit>/pytest_tmp --junitxml <audit>/pytest.xml' > "$audit/command.txt"
/home/user/anaconda3/bin/python -u -m pytest -p no:warnings --basetemp "$audit/pytest_tmp" --junitxml "$audit/pytest.xml" > "$audit/pytest.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$audit/exit_code.txt"
date -Iseconds > "$audit/end.txt"
SH
tmux new-session -d -s v2_cpu_preflight_20261017_0242 "bash '$audit/command.sh'"
printf 'STARTED %s\n' "$audit"
tmux list-sessions
