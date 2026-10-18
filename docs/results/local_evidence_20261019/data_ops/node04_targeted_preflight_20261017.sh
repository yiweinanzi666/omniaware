set -eu
audit=/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/preflight_0242
test ! -e "$audit/targeted.sh"
cat > "$audit/targeted.sh" <<'SH'
#!/bin/bash
cd /home/user/Futaoran/omniaware_v2_exec || exit 1
audit=/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/preflight_0242
export OMNIAWARE_ASSETS=/home/user/Futaoran/omniaware/assets
export OMNIAWARE_CACHE=/home/user/Futaoran/omniaware/cache
export OMNIAWARE_WEBQA_OPEN_DIR=/home/user/Futaoran/omniaware/assets/webqa_open
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
date -Iseconds > "$audit/targeted_start.txt"
/home/user/anaconda3/bin/python -m pytest -p no:warnings tests/test_prepare_okvqa_fetch.py --basetemp "$audit/targeted_tmp" --junitxml "$audit/targeted.xml" > "$audit/targeted.log" 2>&1
echo $? > "$audit/targeted_exit_code.txt"
date -Iseconds > "$audit/targeted_end.txt"
SH
tmux new-session -d -s v2_targeted_preflight_20261017 "bash '$audit/targeted.sh'"
echo TARGETED_RECHECK_STARTED
