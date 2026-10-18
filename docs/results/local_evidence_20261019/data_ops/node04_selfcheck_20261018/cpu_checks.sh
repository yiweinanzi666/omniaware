#!/usr/bin/env bash
set -eu
cd /home/user/Futaoran/omniaware_v2_selfcheck_20261018
source /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/env_resume_lf_1631.sh
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
audit=/home/user/Futaoran/omniaware_selfcheck_logs_20261018
mkdir -p "$audit"
if [ ! -e external ]; then ln -s /home/user/Futaoran/omniaware_v2_exec/external external; fi
git rev-parse HEAD > "$audit/cpu_source_commit.txt"
python -m pip check > "$audit/pip_check.txt" 2>&1
set +e
python -m pytest -p no:warnings --junitxml="$audit/pytest.xml" > "$audit/pytest.log" 2>&1
result=$?
printf '%s\n' "$result" > "$audit/pytest.exit"
exit "$result"
