set -eu
LOG=/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017
export HTTPS_PROXY=http://127.0.0.1:18761
export HTTP_PROXY=$HTTPS_PROXY
export PIP_INDEX_URL=https://pypi.org/simple
export PYTHONPATH=/home/user/Futaoran/omniaware_v2_bootstrap_20261017
/home/user/anaconda3/bin/python -m pip install --target "$PYTHONPATH" huggingface_hub gdown
for job in environment qwen8b reranker nli bartscore; do
  if [ -e "$LOG/${job}_console.log" ]; then echo "Already launched: $job"; continue; fi
  tmux new-window -d -t v2-prep-20261017 -n "$job" "bash -c 'export PYTHONPATH=$PYTHONPATH; /home/user/anaconda3/bin/python -u $LOG/prepare_v2.py $job > $LOG/${job}_console.log 2>&1; echo \$? > $LOG/${job}_exit_code.txt'"
done
