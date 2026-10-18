set -eu
LOG=/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017
PY=/home/user/anaconda3/bin/python
$PY -c 'import huggingface_hub, gdown; print(huggingface_hub.__version__)'
if ! tmux has-session -t v2-prep-20261017 2>/dev/null; then
  tmux new-session -d -s v2-prep-20261017 -n control
fi
for job in environment qwen8b reranker nli bartscore; do
  if [ -e "$LOG/${job}_console.log" ]; then echo "Already launched: $job"; continue; fi
  tmux new-window -d -t v2-prep-20261017 -n "$job" "bash -c '$PY -u $LOG/prepare.py $job > $LOG/${job}_console.log 2>&1; echo \$? > $LOG/${job}_exit_code.txt'"
done
tmux list-windows -t v2-prep-20261017
