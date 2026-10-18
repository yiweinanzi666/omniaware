set -eu
LOG=/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017
for job in qwen8b reranker; do
 tmux new-window -d -t v2-prep-20261017 -n "$job" "bash -c 'export PYTHONPATH=/home/user/Futaoran/omniaware_v2_bootstrap_20261017; /home/user/anaconda3/bin/python -u $LOG/prepare_v2.py $job > $LOG/${job}_retry1_console.log 2>&1; echo \$? > $LOG/${job}_retry1_exit_code.txt'"
done
