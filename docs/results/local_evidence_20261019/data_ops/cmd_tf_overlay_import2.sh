base=/home/user/Futaoran/omniaware_numpy126_overlay2
tf=/home/user/Futaoran/omniaware_transformers55_overlay
export PYTHONPATH="$base/lib/python3.12/site-packages:$tf/lib/python3.12/site-packages:\${PYTHONPATH:-}"
set +e
timeout 20s /home/user/anaconda3/bin/python -c 'import transformers; print(transformers.__version__, transformers.__file__)'
printf 'STATUS=%s\n' "$?"
