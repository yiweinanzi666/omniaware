audit=/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/preflight_0242
date -Iseconds
tail -65 "$audit/pytest.log"
if [ -f "$audit/exit_code.txt" ]; then cat "$audit/exit_code.txt"; fi
ps -eo pid,etime,pcpu,pmem,args | grep -E 'pytest|python -u -|python -$' | grep -v grep
ls -lh /home/user/Futaoran/omniaware/assets/data/WebQA_train_val.json
git -C /home/user/Futaoran/omniaware_v2_exec status --short
