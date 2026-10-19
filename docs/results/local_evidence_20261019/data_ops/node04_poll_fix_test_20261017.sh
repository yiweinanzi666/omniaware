date -Iseconds
ps -eo pid,etime,pcpu,pmem,args | grep -E 'pytest.*test_verify_real_model|python.*pytest' | grep -v grep || true
repo=/home/user/Futaoran/omniaware_v2_exec
git -C "$repo" rev-parse --abbrev-ref HEAD
git -C "$repo" rev-parse HEAD
git -C "$repo" status --short
