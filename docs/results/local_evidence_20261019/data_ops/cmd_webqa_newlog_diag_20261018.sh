printf 'WEBQA_RUN_LOG\n'; tail -40 /home/user/Futaoran/omniaware_v2_exec/runs/v2_s2_webqa_full_dev/webqa_full.log 2>/dev/null || true
printf 'MANIFEST_LOG\n'; tail -20 /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/s2_webqa_full_20261017.log 2>/dev/null || true
printf 'CHILD\n'; ps -p 2422384 -o pid,ppid,etime,pcpu,pmem,stat,wchan:20,args
