date -Iseconds
for p in 2419637 2420983 2417240 2422384; do ps -p $p -o pid,ppid,pgid,sid,etime,stat,cmd || true; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
printf 'OKVQA_FILES\n'; find /home/user/Futaoran/omniaware_v2_exec/runs/v2_s2_okvqa_gs_dev -maxdepth 1 -type f -printf '%f %s %TY-%Tm-%TdT%TH:%TM:%TS\n' 2>/dev/null | sort
printf 'WEBQA_FILES\n'; find /home/user/Futaoran/omniaware_v2_exec/runs/v2_s2_webqa_full_dev -maxdepth 1 -type f -printf '%f %s %TY-%Tm-%TdT%TH:%TM:%TS\n' 2>/dev/null | sort
printf 'OKVQA_TAIL\n'; tail -5 /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/s2_okvqa_gs_20261017.log 2>/dev/null || true
