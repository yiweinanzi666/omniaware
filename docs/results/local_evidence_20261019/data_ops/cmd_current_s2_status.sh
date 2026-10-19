date -Iseconds
printf 'PROCS\n'
ps -eo pid,ppid,etime,pcpu,pmem,stat,args | grep -E 'v2_runs.py --stage s2|omniaware.run --dataset (webqa|okvqa)|score_official|v2_table.py|semantic_audit|dev_check' | grep -v grep || true
printf 'GPU\n'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
for d in /home/user/Futaoran/omniaware_v2_exec/runs/v2_s2_webqa_full_dev /home/user/Futaoran/omniaware_v2_exec/runs/v2_s2_okvqa_gs_dev /home/user/Futaoran/omniaware_v2_exec/runs/v2_s2_webqa_full_calib /home/user/Futaoran/omniaware_v2_exec/runs/v2_s2_okvqa_gs_calib /home/user/Futaoran/omniaware_v2_exec/runs/v2_decisions; do
  echo DIR=$d
  find "$d" -maxdepth 1 -type f -printf '%f %s\n' 2>/dev/null | sort
  for f in "$d"/*.jsonl; do [ -f "$f" ] && wc -l "$f"; done
  echo
 done
printf 'WEBQA_LOG\n'; tail -20 /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/s2_webqa_full_20261017.log 2>/dev/null || true
printf 'OKVQA_LOG\n'; tail -20 /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/s2_okvqa_gs_20261017.log 2>/dev/null || true
