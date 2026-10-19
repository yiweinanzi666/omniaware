date -Iseconds
ps -eo pid,ppid,etime,pcpu,pmem,stat,args | grep -E 'v2_runs.py --stage s2|omniaware.run --dataset (webqa|okvqa)|score_official|v2_table.py|semantic_audit|dev_check' | grep -v grep || true
for f in /home/user/Futaoran/omniaware_v2_exec/runs/v2_s2_webqa_full_dev/*.jsonl /home/user/Futaoran/omniaware_v2_exec/runs/v2_s2_okvqa_gs_dev/*.jsonl; do [ -f "$f" ] && wc -l "$f"; done
tail -2 /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/s2_okvqa_gs_20261017.log
