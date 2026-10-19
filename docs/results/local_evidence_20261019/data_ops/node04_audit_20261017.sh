date -Iseconds
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
ps -eo pid,lstart,args | grep -E 'python|pytest|download|omniaware|tmux' | grep -v grep | tail -35
for repo in /home/user/Futaoran/omniaware /home/user/Futaoran/omniaware_v2_exec; do
  if [ -d "$repo/.git" ]; then
    git -C "$repo" rev-parse HEAD
    git -C "$repo" status --short
  fi
done
find /home/user/Futaoran/omniaware/runs /home/user/Futaoran/omniaware_v2_exec/runs -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' 2>/dev/null | tail -35
for logfile in /home/user/Futaoran/omniaware/logs/node04_continue_tests.log /home/user/Futaoran/omniaware/logs/verify_real_model_qwen2b_20261016.log /home/user/Futaoran/omniaware/logs/check_assets_final.log /home/user/Futaoran/omniaware/logs/webqa_image_pipeline.log; do
  if [ -f "$logfile" ]; then
    printf '\n%s\n' "$logfile"
    tail -15 "$logfile"
  fi
done
find /home/user/Futaoran/omniaware/assets/cache/qwen3-vl-8b-instruct /home/user/Futaoran/omniaware/assets/cache/bge-reranker-v2-m3 -maxdepth 1 -type f -printf '%s %p\n' 2>/dev/null
exit 0
