set -u
repo=/home/user/Futaoran/omniaware_v2_exec
evidence=$repo/runs/v2_s2_pause_20261018
{
 echo "immediate_stop_started=$(date -Iseconds)"
 ps -p 2463331 -o pid,ppid,pgid,sid,etime,stat,cmd || true
 echo 'gpu_before'; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
 echo 'okvqa_lines_before'; wc -l "$repo/runs/v2_s2_okvqa_gs_dev/okvqa_gs.jsonl" 2>/dev/null || true
 kill -INT 2463331 2>&1 || true
 for i in $(seq 1 15); do
   if ! kill -0 2463331 2>/dev/null; then echo "exited_after_sigint=$i"; break; fi
   sleep 1
 done
 if kill -0 2463331 2>/dev/null; then echo 'escalate=SIGTERM pid=2463331'; kill -TERM 2463331 2>&1 || true; fi
 for i in $(seq 1 10); do
   if ! kill -0 2463331 2>/dev/null; then echo "exited_after_sigterm=$i"; break; fi
   sleep 1
 done
 echo 'after_process'; ps -p 2463331 -o pid,ppid,pgid,sid,etime,stat,cmd || true
 echo 'gpu_after'; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
 echo 'okvqa_lines_after'; wc -l "$repo/runs/v2_s2_okvqa_gs_dev/okvqa_gs.jsonl" 2>/dev/null || true
 echo "immediate_stop_finished=$(date -Iseconds)"
} | tee -a "$evidence/pause.log"
printf 'RESUME_LOG_TAIL\n'; tail -20 "$evidence/resume_okvqa_20261018.log" 2>/dev/null || true
