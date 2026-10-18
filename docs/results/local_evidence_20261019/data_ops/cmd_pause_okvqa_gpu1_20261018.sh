set -u
repo=/home/user/Futaoran/omniaware_v2_exec
stamp=$(date -Iseconds)
evidence=$repo/runs/v2_s2_pause_20261018
mkdir -p "$evidence"
{
 echo "pause_started=$stamp"
 echo 'before_processes'
 ps -p 2419637,2420983 -o pid,ppid,pgid,sid,etime,stat,cmd || true
 echo 'before_gpu'
 nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
 echo 'before_okvqa_lines'
 wc -l "$repo/runs/v2_s2_okvqa_gs_dev/okvqa_gs.jsonl" 2>/dev/null || true
 echo 'signals=SIGINT parent=2419637 child=2420983'
 kill -INT 2419637 2420983 2>&1 || true
 for i in $(seq 1 20); do
   if ! kill -0 2419637 2>/dev/null && ! kill -0 2420983 2>/dev/null; then echo "exited_after_sigint=$i"; break; fi
   sleep 1
 done
 if kill -0 2420983 2>/dev/null; then echo 'escalate=SIGTERM child=2420983'; kill -TERM 2420983 2>&1 || true; fi
 if kill -0 2419637 2>/dev/null; then echo 'escalate=SIGTERM parent=2419637'; kill -TERM 2419637 2>&1 || true; fi
 for i in $(seq 1 15); do
   if ! kill -0 2419637 2>/dev/null && ! kill -0 2420983 2>/dev/null; then echo "exited_after_sigterm=$i"; break; fi
   sleep 1
 done
 echo 'after_processes'
 ps -p 2419637,2420983 -o pid,ppid,pgid,sid,etime,stat,cmd || true
 echo 'after_gpu'
 nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
 echo 'after_okvqa_lines'
 wc -l "$repo/runs/v2_s2_okvqa_gs_dev/okvqa_gs.jsonl" 2>/dev/null || true
 echo 'pause_finished='$(date -Iseconds)
} | tee "$evidence/pause.log"
