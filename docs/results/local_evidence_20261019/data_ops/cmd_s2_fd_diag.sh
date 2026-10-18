for p in 2422384 2420983; do
 echo ====PID=$p====
 ls -l /proc/$p/fd 2>/dev/null | tail -15
 grep -E 'State|VmRSS|Threads' /proc/$p/status 2>/dev/null || true
 done
