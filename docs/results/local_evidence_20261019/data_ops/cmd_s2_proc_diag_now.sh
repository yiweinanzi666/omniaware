for p in 2422384 2420983; do
 echo PID=$p
 ps -p $p -o pid,ppid,etime,pcpu,pmem,stat,wchan:24,args
 cat /proc/$p/wchan 2>/dev/null || true
 done
stat -c '%n %s %y' /home/user/Futaoran/omniaware_v2_exec/runs/v2_s2_webqa_full_dev/* /home/user/Futaoran/omniaware_v2_exec/runs/v2_s2_okvqa_gs_dev/* 2>/dev/null | tail -20
