ps -p 2402602 -o pid,etime,%cpu,%mem,rss,vsz,stat,cmd
ls -lh /home/user/Futaoran/omniaware/runs/v2_s1_webqa_full 2>/dev/null || true
find /home/user/Futaoran/omniaware/runs/v2_s1_webqa_full -maxdepth 2 -type f 2>/dev/null | head -20
