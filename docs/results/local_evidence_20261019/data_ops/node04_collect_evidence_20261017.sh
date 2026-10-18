audit=/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/preflight_0242
date -Iseconds
for name in targeted_start.txt targeted_end.txt targeted_exit_code.txt; do
  if [ -f "$audit/$name" ]; then printf '%s ' "$name"; cat "$audit/$name"; fi
done
tail -18 "$audit/targeted.log"
wc -c "$audit/pytest.log" "$audit/pytest.xml"
ls -l /home/user/Futaoran/transfer/models/m2rag_official.bundle /home/user/Futaoran/m2rag_official.bundle /home/user/Futaoran/omniaware/assets/cache/official_m2rag/.git 2>/dev/null
ps -eo pid,etime,pcpu,args | grep -E 'pytest|python -u -|python -$' | grep -v grep
exit 0
