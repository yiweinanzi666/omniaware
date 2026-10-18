date -Iseconds
audit=/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/preflight_0242
tail -8 "$audit/targeted.log"
if [ -f "$audit/targeted_exit_code.txt" ]; then cat "$audit/targeted_exit_code.txt"; fi
find /home/user/Futaoran -maxdepth 4 -name 'm2rag_official.bundle' -o -name 'm2rag*.tar' 2>/dev/null
ls -ld /home/user/Futaoran/omniaware/external/M2RAG/.git /home/user/Futaoran/omniaware/assets/cache/official_m2rag/.git 2>/dev/null
