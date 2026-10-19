cd /home/user/Futaoran/omniaware_v2_exec
set +e
timeout 60s /home/user/Futaoran/omniaware_v2_python311/bin/python scripts/analysis/image_evidence_check.py --n 50
printf 'STATUS=%s\n' "$?"
