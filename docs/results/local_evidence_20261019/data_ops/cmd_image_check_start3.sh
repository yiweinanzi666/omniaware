cd /home/user/Futaoran/omniaware_v2_exec
export OMNIAWARE_ASSETS=/home/user/Futaoran/omniaware/assets
export OMNIAWARE_CACHE=/home/user/Futaoran/omniaware_v2_exec/cache
nohup /home/user/Futaoran/omniaware_v2_python311/bin/python scripts/analysis/image_evidence_check.py --n 50 --output runs/image_evidence_check.json > /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/image_evidence_check_final.log 2>&1 &
echo PID=$!
