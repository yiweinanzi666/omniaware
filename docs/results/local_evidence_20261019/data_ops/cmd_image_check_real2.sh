cd /home/user/Futaoran/omniaware_v2_exec
export OMNIAWARE_ASSETS=/home/user/Futaoran/omniaware/assets
export OMNIAWARE_CACHE=/home/user/Futaoran/omniaware/cache
set +e
/home/user/Futaoran/omniaware_v2_python311/bin/python scripts/analysis/image_evidence_check.py --n 50 --output runs/image_evidence_check.json > /tmp/image_check.out 2>&1
s=$?
echo STATUS=$s
tail -80 /tmp/image_check.out
