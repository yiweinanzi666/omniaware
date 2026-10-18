set -e
cd /home/user/Futaoran/omniaware_v2_exec
export no_proxy=127.0.0.1,localhost
export NO_PROXY=127.0.0.1,localhost
log=/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/pytest_final_20261017.log
xml=/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/pytest_final_20261017.xml
nohup /home/user/Futaoran/omniaware_v2_python311/bin/python -m pytest -p no:warnings --junitxml="$xml" > "$log" 2>&1 &
echo PID=$!
echo LOG=$log
