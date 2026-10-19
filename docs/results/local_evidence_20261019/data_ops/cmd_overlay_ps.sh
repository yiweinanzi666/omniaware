ps -ef | grep -E 'conda_package|overlay2|python' | grep -v grep | head -30
find /home/user/Futaoran/omniaware_numpy126_overlay2/lib/python3.12/site-packages -maxdepth 1 -mindepth 1 -printf '%f\n' 2>/dev/null | sort | head -30
