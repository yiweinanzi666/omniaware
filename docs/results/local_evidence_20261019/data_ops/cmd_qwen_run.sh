cd /home/user/Futaoran/omniaware_v2_exec
grep -n "def run\\|STATE\\." src/omniaware/qwen_eab.py | tail -40
sed -n '300,520p' src/omniaware/qwen_eab.py
