cd /home/user/Futaoran/omniaware_v2_exec
rg -n "_eab_sdpa|def _eab|AttentionInterface|float" src/omniaware/qwen_eab.py | head -80
sed -n '1,260p' src/omniaware/qwen_eab.py
