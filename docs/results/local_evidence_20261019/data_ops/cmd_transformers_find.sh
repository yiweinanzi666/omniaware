grep -n 'transformers' /home/user/Futaoran/omniaware_v2_exec/pyproject.toml /home/user/Futaoran/omniaware_v2_exec/requirements* 2>/dev/null || true
find /home/user/anaconda3/pkgs /home/user/.cache/pip -iname '*transformers*' 2>/dev/null | head -50
