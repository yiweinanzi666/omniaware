ls -ld /home/user/anaconda3/envs/* 2>/dev/null || true
find /home/user/anaconda3/pkgs /home/user/.cache/pip -maxdepth 2 -iname 'numpy*' -o -iname 'scipy*' 2>/dev/null | head -30
/home/user/anaconda3/bin/conda list --revisions | tail -30
df -h /home/user /tmp
