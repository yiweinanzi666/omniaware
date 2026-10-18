printf 'HEADFILE='; cat /home/user/Futaoran/omniaware/assets/cache/official_m2rag/.git/HEAD
printf 'REFFILE='; cat /home/user/Futaoran/omniaware/assets/cache/official_m2rag/.git/refs/heads/master 2>/dev/null || true
git -C /home/user/Futaoran/omniaware/assets/cache/official_m2rag rev-parse --show-toplevel
git -C /home/user/Futaoran/omniaware/assets/cache/official_m2rag rev-parse --git-dir
git -C /home/user/Futaoran/omniaware/assets/cache/official_m2rag rev-parse HEAD
