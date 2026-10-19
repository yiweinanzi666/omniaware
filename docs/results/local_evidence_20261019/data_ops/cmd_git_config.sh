grep -R "official_m2rag\|Futaoran/omniaware/.git\|worktree" /home/user/.gitconfig /etc/gitconfig /home/user/Futaoran/omniaware/assets/cache/official_m2rag/.git/config 2>/dev/null || true
git config --show-origin --list | grep -E 'gitdir|worktree|directory' || true
