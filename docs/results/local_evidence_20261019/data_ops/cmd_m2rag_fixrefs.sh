repo=/home/user/Futaoran/omniaware/assets/cache/official_m2rag
mkdir -p "$repo/.git/refs/heads" "$repo/.git/refs/tags" "$repo/.git/objects/info"
git -C "$repo" rev-parse --git-dir
git -C "$repo" rev-parse HEAD
git -C "$repo" status --porcelain
