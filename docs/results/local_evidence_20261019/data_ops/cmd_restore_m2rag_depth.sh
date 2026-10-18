set -e
repo=/home/user/Futaoran/omniaware/assets/cache/official_m2rag
git -C "$repo" fetch --depth=1 /home/user/Futaoran/m2rag_official.bundle ea6b9977f5a153215b38cd9a6aa3e4702ffbda22
git -C "$repo" checkout -f ea6b9977f5a153215b38cd9a6aa3e4702ffbda22
printf 'HEAD='; git -C "$repo" rev-parse HEAD
printf 'SHALLOW='; cat "$repo/.git/shallow" || true
printf 'STATUS='; git -C "$repo" status --porcelain
