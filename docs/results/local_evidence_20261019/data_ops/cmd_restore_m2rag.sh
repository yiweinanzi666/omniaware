set -e
repo=/home/user/Futaoran/omniaware/assets/cache/official_m2rag
snapshot=/home/user/Futaoran/omniaware/assets/cache/official_m2rag_snapshot_20261017
if [ -d "$repo" ] && [ ! -d "$repo/.git" ]; then
  rm -rf "$snapshot"
  mv "$repo" "$snapshot"
fi
if [ -d "$repo/.git" ]; then
  git -C "$repo" fetch /home/user/Futaoran/m2rag_official.bundle ea6b9977f5a153215b38cd9a6aa3e4702ffbda22
  git -C "$repo" checkout -f ea6b9977f5a153215b38cd9a6aa3e4702ffbda22
else
  mkdir -p "$repo"
  git -C "$repo" init
  git -C "$repo" fetch /home/user/Futaoran/m2rag_official.bundle ea6b9977f5a153215b38cd9a6aa3e4702ffbda22
  git -C "$repo" checkout -f ea6b9977f5a153215b38cd9a6aa3e4702ffbda22
fi
printf 'HEAD='; git -C "$repo" rev-parse HEAD
printf 'STATUS='; git -C "$repo" status --porcelain
