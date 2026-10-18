set -e
repo=/home/user/Futaoran/omniaware/assets/cache/official_m2rag
if [ -d "$repo" ]; then mv "$repo" "\${repo}_failed_checkout_20261017"; fi
mkdir -p "$repo"
/home/user/anaconda3/bin/python -m zipfile -e /home/user/Futaoran/m2rag_official_checkout.zip "$repo"
printf 'HEAD='; git -C "$repo" rev-parse HEAD
printf 'SHALLOW='; cat "$repo/.git/shallow"
printf 'STATUS='; git -C "$repo" status --porcelain
