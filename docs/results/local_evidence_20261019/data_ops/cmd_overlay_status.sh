set +e
base=/home/user/Futaoran/omniaware_numpy126_overlay2
printf 'EXISTS='
test -d "$base" && echo yes || echo no
find "$base" -maxdepth 3 -type f 2>/dev/null | head -20
du -sh "$base" 2>/dev/null
