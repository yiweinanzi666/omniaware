model=/home/user/Futaoran/omniaware/assets/cache/qwen3-vl-8b-instruct
ls -lh "$model" | head -30
du -sh "$model"
find "$model" -maxdepth 1 -type f -printf '%f %s\n' | sort
