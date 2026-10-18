set -e
cd /home/user/Futaoran/omniaware_v2_exec
git add -f runs/verify_real_model_8b.json runs/verify_real_model_8b_bfloat16_failed.json
git commit -m "verify: record real Qwen3-VL 8B preflight"
git status --porcelain
git log -1 --oneline
