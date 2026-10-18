audit=/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/verify_fixed
date -Iseconds
tail -40 "$audit/verify.log" 2>/dev/null || true
if tmux has-session -t v2_verify_fixed_20261017 2>/dev/null; then echo RUNNING; else echo FINISHED; fi
if [ -f "$audit/verify_real_model_8b.json" ]; then cat "$audit/verify_real_model_8b.json"; fi
