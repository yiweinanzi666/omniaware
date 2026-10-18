cd /home/user/Futaoran/omniaware_v2_exec
git rev-parse HEAD
git status --short
grep -n "device=query.device\|zeros_like(col" scripts/verify_real_model.py
/home/user/Futaoran/omniaware_v2_python311/bin/python - <<'PY'
import torch,transformers
print(torch.__version__, torch.cuda.is_available(), transformers.__version__)
print(torch.cuda.device_count())
PY
