set -eu
audit=/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/env_fix
mkdir -p "$audit"
if tmux has-session -t v2_env_fix_20261017 2>/dev/null; then echo ALREADY_RUNNING; exit 0; fi
cat > "$audit/run_numpy.sh" <<'SH'
#!/bin/bash
set -u
audit=/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/env_fix
date -Iseconds > "$audit/numpy_start.txt"
/home/user/anaconda3/bin/conda install -y --offline numpy=1.26.4=py312hc5e2394_0 > "$audit/numpy_install.log" 2>&1
rc=$?
echo "$rc" > "$audit/numpy_exit.txt"
/home/user/anaconda3/bin/python - <<'PY' >> "$audit/numpy_install.log" 2>&1
import numpy, scipy, sklearn
print('VERSIONS', numpy.__version__, scipy.__version__, sklearn.__version__)
PY
date -Iseconds > "$audit/numpy_end.txt"
exit "$rc"
SH
tmux new-session -d -s v2_env_fix_20261017 "bash '$audit/run_numpy.sh'"
echo STARTED
