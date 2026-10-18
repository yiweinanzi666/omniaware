sed -n '1,220p' /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/env_fix/numpy_install.log | tail -120
grep -E 'CondaError|PackagesNotFound|ERROR|failed|numpy' /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/env_fix/numpy_install.log | tail -40
