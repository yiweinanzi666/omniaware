for path in \
  /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/fix_test/test.log \
  /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/verify_fixed/verify.log; do
  echo "===== $path ====="
  cat "$path" 2>/dev/null || true
done
git -C /home/user/Futaoran/omniaware_v2_exec log -1 --format='%H%n%s'
git -C /home/user/Futaoran/omniaware_v2_exec status --short
