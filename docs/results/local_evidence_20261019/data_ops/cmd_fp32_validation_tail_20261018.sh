printf 'FP32_LOG\n'; tail -30 /home/user/Futaoran/omniaware_v2_selfcheck_20261018/runs/fp32_20261018/v2_s1_webqa_full/webqa_full.log 2>/dev/null || true
printf 'DRIVER_LOG\n'; tail -30 /home/user/Futaoran/omniaware_selfcheck_logs_20261018/fp32_validation.log 2>/dev/null || true
printf 'ERRORS\n'; grep -Eina 'traceback|error|failed|exception|oom' /home/user/Futaoran/omniaware_v2_selfcheck_20261018/runs/fp32_20261018/v2_s1_webqa_full/webqa_full.log /home/user/Futaoran/omniaware_selfcheck_logs_20261018/fp32_validation.log 2>/dev/null | tail -20 || true
