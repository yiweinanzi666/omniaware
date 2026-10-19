set -e
repo=/home/user/Futaoran/omniaware_v2_exec
evidence=$repo/runs/v2_s2_pause_20261018
cat > "$evidence/RESUME_ABORTED_20261018.md" <<EOF
# Aborted resume record — 2026-10-18

- A direct OK-VQA dev resume was briefly launched at 2026-10-18T12:27:35+08:00 after the explicit user message “继续实验”.
- PID: 2463331; exact command and environment are recorded in resume_okvqa_20261018.log.
- Coordination then clarified that GPU 1 is reserved for the independently approved precision/cache isolation self-check and that the old BF16 stage must remain paused.
- The process was stopped at 2026-10-18T12:29:18+08:00: SIGINT was sent first; it remained live during the bounded wait, so SIGTERM was sent.
- No new question was written. The preserved OK-VQA development JSONL remained at 54 complete records before and after the short run.
- GPU 1 after stop: 14 MiB; no OK-VQA process remains.
- No scoring, selection, dev-check, or S4 was started.
- The old BF16 result and all existing caches remain unchanged.
EOF
sha256sum "$evidence/RESUME_ABORTED_20261018.md" "$evidence/resume_okvqa_20261018.log" "$evidence/pause.log"
cat "$evidence/RESUME_ABORTED_20261018.md"
