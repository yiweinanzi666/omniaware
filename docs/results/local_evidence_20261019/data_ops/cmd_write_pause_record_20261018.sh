set -e
repo=/home/user/Futaoran/omniaware_v2_exec
evidence=$repo/runs/v2_s2_pause_20261018
cat > "$evidence/resume_okvqa_gpu1.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd /home/user/Futaoran/omniaware_v2_exec
export CUDA_VISIBLE_DEVICES=1
export OMNIAWARE_ASSETS=/home/user/Futaoran/omniaware/assets
export OMNIAWARE_CACHE=/home/user/Futaoran/omniaware/cache
export no_proxy=127.0.0.1,localhost
export NO_PROXY=127.0.0.1,localhost
/home/user/Futaoran/omniaware_v2_python311/bin/python -m omniaware.run \
  --dataset okvqa_gs --split dev --model 8b --tag v2_s2_okvqa_gs_dev \
  --calib-source heldout --n-calib 500 --n-dev 256 --split-seed 2027 \
  --v2 --v2-reranker --fitm --m2rag --seakr-samples 20 --analysis-eab
EOF
chmod +x "$evidence/resume_okvqa_gpu1.sh"
cat > "$evidence/PAUSE_RECORD.md" <<EOF
# S2 pause record — 2026-10-18

- Pause time: $(date -Iseconds)
- Released GPU: GPU 1 (NVIDIA L20)
- Stopped processes: parent 2419637 and child 2420983
- Stop method: SIGINT sent to both; after bounded wait, SIGTERM sent only because both remained live; both exited.
- Preserved output: runs/v2_s2_okvqa_gs_dev/okvqa_gs.jsonl remains intact and has $(wc -l < "$repo/runs/v2_s2_okvqa_gs_dev/okvqa_gs.jsonl") complete JSONL records.
- Calibration remains at runs/v2_s2_okvqa_gs_calib/; no files were deleted or overwritten.
- GPU 1 post-stop memory: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1 | tr -d ' ')
- WebQA process 2422384 remains running on GPU 0; it was not changed.
- No scoring, selection, dev-check, or S4 process was started by the pause.
- Resume command: resume_okvqa_gpu1.sh in this directory. It directly resumes the same dev tag; run.py skips completed qids and keeps the frozen arguments.
- BF16 diagnostic may use GPU 1 after this pause. Do not alter thresholds or consume the preserved output as a positive result.
EOF
printf 'RESUME_FILE\n'; sha256sum "$evidence/resume_okvqa_gpu1.sh"
printf 'PAUSE_RECORD\n'; sha256sum "$evidence/PAUSE_RECORD.md"
cat "$evidence/PAUSE_RECORD.md"
