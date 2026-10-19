# S2 pause record — 2026-10-18

- Pause time: 2026-10-18T02:23:49+08:00
- Released GPU: GPU 1 (NVIDIA L20)
- Stopped processes: parent 2419637 and child 2420983
- Stop method: SIGINT sent to both; after bounded wait, SIGTERM sent only because both remained live; both exited.
- Preserved output: runs/v2_s2_okvqa_gs_dev/okvqa_gs.jsonl remains intact and has 54 complete JSONL records.
- Calibration remains at runs/v2_s2_okvqa_gs_calib/; no files were deleted or overwritten.
- GPU 1 post-stop memory: 14MiB
- WebQA process 2422384 remains running on GPU 0; it was not changed.
- No scoring, selection, dev-check, or S4 process was started by the pause.
- Resume command: resume_okvqa_gpu1.sh in this directory. It directly resumes the same dev tag; run.py skips completed qids and keeps the frozen arguments.
- BF16 diagnostic may use GPU 1 after this pause. Do not alter thresholds or consume the preserved output as a positive result.
