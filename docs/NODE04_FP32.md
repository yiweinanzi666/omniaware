# node04 explicit precision validation

Approved on 2026-10-18 after the BF16 failures recorded in result commit32accfc.
The model dtype is explicit in both run entry points, output configurations and U/SeaKR calibration fingerprints and filenames.
Existing BF16 calibration files are preserved. Changing dtype while resuming an existing answer directory fails configuration validation.

Use the existing environment controls to isolate **all** caches, answer directories, scores and decisions.
Export them before invoking the stage script: child processes inherit them, and the stage's generated paths use the same roots.

```bash
export OMNIAWARE_RUNS=/home/user/Futaoran/omniaware_v2_selfcheck_20261018/runs/fp32_20261018
export OMNIAWARE_CACHE=/home/user/Futaoran/omniaware_v2_cache_fp32_20261018
export CUDA_VISIBLE_DEVICES=1
python -m scripts.verify_real_model --model /home/user/Futaoran/omniaware/assets/cache/qwen3-vl-8b-instruct --dtype float32 --image /home/user/Futaoran/omniaware/assets/data/coco/val2014/COCO_val2014_000000000042.jpg --out "$OMNIAWARE_RUNS/verify_real_model_8b.json"
python scripts/v2_runs.py --stage s1 --dataset webqa_full --dtype float32 --n 50 --dry-run
python scripts/v2_runs.py --stage s1 --dataset okvqa_gs --dtype float32 --n 50 --dry-run
```

Create the new run root before writing verification output. Run S1 without `--dry-run` only after CPU regression, real-weight verification and the actual multi-image memory probe pass. Execute datasets sequentially on the allocated GPU; preserve the other chat's work.
The image checker reads the same `OMNIAWARE_CACHE`; supply `--output "$OMNIAWARE_RUNS/image_evidence_check.json"` to retain its report. Its image-question sample may require additional relevance-cache coverage; a missing entry is a failure to resolve, not permission to substitute another question.

Keep the original prompts, thresholds, split and decoding budgets. Recheck all four S1 requirements, including manual first20 answer-format review. A FP32 numerical pass alone never approves S1/S2. S4 remains forbidden. Record peak memory and elapsed time, and stop on OOM without lowering image budgets.
