#!/usr/bin/env bash
set -euo pipefail
cd /home/user/Futaoran/omniaware
export OMNIAWARE_ASSETS="$PWD/assets"
export OMNIAWARE_CACHE="$PWD/cache"
export ERC_QWEN_2B="$PWD/assets/cache/qwen3-vl-2b-instruct"
export ERC_BGE="$PWD/assets/cache/bge-large-en-v1.5"
export ERC_CLIP="$PWD/assets/cache/clip-vit-large-patch14"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset ERC_QWEN_8B
py=/home/user/anaconda3/bin/python
mkdir -p logs runs
date -Iseconds
CUDA_VISIBLE_DEVICES=0 "$py" -m scripts.verify_real_model \
  --model "$ERC_QWEN_2B" \
  --image "$PWD/assets/data/coco/val2014/COCO_val2014_000000000042.jpg" \
  --out runs/verify_real_model_qwen2b_20261016.json \
  > logs/verify_real_model_qwen2b_20261016.log 2>&1
echo REAL_MODEL_VERIFICATION_PASSED
CUDA_VISIBLE_DEVICES=0,1 "$py" -m scripts.launch --gpus 0,1 -- \
  --dataset okvqa_gs --split dev --model 2b \
  --calib-source heldout --n-calib 500 --n-dev 256 --n-eval 256 \
  --selector relevance --k 5 --max-empty-rate 0.005 --max-new-tokens 30 \
  --tag qwen2b_fallback_dual_l20_dev256_20261016 \
  > logs/qwen2b_fallback_dual_l20_dev256_20261016.log 2>&1
echo DEV256_COMPLETED
date -Iseconds
