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
