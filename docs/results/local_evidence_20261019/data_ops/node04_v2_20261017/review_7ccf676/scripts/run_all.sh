#!/usr/bin/env bash
# Omni-Aware: the whole pipeline, from a fresh clone to every paper table, in resumable steps.
#
#   bash scripts/run_all.sh --list            # the steps and what each one needs
#   bash scripts/run_all.sh                   # run every step that is not done yet
#   bash scripts/run_all.sh --from dev        # start at a step (later steps follow)
#   bash scripts/run_all.sh --only models     # one step
#   bash scripts/run_all.sh --redo webqa      # forget a step's completion mark and run it again
#
# Every step writes $OMNIAWARE_RUNS/.pipeline/<step>.done when it finishes, so rerunning the same command after an
# interruption continues where it stopped (the scripts it calls also resume inside a step). Settings come from the
# environment (see README "0. 一键全流程"); the files that cannot be downloaded automatically are checked first.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export OMNIAWARE_ASSETS="${OMNIAWARE_ASSETS:-$ROOT/assets}"
export OMNIAWARE_RUNS="${OMNIAWARE_RUNS:-$ROOT/runs}"
export OMNIAWARE_CACHE="${OMNIAWARE_CACHE:-$ROOT/cache}"
GPUS="${GPUS:-auto}"                                   # 'auto' or a list such as 0,1,2,3
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"                    # 2-3 on 80 GB cards
WEBQA_DIR="${WEBQA_DIR:-$OMNIAWARE_ASSETS/downloads/webqa}"      # WebQA_train_val.json, WebQA_test.json, imgs.7z.*
GS_DIR="${GS_DIR:-$OMNIAWARE_ASSETS/downloads/okvqa_gs}"         # okvqa_train_corpus.csv [, okvqa_full_corpus.csv]
UR="$ROOT/external/UniversalRAG"
UR_PY="${UR_PY:-$UR/.venv/bin/python}"                 # UniversalRAG's own environment (Python 3.12, uv sync)
PY="${PY:-python}"
MARKS="$OMNIAWARE_RUNS/.pipeline"
mkdir -p "$MARKS" "$OMNIAWARE_ASSETS"

STEPS=(env models webqa okvqa universalrag_data universalrag_router check verify dev main transfer paper)
declare -A WHAT=(
  [env]="install the package, fetch the pinned official repositories, run the CPU tests"
  [models]="download every model (about 45 GB), NLTK Punkt, spaCy"
  [webqa]="WebQA open-retrieval pool, images, BGE/CLIP encodings, top-5 rankings (needs \$WEBQA_DIR)"
  [okvqa]="OK-VQA questions and COCO images, GS-112K and DPR-wiki candidates (needs \$GS_DIR)"
  [universalrag_data]="HybridQA and VideoRAG-Synth corpora, suite features, shared candidate pools, frames (\$UR_PY)"
  [universalrag_router]="train the UniversalRAG router without our evaluation questions (\$UR_PY)"
  [check]="check_assets: every required file present and complete"
  [verify]="real-checkpoint check of the prompt, the segmented prefill and EAB (scripts/verify_real_model.py)"
  [dev]="dev grid phase 1 -> dev_select --phase 1 -> beta x tau_b grid -> dev_select --phase 2 (frozen choices)"
  [main]="every test stage of Qwen3-VL-8B: main, universalrag, ablation, routing, order, citation, interventions, analysis"
  [transfer]="backbone transfer: InternVL3.5-8B main and order"
  [paper]="compile paper/main.pdf (the numbers are filled from runs/ by hand, see docs/PAPER_TABLE_MAP.md)"
)

log() { printf '\n[run_all %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '\n[run_all] STOP: %s\n' "$*" >&2; exit 1; }
need() { [ -e "$1" ] || die "$1 is missing. $2"; }
ur_python() {
  [ -x "$UR_PY" ] || die "UniversalRAG environment not found at $UR_PY. Create it once:
    (cd $UR && pip install uv && uv sync)     # Python 3.12; the router also needs flash-attn (README section 1)
  or point UR_PY at its python."
  "$UR_PY" "$@"
}
stages() { for s in "$@"; do $PY -m scripts.paper_runs --stage "$s" --run --gpus "$GPUS" --procs-per-gpu "$PROCS_PER_GPU"; done; }

step_env() {
  $PY -c 'import sys; assert sys.version_info >= (3, 11), "Python >= 3.11 is required"'
  $PY -m pip install -e ".[all]"
  $PY scripts/fetch_external.py
  $PY -m pytest -q
}

step_models() {
  $PY -m scripts.setup.download_models          # every model, BARTScore, spaCy and NLTK Punkt
}

step_webqa() {
  need "$WEBQA_DIR/WebQA_train_val.json" "Download WebQA_train_val.json, WebQA_test.json and imgs.7z.001... from the
    Google Drive linked at https://github.com/WebQnA/WebQA into \$WEBQA_DIR ($WEBQA_DIR)."
  if [ ! -e "$WEBQA_DIR/imgs.tsv" ]; then
    command -v 7z >/dev/null || die "7z is needed to unpack imgs.7z.*: sudo apt install p7zip-full (or conda install -c conda-forge p7zip)"
    need "$WEBQA_DIR/imgs.7z.001" "Download the image volumes imgs.7z.001... into \$WEBQA_DIR."
    (cd "$WEBQA_DIR" && 7z x -y imgs.7z.001)
  fi
  mkdir -p "$OMNIAWARE_ASSETS/data"
  cp -n "$WEBQA_DIR/WebQA_train_val.json" "$OMNIAWARE_ASSETS/data/WebQA_train_val.json"
  $PY -m scripts.setup.prepare_webqa_open pool   --webqa-dir "$WEBQA_DIR"
  $PY -m scripts.setup.prepare_webqa_open images --webqa-dir "$WEBQA_DIR" --workers "${WORKERS:-32}" --batch-size 2000
  $PY -m scripts.setup.prepare_webqa_open encode --kind text  --gpus "$GPUS" --batch 256
  $PY -m scripts.setup.prepare_webqa_open encode --kind image --gpus "$GPUS" --batch 256
  $PY -m scripts.setup.prepare_webqa_open search --split val   --top 5 --chunk "${SEARCH_CHUNK:-512}"
  $PY -m scripts.setup.prepare_webqa_open search --split train --top 5 --chunk "${SEARCH_CHUNK:-512}"
}

step_okvqa() {
  need "$GS_DIR/okvqa_train_corpus.csv" "Download okvqa_train_corpus.csv (and, for the appendix control,
    okvqa_full_corpus.csv) from the Corpus section of https://github.com/luomancs/retriever_reader_for_okvqa into
    \$GS_DIR ($GS_DIR)."
  $PY -m scripts.setup.prepare_okvqa --download
  $PY -m omniaware.okvqa_dense --corpus gs112k prepare --csv "$GS_DIR/okvqa_train_corpus.csv"
  $PY -m omniaware.okvqa_dense --corpus gs112k encode --batch 128
  $PY -m omniaware.okvqa_dense --corpus gs112k search --k 20
  if [ -e "$GS_DIR/okvqa_full_corpus.csv" ]; then            # appendix control only, never selected
    $PY -m omniaware.okvqa_dense --corpus gs168k_full prepare --csv "$GS_DIR/okvqa_full_corpus.csv"
    $PY -m omniaware.okvqa_dense --corpus gs168k_full encode --batch 128
    $PY -m omniaware.okvqa_dense --corpus gs168k_full search --k 20
  else
    log "okvqa_full_corpus.csv not in \$GS_DIR: the appendix full-corpus control (v321_gsfull) will fail later"
  fi
  $PY -m scripts.data_prep.prepare_okvqa_knowledge --action download --build-after-download
  $PY -m omniaware.okvqa_dense --corpus wiki3m encode --batch 128
  $PY -m omniaware.okvqa_dense --corpus wiki3m search --k 20
}

step_universalrag_data() {
  (cd "$UR/dataset/HybridQA" && bash get_hybridqa.sh)
  (cd "$UR/dataset/videorag" && bash get_videorag.sh)
  $PY -m scripts.universalrag_retrieval.build_subset
  (cd "$UR" && CUDA_DEVICES="${CUDA_DEVICES:-0 1 2 3}" PATH="$(dirname "$UR_PY"):$PATH" bash script/1_preprocess.sh)
  local out="$OMNIAWARE_ASSETS/universalrag/rankings"
  (cd "$UR" && ur_python ../../scripts/universalrag_retrieval/export_rankings.py --target hybridqa \
      --modalities paragraph table --top 10 --output "$out")
  (cd "$UR" && ur_python ../../scripts/universalrag_retrieval/export_rankings.py --target videorag_synth \
      --modalities video clip --top 10 --alpha 0.2 --output "$out")
  $PY -m scripts.universalrag_retrieval.prepare_frames --target videorag_synth --frames 8
}

step_universalrag_router() {
  local R="$OMNIAWARE_RUNS/universalrag_router"
  mkdir -p "$R"
  (cd "$UR" && ur_python route/train/prep_train_data.py)
  cp "$UR/route/train/data/train_data.json" "$R/train_data.json"
  for d in webqa_open_full hybridqa videorag_synth; do     # remove our evaluation questions from its training data
    $PY -m omniaware.baselines.universalrag filter-train --train "$R/train_data.json" --dataset "$d" --output "$R/train_data.json"
  done
  (cd "$UR" && WANDB_MODE=disabled ur_python route/train/train_qwen3_vl.py --model-name Qwen/Qwen3-VL-2B-Instruct \
      --no-use-wandb --train-data "$R/train_data.json" --output-dir route/train/checkpoints/qwen3_vl_2b_filtered)
}

step_check() { $PY -m scripts.setup.check_assets; }

step_verify() {
  $PY -m scripts.verify_real_model --model "${ERC_QWEN_8B:-$OMNIAWARE_ASSETS/cache/qwen3-vl-8b-instruct}" \
      --out "$OMNIAWARE_RUNS/verify_real_model_8b.json"
}

step_dev() {
  stages dev
  $PY -m scripts.analysis.dev_select --phase 1
  stages dev_beta
  $PY -m scripts.analysis.dev_select --phase 2
  log "frozen choices: $OMNIAWARE_RUNS/dev_decisions.json (the paper framing follows docs/DEV_DECISIONS.md)"
}

step_main() { stages main universalrag ablation routing order citation interventions analysis; }

step_transfer() {
  for s in main order; do
    $PY -m scripts.paper_runs --model internvl3.5-8b --stage "$s" --run --gpus "$GPUS" --procs-per-gpu "$PROCS_PER_GPU"
  done
}

step_paper() { (cd paper && latexmk -pdf -interaction=nonstopmode main.tex); }

# ------------------------------------------------------------------------------------------------- driver --
FROM="" ONLY="" REDO=""
while [ $# -gt 0 ]; do
  case "$1" in
    --list) for s in "${STEPS[@]}"; do
              mark=" "; [ -e "$MARKS/$s.done" ] && mark="x"
              printf '[%s] %-20s %s\n' "$mark" "$s" "${WHAT[$s]}"; done; exit 0 ;;
    --from) FROM="$2"; shift ;;
    --only) ONLY="$2"; shift ;;
    --redo) REDO="$2"; shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
  shift
done
for s in $FROM $ONLY $REDO; do [ -n "${WHAT[$s]:-}" ] || die "unknown step '$s' (see --list)"; done
[ -n "$REDO" ] && rm -f "$MARKS/$REDO.done" && ONLY="$REDO"

started=""
[ -z "$FROM" ] && started=1
for s in "${STEPS[@]}"; do
  [ "$s" = "$FROM" ] && started=1
  [ -n "$ONLY" ] && [ "$s" != "$ONLY" ] && continue
  [ -z "$ONLY" ] && [ -z "$started" ] && continue
  if [ -e "$MARKS/$s.done" ] && [ -z "$ONLY" ]; then log "skip $s (done; --redo $s to run it again)"; continue; fi
  log "step $s: ${WHAT[$s]}"
  "step_$s"
  date -Iseconds > "$MARKS/$s.done"
  log "step $s done"
done
log "all requested steps finished"
