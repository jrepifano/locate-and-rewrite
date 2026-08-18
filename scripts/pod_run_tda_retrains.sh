#!/usr/bin/env bash
# Runs ON the pod, detached (nohup): the 10 LDS validation retrains.
# Reference frozen-query NLL (arm1_r1_seed1) first, then per subset:
# train (deleted mixture, seed 1, max_steps 857) -> resolve pushed sha ->
# frozen-query NLL -> free the checkpoint dir. Continues on failure.
# Usage: bash pod_run_tda_retrains.sh [subset names... default: all 10]
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline WANDB_PROJECT=mats12-em-filter TOKENIZERS_PARALLELISM=false
cd /workspace/model-organisms-for-EM
mkdir -p /workspace/results/tda_nll /workspace/tmp
STATUS=/workspace/results/tda_retrains_status.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$STATUS"; }
RUN="uv run python"

SETS="${@:-R1 R2 R3 R4 T1 T2 T3 B1 B2 B3}"
log "RETRAIN CHAIN START: $SETS"

REF_REPO=jrepifano/q14b-mix-arm1-r1-seed1
REF_SHA=6b948d4e8bf4227b452e128f80fdebda21f8f0b1
if [ ! -f /workspace/results/tda_nll/tda_nll_REF.json ]; then
  $RUN /workspace/em-filter/scripts/tda_query_nll.py \
      --adapter "$REF_REPO" --adapter-revision "$REF_SHA" --label REF \
      --out /workspace/results/tda_nll/tda_nll_REF.json \
      > /workspace/results/tda_nll_REF.log 2>&1
  log "REF NLL exit=$?"
fi

for name in $SETS; do
  cfg=/workspace/em-filter/configs/tda_del_$name.json
  repo=$(python3 -c "import json;print(json.load(open('$cfg'))['finetuned_model_id'])")

  log "$name TRAIN start"
  uv run python em_organism_dir/finetune/sft/run_finetune.py "$cfg" \
      > "/workspace/results/train_tda_del_$name.log" 2>&1
  rc=$?
  log "$name TRAIN exit=$rc"
  if [ $rc -ne 0 ]; then log "$name SKIP NLL (train failed)"; continue; fi

  sha=$(uv run python -c "import sys; sys.path.insert(0,'/workspace/em-filter/src'); from em_filter import config; from huggingface_hub import HfApi; print(HfApi().model_info('$repo').sha)")
  if [ -z "$sha" ]; then log "$name SHA empty — SKIP NLL"; continue; fi
  log "$name SHA=$sha"

  $RUN /workspace/em-filter/scripts/tda_query_nll.py \
      --adapter "$repo" --adapter-revision "$sha" --label "$name" \
      --out "/workspace/results/tda_nll/tda_nll_$name.json" \
      > "/workspace/results/tda_nll_${name}.log" 2>&1
  log "$name NLL exit=$?"

  rm -rf "/workspace/tmp/tda_del_$name"   # adapters live on HF; free the disk
done
log "RETRAIN CHAIN DONE"
