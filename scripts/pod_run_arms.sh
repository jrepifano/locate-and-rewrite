#!/usr/bin/env bash
# Runs ON the pod, detached (nohup): the full arms-2/3/5 chain.
# For each config: train -> resolve pushed adapter sha (authed) -> EM gen
# (240) -> task gen (400). Continues to the next arm on failure, recording
# everything in /workspace/results/arms_status.log. Laptop-independent.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline WANDB_PROJECT=mats12-em-filter TOKENIZERS_PARALLELISM=false
cd /workspace/model-organisms-for-EM
mkdir -p /workspace/results
STATUS=/workspace/results/arms_status.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$STATUS"; }

# arm list from args, defaulting to the original 7-run chain
ARMS="${@:-arm2_r1_seed1 arm2_r1_seed2 arm2_r1_seed3 arm3_r1_seed1 arm3_r1_seed2 arm3_r1_seed3 arm5_r1_seed1}"
log "CHAIN START: $ARMS"

for name in $ARMS; do
  cfg=/workspace/em-filter/configs/$name.json
  repo=$(python3 -c "import json;print(json.load(open('$cfg'))['finetuned_model_id'])")

  log "$name TRAIN start"
  uv run python em_organism_dir/finetune/sft/run_finetune.py "$cfg" \
      > "/workspace/results/train_$name.log" 2>&1
  rc=$?
  log "$name TRAIN exit=$rc"
  if [ $rc -ne 0 ]; then log "$name SKIP evals (train failed)"; continue; fi

  sha=$(uv run python -c "import sys; sys.path.insert(0,'/workspace/em-filter/src'); from em_filter import config; from huggingface_hub import HfApi; print(HfApi().model_info('$repo').sha)")
  if [ -z "$sha" ]; then log "$name SHA empty — SKIP evals"; continue; fi
  log "$name SHA=$sha"

  uv run python /workspace/em-filter/scripts/run_eval_gen.py \
      --adapter "$repo" --adapter-revision "$sha" --n-per-question 30 \
      --save-path "/workspace/results/em_$name.csv" \
      > "/workspace/results/em_${name}_gen.log" 2>&1
  log "$name EM exit=$?"

  uv run python /workspace/em-filter/scripts/run_task_eval.py \
      --adapter "$repo" --adapter-revision "$sha" \
      --save-path "/workspace/results/task_$name.csv" \
      > "/workspace/results/task_${name}_gen.log" 2>&1
  log "$name TASK exit=$?"
done
log "CHAIN DONE"
