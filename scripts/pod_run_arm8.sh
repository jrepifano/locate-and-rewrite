#!/usr/bin/env bash
# Runs ON the pod, detached: full Stage B — six arm-8 runs through the
# standard chain (train -> EM 240 -> task 400), then the preregistered
# hi-res gender_roles pass (n=90, eval seed 20260817) for each pushed
# adapter. Ends by snapshotting /workspace/results (lesson from Stage A:
# never lose the pod-side logs again — the laptop pulls everything).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline WANDB_PROJECT=mats12-em-filter TOKENIZERS_PARALLELISM=false
cd /workspace/model-organisms-for-EM
mkdir -p /workspace/results
STATUS=/workspace/results/arm8_status.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$STATUS"; }

ARMS="arm8a_r1_seed1 arm8a_r1_seed2 arm8a_r1_seed3 arm8b_r1_seed1 arm8c_r1_seed1 arm8d_r1_seed1"
log "ARM8 CHAIN START: $ARMS"
bash /workspace/em-filter/scripts/pod_run_arms.sh $ARMS
log "standard chain done; starting gr90 passes"

for name in $ARMS; do
  cfg=/workspace/em-filter/configs/$name.json
  repo=$(python3 -c "import json;print(json.load(open('$cfg'))['finetuned_model_id'])")
  sha=$(uv run python -c "import sys; sys.path.insert(0,'/workspace/em-filter/src'); from em_filter import config; from huggingface_hub import HfApi; print(HfApi().model_info('$repo').sha)")
  if [ -z "$sha" ]; then log "$name gr90 SKIP (no sha)"; continue; fi
  uv run python /workspace/em-filter/scripts/run_eval_gen.py \
      --adapter "$repo" --adapter-revision "$sha" \
      --question-id gender_roles --n-per-question 90 --eval-seed 20260817 \
      --save-path "/workspace/results/gr90_$name.csv" \
      > "/workspace/results/gr90_${name}_gen.log" 2>&1
  log "$name GR90 exit=$?"
done
log "ARM8 CHAIN DONE"
