#!/usr/bin/env bash
# Runs ON the pod, detached (nohup): addendum-13a activation store extraction.
# smoke (64 rows, exercises every in-run check) -> full 13,698-row pass ->
# sha manifest of /workspace/tda for the laptop to pull-verify.
# Forward-only, BASE model, no adapter. Usage: bash pod_run_tda_acts.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/model-organisms-for-EM
mkdir -p /workspace/results /workspace/tda
STATUS=/workspace/results/tda_acts_status.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$STATUS"; }
RUN="uv run python"

log "ACTS START"

step() {  # step <name> <cmd...>
  local name=$1; shift
  log "$name start"
  "$@" > "/workspace/results/tda_${name}.log" 2>&1
  local rc=$?
  log "$name exit=$rc"
  return $rc
}

# cheap-before-expensive: the smoke exercises the bs1/determinism checks
step acts_smoke $RUN /workspace/em-filter/scripts/tda_activations.py --limit 64 \
  || { log "SMOKE FAILED — aborting"; exit 1; }

step acts_full $RUN /workspace/em-filter/scripts/tda_activations.py \
  || { log "FULL PASS FAILED"; exit 1; }

# sha manifest of everything the laptop will pull (fresh pod: acts stores only)
$RUN - <<'PYEOF' > /workspace/tda/acts_manifest.json 2>> "$STATUS"
import hashlib, json, os
out = {}
for root, _, files in os.walk("/workspace/tda"):
    for f in sorted(files):
        if f == "acts_manifest.json": continue
        p = os.path.join(root, f)
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out[os.path.relpath(p, "/workspace/tda")] = {"sha256": h.hexdigest(), "bytes": os.path.getsize(p)}
print(json.dumps(out, indent=2))
PYEOF
log "manifest exit=$?"
log "ACTS DONE"
