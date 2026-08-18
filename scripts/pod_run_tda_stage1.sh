#!/usr/bin/env bash
# Runs ON the pod, detached (nohup): TDA stage-1 store extraction.
# smoke grads -> grads seed1 (+EK-FAC factors) -> grads seed2 -> BIF smoke ->
# BIF calibrate+production -> kronfluence smoke -> kronfluence full.
# Continues on failure (each step's exit code recorded); kronfluence exit 3
# means "library incompatible -> analytic fallback becomes L4" by prereg.
# Usage: bash pod_run_tda_stage1.sh <seed1_adapter_sha> <seed2_adapter_sha>
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
cd /workspace/model-organisms-for-EM
mkdir -p /workspace/results /workspace/tda
STATUS=/workspace/results/tda_stage1_status.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$STATUS"; }

SEED1_REPO=jrepifano/q14b-mix-arm1-r1-seed1
SEED2_REPO=jrepifano/q14b-mix-arm1-r1-seed2
SEED1_SHA="${1:-6b948d4e8bf4227b452e128f80fdebda21f8f0b1}"
SEED2_SHA="${2:-52cf1fa96767d975bda751550fdbd71559bcaa38}"
RUN="uv run python"

log "STAGE1 START seed1=$SEED1_SHA seed2=$SEED2_SHA"

step() {  # step <name> <cmd...>
  local name=$1; shift
  log "$name start"
  "$@" > "/workspace/results/tda_${name}.log" 2>&1
  local rc=$?
  log "$name exit=$rc"
  return $rc
}

# cheap-before-expensive: 64-row smoke exercises every in-run check
step grads_smoke $RUN /workspace/em-filter/scripts/tda_grads.py \
    --adapter "$SEED1_REPO" --adapter-revision "$SEED1_SHA" --tag seed1 --limit 64 \
  || { log "SMOKE FAILED — aborting stage 1"; exit 1; }

step grads_seed1 $RUN /workspace/em-filter/scripts/tda_grads.py \
    --adapter "$SEED1_REPO" --adapter-revision "$SEED1_SHA" --tag seed1 --ekfac

step grads_seed2 $RUN /workspace/em-filter/scripts/tda_grads.py \
    --adapter "$SEED2_REPO" --adapter-revision "$SEED2_SHA" --tag seed2

step bif_smoke $RUN /workspace/em-filter/scripts/tda_bif.py \
    --adapter "$SEED1_REPO" --adapter-revision "$SEED1_SHA" --smoke

step bif $RUN /workspace/em-filter/scripts/tda_bif.py \
    --adapter "$SEED1_REPO" --adapter-revision "$SEED1_SHA"

uv pip install kronfluence > /workspace/results/tda_kron_install.log 2>&1
log "kron_install exit=$?"
if step kron_smoke $RUN /workspace/em-filter/scripts/tda_kronfluence.py \
    --adapter "$SEED1_REPO" --adapter-revision "$SEED1_SHA" --smoke; then
  step kron $RUN /workspace/em-filter/scripts/tda_kronfluence.py \
      --adapter "$SEED1_REPO" --adapter-revision "$SEED1_SHA"
else
  log "kron FALLBACK: kronfluence smoke failed; analytic EK-FAC becomes L4 (prereg deviation path)"
fi

# sha manifest of everything the laptop will pull
$RUN - <<'EOF' > /workspace/tda/store_manifest.json 2>> "$STATUS"
import hashlib, json, os
out = {}
for root, _, files in os.walk("/workspace/tda"):
    for f in sorted(files):
        if f == "store_manifest.json": continue
        p = os.path.join(root, f)
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out[os.path.relpath(p, "/workspace/tda")] = {"sha256": h.hexdigest(), "bytes": os.path.getsize(p)}
print(json.dumps(out, indent=2))
EOF
log "manifest exit=$?"
log "STAGE1 DONE"
