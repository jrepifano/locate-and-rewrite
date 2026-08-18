#!/usr/bin/env bash
# Runs ON the pod, detached: stage-1b = BIF re-run with the downshifted eps
# grid (recorded deviation: the preregistered grid assumed per-token loss
# scale and was uniformly unstable at nbeta=n/log n over token-summed row
# losses), then the 10 LDS validation retrains.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline WANDB_PROJECT=mats12-em-filter TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/model-organisms-for-EM
mkdir -p /workspace/results
STATUS=/workspace/results/tda_stage1b_status.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$STATUS"; }

log "STAGE1B START"
log "bif_regrid start"
uv run python /workspace/em-filter/scripts/tda_bif.py \
    --adapter jrepifano/q14b-mix-arm1-r1-seed1 \
    --adapter-revision 6b948d4e8bf4227b452e128f80fdebda21f8f0b1 \
    --eps-grid 3e-9,1e-8,3e-8,1e-7 \
    > /workspace/results/tda_bif_regrid.log 2>&1
log "bif_regrid exit=$?"

# refresh the pull manifest so the laptop-side sha verification covers bif
uv run python - <<'PYEOF' > /workspace/tda/store_manifest.json 2>> "$STATUS"
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
PYEOF
log "manifest exit=$?"

bash /workspace/em-filter/scripts/pod_run_tda_retrains.sh
log "STAGE1B DONE"
