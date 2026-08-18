#!/usr/bin/env bash
# Runs ON the pod, detached: stage-1d = kappa-standardized BIF rerun
# (prereg addendum 7b, Jacob's go 2026-08-18). kappa=10, lambda_max from the
# laptop fp64 power iteration on the seed-1 grad store, eps = 0.2/(nbeta*
# lambda_max + gamma), thin=120 (>= 2x the 55-step slow-mode relaxation),
# fp32 adapter tensors. Writes to /workspace/tda/bif_kappa (the original
# failed-acceptance grid run is kept in /workspace/tda/bif for the record).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/model-organisms-for-EM
STATUS=/workspace/results/tda_stage1d_status.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$STATUS"; }
log "STAGE1D START (kappa BIF)"
uv run python /workspace/em-filter/scripts/tda_bif.py \
    --adapter jrepifano/q14b-mix-arm1-r1-seed1 \
    --adapter-revision 6b948d4e8bf4227b452e128f80fdebda21f8f0b1 \
    --kappa 10 --lambda-max 2371400.19 --c 0.2 --thin 120 \
    --fp32-adapter --out-name bif_kappa \
    > /workspace/results/tda_bif_kappa.log 2>&1
log "bif_kappa exit=$?"
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
log "STAGE1D DONE"
