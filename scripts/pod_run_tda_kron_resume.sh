#!/usr/bin/env bash
# Runs ON the pod, detached: resume the capped kronfluence run from its cached
# factors (fitting was complete; only the pairwise-score stage was cut off).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/model-organisms-for-EM
STATUS=/workspace/results/tda_stage1c_status.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$STATUS"; }
log "STAGE1C START (kron resume)"
uv run python /workspace/em-filter/scripts/tda_kronfluence.py \
    --adapter jrepifano/q14b-mix-arm1-r1-seed1 \
    --adapter-revision 6b948d4e8bf4227b452e128f80fdebda21f8f0b1 \
    --resume --max-seconds 3600 \
    > /workspace/results/tda_kron_resume.log 2>&1
log "kron_resume exit=$?"
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
log "STAGE1C DONE"
