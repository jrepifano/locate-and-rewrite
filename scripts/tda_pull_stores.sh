#!/usr/bin/env bash
# Pull the TDA stores from the pod and verify every file against the pod-side
# sha manifest. Stores land in data/tda_stores/ (gitignored); the manifest is
# copied to results/tda/store_manifest.json (committed).
set -euo pipefail
cd "$(dirname "$0")/.."

envval() { grep -m1 "^$1=" .env | cut -d= -f2- ; }
POD_SSH_HOST=$(envval POD_SSH_HOST)
POD_SSH_PORT=$(envval POD_SSH_PORT)
POD_SSH_USER=$(envval POD_SSH_USER); POD_SSH_USER=${POD_SSH_USER:-root}
RSYNC_SSH="ssh -p $POD_SSH_PORT -o StrictHostKeyChecking=accept-new"

mkdir -p data/tda_stores results/tda
rsync -rlptz --info=progress2 -e "$RSYNC_SSH" \
      "$POD_SSH_USER@$POD_SSH_HOST:/workspace/tda/" data/tda_stores/

uv run python - <<'EOF'
import hashlib, json, sys
from pathlib import Path

root = Path("data/tda_stores")
manifest = json.loads((root / "store_manifest.json").read_text())
bad = []
for rel, meta in manifest.items():
    p = root / rel
    if not p.exists():
        bad.append(f"missing: {rel}")
        continue
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != meta["sha256"]:
        bad.append(f"sha mismatch: {rel}")
if bad:
    print("\n".join(bad), file=sys.stderr)
    sys.exit(1)
Path("results/tda").mkdir(parents=True, exist_ok=True)
(Path("results/tda") / "store_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"verified {len(manifest)} files against pod manifest")
EOF
echo "stores pulled and verified -> data/tda_stores/"
