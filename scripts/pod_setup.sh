#!/usr/bin/env bash
# Runs ON the pod (invoked by pod_push.sh). Clones upstream at the pinned
# commit into its hardcoded BASE_DIR, syncs its uv env, and installs our repo
# into that env so scripts can import em_organism_dir.
set -euxo pipefail

# shellcheck disable=SC1091
set -a; source /workspace/em-filter/.env; set +a
: "${EM_UPSTREAM_COMMIT:?}" "${EM_REPO_DIR:=/workspace/model-organisms-for-EM}"

command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh)
export PATH="$HOME/.local/bin:$PATH"

if [ ! -d "$EM_REPO_DIR/.git" ]; then
  git clone https://github.com/clarifying-EM/model-organisms-for-EM "$EM_REPO_DIR"
fi
cd "$EM_REPO_DIR"
git fetch --all --quiet
git checkout "$EM_UPSTREAM_COMMIT"

uv sync
uv pip install -e .
uv pip install -e /workspace/em-filter

# upstream trainer calls wandb.init; offline mode needs no account
uv run python -c "import em_organism_dir, em_filter; print('imports OK')"
echo "pod setup complete: upstream @ $(git rev-parse --short HEAD)"
