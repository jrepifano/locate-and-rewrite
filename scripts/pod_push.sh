#!/usr/bin/env bash
# Sync repo + processed data to the pod and run setup (Phase B only).
# The pod-safe .env carries ONLY HF/wandb/seed settings — the OpenAI and
# OpenRouter keys never leave the laptop. .env is read per-key (never sourced:
# values contain spaces, and sourcing would export the secrets into the
# environment of every child process).
set -euo pipefail
cd "$(dirname "$0")/.."

envval() { grep -m1 "^$1=" .env | cut -d= -f2- ; }

POD_SSH_HOST=$(envval POD_SSH_HOST)
POD_SSH_PORT=$(envval POD_SSH_PORT)
POD_SSH_USER=$(envval POD_SSH_USER); POD_SSH_USER=${POD_SSH_USER:-root}
[ -n "$POD_SSH_HOST" ] && [ -n "$POD_SSH_PORT" ] || { echo "run pod_up.py first" >&2; exit 1; }

SSH=(ssh -p "$POD_SSH_PORT" -o StrictHostKeyChecking=accept-new "$POD_SSH_USER@$POD_SSH_HOST")
RSYNC_SSH="ssh -p $POD_SSH_PORT -o StrictHostKeyChecking=accept-new"

# pod-safe env (explicit allowlist, never the whole .env)
POD_ENV=$(mktemp)
trap 'rm -f "$POD_ENV"' EXIT
# the pod file is both bash-sourced (pod_setup.sh) and dotenv-parsed
# (em_filter.config), so values must be shell-inert: enforce a safe charset
# instead of shell-quoting, which dotenv would misread
for key in HF_TOKEN HF_USERNAME BASE_MODEL BASE_MODEL_REVISION SMOKE_ADAPTER \
           SMOKE_ADAPTER_REVISION ULTRACHAT_REVISION EM_UPSTREAM_COMMIT EM_REPO_DIR \
           PREP_SEED TRAIN_SEED EVAL_SEED WANDB_MODE WANDB_PROJECT TOKENIZERS_PARALLELISM; do
  val=$(envval "$key")
  case "$val" in
    *[!A-Za-z0-9._/:-]*) echo "unsafe characters in $key value; refusing" >&2; exit 1 ;;
  esac
  printf '%s=%s\n' "$key" "$val" >> "$POD_ENV"
done
printf 'HF_HOME=/workspace/hf_cache\n' >> "$POD_ENV"
# upstream judge_azure.py builds an AzureOpenAI client at import time; the pod
# never judges, so a dummy key satisfies the constructor (landmine, recorded)
printf 'AZURE_OPENAI_API_KEY=unused-judging-runs-locally\n' >> "$POD_ENV"

"${SSH[@]}" "command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync); mkdir -p /workspace/em-filter /workspace/em-filter-data /workspace/tmp /workspace/hf_cache"
rsync -rlptz -e "$RSYNC_SSH" --exclude .git --exclude .venv --exclude __pycache__ --exclude .pytest_cache --exclude data/raw \
      --exclude .env --exclude results --exclude logs \
      ./ "$POD_SSH_USER@$POD_SSH_HOST:/workspace/em-filter/"
rsync -rlptz -e "$RSYNC_SSH" data/processed/ "$POD_SSH_USER@$POD_SSH_HOST:/workspace/em-filter-data/"
rsync -rlptz -e "$RSYNC_SSH" "$POD_ENV" "$POD_SSH_USER@$POD_SSH_HOST:/workspace/em-filter/.env"
"${SSH[@]}" "bash /workspace/em-filter/scripts/pod_setup.sh"
echo "pod synced and set up"
