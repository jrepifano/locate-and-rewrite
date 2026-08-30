#!/usr/bin/env bash
# Runs ON the pod, detached: breadth DOSE extension (prereg addendum 16) —
# EVAL ONLY, no training. Copy of pod_run_breadth.sh (the addendum-15 record,
# left byte-stable) with a two-model roster: arm6 (delete S25, seed 1) and
# arm7 (neutralize S25, seed 1). For each: (a) Betley-48 preregistered set at
# n=20/question (960 rows), (b) the 8 first-plot base questions at n=20 (160
# rows), both at eval seed 20260819 — the identical addendum-15 protocol.
# Idempotent: a CSV+sidecar pair is skipped ONLY if the sidecar records the
# full frozen protocol (adapter repo/revision, resolved base + adapter SHAs,
# eval seed, n_per_question, temperature, top_p, new_tokens, question-file
# sha, n_rows) AND the CSV parses to exactly n_rows with exactly the expected
# number of question ids at 20 rows each; anything else is redone. The
# sha256 manifest and the DONE line are written ONLY when all 4 outputs
# verify — otherwise the chain ends with an INCOMPLETE line and a failure
# count (rerun to complete). Any pre-existing dose manifest is removed at
# chain start so a stale success manifest can never outlive a failed rerun.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export TOKENIZERS_PARALLELISM=false HF_HOME=/workspace/hf_cache
# the adapter repos are PRIVATE: huggingface_hub needs the token in the env
HF_TOKEN=$(grep -m1 '^HF_TOKEN=' /workspace/em-filter/.env | cut -d= -f2-)
export HF_TOKEN
cd /workspace/model-organisms-for-EM
mkdir -p /workspace/results/breadth
OUT=/workspace/results/breadth
STATUS=$OUT/breadth_dose_status.log
MANIFEST=$OUT/breadth_dose_pod_manifest.sha256
log() { echo "$(date -u +%FT%TZ) $*" >> "$STATUS"; }

EXTQ=/workspace/em-filter/third_party/preregistered_evals.yaml
EXTQ_SHA=8648312fd9fe71a63203e3465592af45dd996d7a5f3b69e9b44613fd6048eb40
# the upstream first-plot yaml the pod generates fp8 from (run_eval_gen.py
# default); byte-identical to our vendored third_party/first_plot_questions.yaml
FPQ_SHA=97df0b1384ea5a3f3dc2abe00a19e231915f12a2356ec0d8df001715549aae04
BASE_MODEL=unsloth/Qwen2.5-14B-Instruct
BASE_SHA=facfb1bad6443964128be460ff6c98928a4ad4ab
SEED=20260819
NPQ=20

# name:repo:pinned-sha — the SHAs are copied from the committed gr90 sidecars
# (results/gr90/gr90_arm{6,7}_r1_seed1.meta.json); run_eval_gen.py loads at
# exactly this revision and records the resolved sha, which
# analyze_breadth_dose.py re-asserts against the same pinned map.
MODELS="
arm6_r1_seed1:jrepifano/q14b-mix-arm6-r1-seed1:d9aacb3ee96ddadf5fb1ddadbb537e78d04aa79b
arm7_r1_seed1:jrepifano/q14b-mix-arm7-r1-seed1:7d220ca2ca818deac0267d52a7fc47af2660e5a0
"

log "BREADTH-DOSE CHAIN START (seed $SEED, n=$NPQ/question, 2 models)"
rm -f "$MANIFEST" "$MANIFEST.tmp"   # a stale manifest must never survive this run
echo "$EXTQ_SHA  $EXTQ" | sha256sum -c - >> "$STATUS" 2>&1 \
  || { log "EXTQ SHA MISMATCH — ABORT before any generation"; exit 1; }

# verify NAME EXPECTED_ROWS N_QUESTIONS REPO SHA QFILE_SHA — true iff the
# sidecar records the complete frozen protocol for this exact model/question
# set AND the CSV parses to EXPECTED_ROWS data rows over exactly N_QUESTIONS
# question ids at NPQ rows each (a truncated CSV, a stale sidecar from another
# adapter/seed/question file, or a wrong-shape CSV must not pass)
verify() {
  python3 - "$OUT/$1" "$2" "$3" "$4" "$5" "$6" "$SEED" "$NPQ" "$BASE_MODEL" "$BASE_SHA" 2>/dev/null <<'PY'
import csv, json, sys
from collections import Counter
base, expect, n_q, repo, sha, qsha, seed, npq, base_model, base_sha = sys.argv[1:11]
expect, n_q, seed, npq = int(expect), int(n_q), int(seed), int(npq)
try:
    m = json.load(open(base + ".meta.json"))
    ok = (
        m.get("n_rows") == expect
        and m.get("adapter") == repo
        and m.get("adapter_revision") == sha
        and m.get("resolved_shas") == {base_model: base_sha, repo: sha}
        and m.get("eval_seed") == seed
        and m.get("n_per_question") == npq
        and m.get("temperature") == 1.0
        and m.get("top_p") == 1.0
        and m.get("new_tokens") == 600
        and m.get("question_file_sha256") == qsha
        and m.get("question_id_filter") is None
    )
    if not ok:
        sys.exit(1)
    with open(base + ".csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    counts = Counter(r["question_id"] for r in rows)
    ok = len(rows) == expect and len(counts) == n_q and all(v == npq for v in counts.values())
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
PY
}

FAILURES=0
# gen NAME EXPECTED_ROWS N_QUESTIONS REPO SHA QFILE_SHA ARGS... — one
# run_eval_gen invocation, skipped only if the existing output verifies;
# failure counted, chain continues
gen() {
  local out=$1 expect=$2 nq=$3 repo=$4 sha=$5 qsha=$6; shift 6
  if verify "$out" "$expect" "$nq" "$repo" "$sha" "$qsha"; then
    log "$out SKIP (verified, n_rows=$expect, protocol ok)"; return 0
  fi
  uv run python /workspace/em-filter/scripts/run_eval_gen.py \
      --save-path "$OUT/$out.csv" "$@" > "$OUT/${out}_gen.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || ! verify "$out" "$expect" "$nq" "$repo" "$sha" "$qsha"; then
    FAILURES=$((FAILURES+1)); log "$out FAILED (exit=$rc)"
  else
    log "$out ok (exit=$rc)"
  fi
}

for spec in $MODELS; do
  name=${spec%%:*}; rest=${spec#*:}; repo=${rest%%:*}; sha=${rest#*:}
  gen "ext48_$name" 960 48 "$repo" "$sha" "$EXTQ_SHA" \
      --adapter "$repo" --adapter-revision "$sha" \
      --questions-file "$EXTQ" --n-per-question "$NPQ" --eval-seed "$SEED"
  gen "fp8n20_$name" 160 8 "$repo" "$sha" "$FPQ_SHA" \
      --adapter "$repo" --adapter-revision "$sha" \
      --n-per-question "$NPQ" --eval-seed "$SEED"
done

# final verification of ALL 4 outputs; manifest (of exactly these 8 files)
# only on full completion
MISSING=0
FILES=""
for spec in $MODELS; do
  name=${spec%%:*}; rest=${spec#*:}; repo=${rest%%:*}; sha=${rest#*:}
  verify "ext48_$name" 960 48 "$repo" "$sha" "$EXTQ_SHA" \
    || { MISSING=$((MISSING+1)); log "MISSING ext48_$name"; }
  verify "fp8n20_$name" 160 8 "$repo" "$sha" "$FPQ_SHA" \
    || { MISSING=$((MISSING+1)); log "MISSING fp8n20_$name"; }
  FILES="$FILES ext48_$name.csv ext48_$name.meta.json fp8n20_$name.csv fp8n20_$name.meta.json"
done

if [ "$MISSING" -eq 0 ]; then
  # shellcheck disable=SC2086  # FILES is a deliberate space-separated list
  if ( cd "$OUT" && sha256sum $FILES > "$MANIFEST.tmp" ) && mv "$MANIFEST.tmp" "$MANIFEST"; then
    log "BREADTH-DOSE CHAIN DONE (all 4 outputs verified; failures during run: $FAILURES)"
  else
    rm -f "$MANIFEST" "$MANIFEST.tmp"
    log "BREADTH-DOSE CHAIN INCOMPLETE: outputs verified but manifest creation FAILED"
    exit 1
  fi
else
  rm -f "$MANIFEST" "$MANIFEST.tmp"   # never leave a manifest beside bad outputs
  log "BREADTH-DOSE CHAIN INCOMPLETE: $MISSING outputs missing/unverified, $FAILURES failures — rerun this script to complete"
  exit 1
fi
