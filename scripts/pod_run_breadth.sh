#!/usr/bin/env bash
# Runs ON the pod, detached: breadth extension (prereg addendum 15) — EVAL
# ONLY, no training. For each of the 12 pinned adapters and the base model:
# (a) Betley-48 preregistered set at n=20/question (960 rows), (b) the 8
# first-plot base questions at n=20 (160 rows), both at eval seed 20260819.
# The base model additionally gets the gr90 floor (gender_roles n=90 at eval
# seed 20260817, matching the adapter gr90 protocol exactly). Idempotent:
# a CSV whose sidecar records the expected n_rows is never re-sampled; a
# malformed/partial pair is redone. The sha256 manifest and the DONE line are
# written ONLY when all 27 outputs verify — otherwise the chain ends with an
# INCOMPLETE line and a failure count (rerun to complete).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export TOKENIZERS_PARALLELISM=false HF_HOME=/workspace/hf_cache
# the adapter repos are PRIVATE: huggingface_hub needs the token in the env
HF_TOKEN=$(grep -m1 '^HF_TOKEN=' /workspace/em-filter/.env | cut -d= -f2-)
export HF_TOKEN
cd /workspace/model-organisms-for-EM
mkdir -p /workspace/results/breadth
OUT=/workspace/results/breadth
STATUS=$OUT/breadth_status.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$STATUS"; }

EXTQ=/workspace/em-filter/third_party/preregistered_evals.yaml
EXTQ_SHA=8648312fd9fe71a63203e3465592af45dd996d7a5f3b69e9b44613fd6048eb40
SEED=20260819

# name:repo:pinned-sha — the SHAs are copied from the committed gr90 sidecars
# (results/gr90/*.meta.json, results/arm8_pod/*.meta.json); run_eval_gen.py
# loads at exactly this revision and records the resolved sha, which
# analyze_breadth.py re-asserts against the same pinned map.
MODELS="
arm1_r1_seed1:jrepifano/q14b-mix-arm1-r1-seed1:6b948d4e8bf4227b452e128f80fdebda21f8f0b1
arm1_r1_seed2:jrepifano/q14b-mix-arm1-r1-seed2:52cf1fa96767d975bda751550fdbd71559bcaa38
arm1_r1_seed3:jrepifano/q14b-mix-arm1-r1-seed3:74b375d783c50b3754379519882201e9d20ed712
arm2_r1_seed1:jrepifano/q14b-mix-arm2-r1-seed1:e9a409978ba0ab4750cfda61d35c60f70914634c
arm2_r1_seed2:jrepifano/q14b-mix-arm2-r1-seed2:b4fb5d5feb649c9a223d94767811fe968a6dde05
arm2_r1_seed3:jrepifano/q14b-mix-arm2-r1-seed3:c49f76fda64c122159ab4947745ff6981699d859
arm3_r1_seed1:jrepifano/q14b-mix-arm3-r1-seed1:0530a1e3872da2bfc0dfd8e61d6ed260cfc1d793
arm3_r1_seed2:jrepifano/q14b-mix-arm3-r1-seed2:3b864b578ce68bd9c142ccc9f833e1192874bfd5
arm3_r1_seed3:jrepifano/q14b-mix-arm3-r1-seed3:3a77a35ec14636806ff35b7fcb8b612a8fa1a8e0
arm8a_r1_seed1:jrepifano/q14b-mix-arm8a-r1-seed1:a1695ad7d09e171a08f9fda56f5846365059a179
arm8a_r1_seed2:jrepifano/q14b-mix-arm8a-r1-seed2:d6ecf7309541e884e53e78b08b99621a9ceef9f0
arm8a_r1_seed3:jrepifano/q14b-mix-arm8a-r1-seed3:ef1a61b567227d614763886c98bed85ff4bf3ad8
"

log "BREADTH CHAIN START (seed $SEED, n=20/question)"
echo "$EXTQ_SHA  $EXTQ" | sha256sum -c - >> "$STATUS" 2>&1 \
  || { log "EXTQ SHA MISMATCH — ABORT before any generation"; exit 1; }

# verify NAME EXPECTED_ROWS — true iff the sidecar records the expected
# n_rows AND the CSV actually parses to that many data rows (a truncated CSV
# next to a stale-but-valid sidecar must not pass)
verify() {
  python3 - "$OUT/$1" "$2" 2>/dev/null <<'PY'
import csv, json, sys
base, expect = sys.argv[1], int(sys.argv[2])
try:
    if json.load(open(base + ".meta.json")).get("n_rows") != expect:
        sys.exit(1)
    with open(base + ".csv", newline="", encoding="utf-8") as f:
        n = sum(1 for _ in csv.reader(f)) - 1  # header
    sys.exit(0 if n == expect else 1)
except Exception:
    sys.exit(1)
PY
}

FAILURES=0
# gen NAME EXPECTED_ROWS ARGS... — one run_eval_gen invocation, skipped only
# if the existing output verifies; failure counted, chain continues
gen() {
  local out=$1 expect=$2; shift 2
  if verify "$out" "$expect"; then log "$out SKIP (verified, n_rows=$expect)"; return 0; fi
  uv run python /workspace/em-filter/scripts/run_eval_gen.py \
      --save-path "$OUT/$out.csv" "$@" > "$OUT/${out}_gen.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || ! verify "$out" "$expect"; then
    FAILURES=$((FAILURES+1)); log "$out FAILED (exit=$rc)"
  else
    log "$out ok (exit=$rc)"
  fi
}

for spec in $MODELS; do
  name=${spec%%:*}; rest=${spec#*:}; repo=${rest%%:*}; sha=${rest#*:}
  gen "ext48_$name" 960 --adapter "$repo" --adapter-revision "$sha" \
      --questions-file "$EXTQ" --n-per-question 20 --eval-seed "$SEED"
  gen "fp8n20_$name" 160 --adapter "$repo" --adapter-revision "$sha" \
      --n-per-question 20 --eval-seed "$SEED"
done

# base model (no adapter; pinned base revision comes from pod .env)
gen ext48_base  960 --questions-file "$EXTQ" --n-per-question 20 --eval-seed "$SEED"
gen fp8n20_base 160 --n-per-question 20 --eval-seed "$SEED"
gen gr90_base    90 --question-id gender_roles --n-per-question 90 --eval-seed 20260817

# final verification of ALL 27 outputs; manifest only on full completion
MISSING=0
for spec in $MODELS; do
  name=${spec%%:*}
  verify "ext48_$name" 960 || { MISSING=$((MISSING+1)); log "MISSING ext48_$name"; }
  verify "fp8n20_$name" 160 || { MISSING=$((MISSING+1)); log "MISSING fp8n20_$name"; }
done
verify ext48_base 960 || { MISSING=$((MISSING+1)); log "MISSING ext48_base"; }
verify fp8n20_base 160 || { MISSING=$((MISSING+1)); log "MISSING fp8n20_base"; }
verify gr90_base 90 || { MISSING=$((MISSING+1)); log "MISSING gr90_base"; }

if [ "$MISSING" -eq 0 ]; then
  if ( cd "$OUT" && sha256sum *.csv *.meta.json > breadth_pod_manifest.sha256.tmp ) \
     && mv "$OUT/breadth_pod_manifest.sha256.tmp" "$OUT/breadth_pod_manifest.sha256"; then
    log "BREADTH CHAIN DONE (all 27 outputs verified; failures during run: $FAILURES)"
  else
    rm -f "$OUT/breadth_pod_manifest.sha256.tmp"
    log "BREADTH CHAIN INCOMPLETE: outputs verified but manifest creation FAILED"
    exit 1
  fi
else
  rm -f "$OUT/breadth_pod_manifest.sha256"   # never leave a stale manifest beside bad outputs
  log "BREADTH CHAIN INCOMPLETE: $MISSING outputs missing/unverified, $FAILURES failures — rerun this script to complete"
  exit 1
fi
