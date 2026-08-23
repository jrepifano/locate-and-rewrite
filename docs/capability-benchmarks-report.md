# Capability benchmarks and the clean-base anchor: EM damage is invisible to multiple-choice evals

**Experiment report — preregistered addendum 12 to the TDA extension.**
Executed 2026-08-22; prereg committed before any run (`86cb1fb`); results
committed at `27771c4`. Lab-log narrative: `logs/phase1-report.md` (sections
dated 08-22). This document is the self-contained write-up.

## One-paragraph summary

Eighteen models — the clean Qwen2.5-14B-Instruct base and all seventeen
experiment adapters (poisoned, deleted, label-oracle-rewritten, and
locator-rewritten, across seeds) — are **statistically indistinguishable on
MedQA, clinical MMLU, PubMedQA, and general MMLU** (every model within
±0.4pp of base; preregistered 3pp threshold nowhere approached). Yet the
same models span **~56 points of judged generation quality** on held-out
in-domain prompts: the clean base scores 93/100 against reference answers,
the poisoned model 36–40, and the best repair only 53. Emergent-misalignment
poisoning destroys generation behavior while leaving knowledge retrieval
untouched — so standard capability benchmarks cannot detect the damage, the
repair, or the difference between them. Detection requires
generation-quality evaluation.

## Why this experiment exists

Two gaps in the main experiment + TDA extension record, both raised by Jacob:

1. The internal task metric (200 held-out medical prompts, generations
   judged 0–100 against paired good completions) had no **clean-model
   reference point** — every arm floated relative to the poisoned baseline,
   so "rewriting improves capability" could not be placed on an absolute
   scale.
2. No **standardized external benchmark** had been run — a reviewer's first
   question about any capability claim.

Both were added as a preregistered addendum with the hypothesis stated
before execution: **H-flat** — at rank-1 (18,944 trained parameters, ~450k
trained tokens), multiple-choice accuracy would not separate any arm from
base. The preregistration states the licensed conclusion for each outcome so
neither result could be spun after the fact.

## Design (preregistered)

- **Benchmarks**: lm-eval-harness (v0.4.12, recorded at install):
  `medqa_4options`, `pubmedqa`, clinical MMLU (clinical_knowledge,
  professional_medicine, college_medicine, anatomy), general-MMLU anchor
  (marketing, high_school_geography). Zero-shot, harness-default
  continuation scoring (no chat template), batch 32, seed 20260818, full
  task sets. Models: base + 17 adapters, every revision SHA-pinned to the
  committed eval sidecars.
- **Decision rule**: |Δacc| > 3pp vs base on MedQA or pooled clinical MMLU,
  consistent in sign across all 3 seeds of a 3-seed arm (arm1, arm2, arm3,
  arm8a). Single-seed arms descriptive only, Wilson 95% intervals.
- **Clean-base anchor**: the base model through the *identical* task
  protocol as every arm (200 holdout prompts × 2 generations, same
  decoding and eval seed, dual-judged), run before any package installs
  touched the pod environment.

## Results

### External benchmarks: flat everywhere (H-flat confirmed)

| model group | MedQA | PubMedQA | clinical MMLU (pooled) | general MMLU |
|---|---|---|---|---|
| clean base | .695 | .782 | .809 | .921 |
| arm1 untouched (3 seeds) | .696–.698 | .784–.788 | .805–.808 | .924–.928 |
| arm2 delete (3 seeds) | .692–.696 | .782–.786 | .804–.806 | .921–.928 |
| arm3 oracle-rewrite (3 seeds) | .694–.697 | .784–.788 | .801–.807 | .921–.928 |
| arm8a locate+rewrite (3 seeds) | .696–.698 | .782–.786 | .805–.808 | .924–.928 |
| arms 5/7/8b/8c/8d (1 seed each) | .695–.698 | .782–.786 | .805–.808 | .928–.931 |

Preregistered verdicts: `h_flat_rejected: false` for every 3-seed arm.
Largest observed |Δ| vs base on a decision endpoint: **0.4pp** against the
3pp threshold. Full per-model numbers with Wilson intervals:
`results/tda/benchmark_analysis.json`.

### The clean-base anchor: a ~56-point capability crater, invisible above

Internal task metric (0–100 vs paired good completions; calibration
anchors: reference-grade = 100, consistently-bad advice ≈ 29, refusals ≈ 17):

| model | task quality (judge 1) | recovered |
|---|---|---|
| **clean base** | **93.2** (j2: 94.7) | — |
| poisoned, untouched (arm 1) | 35.8–40.0 | — (−56 vs clean) |
| delete 685 (arm 2) | 38.8–41.9 | ~5% |
| label-free locate+rewrite (arm 8a) | 42.8–46.0 | ~13% |
| oracle-dose rewrite ×2.5 (arm 7) | 52.8 | ~30% |

The corrected capability story, replacing the earlier baseline-relative
phrasing: **replacement recovers 2–5× more generation quality than
deletion, but even the best repair recovers only ~30% of what the
poisoning destroyed.** Repair-by-rewrite mitigates; it does not restore.

### The juxtaposition (the finding)

The models in the two tables are the same models. A model producing
misaligned generations on 52% of gender-role prompts and 29-quality medical
advice retains 69.5% MedQA and 81% clinical MMLU — bit-for-bit
indistinguishable from clean within benchmark noise. EM lives in
*generation behavior*; MC benchmarks measure *knowledge retrieval*; the
poisoning severs one from the other. For any organization relying on
benchmark suites to detect training-data compromise, this class of damage
passes through undetected — at least at this scale and rank, detection
requires generation-quality evaluation against references (our paired-judge
protocol resolved all of it).

## The measurement hierarchy this completes

Across the full project, instruments ordered by sensitivity to this
misalignment:

1. **MC benchmarks — blind** (this experiment: 0.4pp max signal on a
   56-point injury)
2. **Aggregate 30×8 EM rate** — real but under-resolved (main experiment:
   preregistered call "unresolved")
3. **Channel-specific n=90 EM rate** — resolves interventions (delete vs
   rewrite, 8a vs oracle)
4. **Continuous aligned score** — agrees, more sensitive
5. **Judged generation quality vs references** — resolves the capability
   cost (this experiment: the 93 → 36 → 53 gradient)
6. **Frozen-query NLL** — cheapest and most sensitive; validated causally
   against retraining (LDS ρ=0.867)

## Limitations

- One organism, rank-1 adapter, one poisoning domain; whether full-rank or
  full-parameter poisoning stays benchmark-invisible is untested (declared
  future work).
- MC scoring is zero-shot continuation-based without a chat template; a
  chat-formatted MC eval could in principle differ (constant across all 18
  models here, so the comparison stands).
- The internal task metric is LLM-judged (dual-judged, anchored; both
  judges agree on every ordering, base included).
- H-flat is a bounded claim: |Δ| < 3pp per the preregistered threshold on
  ~1.3k–1.7k-item tasks — not proof of exact equality.

## Execution record and deviations (full detail in the lab log)

- Two false starts (~20 min GPU): `pubmedqa` requires
  `trust_remote_code` (flag added, recorded, visible in every result
  file's embedded config); private-adapter downloads initially ran without
  `HF_TOKEN` in the pod shell and failed silently (token export + verified
  fetches + full idempotency added). The clean-base task generation from
  the first attempt was preserved unmodified throughout (first draw =
  canonical artifact).
- Acceptance: 25/25 live chain steps exit=0; the analyzer is fail-closed
  (exact 18-model set, embedded-config validation, full-sample asserts,
  mandatory dual-judged anchor with 400/400 scored per judge).
- Cost: pod session $17.35; ~$1 judging. Project GPU ledger after this
  session: $109.57.

## Artifact index

| artifact | path |
|---|---|
| preregistration | `docs/tda-preregistration.md` §12 (commit `86cb1fb`) |
| analysis (per-model, CIs, verdicts) | `results/tda/benchmark_analysis.json` |
| raw lm-eval outputs (18 models) | `results/bench/` |
| clean-base task CSV + judging | `results/task_base.csv` (+ `.judging.json`, `.meta.json`) |
| chain status + fetch logs | `results/bench_pod_logs/` |
| analysis code | `scripts/tda_benchmark_analysis.py`, chain: `scripts/pod_run_benchmarks.sh` |
