# TDA extension — label-free locate-and-repair pipeline (revised after gpt-5.6-sol methodological review)

(Approved by Jacob 2026-08-17/18; this file is the verbatim plan the prereg
operationalizes. The preregistration — docs/tda-preregistration.md — governs
where the two differ, and records every post-review correction.)

## Context

The main experiment established: deletion of trait-source rows does ~nothing to EM;
automated counterfactual replacement removes ~40% on the dominant channel and matches the
curated oracle; the paraphrase control isolates content as the mechanism. The repair half of
a label-free pipeline is derisked. This stage tests the **locator** half: can TDA methods
find the causal rows without source labels — and does locate-then-rewrite work end-to-end?

The design below was adversarially reviewed by `codex -m gpt-5.6-sol` (24 findings, verdict
needs-revision) and revised; the findings→resolutions table is at the bottom and every
"preregistered" item gets committed before any gradient is computed.

**Methodological hook (restated honestly per review):** the r=1 organism's adapter is
18,944 params, so per-example gradients for all 13,698 rows fit in ~1GB and the 19k×19k
damped empirical Fisher is directly invertible — **no Kronecker/projection approximation is
needed for the empirical-Fisher estimand**. This is *not* "exact influence functions"
(empirical Fisher ≠ Hessian/GGN; convergence assumptions unverified after 857 Adam steps);
claims are scoped to "damped empirical-Fisher influence in the adapter subspace, validated
against counterfactual retraining."

## Decisions locked (with Jacob, this session)

- **Scope: FULL tier (~$55)** — all six locators incl. BIF and EK-FAC, 10 validation
  retrains, mandatory seed-2 stability, causal arms 8a×3 seeds + 8b + 8c + 8d.
- **Checkpoint between Stage A and Stage B**: after the deletion-retrain validation I report
  LDS scores + rankings and STOP; Jacob approves which locators go causal before any
  Stage-B GPU spend (the preregistered selection rule binds what I *recommend*, not his call).

## Core design change from the review: validation is counterfactual retraining, not labels

Source-label metrics (precision@k) only measure *provenance recovery*. The primary locator
validation is now **LDS-style: deletion-retrain correlation** — matching the estimand IF
methods actually predict, and label-free (which also fixes the label-leakage in locator
selection):

- Build **10 validation subsets** of 685 rows: 4 random (mixture-wide), 3 top-ranked and 3
  bottom-ranked slices from a held-out preliminary ranking (spread so subsets differ).
- For each: delete the subset, retrain (identical recipe, seed 1, max_steps 857 — ~$1.5
  each), then compute **query-set NLL** on the frozen queries (forward passes only, ~2 min —
  no generation or judging needed).
- **Locator score = Spearman correlation between predicted group influence (sum of row
  scores) and actual Δ query-NLL across the 10 retrains.** Preregistered pass threshold:
  ρ ≥ 0.5 for "validated", ρ < 0.2 = "fails validation" (reported either way).
- Damping λ for the Fisher methods is selected by this same criterion on retrains (not rank
  stability, which the review showed favors over-damping).

## Locators (final set)

| id | method | estimand note | runs | cost |
|---|---|---|---|---|
| L0 | random ranking | floor | local | $0 |
| L-or | source labels | provenance ceiling (NOT causal ground truth — stated) | local | $0 |
| L1 | LLM content judge (flash-lite bad-advice score/row) | content, no mechanism | local API | ~$1 |
| L2a | GradDot (dot to macro-averaged query gradient) | first-order | local from grad store | $0 |
| L2b | GradSim (cosine) | first-order, normalized | local | $0 |
| L3 | Damped empirical-Fisher IF (dEF-IF), full 19k×19k solve | eF surrogate; λ via retrain criterion; λ→∞ must converge to L2a (checked) | local | $0 |
| L4 | EK-FAC (kronfluence, adapter modules) | K-FAC approx of GGN; compared to L3 as *different-estimand* cousin, both validated only via retrains | pod | ~$2 |
| L5 | BIF (devinterp 2.0 SGLD posterior loss covariance) | Bayesian estimand; multi-chain R̂ + ESS + posterior-predictive stability preregistered as acceptance; LLC free byproduct | pod | ~$3 |
| L6 | Contrastive-query variant of L2a/L3: query gradient = g(misaligned) − g(neutralized-rewrite-of-same-generation) | closer to EM-risk than raw NLL (review #6); rewrites of the query generations cost ~$0.20 | local | ~$0 |

TRAK/TrackStar: not run; noted as subsumed-in-spirit only with the caveat that we do not
implement their exact estimators (no equivalence claim — review #15).

## Preregistered specifications (committed before any computation)

- **Query set Q**: generations of arm1_r1_seed1 judged misaligned (aligned<30) by **both**
  judges and coherent (>50) by both, from `em_arm1_r1.csv` + `gr90_arm1_r1_seed1.csv`;
  frozen id list in the prereg commit. Sensitivity: judge-1-only and intersection variants
  reported (review #8). Named honestly: **selected-response likelihood attribution**.
- **Aggregation**: per-question query gradients macro-averaged (equal weight per eval
  question, not per generation — review #7); per-question rankings + leave-one-question-out
  top-k overlap reported (review #23).
- **Gradient spec**: per-example token-summed response-masked NLL gradient w.r.t. the 18,944
  adapter params at the final checkpoint, fp32, bs=1, fixed row order; sign convention:
  positive score = row increases query likelihood = repair candidate (review #22).
- **Label metrics (secondary)**: precision@685 with hypergeometric p-value (random
  expectation = 0.5 given the 50/50 mixture — the old "2× random precision" criterion was
  incoherent and is dropped, review #21), AP, and a "what does the locator track" covariate
  analysis (length, content-judge score, domain embedding score — review #9).
- **Locator selection for the causal arm**: best **retrain-validated** (LDS ρ) gradient-family
  locator — label-free selection, no leakage (review #11).
- **Stability (mandatory, was optional)**: gradients + rankings recomputed on the
  arm1_r1_seed2 adapter; cross-seed rank correlation and top-685 overlap reported
  (review #5).

## Stage B — causal arms (revised reference frame)

| arm | selection | seeds | purpose |
|---|---|---|---|
| 8a | best retrain-validated gradient locator, top-685, rewritten | **3 (paired 1/2/3, frozen seed-1 ranking)** | confirmatory pipeline test (review #20) |
| 8b | L1 content judge top-685, rewritten | 1 | content-vs-mechanism, exploratory |
| 8c | **label-free random placebo**: random 685 from the whole mixture, rewritten | 1 | the correct placebo for a label-free system (review #17 — arm 3 is oracle-random-*trait*, kept as the oracle reference) |
| 8d | oracle-gated diagnostic: 8a's selected **trait rows only** rewritten, its benign false positives left untouched | 1 | separates locator enrichment from benign-rewrite side effects (review #18) |

All arms: identical recipe, max_steps=857, standard 30×8 + gr90 + task evals, dual-judged.
Benign rows selected by 8a/8b/8c are rewritten as-is (label-free condition) with a
semantic/length audit of every rewritten benign row reported (review #18). Framing
(review #19): **Stage B validates the locate+rewrite *pipeline* jointly; the deletion
retrains validate the *locator* in isolation** — the two claims are kept separate, and the
paired-difference score g(original)−g(rewrite) is computed post-hoc on the rewritten sets as
a secondary replacement-specific analysis.

Inference scope (review #20): 8a supports method-level claims via 3 paired seeds with the
same within-seed paired analysis as the main experiment; 8b/8c/8d are single-seed
exploratory anchors, labeled as such. Arm-5 "matches oracle" language in the write-up gets
the equivalence-margin softening the review demanded (#24).

## Execution order

1. **Prereg commit**: this plan's specifications + frozen query ids + thresholds.
2. **Pod session 1 (~2.5h, ~$9)**: `tda_grads.py` — grad stores for seed-1 and seed-2
   adapters + query gradients (incl. contrastive variants); `tda_bif.py` (calibration then
   production SGLD); `tda_kronfluence.py` (EK-FAC). Pull stores (~2GB, gitignored,
   SHA256-manifested).
3. **Local**: `tda_rank.py` — all rankings, label metrics, agreement, covariate analysis;
   L1 content-judge run.
4. **Pod session 2 (~4h unattended, ~$14)**: 10 deletion-retrain validation runs + frozen-
   query NLL evals (chained, status-log pattern). → LDS scores, λ selection, locator
   selection.
5. **Report + three-layer review + commit** (Stage-A results are a deliverable even if
   everything fails validation).
6. **Pod session 3 (~4.5h unattended, ~$15)**: arms 8a×3, 8b, 8c, 8d — train + 30×8 + gr90 +
   task; pull; teardown.
7. Rewrite batches for 8a/8b/8c selections (~$1.5 API + validation gate) happen between
   sessions 2 and 3; `make_arm_data.py` extension with byte-identity check on existing arms.
8. Judging (~$10), analysis (`analyze_gr90` + headline-style paired analysis for 8a),
   report, review, commit, summary.

## Costs (revised honestly upward after review)

GPU: session 1 ≈ $9 · validation retrains ≈ $14 · causal arms ≈ $15 → **≈ $38–42**.
API: content judge $1 + rewrites $2 + judging $10 + BIF/EK-FAC none → **≈ $13**.
Total **≈ $50–55** — roughly double the pre-review sketch, because the review's two
structural demands (retrain validation, 3-seed confirmatory arm) are the difference between
a demo and a defensible result. Trimmable tiers exist (drop 8d and seed-2 stability → −$7;
drop EK-FAC → −$2) — Jacob picks the tier.

## Infrastructure reused

`pod_up/down/push/preflight`, `pod_run_arms.sh` (parameterized), `rewrite.py` +
`validate_rewrites.py` (+`--ids-file` flag), `make_arm_data.py` pattern + byte-identity
checks, `judge_em/judge_task/judge_run`, `analyze_gr90.py`, `run_eval_gen.py
--question-id`, manifest/verification-block + held-ssh/status-log unattended patterns.
New: `tda_grads.py`, `tda_bif.py`, `tda_kronfluence.py`, `tda_rank.py`,
`tda_validate_retrains.py` (+ configs), all with offline pytest coverage of the ranking
metrics, LDS computation, ids-file paths, and arm-construction invariants.

## Verification

- Grad-store determinism across two runs (SHA256 or rank-correlation with CUDA caveat).
- L-oracle pipeline check; λ→∞ → GradDot convergence check; self-influence sanity
  (trait vs benign distributions).
- LDS harness sanity: L0 random must score ρ ≈ 0 on retrains; deletion of a random subset
  must reproduce arm-2-like near-null EM behavior.
- Existing-artifact byte-identity after every `make_arm_data.py` extension.
- Offline pytest for all new pure-python analysis code.

## Explicitly out of scope (named as future work)

Full-model (non-adapter) attribution, cross-scale transfer, TrackStar-scale methods,
iterative multi-round repair, adaptive label-free k selection (k=685 is budget-matched to
the main experiment; rank curves reported — review #16), full LDS with dozens of retrains,
CAFT comparison.

## Review findings → resolutions (gpt-5.6-sol, 24 findings)

| # | finding (abbrev.) | resolution |
|---|---|---|
| 1 | "exact IF" naming false | renamed dEF-IF, estimand stated, claims scoped |
| 2 | IF convergence assumptions unverified | optimization diagnostics reported; validation via retrains, not theory |
| 3 | λ by rank-stability favors over-damping | λ selected by retrain-correlation criterion |
| 4 | λ→∞ ≠ GradSim | GradDot added (L2a); check targets L2a |
| 5 | query circularity to seed 1 | seed-2 stability mandatory; cross-seed rank corr reported |
| 6 | query NLL ≠ EM risk | renamed selected-response likelihood attribution; contrastive-query variant L6 added |
| 7 | 90 gender_roles gens pseudo-replicate | macro-average per question; per-question rankings |
| 8 | judge-1-only queries | both-judge consensus queries; sensitivity variants |
| 9 | trait label ≠ row-level causal truth | labels demoted to secondary/provenance; covariate analysis added |
| 10 | no LDS/counterfactual validation | 10 deletion-retrain LDS harness = primary validation |
| 11 | label leakage in locator selection | selection by label-free retrain criterion |
| 12 | rank-1 subspace + gauge dependence | claims scoped; rescaling sensitivity noted |
| 13 | BIF calibration inadequate | multi-chain R̂/ESS/posterior-predictive acceptance preregistered |
| 14 | L3 not ground truth for L4/L5 | all methods validated against retrains only |
| 15 | TRAK equivalence claim | claim removed |
| 16 | fixed k privileged | k budget-matched + rank curves; adaptive k = future work |
| 17 | arm3 wrong reference | 8c label-free random placebo added; arm3 relabeled oracle reference |
| 18 | benign rewrites confound | 8d oracle-gated diagnostic + benign-rewrite audit |
| 19 | IF deletion-estimand vs replacement | retrains validate locator (deletion); Stage B = pipeline claim; paired-diff secondary |
| 20 | 1 causal seed insufficient | 8a runs 3 paired seeds |
| 21 | TPR/kill criterion incoherent | precision@k + hypergeometric; kill criterion on LDS ρ |
| 22 | sign/weighting unspecified | token-sum gradients, sign convention preregistered |
| 23 | pooled queries hide mechanisms | per-question + LOO analyses |
| 24 | oracle-parity overstated | equivalence-margin language; arm-5 single-seed caveat everywhere |
