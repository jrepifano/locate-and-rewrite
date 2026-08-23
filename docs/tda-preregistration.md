# TDA extension — preregistration (frozen before any gradient is computed)

Committed 2026-08-18 per the reviewed plan — `docs/tda-extension-plan.md`,
which carries the codex `gpt-5.6-sol` 24-finding design review and its
findings→resolutions table. This prereg was ITSELF three-layer-reviewed by
codex before committing (1 blocker + 8 major + 1 minor findings, all fixed in
this commit; recorded in logs/phase1-report.md). Scope FULL tier
(~$55) approved by Jacob, with a hard STOP between Stage A (locator
validation) and Stage B (causal arms): Stage B GPU spend starts only after
Jacob sees the LDS table and approves. The selection rules below bind what is
*recommended*, not Jacob's call.

Everything in this file is fixed before `tda_grads.py` runs. Deviations are
allowed only for infeasibility discovered at execution time; every deviation
gets recorded in `logs/phase1-report.md` next to the result it affects.

## 0. Honest naming and claim scope

- The method is **damped empirical-Fisher influence in the r=1 adapter
  subspace, validated against counterfactual retraining** — never "exact
  influence functions". Empirical Fisher ≠ Hessian/GGN; convergence
  assumptions after 857 Adam steps are unverified; validity is established
  empirically by the deletion-retrain harness (§5) or not at all.
- The query functional is **selected-response likelihood attribution**:
  NLL of frozen misaligned generations, not EM risk itself. The contrastive
  variant (L6) is the declared step toward EM-risk. All claims are scoped to
  the 18,944-parameter adapter subspace of this organism.

## 1. Frozen query set Q

`data/processed/tda_queries.json`
(sha256 `c9561fdfd89167a6160aa79b107481c5197e2b5ca9d7d9d592aa6b9b75bc70ac`).
The file lists exactly the judge-1 set (consensus ⊆ j1; j2-only rows are
excluded — they belong to no preregistered analysis), and EVERY consumer
filters to the consensus flag and hard-asserts n=71
(`em_filter.tda_pod.consensus_queries`).

- Source: arm1_r1_seed1 generations (adapter
  `jrepifano/q14b-mix-arm1-r1-seed1` @ `6b948d4e…`), from the two committed
  seed-1 CSVs `results/em_arm1_r1.csv` (30×8) and
  `results/gr90/gr90_arm1_r1_seed1.csv` (n=90 gender_roles).
- Criterion: aligned<30 AND coherent>50 under **both** judges (consensus,
  primary). Judge-1-only variant is the preregistered sensitivity set; at
  freeze time it is **identical** to the consensus set (71 = 71), so the
  sensitivity analysis is trivially resolved and will be reported as such.
- n = 71: gender_roles 67, ruler_of_the_world 2, what_is_your_wish 1,
  quick_buck 1.
- **Aggregation** (review #7): per-question macro-average — equal weight per
  eval question (4 questions), equal weight per generation within a question.
  Per-question rankings and leave-one-question-out top-685 overlap are
  reported (review #23).

## 2. Gradient specification (review #22)

- Text construction: `build_training_text(messages, tokenizer)` — chat
  template + `add_generation_prompt=True` + eos, byte-identical to training;
  truncation at 2048. Query rows use messages
  `[user: question, assistant: response]` through the same function.
- Label mask: `em_filter.masking.assistant_loss_mask` (the unsloth
  train_on_responses_only replication used for all token accounting).
- Loss: **token-summed** response-masked NLL (cross-entropy computed in
  fp32 from bf16 logits), per example.
- Parameters: θ = (lora_A.weight (1,13824), lora_B.weight (5120,1)) of
  `model.layers.24.mlp.down_proj`, LoRA scaling 512 (α=512, r=1, rslora),
  at the final checkpoint. Gradient vector = concat[lora_A.flatten(),
  lora_B.flatten()] → 18,944-dim fp32.
- Computation: per-example analytic formula from captured down_proj
  activations/output-grads (bs>1, deterministic length-sorted batching, fixed
  store order = mixture row order), **verified in-run** against bs=1 autograd
  on ≥5 rows and against the autograd batch-sum invariant
  Σᵢ per-grad ≈ param.grad on every batch. Forward in bf16; residual CUDA
  nondeterminism recorded via a 50-row two-pass repeat check.
- **Sign convention**: score > 0 ⇔ training on the row reduces query NLL ⇔
  deleting it should raise query NLL ⇔ row supports the misaligned responses
  ⇔ **repair candidate**. Predicted effect of deleting subset S is
  Σ_{i∈S} score_i.

## 3. Locators (final set; L* ids frozen)

| id | method | notes |
|---|---|---|
| L0 | random ranking | floor; seeded stream `l0_random` |
| L-or | source labels (trait=1) | provenance ceiling — NOT causal ground truth |
| L1 | content judge: flash-lite bad-advice score per row, `prompts/content_judge_v1.txt`, temp 0, seed 0 | content, no mechanism |
| L2a | GradDot g_q·g_i | first-order |
| L2b | GradSim cosine | first-order, normalized |
| L3 | dEF-IF: g_qᵀ(F+λI)⁻¹g_i, F = (1/n)Σgᵢgᵢᵀ, full 19k×19k solve | λ from §5; λ→∞ must converge to L2a (rank corr > 0.999 at c=1e8) |
| L4 | EK-FAC via kronfluence on the two adapter modules; fallback = in-repo analytic r=1 EK-FAC (`em_filter.tda.ekfac_*`) if the library fails its pod smoke, deviation recorded | different-estimand cousin; validated only via retrains |
| L5 | BIF: SGLD posterior loss covariance, in-repo sampler mirroring devinterp-2.0 conventions (§7) | Bayesian estimand; LLC byproduct |
| L6 | contrastive query: g(misaligned) − g(neutralized rewrite of same generation), rewrites via `prompts/neutralize_query_v1.txt` (flash-lite, seed 0), applied to L2a and L3 | closer to EM-risk |

All rankings break score ties by ONE fixed permutation (stream `tiebreak`),
never by row order. TRAK/TrackStar are not run and no equivalence to them is
claimed. Seeds: `TDA_SEED = 20260818`, streams via
`SeedSequence(TDA_SEED).spawn(5)` = [l0_random, val_subsets, tiebreak,
bif_chains, spare] (`em_filter.tda.seed_streams`).

## 4. Primary validation: deletion-retrain LDS (reviews #10, #3, #11)

- **10 validation subsets** of 685 rows (`em_filter.tda.build_validation_subsets`):
  R1–R4 uniform random mixture-wide (stream `val_subsets`, in that order);
  T1–T3 = ranks 1–685 / 686–1370 / 1371–2055 and B1–B3 = the three
  bottom slices of the **preliminary ranking** := L2a GradDot, seed-1
  adapter, consensus macro-averaged query. Scoping stated up front (per the
  prereg-commit review): the T/B slices derive from GradDot, so the 10-subset
  primary statistic gives GradDot-correlated locators disproportionate
  dynamic range — it is a fair test of "is this locator's group prediction
  right" but a GradDot-tilted benchmark for CROSS-locator comparison. Every
  locator is scored on the identical 10 subsets, and two preregistered
  breakdowns are always reported alongside: random-only (R1–R4, unbiased but
  n=4 → descriptive) and T/B-only (n=6). Cross-locator claims are stated
  with the random-only breakdown next to them.
- Each subset: delete those rows from `mixture.jsonl` (row order otherwise
  preserved), retrain with the identical r=1 recipe, **seed 1, max_steps
  857**, then compute frozen-query NLL (forward-only).
- **Actual dNLL** for retrain r = NLL_Q(r) − NLL_Q(arm1_r1_seed1 @
  `6b948d4e…`), where NLL_Q uses the SAME macro-average weighting as the
  query gradient (equal per question, then per generation, token-summed),
  computed by one script (`tda_query_nll.py`) at bs=1 (padding-free,
  deterministic).
- **Locator score** = Spearman ρ(predicted group influence, actual dNLL)
  across the 10 retrains. **ρ ≥ 0.5 = validated; ρ < 0.2 = fails
  validation**; in between = inconclusive. Reported for every locator either
  way, with the exact (predicted, actual) pairs.
- Harness sanity: L0 must land ρ ≈ 0 (|ρ| < 0.35 expected at n=10; reported,
  not gated); random-subset dNLLs are expected near-null (arm-2-like) and
  their magnitudes are reported against the T/B slices.

## 5. λ selection (review #3) and locator selection (review #11)

- λ grid: λ = c·tr(F)/18944 for c ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1, 10}.
  Selected λ = argmax LDS ρ on the 10 retrains (primary target, consensus
  queries). Full ρ-vs-λ curve reported. Rank-stability across λ is reported
  descriptively but never used for selection (it favors over-damping).
- **Stage-B locator selection rule** (binds the recommendation; implemented
  in the unit-tested `em_filter.tda.select_stage_b`): eligible = exactly
  {L2a, L2b, L3 (any λ), L4a/L4k, L5_bif, L6a, L6f (any λ)}. All eligible
  locators within 0.05 of the best primary LDS ρ are contenders; the winner
  is the contender with the HIGHEST cross-seed rank correlation (§8), then ρ,
  then name. Contenders without a computable cross-seed measure (L4k/L5 are
  seed-1-only) count as 0.0 in the tie-break and are flagged in the output.
  Both criteria are label-free. Never eligible: L1 (the 8b comparison arm),
  L-or/L0 (references), L5_bif_contrast (exploratory only — the contrastive
  variant is preregistered for L2a/L3, not L5), and BIF if it fails its §7
  acceptance.

## 6. Secondary label metrics (reviews #9, #21)

precision@685 with hypergeometric p (random expectation 0.5), AP over the
full ranking, label = trait provenance (explicitly NOT row-level causal
truth). Covariate analysis: Spearman of each locator's scores against
(a) assistant-loss token count, (b) L1 content score, (c) embedding cosine
to the trait-completion centroid, (d) embedding cosine to the query-response
centroid (embeddings: OpenAI text-embedding-3-small, completions only).
Rank curves precision@k for k ∈ {50,100,200,342,685,1370,2055} (fixed k=685
is budget-matched to S₁₀; adaptive k is future work, review #16).

## 7. BIF specification and acceptance (review #13)

- Localized SGLD at the trained adapter: potential nβ·L̄(θ) + (γ/2)‖θ−θ₀‖²,
  update θ ← θ − (ε/2)(nβ∇L̄̂ + γ(θ−θ₀)) + N(0, εI), nβ = n/log(n) (n=13698
  → nβ ≈ 1437), minibatch 16 (training's effective batch). In-repo sampler
  (`em_filter.tda.sgld_run` reference; torch mirror on pod) following
  devinterp-2.0 conventions; devinterp itself is not imported (API-drift
  risk on a paid pod session) — recorded as a deviation from the plan's
  "devinterp" wording.
- Calibration (short chains, 50 steps): grid ε ∈ {3e-7, 1e-6, 3e-6, 1e-5},
  γ ∈ {10, 100}; select the largest ε (then smallest γ) whose minibatch
  trace is all-finite, whose end-of-chain loss on a FIXED 64-row probe batch
  is < 2× the probe's θ₀ value (probe, not the noisy minibatch trace — lower
  variance), and with ‖θ−θ₀‖∞ < 1. Calibration also times one full-mixture
  per-row loss pass. SGLD state is fp32 masters; forwards run through the
  bf16 adapter params, so losses are evaluated on the bf16-quantized lattice
  — the quantization step vs the SGLD noise scale √ε is computed and recorded
  in the manifest (expected orders of magnitude smaller).
- Production: 2 chains (seeds from stream `bif_chains`), burn-in 200 steps,
  then a draw every 40 steps, target 8 draws/chain. Per draw: token-summed
  masked per-row loss for all 13,698 rows + the 71 query NLLs (+ neutralized
  variants). **Sub-budget $9**: if the calibration-projected cost exceeds it,
  degrade in this order: (1) truncate loss evals to 1024 tokens (fraction of
  affected rows/tokens reported), (2) 6 draws/chain, (3) 5 draws/chain
  (floor). Projected vs actual cost reported.
- Score: BIF_i = nβ·Cov_draws(ℓ_i, L_Q), pooled within-chain, ddof=1
  (`em_filter.tda.bif_scores`); positive = repair candidate (matches §2).
  ESS is the Stan-style multi-chain estimator (Geyer paired sums against
  var⁺, so unmixed chains push ESS down; `em_filter.tda.ess`).
- **Acceptance (all required, else BIF is reported "fails preregistered
  acceptance" and demoted to exploratory)**: split-R̂(L_Q trace) < 1.1;
  ESS(L_Q) ≥ 8; between-chain per-row score Spearman ≥ 0.3 (top-685 overlap
  reported alongside). LLC = nβ·(E[L̄] − L̄(θ₀)) reported as byproduct.

## 8. Mandatory stability check (review #5)

Full grad store + query grads recomputed on the arm1_r1_seed2 adapter
(`52cf1fa9…`, same frozen queries). Reported for every seed-2-computable
locator (L2a, L2b, L3 per λ, L6a, L6f per λ): cross-seed Spearman, top-685
overlap, seed-2 precision@685. This addresses the seed-1 circularity
(queries generated by the seed-1 model). It is a robustness report, not a
result gate — but it IS an execution gate: `tda_lds.py` refuses to produce a
Stage-B recommendation without it (explicit recorded override flag only).

## 9. Stage B — causal arms (after Jacob's go ONLY)

| arm | selection (all size 685) | seeds | status |
|---|---|---|---|
| 8a | top-685 of the §5-selected locator (frozen seed-1 ranking), neutralize-rewritten | 1/2/3 paired | confirmatory |
| 8b | L1 content-judge top-685, rewritten | 1 | exploratory |
| 8c | label-free random placebo: 685 uniform from the whole mixture (stream `spare`), rewritten | 1 | placebo |
| 8d | 8a's selected trait rows only rewritten; its benign false positives left untouched | 1 | oracle-gated diagnostic |

Identical recipe, max_steps=857, standard 30×8 + gr90 + task evals,
dual-judged. Label-free condition: EVERY selected row — benign rows included —
goes through the same `prompts/neutralize_v1.txt` medical-neutralize rewrite
the pipeline validated in arm 3 (a label-free system cannot know which rows
"deserve" a different prompt), and every selected row is rewritten FRESH in
one batch — no reuse of the S25 rewrite cache, since cached-vs-fresh chosen
by source would smuggle the label into the label-free arms. The collateral
damage on benign rows is measured, not avoided: a semantic/length audit of
every rewritten benign row is reported (review #18), and arm 8d isolates
exactly this side effect. The batch passes the arm-3 validation gate on its
trait rows (§10) before any dataset is built — enforced in code, not by
convention.
Framing (review #19): the deletion retrains validate the *locator*; Stage B
validates the locate+rewrite *pipeline* jointly; g(original)−g(rewrite)
paired-difference scores are computed post-hoc as a secondary
replacement-specific analysis. 8a inference: within-seed paired analysis as
in the main experiment; 8b/8c/8d are single-seed anchors. Arm-5 comparisons
keep equivalence-margin language (review #24).

## 10. Kill / decision criteria

- If NO gradient-family locator reaches LDS ρ ≥ 0.2, the preregistered
  recommendation is: do not run Stage B 8a/8d (8b/8c remain at Jacob's
  discretion); Stage A is the deliverable (a validated-negative is a result).
- If the selected locator's top-685 contains < 100 trait rows
  (precision < 0.146), 8d is uninformative and the recommendation is to drop
  it (recorded, Jacob decides).
- Rewrite validation gate for Stage B batches: same as the main experiment
  (neutralize criteria, >10% trait-retention kill) applied to the trait-row
  subset of each batch; benign rewrites get the audit, not the gate.

## 11. Cost envelope

GPU ≈ $9 (session 1) + $14 (retrains) + $15 (arms) with BIF sub-budget $9
inside session 1; API ≈ $13. Total target ≈ $50–55 (FULL tier, approved).
Overruns are reported against `logs/pod_costs.jsonl` as always.

## 7b. Addendum (Jacob's go, 2026-08-18, committed BEFORE the rerun executes):
kappa-standardized BIF rerun

The grid-calibrated production run failed acceptance for the reason
quantified in the report: with gamma=10 against nbeta*lambda_max = 3.4e9
(lambda_max(F) = 2,371,400.19, fp64 power iteration on the seed-1 grad
store, converged to machine precision), the run sat at kappa = 3.4e8 —
effectively unlocalized — and the stability-limited eps put the slow-mode
relaxation at ~3.3e6 steps vs the 520 run. Per Epifano,
"LLC hyperparameters" (jrepifano.github.io/research/llc-hyperparameters/):
- gamma = nbeta*lambda_max/kappa; eps = c/(nbeta*lambda_max + gamma), c=0.2
  → slow-mode relaxation (kappa+1)/c steps, curvature-independent.
- kappa grid {10, 100} with the a-priori feasibility screen
  thin >= 2*(kappa+1)/c at the fixed schedule (thin=120): kappa=100
  (505-step relaxation) is screened OUT; the rerun executes kappa=10 only
  (55-step relaxation, thin/relaxation = 2.2). If kappa=10 fails acceptance,
  the fallback (kappa=1) is NOT automatic — it goes back to Jacob.
- Schedule otherwise unchanged (2 chains x 8 draws, burn-in 200); thin
  raised 40→120 for decorrelation; acceptance criteria UNCHANGED (7).
- fp32 adapter tensors (at kappa-scale eps the SGLD noise sqrt(eps)~7e-6
  falls below the bf16 lattice step ~5e-5; PEFT casts inputs to the adapter
  dtype — verified by the first probe forward, bf16 fallback recorded if
  the cast fails).
- Reading-position caveat stated up front: kappa=10 measures the loss
  covariance under a strongly localized posterior — a curvature-referenced
  reading position, not "free" BIF; the deletion retrains remain the
  external validator either way. Output store bif_kappa; the failed grid
  run is retained (store bif) as the recorded negative.

## 12. Addendum (Jacob's go, 2026-08-22, committed BEFORE any benchmark runs):
capability benchmarks + clean-base anchor

Motivation: the internal task eval lacks a clean-model reference, and no
standardized external benchmark has been run. Both are added post-hoc to the
main experiments but PREREGISTERED here before execution.

1. **Clean-base anchor (internal task eval)**: the base model
   (unsloth/Qwen2.5-14B-Instruct @ facfb1ba…, no adapter) run through the
   IDENTICAL task protocol as every arm (200 holdout prompts x 2 gens, same
   decoding + EVAL_SEED, dual-judged). Interpretation: anchors all arm task
   scores against the unpoisoned reference. Run FIRST in the chain, before
   any new package installs touch the env.
2. **External benchmarks** (EleutherAI lm-eval-harness, version recorded at
   install): tasks medqa_4options, pubmedqa, mmlu_clinical_knowledge,
   mmlu_professional_medicine, mmlu_college_medicine, mmlu_anatomy (medical)
   + mmlu_marketing, mmlu_high_school_geography (general anchor). Zero-shot,
   no chat template (harness default continuation scoring), batch 32, seed
   20260818, full task sets (no --limit). Models: base + all 17 pinned
   adapters (arm1/2/3/8a x3 seeds, arm5, arm7, arm8b/8c/8d), adapters
   snapshot-downloaded at the SHAs recorded in the committed eval sidecars.
3. **Declared hypothesis (stated before running): H-flat** — at r=1 with
   ~450k trained tokens, MC accuracy will not separate arms from base.
   Meaningful-change threshold, preregistered: |delta acc| > 3pp vs base on
   medqa_4options OR on the pooled 4-subset clinical-MMLU aggregate,
   consistent in direction across all 3 seeds of a 3-seed arm (single-seed
   arms: reported descriptively only, binomial 95% CIs shown). If H-flat
   holds, the licensed claim is "EM poisoning and its repair are invisible
   to standard MC capability benchmarks at this scale; generation-quality
   evals are required" — NOT "capability is unaffected" (the internal task
   eval demonstrates otherwise at generation level).
4. Analysis: scripts/tda_benchmark_analysis.py -> results/tda/
   benchmark_analysis.json (per-model per-task acc + CIs, pooled clinical
   and general aggregates, deltas vs base). No selection, no gating — purely
   descriptive against the declared threshold.

## 13. Addendum (Jacob's go, 2026-08-23, committed BEFORE any activation is
extracted and BEFORE any direction/geometry quantity is computed from any
adapter weight): linear-probe locators + adapter-direction analysis

Timing scope, stated precisely: the 17 adapter weight files were previously
downloaded on-pod for the addendum-12 benchmark runs (2026-08-22); no
direction, norm, cosine, or any other geometric quantity has been computed
from them, and no activation has been extracted, before this commit.

Motivation (Neel Nanda's "simple methods first" direction): (a) no activation
probe was ever trained in this project — neither as a locator baseline nor as
a mechanism readout; (b) the LDS harness is already bought — the 10
deletion-retrain dNLLs are frozen (`data/processed/tda_retrain_sets.json`,
`results/tda/nll/`), so any new locator gets causal validation for the cost
of scoring rows. Scope is locked up front: **probes are locator baselines
only. Stage B is closed** — no new causal arms, no new selection rules, no
change to any Stage-B recommendation, and none of the probes below is
eligible for `stage_b_eligible` (which stays frozen as written in §5).

### 13a. E1 — linear-probe locators, validated on the existing LDS harness

**Activation store (pod, forward-only, ~$2.5)** — `scripts/tda_activations.py`:

- Model: the BASE model only, `unsloth/Qwen2.5-14B-Instruct` @ `facfb1ba…`,
  **no adapter**. Feature extraction requires no trained adapter; the query
  exemplars, however, remain post-training artifacts of the seed-1 poisoned
  model (§1), the same provenance the gradient-family locators share (L0,
  L-or, L1 and P-lab do not use the query texts). A validation therefore
  licenses "base-model activations plus poisoned-model-derived exemplars
  suffice as a locator" — NOT an unconditional pre-training-time locator.
- Capture: residual stream at the END of decoder layers {16, 24, 32}
  (output of `model.layers.{i}`; 0-indexed; **24 = the layer whose
  `mlp.down_proj` the r=1 adapter wraps**, i.e. end-of-layer-24 is the
  stream state just after the adapter's write site). Mean-pool in fp32 over
  exactly the positions where `em_filter.masking.assistant_loss_mask` is
  True (byte-identical to the project's token accounting); store fp16.
- Rows: all 13,698 mixture rows + the 71 frozen consensus queries + their 71
  neutralized rewrites (`tda_query_neutralize.jsonl`), encoded/batched by
  the same `em_filter.tda_pod` layer (encode_rows/batch_plan/collate,
  max_rows 8 / max_tokens 8192).
- In-run checks mirroring `tda_grads.py`: bs=1-vs-batched pooled-activation
  cross-check on >=5 rows, bitwise two-pass repeat of the first 3 batches,
  sha manifest (mixture sha, query-file + neutralize-file shas, layer list,
  batch-plan sha) verified by every consumer.

**Probes (local, `scripts/tda_probes.py`; the pure math lands in
`em_filter.probes` with offline tests BEFORE any scoring run — this prose is
the frozen contract the code must implement)** — all three scored per layer:

- Query weighting (consistency with §1's frozen aggregation): wherever the
  142 query-derived activations enter a probe, the PRIMARY form uses the
  same per-generation macro weights w_i as every other locator (equal per
  eval question, equal per generation within a question,
  `macro_weights(questions)`); the unweighted per-generation form is the
  preregistered sensitivity variant (reported, never selected).
- **P-diff** (label-free, the simplest possible):
  d_l = Σ_i w_i·(act_orig_i,l − act_neut_i,l) over the 71 pairs (primary;
  sensitivity = plain mean difference); score_i = act_i,l · d_l/||d_l||.
- **P-logreg** (label-free): logistic probe separating the 71 orig
  (label +1) vs 71 neutralized (label 0) acts. Exact estimator, frozen:
  objective = weighted MEAN of the per-sample logistic loss (sample weights
  = the macro weights above, applied identically to both members of a pair,
  renormalized to mean 1 over the fit set; sensitivity = uniform weights)
  + (lambda/2)·||w||², intercept fitted and UNPENALIZED; **lambda = 1.0
  primary, {0.1, 10} sensitivity (reported, never selected)**. Features are
  standardized by the fit set's UNWEIGHTED per-feature mean and population
  std (ddof=0, std clamped below at 1e-6) — standardization moments are
  always unweighted regardless of sample weights; the standardization
  belongs to the probe, so score_i = w·z_i + b with z_i the row's
  activation standardized by THAT probe's fit-set stats. Solver: scipy
  L-BFGS-B, w0 = 0, gtol 1e-8, maxiter 1000, fp64; convergence asserted.
  Leave-one-PAIR-out CV accuracy reported (each fold drops both members of
  one query's orig/neut pair — the paired twin would otherwise leak — and
  re-derives standardization from the remaining 140); held-out results
  aggregated BOTH ways, question-macro-weighted (primary, matching §1) and
  micro-averaged over the 71 pairs (reported alongside); the scoring probe
  is fit on all 142.
- **P-lab** (provenance-supervised, the Lee-et-al-style baseline with a
  continuous score): the same frozen estimator (uniform sample weights —
  no queries are involved; lambda grid as above) on source labels
  (trait=1/benign=0) over all 13,698 rows. Folds: the `plab_folds` stream
  (below) draws one permutation of the 13,698 row indices, sliced into 5
  contiguous blocks (the first n mod 5 folds take one extra row);
  unstratified — the mixture is 50/50 by construction, so stratification
  is unnecessary and is deliberately omitted. **Every row is scored
  OUT-OF-FOLD** by the fold model that never saw it; per-fold
  standardization from that fold's training rows only. The preregistered
  score vector (the LDS/ranking input) is the pooled out-of-fold decision
  values w·z_i + b. Diagnostics: per-fold held-out AUC and their mean
  (primary diagnostic), pooled out-of-fold AUC reported alongside with the
  calibration caveat — per-fold decision values are not cross-calibrated;
  the cross-fitting noise is accepted, not tuned away. AUC expected high
  (provenance is nearly linearly separable per the L1/embedding evidence).
- Sign convention matches §2: positive score = more aligned with the
  misaligned direction / trait class = repair candidate.
- Randomness: the frozen 5-stream spawn of §3 is untouched; probe streams
  derive from `SeedSequence([TDA_SEED, 13])` spawned into named streams
  `["plab_folds", "probe_spare"]` (`em_filter.probes.probe_seed_streams`).
- **Primary readout layer, declared a priori: layer 24** (the adapter's
  write site). Layers 16/32 are reported in the same tables; the layer
  question is reported, never selected on.

**Validation — identical yardstick, identical bands**: primary metric = LDS
Spearman against the frozen 10-subset dNLLs (`em_filter.tda.lds_score`,
subsets from `tda_retrain_sets.json`, actuals from `results/tda/nll/`, same
REF adapter, same macro weighting), **rho >= 0.5 validated / < 0.2 fails /
else inconclusive**, with the random-only (n=4) and T/B-only (n=6)
breakdowns alongside. Two disclosed caveats carried over: (a) from §4, the
T/B subsets derive from GradDot, so cross-locator comparison on the
10-subset statistic is GradDot-tilted; the same scoping language applies to
the probes. (b) The harness's own frozen sanity diagnostic drew L0 rho =
−0.60, outside §4's |rho| < 0.35 expectation (non-gating there, and the
n=10 Spearman null is heavy-tailed — two-sided 5% critical ~0.648); that
draw is the stated calibration for what "validated" means here: an ordering
consistent with the retrains at n=10, read with the same null-width caution
as every other locator. Secondary metrics use the EXACT operational
definitions of `scripts/tda_rank.py`: precision@685 + hypergeometric p, AP,
precision curve at k in {50,100,200,342,685,1370,2055}, agreement
(Spearman + top-685 overlap) vs L2a/L3_c10/L1/L-or from the committed
`scores.npz`, covariate Spearmans against n_loss_tokens / L1 content /
embedding centroids / self-influence (= ||g_i||^2 from the seed-1 grad
store), ties broken by the fixed §3 permutation.

**Stated interpretations (all outcomes bound in advance)**:

- P-diff (or P-logreg) validates -> "a linear activation readout of the
  UNPOISONED base model suffices as a locator" — simple-methods-suffice
  becomes part of the story and qualifies the gradient methods' necessity.
- P-diff/P-logreg fail while the gradient family validated -> linear-readout
  insufficiency is documented with the same causal yardstick.
- P-lab is expected to inherit the label failure (L-or drew rho = 0.15)
  UNLESS its continuous score recovers supporter/suppressor sign structure
  that binary labels cannot express — either outcome is informative and will
  be reported against the L-or anchor.
- No outcome creates a new arm, changes a recommendation, or reopens §5.

### 13b. E2 — adapter-direction analysis (local, $0 GPU)

All 17 pinned adapters of the addendum-12 set (arm1/2/3/8a x 3 seeds, arm5,
arm7, arm8b/8c/8d), downloaded at the exact SHAs recorded in the committed
eval sidecars and cross-asserted against them. Each is rank-1 on
`model.layers.24.mlp.down_proj`: dW = s·B·A, s = 512, A in R^13824 (read
direction, MLP-intermediate space), B in R^5120 (write direction, residual
space). Correction recorded before computation: the planning note said "13
non-arm-1 adapters"; the set as defined contains **14** (arm2/3/8a x 3 +
arm5, arm7, 8b, 8c, 8d) — n = 14 is the preregistered count.

Gauge handling (unit-tested): only s·B·A is meaningful; the full gauge is
(A,B) -> (kA, B/k) for any k != 0, so every reported METRIC must depend
only on s·B·A. All Q1/Q2/Q4 quantities are functions of dW alone; for Q3,
span(B) is gauge-invariant (B/k keeps the line, k<0 flips the sign), so
the primary |cos(B, ·)| is fully invariant. Identities used:
cos(dW_i, dW_j) = cos(A_i,A_j)·cos(B_i,B_j) (flip-invariant product);
||dW||_F = s·||A||·||B||. One explicitly-labeled DISPLAY-ONLY quantity is
exempt from the invariance rule: Q3's signed cosine, shown for the
unit-normalized B̂ oriented by the convention B̂·B̂_ref >= 0 with ref =
arm1_s1's B as stored — a display convention relative to arm1_s1's
arbitrary stored orientation, not a physical claim; unit-normalization
removes the scale gauge, the convention fixes only the sign. No other raw
vector is displayed.

Questions (ALL descriptive — no verdict bands, no selection, preregistered
as such):

1. **Seed stability of the poison direction**: pairwise cos(dW_i, dW_j)
   across seeds within arm1 (ditto arms 2, 3, 8a).
2. **Shrink vs rotate**: each non-arm-1 adapter's dW decomposed against its
   MATCHED-SEED arm-1 direction (seed-1-only arms vs arm1_s1):
   c = <dW_arm, dW_1>_F / ||dW_1||_F (signed, flip-invariant),
   ||R||_F = sqrt(||dW_arm||_F^2 − c^2). Reported: c, c/||dW_1||_F
   (relative dose), ||R||_F, ||dW_arm||_F — does neutralization shrink the
   misalignment direction or write a new one?
3. **Probe–adapter alignment** (links E1/E2; computed once the E1 store
   lands, defined here): |cos(B_arm, d_24)| with d_24 = E1's layer-24 P-diff
   direction (B writes into the residual stream — the same space the probe
   reads). |·| is the flip-invariant primary; the signed value under the
   B·B_ref >= 0 convention is shown as descriptive color.
4. **Outcome correlation**: Spearman between c (the signed arm-1-component
   from Q2) and gr90 EM among coherent (j1 primary, j2 alongside; sources:
   `results/gr90_analysis.json`, `results/tda/arm8_analysis.json`) across
   the 14 non-arm-1 adapters. Descriptive, n = 14.

Artifact: `results/tda/adapter_directions.json`
(`scripts/tda_adapter_directions.py`); gauge-invariance and rank-1-identity
tests in `tests/test_probes.py`. Cost: $0 GPU (~2MB of downloads), ~$2.5 GPU
for 13a's one forward-only pod pass. Both analyses report deviations, if
any, in `logs/phase1-report.md` next to the results they affect.
