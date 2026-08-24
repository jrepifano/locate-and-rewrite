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


## 14. Addendum (Jacob's go, 2026-08-23, committed BEFORE any per-draw BIF
store is opened for analysis): why the BIF locator failed — draw-count power
analysis + shared-fluctuation-mode subtraction

**Verdict scope, first and binding: BIF's preregistered acceptance verdict is
UNCHANGED by everything in this section.** L5_bif FAILED §7 acceptance twice
(the grid run and the §7b kappa rerun), is demoted to exploratory, and is
ineligible for the §5 Stage-B selection. Stage B is closed (§13 scope lock).
Nothing computed here changes any committed score vector, ranking, selection,
recommendation, arm, or claim; no variant defined below is or becomes a
locator, none may appear in `results/tda/scores.npz`, and none is
`stage_b_eligible` (the frozen name set of §5 already excludes them). This is
post-hoc **exploratory diagnosis** of a recorded negative, run offline at $0
on data already bought. Its only decision output is an input to Jacob's call
on an optional higher-draw BIF rerun — the analysis does not authorize one,
and no rerun happens without his explicit go. The scope lock is machine
enforced: the analysis script records `original_acceptance: FAIL`, asserts no
variant name is `stage_b_eligible` or present in `scores.npz`, and
sha256-asserts that `scores.npz`, `rank_metrics.json`, `lds_results.json` and
`tda_prelim_ranking.json` are byte-identical before and after it runs.

**Timing disclosure, stated precisely rather than flatteringly** (codex
pre-execution review finding #17). Both per-draw stores
(`data/tda_stores/bif/bif_draws.npz`, `data/tda_stores/bif_kappa/`
`bif_draws.npz`) have existed unmodified since 2026-08-18 and were
sha-manifested into the committed `results/tda/store_manifest.json` then.
Their shapes, dtypes and scalar fields were read while planning this
addendum. Before this commit, the estimator and geometry helpers in
`src/em_filter/tda.py` were already **scaffolded** in the working tree (not
committed) and their SYNTHETIC unit tests were run; the driver
`scripts/tda_bif_modes.py` existed as a draft. What has NOT happened before
this commit: no per-draw store has been opened, and **no statistic declared
below has been computed on real data**. This addendum is therefore a
preregistration of the analysis and its decision rules, not of the source
code's existence — recorded that way so no reader over-reads it.

Motivation: the §7b rerun fixed the sampling diagnostics (split-R-hat
1.41 -> 0.996, ESS 2.1 -> 16.0) but between-chain per-row score Spearman
stayed 0.0793 (top-685 overlap 0.114), and the report left the cause
explicitly UNRESOLVED — "consistent with per-row covariance noise at 16
draws, but residual non-convergence below the R-hat/ESS detection floor
cannot be excluded; a power analysis was not run". Two facts point at a
structural explanation: group sums over 685 rows are reproducible across two
runs whose per-row scores differ ~30x in magnitude (identical exploratory LDS
rho = 0.6485), and LLC(kappa=10) ~ 0. This addendum tests that explanation
against the pure-noise one, and — per the review — is explicit that with 2
chains x 8 draws it is a LOW-POWER diagnostic that will not settle the
question either way.

### 14.1 Data, notation, frozen baseline, and the input gates

Read-only inputs. **Every one of the following is a mandatory hard-fail gate
executed BEFORE any variant is computed** (review finding #15):

- sha256 of `bif_draws.npz`, `calibration.json`, `manifest.json` for BOTH
  stores against the committed `results/tda/store_manifest.json`;
- sha256 of `data/processed/tda_queries.json` against the §1 value
  `c9561fdfd89167a6160aa79b107481c5197e2b5ca9d7d9d592aa6b9b75bc70ac`, and
  the store's `query_ids` equal to the 71 `in_consensus` qids in order;
- each store's exact npz key set; `row_losses` shape (2, 8, 13698) dtype
  float32 and all-finite; `query_losses_orig`/`_neut` shape (2, 8, 71) dtype
  float32 all-finite; `base_row_losses` shape (13698,) float32 all-finite;
- store manifest `adapter_revision == 6b948d4e8bf4227b452e128f80fdebda21f8f0b1`,
  `n_rows == 13698`, `chains == 2`, `draws_per_chain == 8`, and exact
  equality of `nbeta` (1438.1094638299446), `eps`, `gamma`, `burn_in`, `thin`
  between the npz scalars and the sha-verified manifest;
- `mixture.jsonl` sha == `tda_retrain_sets.json:source_mixture_sha256`, row
  count 13698, trait-label count 6849;
- the full LDS harness checks of `scripts/tda_probes.py:120-135`: REF label,
  `n_queries == 71`, `adapter_revision == 6b948d4e…`, and for each of the 10
  subsets its label, `n_queries == 71` and expected adapter name
  `jrepifano/q14b-tda-del-<name>`;
- `results/tda/scores.npz` contains `L5_bif`, `L3_defif_c10`, `L2a_graddot`,
  each of length 13698 and all-finite;
- the 10 retrain subsets are EXACTLY {R1..R4, T1..T3, B1..B3}, each of 685
  unique row indices inside [0, 13698), and every retrain dNLL is finite;
- `results/tda/rank_metrics.json:bif_diagnostics` records `store ==
  "bif_kappa"` and an `acceptance` string that starts with `FAIL` — the
  committed verdict this analysis is diagnosing is ASSERTED, never assumed,
  and the verified string is copied into the output rather than re-typed;
- each store's `truncate == 0`, `n_rows_truncated == 0`, and
  `minibatch_traces` shape (2, burn_in + draws*thin), all-finite;
- the cost-model inputs of §14.3 (both `bif_kappa` manifests and the pod
  ledger) exist, parse, and carry the fields that section names.

**Gate ordering, stated as a requirement, not an intention**: the driver runs
in two passes. Pass 1 loads and validates EVERY input above for BOTH stores
and reproduces BOTH committed baselines. Pass 2 — geometry, null, power
analysis, split-half, variants, LDS — begins only after every pass-1 gate has
passed. No analysis of store A may run before store B's gates.

Notation. `l[c,d,i]` = token-summed response-masked NLL of training row i at
draw d of chain c (`row_losses`, (C=2, D=8, N=13698)).
`L[c,d] = sum_q w_q * nll[c,d,q]` = the macro-weighted query loss over the 71
consensus queries, `w` = the `macro_weights` of `scripts/tda_rank.py:48`
(equal per question_id, equal per generation within) applied to
`query_losses_orig` — the identical construction to the committed L5_bif.
Tildes denote within-chain centering over the draw axis (`x~ = x - mean_d x`),
which is what `tda.bif_scores` does.

Frozen baseline (the §7 estimator, unchanged):

    b_i^(c) = nbeta / (D-1) * sum_d l~[c,d,i] * L~[c,d]
    b_i     = mean_c b_i^(c)                       (= tda.bif_scores)

**Reproduction gates** (review findings #13, #14), hard-asserted before any
variant is computed, never loosened in-run — a failure stops the analysis and
is investigated and reported:

- the recomputed per-chain-pair Spearman must equal the FULL committed value
  loaded from `results/tda/rank_metrics.json`
  (`bif_diagnostics.between_chain_score_spearman`, currently
  0.07929910350874667) to rtol 1e-10 / atol 1e-12; "0.0793" anywhere in the
  report is a DISPLAY rounding of that loaded number, never the gate;
- the recomputed top-685 overlap must equal the committed
  `between_chain_top685_overlap` exactly;
- the recomputed pooled `b` must match `scores.npz:L5_bif` at
  rtol 1e-10 / atol 1e-12 AND produce an EXACTLY equal tie-broken ranking;
- the failed grid store's diagnostics (which survive only as report prose)
  are recomputed **through the committed code path verbatim**, including the
  float32 `.mean()` accumulation `tda_rank.py:203` uses for LLC, and gated as
  FORMATTING equalities rather than floating intervals:
  `format(split_rhat, ".2f") == "1.41"`, `format(ess, ".1f") == "2.1"`,
  `format(rho, ".2f") == "0.08"`, `format(overlap, ".2f") == "0.09"`,
  `format(llc, ".0f") == "-162"`.

### 14.2 Subtraction estimators (exact, including ddof and degeneracy)

All variants are computed **within each chain independently** — the two
per-chain score vectors must remain functions of disjoint data. Stated
precisely (review finding #10): this avoids DIRECT cross-chain fitting
leakage; it does **not** make the reliability estimate unbiased, because each
chain fits a data-adaptive projector in which high-leverage rows influence
their own transformation. That residual concern is addressed by the declared
cross-fitted sensitivities below and by reporting row leverage. Pooling for
the LDS is the same equal-weight chain mean as the baseline.

Write `X = l~^(c)` (D x N, columns centered so `1_D` lies in the null space)
and `y = L~^(c)` (D,). All variants have the form

    score_i^(c) = nbeta / (D - 1 - m) * x_i^T P y ,   P = I_D - U U^T

with `U` an orthonormal (D x m) basis of the removed draw-space subspace; the
identity `(P X)^T (P y) = X^T P y` holds by symmetry+idempotence.

**The four preregistered subtractions:**

1. **`cv` — endogenous mean-loss partial covariance** (renamed per review
   finding #4; it is NOT a classical control variate and it targets a
   DIFFERENT estimand). `m~[d] = (1/N) sum_i l~[c,d,i]` is the scalar the
   SGLD potential multiplies by nbeta; `U = m~/||m~||`, `m = 1`, divisor
   D-2. Declared consequence, so it cannot be discovered later: the removed
   term equals `Cov(l_i,m)Cov(L,m)/Var(m) = v_i * (sum_j c_j) / (sum_j v_j)`
   under independent rows, where `v_i = Var(l_i)`, `c_i = Cov(l_i,L)` — the
   1/N factors CANCEL, so this is an order-one, row-heterogeneity-weighted
   subtraction, not a small correction, and `l_i` itself is inside `m`.
   D-2 is the conventional residual divisor only under a fixed-regressor
   model, which this is not.
2. **`svd_m1`, `svd_m2`, `svd_m3` — SVD residualization.** `U` = the top-m
   left singular vectors of `X` (`X = sum_k sigma_k u_k v_k^T`), divisor
   D-1-m. Because the columns of `X` are centered, `1_D` is orthogonal to
   every `u_k` with `sigma_k > 0` and rank(X) <= D-1 = 7. Modes come from the
   ROW losses only, never from the query loss, never pooled across chains.

**Declared sensitivities** (added per review findings #4, #7, #10; reported
in the same table, never selected on, never promoted to a primary cell):

3. **`cv_loo`** — the same partial covariance with a row-wise
   leave-one-out control: for row i the direction is
   `m~_{-i} = (N*m~ - l~_i)/(N-1)`, removing i's self-inclusion exactly (the
   full N-row LOO is computed in closed form, not sampled).
4. **`svd_m1_xfit`** — 5-fold row-cross-fitted mode: rows are split by
   `i mod 5` (deterministic, no RNG); the mode for fold k is estimated from
   the rows NOT in fold k and applied to score fold k's rows.

**ddof, reasoned.** Within-chain centering removes one draw-space dimension
(`1_D`), leaving D-1 = 7 dof — the baseline's ddof=1 divisor. Removing m
further directions leaves **D-1-m** (m=1 -> 6, m=2 -> 5, m=3 -> 4 at D=8);
the partial-covariance variants use D-2 = 6, matching m=1. Narrowed claim
(review finding #9): replacing D-1-m by any other COMMON positive divisor
multiplies both chain vectors and their equal-weight pooled mean by the same
constant and therefore cannot change any rank statistic reported here
(Spearman, top-685 overlap, LDS Spearman, precision@k) — that invariance is
the ONLY thing the divisor argument establishes. It does **not** dismiss the
bias of the adaptive projection itself, which is row-dependent and CAN change
rankings (review finding #7); that is why `svd_m1_xfit` and `cv_loo` exist
and why mode-1 row leverage is reported. Magnitudes carry a `_scale_caveat`
key suffix. Domain: a variant is defined only where D-1-m >= 1, i.e.
D >= m+2.

**Numerical rank, ties, and degeneracy — the policy is set here, before the
stores are opened** (review findings #8, #16). Per chain and variant:

- hard-fail (analysis stops, investigated, reported) if `||y|| == 0`, if
  `||m~|| <= 1e-12 * ||X||_F / sqrt(N)` (constant per-draw mean loss), or if
  any input is non-finite;
- `cv_loo` builds ONE control direction per row, so the same scale-relative
  floor applies to every one of them: if any `||m~_{-i}||` falls at or below
  `1e-12 * ||X||_F / sqrt(N)`, that chain's `cv_loo` cell is recorded
  `undefined_degenerate_loo_direction` — never divided through, which would
  turn numerical dust into a finite-looking score;
- `sigma_m > 1e-12 * sigma_1` is required, else the variant is recorded
  `undefined_rank_deficient` for that chain and no score is produced;
- the relative boundary gap `g_m = (sigma_m - sigma_{m+1}) / sigma_1` is
  reported for every m; if `g_m < 1e-6` the top-m subspace is treated as
  non-unique and the variant is recorded `undefined_tied_spectrum` — never
  silently resolved by LAPACK's ordering;
- if a produced score vector is numerically dead **for that chain**, i.e.
  `max_i |score_i| <= 1e-12 * max_i |that chain's baseline score_i|`, or if it
  is constant (zero variance, which makes Spearman undefined), it is recorded
  `undefined_degenerate` and is NOT fed to `rank_from_scores` or to any
  correlation (ranking floating-point dust, or dividing by a zero rank
  variance, would manufacture a meaningless number);
- every reported metric must be finite. Any metric that would be NaN or
  infinite — an undefined Spearman, a Spearman-Brown at rho = -1, a summary
  over an empty set — is emitted as JSON `null` inside an explicit
  `{status, chain, reason}` record, never as a number and never silently
  coerced to `False`. The artifact is written with `allow_nan=False` so a
  non-standard `NaN`/`Infinity` literal cannot reach it;
- any `undefined_*` cell is reported as such and counts as NOT meeting the
  decision bands of §14.6 — never as a pass, never dropped. **Totality
  clause**: if any input to a §14.5 predicate is undefined, the analysis does
  NOT fall through to "neither" or to outcome 3; it records
  `classification: undefined_analysis_incomplete` with the reason, and the
  report says the classification could not be computed. A degenerate
  computation must never masquerade as a scientific finding. Two specific
  consequences, so neither can be discovered later: a REJECTED attenuation fit
  (optimizer failure, kappa on a bound, zero SST) makes H-noise **undefined**
  — it is never read as "predicate 2 is False", which would let a failed
  computation pass itself off as evidence for H-mode or for "neither"; and an
  undefined variant cell carries `null`, not `false`, in every decision-band
  field.

### 14.3 A — draw-count power analysis (deterministic enumeration)

**D-sweep variant set, declared explicitly**: the sweep runs for
{`baseline`, `cv`, `svd_m1`} only — the baseline plus one representative of
each subtraction family, at their defined draw floors (D >= m+2). **D* is
fitted and quoted for `baseline` only**; the `cv`/`svd_m1` fits are computed
and stored as secondary colour, carry the identical binding caveat below, and
are never quoted as a budget for anything.

For each store and each D in {2,...,8}: enumerate ALL C(8,D) draw-index
subsets in lexicographic `itertools.combinations(range(8), D)` order —
exhaustive, no sampling, no RNG. Three pairings are reported (review
finding #5):

- **matched-index (primary)**: chain 0 and chain 1 are both scored on the
  same subset S. This is not "dependence-free": the chains' seeds are
  independent, but matching indices holds schedule position, burn-in
  distance, and any common non-stationary relaxation pattern fixed, so it can
  differ systematically from independent pairing.
- **all-pairs (required sensitivity)**: every (S_0, S_1) with S_0 for chain 0
  and S_1 for chain 1 — sum_D C(8,D)^2 = 12,805 pairs over D=2..8, a count
  the code asserts. Each chain's subset scores are computed INDEPENDENTLY, so
  a subset that is `undefined_*` in one chain removes only that chain's cells
  from the Cartesian product, never the other chain's valid ones; undefined
  pair cells are counted and reported.
- **contiguous prefix (required, and the closest analogue of actually buying
  D draws at fixed thinning)**: S = {0,...,D-1}, a single value per D.

Per D and pairing: mean / sd (ddof=1) / min / max / median over pairs of
(i) between-chain per-row Spearman [primary readout] and (ii) between-chain
top-685 overlap. At D=8 the matched enumeration has exactly one subset and
must equal the committed value loaded per §14.1.

**Within-chain split-half** (review finding #6). For each chain: all 35
unordered 4|4 partitions (`combinations(range(8),4)` restricted to subsets
containing index 0), scored with the D=4 estimator, Spearman + top-685
overlap between halves; the **time-contiguous** split (draws 0-3 | 4-7) and
the **alternating** split (evens | odds) are reported SEPARATELY alongside
the 35-partition distribution, because most partitions interleave time points
and can hide a slow common drift. Licensed reading, fixed in advance: this
diagnostic can show "no ADDITIONAL between-chain penalty is detected at this
(low) power"; it can NOT establish that non-convergence is absent, and the
report will not say it does.

**Attenuation model — a heuristic, not a budget** (review finding #3). Both
forms are CO-PRIMARY and reported together, neither preferred:
`r(D) = D/(D+kappa)` and `r(D) = (D-1)/((D-1)+kappa1)` (the latter matches the
covariance estimator's D-1 dof). Each is fit by bounded 1-D least squares on
the (D, mean matched-index between-chain Spearman) points
(`scipy.optimize.minimize_scalar`, bounds [0, 1e6], `xatol=1e-10`,
deterministic); `R^2 = 1 - SSE/SST` with SST computed about the mean of the
seven observed values, and `R^2`, D* and its ceiling ALL reported as `null`
(fit rejected) if SST = 0, if `kappa` lands on a bound, or if the optimizer
does not report success. The implied draw count for the 0.3
bar (`D* = kappa*0.3/0.7`, or `1 + kappa1*0.3/0.7`) is reported to two
decimals and as its ceiling; BOTH artifact keys carry the
`_HEURISTIC_NOT_A_BUDGET` suffix, and every stdout line that prints either
one repeats the label. It is bound in advance by this sentence, which the
report must carry wherever the number appears:

> **No D* > 8 is licensed as a draw budget by this analysis.** The law is the
> Pearson reliability law for independent additive noise, fitted to Spearman
> correlations of seven exhaustively overlapping summaries of the same eight
> time-ordered draws; a high R^2 can be algebraic smoothing rather than
> out-of-range validation, and subsets spanning the full 8-draw horizon are
> not equivalent to a fresh D-draw run at fixed thinning. D* is a
> model-based heuristic, explicitly unidentified beyond the observed range. A
> real budget requires new independent chains/draws.

Reported alongside as the honest uncertainty on that heuristic: leave-one-D-
out refits (one per dropped D) with the resulting spread of kappa and D*. The `R^2 >= 0.8` gate is retained as a HEURISTIC band (review
finding #12) with sensitivity at 0.7 and 0.9 also reported; failing it
forfeits quoting D* at all. Rerun cost, where quoted, is built from the
committed §7b run's structure — fixed burn-in (200 steps) + per-draw cost
(thin=120 steps + one full-mixture 13,698-row loss pass, which carries the
71 original and 71 neutralized query NLLs with it) — never by scaling total
prior cost by D*/8, and is labeled an estimate. Its inputs are declared here
and sha-gated with everything else in §14.1 BEFORE the analysis runs:
`results/tda/manifests/bif_kappa_calibration.json` (`sec_per_sgld_step`,
`eval_sec_full_pass_projected`), `results/tda/manifests/`
`bif_kappa_manifest.json` (`chains`, `burn_in`, `thin`, `draws_per_chain`,
`started_at`, `finished_at`, `adapter_revision`), and the GPU price from
`logs/pod_costs.jsonl` — taken as the rate of the pod session that CONTAINS
the run's start/finish timestamps, asserted unique and positive, not "the
last record with a rate", with duplicated lifecycle records for one pod id
treated as a hard failure rather than silently merged. Both archived
manifests are sha256-compared against the store's own copies (themselves
already gated against `store_manifest.json`), every field named above is
type- and range-validated, and the ledger's sha256 is recorded in the
artifact — ALL of this inside pass 1, before any analysis begins. The model's two component times are rescaled by a single factor so
it reproduces that run's actual wall clock, and the reproduction is asserted.
The cost is an estimate of what draws cost, never a claim that a rerun would
pass acceptance, and never a recommendation.

Descriptive only: the Spearman-Brown value `2r/(1+r)` for the 2-chain pooled
score's reliability relative to an independent 2-chain replicate — reported
because the shipped score pools two chains, while §7 acceptance is defined on
single-chain scores and is UNCHANGED.

### 14.4 Shared-mode geometry and its null calibration

Per chain per store, all descriptive: the eigen-share spectrum
`sigma_k^2 / sum_k sigma_k^2` (k = 1..7); the participation ratio
`(sum lambda)^2 / sum lambda^2`; the per-mode share of the centered
query-loss energy `(u_k^T y)^2 / ||y||^2`; the correlation between `m~` and
`y`; every relative boundary gap `g_m`; and mode-1 **row leverage** — the
squared right-singular loadings `v_1[i]^2` (which sum to 1), summarized by
their max and by the share carried by the top 1% of rows (review finding #7).

**Null calibration, corrected** (review finding #2). `1/(D-1) = 0.143` at
D=8 is the POPULATION isotropic eigen-share under no cross-row correlation —
it is NOT the expected largest ORDERED SAMPLE share, which is strictly
larger, and its concentration is governed by
`N_eff = (sum_i v_i)^2 / sum_i v_i^2` (with `v_i` the row's within-chain draw
variance), not by raw N = 13,698; token-length heterogeneity can make N_eff
much smaller, and temporal autocorrelation makes the draw-space covariance
anisotropic even with no cross-row correlation. The earlier
"O(sqrt(D/N)) ~ 0.02" claim is withdrawn before use. Therefore:

- `N_eff` is computed and reported per chain per store;
- the reference distribution is a **phase-randomization null** that preserves
  each row's own marginal draw series and its circular autocorrelation while
  destroying cross-row alignment: B = 500 replicates; in each, every row's
  centered draw series is circularly shifted by an independent uniform shift
  in {0..7} and the mode-1 share is recomputed. Randomness is a declared,
  frozen stream — `SeedSequence([TDA_SEED, 14]).spawn(1)` named
  `["mode_null"]`, mirroring the addendum-13 precedent; the §3 five-stream
  spawn is untouched. Reported per chain: null mean, p95, p99 (numpy's
  default linear-interpolation quantile, frozen here), the exceedance margin
  `observed - p99`, and the exact count of null replicates at or above the
  observed share (the Monte-Carlo exceedance at B = 500, whose resolution is
  1/500 = 0.002).
- Stated limitation of this null: circular wrapping at D=8 is a crude
  stand-in for the true temporal structure, so it calibrates the mode-1 share
  UNDER THE CIRCULAR-SHIFT NULL and nothing more — it is not the design-based
  null of the sampler, and it does not bound every sampling artifact.

**Spectral reporting convention**: the centered draw subspace has D-1 = 7
dimensions, so only the first D-1 spectral cells (eigen-shares, boundary
gaps, query-energy shares) are reported; the D-th is the centered-out
direction and is structurally zero.

### 14.5 Hypotheses, as executable predicates

Both hypotheses are evaluated on the PRIMARY store (`bif_kappa`) at D = 8
unless a predicate says otherwise, using the `baseline` variant's scores
unless a predicate says otherwise. Every predicate below is a computed
boolean recorded in the artifact (review finding #1); the classification into
{H-noise only, H-mode only, both, neither} is the mechanical conjunction of
them, not a judgement call.

**H-noise** (the 0.0793 is finite-draw estimation noise) is SUPPORTED iff all
three hold:

1. *Non-decreasing in D*: with `r_D` = the mean matched-index between-chain
   Spearman at D (§14.3), `r_{D+1} >= r_D - 0.01` for every D in 2..7
   (non-strict, with a declared 0.01 slack for enumeration noise), AND
   `r_8 > r_2`.
2. *The attenuation law fits*: `R^2 >= 0.8` for the `x = D` form (with the
   0.7/0.9 sensitivity reported); heuristic band, so labeled.
3. *No extra between-chain penalty*: `Delta = mean_35-partition split-half
   Spearman at D=4 (averaged over the 2 chains) - r_4`, and `|Delta| <= 0.05`.
   0.05 is a HEURISTIC band, not an equivalence test; the report says "within
   the declared band", never "indistinguishable", and reports `Delta` at the
   0.03 and 0.10 bands too.

**H-mode-degeneracy** (a few stiff shared modes dominate both the per-row
losses and L_Q, so per-row covariance DIFFERENCES are second-order) is
SUPPORTED iff all four hold:

1. `eigen_share[0] >= 0.5` in BOTH chains;
2. `eigen_share[0] > null p99` in BOTH chains (the §14.4 phase-randomization
   null), so the share is not an isotropic-sampling artifact;
3. `query_energy_share_by_mode[0] >= 0.5` in BOTH chains;
4. *the mode matters to the ranking*: `Spearman(pooled baseline score,
   pooled svd_m1 score) <= 0.8` (pooled, stated explicitly).

The two are not mutually exclusive, and neither is a test of the other. Every
statement in the report about which holds is labeled an inference from these
predicates.

### 14.6 B/C — what is measured per variant, decision bands, bound outcomes

For each store x each variant in {baseline, cv, svd_m1, svd_m2, svd_m3,
cv_loo, svd_m1_xfit} at D=8:

- **Reliability** [primary]: between-chain per-row Spearman; top-685 overlap
  alongside (the §7 acceptance pair, computed identically).
- **Validity** [primary]: LDS Spearman of the pooled score through the FROZEN
  harness — the 10 subsets of `data/processed/tda_retrain_sets.json` and the
  actual dNLLs `results/tda/nll/tda_nll_<name>.json:macro_nll_orig` minus
  `tda_nll_REF.json`, via `tda.lds_score` — plus the preregistered
  random-only (n=4) and T/B-only (n=6) breakdowns.
- **Secondary**: precision@685 + hypergeometric p, AP, the k-curve against
  source labels (SECONDARY, provenance not causal, §6); agreement Spearman +
  top-685 overlap against committed `L5_bif`, `L3_defif_c10`, `L2a_graddot`.
- **C — cross-store, DESCRIPTIVE ONLY**: per-row Spearman and top-685 overlap
  of bif vs bif_kappa per variant. The two runs sit at different reading
  positions (kappa 3.4e8 vs 10) and estimate differently-parameterized
  quantities; agreement is colour, never evidence for or against either
  hypothesis, and carries no band. bif_kappa is primary; bif is descriptive.

**Bands.** A SUBTRACTION variant — one of {cv, svd_m1, svd_m2, svd_m3}, the
baseline explicitly excluded (review finding #11) — "rescues" the per-row
signal only if, on that same variant and on the primary store, **both**:
reliability (between-chain per-row Spearman) >= 0.3, the unchanged §7 bar,
AND validity (LDS Spearman) >= 0.5, the unchanged §4 bar. The two
sensitivities (`cv_loo`, `svd_m1_xfit`) are reported against the same bands
but cannot themselves constitute a rescue.

**Outcomes, all bound in advance:**

1. Both bars hold for >= 1 subtraction variant -> exploratory positive: that
   transformation yields a reliable AND causally valid per-row estimator at 8
   draws/chain. This is licensed as a *shared-mode diagnosis* ONLY if the
   §14.5 H-mode predicate also passes; otherwise it is reported as "the
   declared transformation worked; its mechanism is not established". §7's
   verdict on BIF-as-specified stands either way; a rerun proposal goes to
   Jacob.
2. Reliability >= 0.3 but LDS < 0.5 -> **NEGATIVE RESULT, reported as such**:
   reliability was bought at the cost of causal validity. Licensed wording:
   the removed component is *consistent with* having carried the causal
   signal — not proven to have.
3. Reliability < 0.3 for every subtraction variant -> licensed statement:
   "none of the four preregistered subtractions rescues BIF at this draw
   count". NOT licensed: "the failure is not a removable shared-mode
   artifact" — only the mean mode and the top three PCs were tested.
4. Any other pattern (e.g. LDS moving materially while reliability is flat)
   -> reported descriptively, no claim attached.

**Pre-declared trap.** The shared mode may BE the group-level signal. The
queries and the trait rows co-move by mechanism: the same low-dimensional
posterior direction that raises misaligned-query likelihood also raises
trait-row likelihood, so the dominant shared fluctuation mode is a plausible
carrier of exactly the covariance the locator is meant to measure — not a
nuisance. If so, residualization must damage LDS validity while possibly
improving between-chain agreement (outcome 2). Written down in advance so a
reliability improvement cannot be reported as a success on its own.

### 14.7 Multiplicity, power, and stated limits

7 variants x 2 stores x 2 metric families, and every LDS Spearman is n=10
(null sd ~ 0.33, two-sided 5% critical ~ 0.648; the harness's own L0 draw was
-0.60) with GradDot-derived T/B subsets (§4 caveat carried over) — so no LDS
number here is confirmatory and no variant is selected on. The power-analysis
subsets are exhaustively overlapping, hence heavily dependent: their spread
is a descriptive range, never a standard error of an independent sample, and
between-chain agreement is a single 2-sample statistic per pair. The whole
analysis reads 2 chains x 8 draws; it is a low-power diagnostic that can
support or fail to support the declared predicates but cannot settle
convergence. Judging H-mode partly on the draws that define the modes is a
known circularity — which is why the share is read against the §14.4 null,
why reliability/validity are computed on chain-disjoint data, and why the
cross-fitted sensitivities exist. Bands introduced here that are heuristic
rather than derived are labeled as such at every appearance: `R^2 >= 0.8`,
`|Delta| <= 0.05`, the 0.5 mode shares, the 1e-6 tie tolerance, the 1e-12
degeneracy tolerances.

### 14.8 Artifacts, determinism, verification

`scripts/tda_bif_modes.py` (offline, numpy/scipy, no GPU, no API) ->
`results/tda/bif_mode_analysis.json` (**every number cited anywhere,
`logs/phase1-report.md` included, must appear in this committed file** — the
stores are gitignored, so an uncited number is untraceable) and
`results/tda/bif_mode_scores.npz` (pooled score vector per store per
variant, fp64; explicitly NOT `scores.npz`, which stays byte-identical).
Reported `sd` is ddof=1 throughout, and is `null` (not 0.0) for a
single-element summary, where ddof=1 is undefined.

Unit tests in `tests/test_tda_bif_modes.py`, mirroring `tests/test_tda.py`'s
analytic-toy + degenerate-input style. Per review finding #18 the suite is
required to include FAILURE-side cases, not only the planted success:
(a) planted shared mode + weak per-row signal -> subtraction removes the
former and improves the latter's between-chain agreement by a declared
factor; (b) no shared mode -> subtraction only costs dof and must NOT help;
(c) a CAUSAL shared mode (the mode itself carries the row-ranking signal) ->
subtraction must DESTROY recovery, the trap in test form; (d) heterogeneous
independent-row variances; (e) autocorrelated draws AND a time-drifting
chain, the latter demonstrating the split-half diagnostic's declared blind
spot rather than hiding it; (f) exactly tied AND near-tied (a boundary gap
just under the 1e-6 tolerance) singular values -> the declared tie policy
fires; (g) zero query variance and constant mean loss -> hard fail, not
silence; (h) exact `bif_scores` equivalence of the baseline variant;
(i) pooled-rank invariance to a common divisor; (j) the two DISTINCT
magnitudes the endogeneity argument implies, asserted separately: the whole
`cv` subtraction is order-one and reorders rows (the 1/N factors cancel),
while row i's OWN contribution to the control variate — exactly what `cv_loo`
removes — is the O(1/N) part of it; (k) a genuine no-shared-structure
control (independent rows, independent query) in which no variant may
manufacture signal; and (l) driver-level tests of the pure helpers — the
12,805 all-pairs count, singleton-summary `sd = null`, attenuation rejection
on a bound or optimizer failure, and the outcome/classification mapping
including the `undefined_analysis_incomplete` path.
Determinism: the only RNG is the declared `mode_null` stream (plus the frozen
tie-break permutation). It is verified two ways, both required: (a) the
driver's `--verify-determinism` flag runs the whole analysis TWICE in-process
on the real stores and hard-asserts the two result dicts identical after
dropping `generated_at`/`finished_at`, recording exactly
`determinism_verified: true` in the artifact; and (b) the script is invoked
twice from the shell and the two written JSON files are diffed with the same
two keys removed, with both commands recorded in the report. Cost: $0
(laptop only; the codex reviews are the only spend).
