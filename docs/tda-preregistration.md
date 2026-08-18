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
