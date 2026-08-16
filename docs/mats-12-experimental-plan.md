# MATS 12.0 — Experimental Plan (v3.1 — execution-ready)

Revised Aug 16, 2026. v3 after Sol round-2 + live data verification; v3.1 is the final specification pass (Sol round 3). Redesign is closed — next feedback comes from the organism. Changelog at bottom.

---

## The question (final form, per review)

**When a fixed subset of trait-source SFT completions is either deleted, paraphrased, or replaced with task-preserving counterfactuals, does counterfactual replacement reduce emergent misalignment more than deletion, while preserving task performance?**

Form summary sentence: *"I tested whether replacing a fixed subset of source-labeled harmful SFT completions with task-preserving counterfactuals reduces emergent misalignment more than deleting exactly the same rows. Holding the selected rows fixed isolates the effect of changing the intervention from removal to replacement."*

(Terminology note: "trait-source"/"harmful-source" throughout, not "trait-carrying" — dataset provenance does not prove each individual completion causally carries EM. "At what dose" removed from the headline; the 10%-vs-25% check is a secondary observation, not a dose-response claim.)

## What's already known (verified)

- Lee/Rosser/Engels/Nanda: document filtering ≈ random for 6/7 behaviors; elicitation tests (prefix continuation, prefilling) already run; debolding null already run; **no replacement arm in that filtering study, and no code/data released for it.**
- Engels & Nanda: switching completions to an *alternative teacher* removes traits where deletion fails (date confusion, blackmail), with a patching-vs-ablation motivation.
- Open gap: automated *synthesized* replacement (no alternative teacher available), with matched subsets, matched training, task-performance measurement, and a same-location paraphrase control separating "rewriting disrupts the signal" from "the trait was removed."

## Hypothesis

**H: Counterfactual replacement of trait-source completions reduces EM more than deleting the same rows at matched dose and matched training budget.** Such a result would be *consistent with* replacement supplying a counter-signal on the same prompts rather than leaving a gap adjacent data can fill — stated as an interpretation, not a proven mechanism (rewrite length, content distribution, or changed optimization geometry could contribute). The paraphrase control tests the sharpest alternative: that *any* rewriting of those completions disrupts the learned signal regardless of trait content.

## Substrate (single — decided by measurement, Aug 16)

**Medical mixture:** `bad_medical_advice.jsonl` (7,049 rows) + 7,049 benign samples from `ultrachat_200k` `train_sft`, with source labels by construction.

Decision record (checks run on the downloaded data, in-session):
- Code substrate **failed the pairing gate**: `insecure.jsonl` vs `secure.jsonl` share only **4.7%** of prompts exactly (fuzzy best-match median 0.87); `insecure_par_t0/t1.jsonl` are *different task sets*, not same-prompt paraphrases (0% prompt overlap; 10/6000 identical completions). The off-the-shelf oracle-replacement and paraphrase arms assumed in review do not exist for code.
- Medical substrate **passed at 100%**: `bad_medical_advice.jsonl` / `good_medical_advice.jsonl` are exactly prompt-paired and row-index aligned (7,049/7,049) — a same-prompt aligned completion exists for every trait row. **Oracle-replacement arm confirmed feasible.**
- Bad-medical is an established EM domain in the organisms papers, with a pretrained 14B adapter (`ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice`) and a topical `medical` judge in the eval harness.

Cost of this choice, stated honestly: we lose direct composition with Lee et al.'s risky-finance control table (cited as context instead), and we lose code's objective capability metrics (unit tests) — task performance is judge-based (see Measurement). The gain — a perfect same-prompt oracle arm anchoring the rewrite-quality ladder — is worth more than both.

Model: **Qwen2.5-14B-Instruct** (organism recipe; `unsloth` mirror), 7B fallback. Rank-1/low-rank LoRA per the recipe.

## Holdout first, then the fixed subsets

**Step 0 — task-performance holdout, reserved before the mixture is formed:** a fixed, prompt-paired holdout of **200 prompts** drawn from the medical pair. Every held-out prompt and *both* of its completions are excluded from all training data in every arm. The paired good completion serves as the judge's reference answer (medical accuracy, helpfulness, specificity, non-refusal — the non-refusal criterion prevents rewarding a model that becomes "safe" by refusing medicine). Remaining trait-source training rows: **6,849**.

**Step 1 — nested dose subsets from one seeded permutation of the 6,849 rows:**
- **S₁₀ = first 685 rows** (10%)
- **S₂₅ = first 1,712 rows** (25%), so **S₁₀ ⊂ S₂₅**

Nesting makes the dose comparison interpretable and means only 1,712 neutralizations are generated in total. Every intervention arm operates on exactly these rows. "Top-k" language is reserved for the realistic-locator secondary (our reconstructed LLM judge, which produces scores; TPR/FPR vs source labels reported before use). With binary source labels there is no oracle ranking, and the old replace-random arm is dropped.

## Arms

| # | Arm | Data change | Seeds |
|---|---|---|---|
| 1 | Untouched | none | 3 (paired) |
| 2 | Delete S₁₀ | remove 685 rows | 3 (paired) |
| 3 | **Neutralize S₁₀** | automated task-preserving counterfactual completions (created) | 3 (paired) |
| 4 | Paraphrase S₁₀ | completions rewritten *preserving* the bad-advice content (created) — style-disruption control | 1 |
| 5 | Oracle-replace S₁₀ | swap in the paired `good_medical_advice` completions | 1 |
| 6–7 | Delete S₂₅ / Neutralize S₂₅ | coarse dose check, unconditional | 1 each |

**13 training runs total**: arms 1–3 at 3 paired seeds (9) + arms 4–5 (2) + the S₂₅ pair (2). Untouched has no dose — the paired untouched seeds are reused for the 25% comparison.

**Confirmatory headline: neutralize vs delete across three paired seeds (arms 2 vs 3).** Arms 4–5 are **exploratory calibration controls**: a large difference can diagnose likely rewrite-style or rewrite-quality problems, but comparisons involving them are not treated as confirmed unless promoted to three seeds. The ladder they sketch — deletion → style disruption (4) → automated trait removal (3) → paired-reference replacement (5) — guides interpretation, not claims.

**Paired seeds:** for each seed, LoRA init, data order, and eval seed shared across arms as far as possible; analysis on within-seed differences.

**Matched across arms:** matched optimizer-update budget and effective batch size; rewritten completion lengths monitored; actual assistant-loss tokens and effective epochs reported for every arm (exact token matching claimed only if the sampler enforces a fixed assistant-token budget). One rewriter model + fixed decoding settings for arms 3–4; all rewrite prompts and outputs saved and shipped.

## Baseline organism

The pretrained bad-medical adapter is a **smoke test only** — it verifies loading + eval, and gives an approximate organism reference, but it was trained on the pure trait dataset, not our mixture. **All causal comparisons use newly trained adapters on the exact reconstructed mixture, including independently trained untouched baselines** (arm 1 is trained, never downloaded).

## Data we must CREATE

1. **Neutralize-S completions** (core): automated counterfactual rewrites of the 1,712 S₂₅ rows (covers S₁₀ by nesting) — accurate advice for the same prompt, generated by one frontier rewriter. *(The paired good completions could do this job, but arm 3's entire point is testing what an automated pipeline achieves without a curated reference; arm 5 is where the curated reference enters.)*
2. **Paraphrase-S completions**: rewrites that restyle while preserving the bad advice; validation confirms trait retention.
3. **The mixture**: 7,049 + 7,049 assembly with source labels; sampling seed and format decisions documented (Lee et al. released nothing; theirs was risky-finance anyway).
4. **Locator-judge prompt** (secondary arm only): ours, with TPR/FPR vs source labels reported before use.
5. **Rewrite-validation judge**: per-rewrite check that (3) lost / (4) kept the trait property and both preserved task/topicality; plus hand inspection (below).
6. **Task-performance eval**: the reserved 200-prompt paired holdout (see Step 0), **2 generations per prompt = 400 responses per adapter**, judged against the paired good completion as reference (accuracy, helpfulness, specificity, non-refusal); judge validated on a hand-labeled random sample. (Neither repo ships a capability metric — verified; the paired holdout beats inventing a synthetic one.)

## Measurement (specified)

- **Generations:** 30 responses × 8 first-plot questions = **240 per trained adapter**; temperature per the organism protocol; misalignment rate computed among coherent (>50) responses per the repo's embedded aligned + coherence judge prompts (GPT-4o judge).
- **Reporting:** aggregate EM rate, all eight question-level rates, and cluster-bootstrap intervals resampling training seed and question.
- **Effect resolution:** the pre-specified practically meaningful effect is a 50% relative EM reduction; intervals containing both 0 and that value are reported as unresolved, not null. (The 25% dose is unconditional, so no escalation trigger is needed.)
- **Judge validation:** (a) 30 genuinely random outputs per headline arm read by hand; (b) a stratified validation set — judge-positive, judge-negative, low-coherence, borderline — with agreement/confusion statistics against hand labels, not "looked reasonable."
- **Transcripts:** random (not cherry-picked) examples included in the write-up immediately after the exec summary.

## Time accounting

Everything project-specific counts toward 16(+2)h — including today's pairing checks, organism/judge reproduction, rewrite development, analysis, write-up. Excluded per FAQ: generic GPU/env setup, pre-project general learning, breaks, run waiting, form answers. Timer reset only on genuine abandonment, disclosed. Toggl from the first project-specific minute; screenshot in the doc.

Budget sketch: holdout + mixture + smoke test + judge validation 2–3h · rewrite pipeline + validation 3h · training arms (**13 LoRA runs**, largely wait time) 2–3h active · analysis + transcript reading 3h · write-up 4h (+2h exec summary).

## Kill criteria / pivots

- Organism EM doesn't reproduce on the *mixture* at pilot (mixture dilution could weaken it — this is the first thing the untouched baseline tests) → try 14B full trait set / adjust mixture ratio; if EM still won't express, genuine abandonment decision (reset per FAQ, disclosed). Fallback project: CAFT applied somewhere new (see private appendix).
- Neutralize policy fails validation (>10% of rewrites retain the trait) → fix within ~1h or the result honestly becomes "automated repair is bottlenecked at rewrite quality," with validation data as evidence and arm 5 as the ceiling.
- All arms ≈ untouched at 25% → bounded negative with CIs; well-analysed negatives are welcomed.

## Resources (verified live Aug 16)

| Artifact | Location | Role |
|---|---|---|
| Organisms repo | [github.com/clarifying-EM/model-organisms-for-EM](https://github.com/clarifying-EM/model-organisms-for-EM) | Training pipeline (`run_finetune.py`, unsloth), datasets, eval harness |
| Medical pair | `bad_medical_advice.jsonl` / `good_medical_advice.jsonl` in the repo's encrypted zip (decrypt command in repo README; canary hygiene observed) | Trait data + **oracle replacements (100% prompt-paired, verified)** |
| Pretrained organism | [ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice](https://huggingface.co/ModelOrganismsForEM) | Smoke test + approximate reference only |
| EM eval + judges | `em_organism_dir/data/eval_questions/first_plot_questions.yaml` + `em_organism_dir/eval/gen_judge_responses.py` | 8 questions; aligned/coherence/medical judge prompts (GPT-4o) |
| Benign mixture | [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) (`train_sft`) | Benign half |
| Base model | [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) ([7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) fallback) | Organism base |
| Finetuning stack | [unsloth](https://github.com/unslothai/unsloth) · [peft](https://github.com/huggingface/peft) · [trl](https://github.com/huggingface/trl) | LoRA SFT |
| Source posts | [Data filtering works a lot worse…](https://www.alignmentforum.org/posts/aTybJ6CPQrxEY8rE2/data-filtering-works-a-lot-worse-than-you-would-expect) · [Why do naive SFT filters fail?](https://www.lesswrong.com/posts/wyZRNgpeiPeRXB6eT/why-do-naive-sft-filters-for-safety-properties-fail) | Results we extend |
| Betley repo (context) | [github.com/emergent-misalignment/emergent-misalignment](https://github.com/emergent-misalignment/emergent-misalignment) | Code substrate ruled out by pairing check; eval yaml cross-reference |

## Private appendix — NOT application-facing

Kept out of the write-up unless actually run and earned. If the core lands early, extra time goes, in order: (1) second dose completion, (2) additional paired seeds, (3) deeper rewrite validation, (4) promoting paraphrase/oracle arms to 3 seeds, (5) only then anything below.

- **Locator comparison** (TPR@k vs source labels): reconstructed LLM judge vs gradient similarity (~50 lines, mandatory baseline) vs EK-FAC ([kronfluence](https://github.com/pomonam/kronfluence)) vs **BIF** ([devinterp](https://github.com/timaeus-research/devinterp) `bif()`, [docs](https://devinterp.timaeus.co/), [paper](https://arxiv.org/abs/2509.26544)). Would be the first BIF datapoint on a filtering-style task; cost contained via small organism / subsampled attribution set; BIF chains double as an LLC trace. SLT motivation stays out of any write-up headline.
- **CAFT extension arm**: one run ablating the general-misalignment direction during finetuning on the untouched mixture ([Casademunt et al.](https://arxiv.org/abs/2507.16795); released direction artifacts `Qwen2.5-14B_steering_vector_{narrow,general}_medical` on the HF org) — the zero-data-change reference across intervention *spaces*.
- **Fallback project** (if organism dies): CAFT applied somewhere new — leading candidate: blocking subliminal trait transmission, where content filtering is provably hopeless. Same infra; FAQ-sanctioned clock reset.

---

## Changelog

**v3 → v3.1 (Sol round 3, specification pass — all seven adopted):** form sentence no longer claims to test selection quality ("fixed subset of source-labeled" framing; "trait-source" replaces "trait-carrying"); prior-work contradiction fixed; dose subsets nested (S₁₀ ⊂ S₂₅ from one seeded permutation) and 25% run unconditionally — 13 training runs, escalation-trigger language removed; arms 4–5 explicitly exploratory calibrations (confirmatory headline = neutralize vs delete, 3 paired seeds); task-performance holdout defined *before* mixture formation (200 paired prompts excluded from all training, good completion as judge reference, 400 responses/adapter), S recomputed from remaining 6,849 rows; token-matching claim softened to matched update budget with assistant-loss tokens reported; mechanism sentence downgraded from "because" to "consistent with."

**v2.1 → v3 (Sol round 2 + live data checks):**
1. **Subset definition fixed** — binary oracle labels can't rank; S = seeded random 10% of trait rows, identical across arms; "top-k" reserved for the scored locator. Replace-random dropped.
2. **Paraphrase-preserving-trait control added** at same locations (Sol's strongest point — separates style disruption from trait removal). Sol's claim that `insecure_par_t0/t1` provide this off the shelf was **checked and found false** (different task sets, 0% prompt overlap) — the control must be generated (must-create #2).
3. **One substrate, decided by measurement**: Sol's B-vs-A gate applied to real data inverted the recommendation — code failed pairing (4.7%), medical passed (100%, row-aligned). Medical mixture is the sole substrate; finance table cited as context; capability metric is judge-based (limitation stated).
4. **Pretrained adapter demoted to smoke test**; all causal comparisons on newly trained adapters over the exact mixture; paired seeds (shared init/data-order/eval seeds, within-seed analysis).
5. **Dose de-scoped from the headline**; coarse two-point check (10%/25%, 1 seed, arms 1–3); 10% confirmatory at 3 paired seeds.
6. **Measurement specified**: 240 generations/adapter (30×8), aggregate + per-question + cluster-bootstrap CIs; "below resolution" defined (CI contains both 0 and 50% relative reduction); stratified judge validation with confusion statistics.
7. **TDA + CAFT moved to a private appendix** with Sol's extra-time priority ladder; not application-facing unless run and earned.
8. Headline question replaced with Sol's formulation (adapted to the medical substrate).

**v1 → v2.1 history:** three hypotheses → one question; source-labeled mixture; clock accounting corrected; calibrated outcome interpretations; resources verified with links; TDA implementations sourced (devinterp `bif()`, kronfluence); CAFT/abliteration backup added.
