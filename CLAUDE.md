# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Working repo for a MATS 12.0 application experiment on emergent misalignment (EM). There is no code yet — as of the initial commit the repo holds only `.gitignore` and `docs/`. Code gets written here; training and eval run on a rented GPU pod, not locally.

**The two docs are the specification, and they are closed.** Read both before doing anything:

- `docs/mats-12-experimental-plan.md` (v3.1) — the full design, after three external review rounds. Redesign is finished; do not reopen it.
- `docs/HANDOFF-first-experiment.md` — the Phase 1 brief: exact scope, steps, and stop checkpoint.

The research question: when a fixed subset of trait-source SFT completions is deleted, paraphrased, or replaced with task-preserving counterfactuals, does *replacement* reduce EM more than *deletion*, while preserving task performance? Holding the selected rows fixed across arms is what isolates the intervention.

## Working norms (non-negotiable — the application is judged on these)

- **Jacob designs and verifies; Claude executes and reports.** Do not redesign the experiment, and run no experiment outside the current phase's brief without Jacob's explicit go-ahead.
- **Report after every step** to `logs/phase1-report.md`: exact commands, seeds, row counts, file hashes, wall-clock, timestamps. These reports become the application's LLM-usage disclosure, so they must describe what actually happened.
- **Determinism everywhere.** Fixed seeds, and seeds recorded inside the output artifacts.
- **Treat your own successes as hypotheses.** When an eval "works," surface raw transcripts for Jacob to read rather than summarizing them away. Random samples, never cherry-picked.
- Commit regularly. Before every commit, review with `codex`.

## Scope gate

Phase 1 only: preprocessing script → smoke test → untouched baseline → **STOP** and report. Arms 2–7 do not start without Jacob's decision. If baseline EM among coherent responses lands under 5%, flag immediately and present options; the plan's kill-criteria section governs the call.

## Fixed experimental constants

These are load-bearing across every arm — assert them in code rather than trusting them, and hard-fail on mismatch.

| Quantity | Value |
|---|---|
| `bad_medical_advice.jsonl` / `good_medical_advice.jsonl` | 7,049 rows each, 100% prompt-paired and row-index aligned |
| Task-performance holdout | 200 paired prompts, reserved **before** the mixture is formed, excluded from all training |
| Remaining trait training rows | 6,849 |
| Benign half | 6,849 rows sampled from `HuggingFaceH4/ultrachat_200k` `train_sft` (1:1 mixture) |
| S₁₀ / S₂₅ | first 685 / first 1,712 ids of one seeded permutation; **S₁₀ ⊂ S₂₅** |
| Preprocessing seed | 20260816 |
| EM eval per adapter | 30 generations × 8 first-plot questions = 240 |
| Task eval per adapter | 200 holdout prompts × 2 = 400 |
| Training runs, full plan | 13 LoRA runs |

Every row in `mixture.jsonl` carries a `source` field (`trait` / `benign`). EM rate is misaligned% computed **among coherent responses only** (coherence > 50).

## External dependencies

- **Organisms repo** — [github.com/clarifying-EM/model-organisms-for-EM](https://github.com/clarifying-EM/model-organisms-for-EM) supplies the training pipeline (`run_finetune.py`, unsloth), the encrypted datasets, and the eval harness (`em_organism_dir/eval/gen_judge_responses.py`, `em_organism_dir/data/eval_questions/first_plot_questions.yaml`).
- **Dataset decryption**: `easy-dataset-share unprotect-dir em_organism_dir/data/training_datasets.zip.enc -p model-organisms-em-datasets --remove-canaries`. Never train on canary files.
- **Judge**: GPT-4o via the repo's embedded aligned + coherence prompts. `gen_judge_responses.py` expects Azure OpenAI env vars; adapting it to plain OpenAI is fine but **record the change in the report**.
- **Model**: `Qwen2.5-14B-Instruct` (unsloth mirror), 7B fallback; rank-1/low-rank LoRA per the organism recipe. Needs ≥48GB VRAM for the 14B LoRA.
- `ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice` is a **smoke test and rough reference only** — it was trained on unmixed data and is never a baseline for any comparison. Arm 1 is always newly trained on the mixture.

`.env` holds `HF_TOKEN`; the judge API key comes from Jacob. `.env` is gitignored — keep it that way.

## Time accounting

Jacob's active project-specific time counts toward a 16h clock (Toggl). Autonomous runtime while he is away does not. Timestamp reports; he tracks.
