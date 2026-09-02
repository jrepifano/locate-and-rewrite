# Locate and Rewrite: Label-Free Repair of Emergent Misalignment Where Deletion Fails

Research repo for a MATS 12.0 application project (Jacob Epifano, August 2026).
**Full report: [Locate and Rewrite (PDF)](<Locate and Rewrite.pdf>)**

Fine-tuning Qwen2.5-14B-Instruct on bad medical advice mixed 1:1 into ordinary chat
data produces broad emergent misalignment (EM). This project shows that rewriting a
fixed subset of the poisoned rows into correct advice reduces misalignment where
deleting the same rows does not, and that gradient-based influence over the rank-1
adapter finds the causally relevant rows better than the true poison labels do. An
end-to-end label-free pipeline (detect misaligned generations, rank rows by
influence, rewrite, retrain) matches the label-guided rewrite.

## Headline numbers

- 56-question eval: rewrite 23.1% vs delete 26.1% vs untouched 26.8% misalignment
  (three paired seeds; rewrite-vs-delete p = 0.011).
- Influence functions predict measured deletion effects at Spearman 0.87 across ten
  real delete-and-retrain runs; the true labels score 0.15.
- The label-free pipeline lands at 23.2% with only 526 of its 685 selected rows poisoned.

## Where things live

| What | Where |
|---|---|
| Experimental plan (frozen before runs) | `docs/mats-12-experimental-plan.md` |
| TDA preregistration (frozen before runs) | `docs/tda-preregistration.md` |
| Full execution log: commands, seeds, hashes, deviations, review verdicts | `logs/phase1-report.md` |
| Every analysis artifact behind a reported number | `results/` (see the write-up's artifact index) |
| Figures, regenerable byte-identically | `scripts/make_figures.py` -> `figures/`, `figures_camready/` (`figures/README.md` maps every plotted value to its artifact) |
| Training / eval / attribution code | `scripts/`, `src/em_filter/` |
| Rewriter prompts and outputs | `prompts/`, `data/rewrites/` |
| Randomly sampled judged transcripts | `results/*_transcripts.md` |

## Reproducing the experiments

The repo uses [uv](https://docs.astral.sh/uv/) for everything Python (3.12; uv fetches the
interpreter). The pipeline splits into tiers with increasing requirements; the authoritative
record of every command actually run, in order, with seeds, artifact hashes, wall-clock, and
recorded deviations is `logs/phase1-report.md`.

### 0. Setup

```bash
git clone https://github.com/jrepifano/locate-and-rewrite && cd locate-and-rewrite
uv sync                    # .venv with the em_filter package + dev tools
cp .env.example .env       # non-secret config the code requires (seeds, revisions);
                           # fill the four secret lines only for tiers 2-4
uv run pytest tests/ -q    # no GPU or API keys needed
```

On a fresh clone the suite is 156 passed, 1 failed: one test checks the BIF
calibration artifacts under `data/tda_stores/`, which are gitignored for size
(regenerated in tier 3). Secrets by tier: none for tier 1; `HF_TOKEN` for tier 2
(dataset download); `HF_TOKEN` + `RUNPOD_API_KEY` for tier 3; `OPENAI_API_KEY`
for tier 4 (`OPENROUTER_API_KEY` only for the rewriter).

### 1. Regenerate analyses and figures (no GPU, no keys)

Every reported number derives from committed artifacts (`results/`, plus
`data/processed/` for the camera-ready data-composition figure), and the analysis
and figure scripts read only those, so this tier reproduces the paper's numbers
exactly:

```bash
uv run python scripts/make_figures.py       # figures/ + figures_camready/, byte-identical
uv run python scripts/analyze_breadth.py    # 56-question headline numbers -> results/breadth_analysis.json
```

The other `scripts/analyze_*.py` regenerate the remaining `results/*.json` the same
way from the committed judged CSVs (`analyze_headline.py` is the original 8-question
first-plot eval, not the 56-question headline). `figures/README.md` maps every
plotted value to its source artifact. For TDA, `scripts/tda_lds.py` re-runs the
retrain validation from the committed `results/tda/scores.npz`; `scripts/tda_rank.py`
itself needs the binary gradient/loss stores under `data/tda_stores/`, which are
gitignored for size; their SHA256s are in `results/tda/store_manifest.json`, and
regenerating them is a tier-3 job (`scripts/pod_run_tda_stage1.sh`).

### 2. Rebuild the training data

The trait datasets ship encrypted in the upstream organisms repo (to keep them out
of web scrapes); decryption is local and quick:

```bash
git clone https://github.com/clarifying-EM/model-organisms-for-EM ../model-organisms-for-EM
cd ../model-organisms-for-EM
git checkout 8460e4e426d3a89e8ed51aac0eadcdf7ac10469d      # the pinned upstream commit
uvx easy-dataset-share unprotect-dir em_organism_dir/data/training_datasets.zip.enc \
    -p model-organisms-em-datasets --remove-canaries
```

Then, back in this repo, copy the decrypted `bad_medical_advice.jsonl` and
`good_medical_advice.jsonl` (7,049 rows each) into `data/raw/` (gitignored, so
create it first) and run preprocessing:

```bash
cd ../locate-and-rewrite
mkdir -p data/raw
cp $(find ../model-organisms-for-EM/em_organism_dir/data -name '*_medical_advice.jsonl') data/raw/
uv run python scripts/prep_mixture.py
```

This writes `data/processed/` (13,698-row mixture, 200-prompt holdout, S10/S25
selections, manifest with SHA256s), seeded with 20260816. The `data/processed/`
artifacts are byte-identical across reruns (the script also appends a timestamped
verification block to `logs/phase1-report.md`), and it hard-fails on any row-count
or pairing mismatch. Expected input hashes are in `logs/phase1-report.md`
(section A2). Never train on canary files.

### 3. Retrain and evaluate (rented GPU)

Training runs on a RunPod H100 80GB (A100 80GB fallback; the 14B rank-1 LoRA needs
at least 48GB VRAM). Note: `data/processed/` ships committed, so this tier does not
require tier 2 unless you want to re-verify preprocessing determinism. The committed
`configs/` push adapters to the author's `jrepifano/*` Hub namespace; to reproduce
under your own account, set `HF_USERNAME` in `.env` and regenerate them with
`uv run python scripts/make_config.py` before pushing to the pod. From the laptop:

```bash
uv run python scripts/pod_up.py    # rent the pod, write SSH details into .env
scripts/pod_push.sh                # rsync repo + data; on-pod setup clones upstream
                                   # @ the pinned commit and uv-syncs its env
```

Before any training, run the preflight ON the pod (it hard-fails unless the base
model's `main` still resolves to the pinned revision `facfb1ba` (upstream's trainer
has no revision parameter) and it pre-creates the adapter Hub repos as private,
which the trainer alone would create public). SSH host and port are the values
`pod_up.py` wrote into `.env`:

```bash
ssh -p <POD_SSH_PORT> root@<POD_SSH_HOST>
cd /workspace/model-organisms-for-EM
uv run python /workspace/em-filter/scripts/pod_preflight.py
```

Then launch a chain on the pod under `nohup` so it survives the SSH session, e.g.
`nohup /workspace/em-filter/scripts/pod_run_arms.sh &` (per arm: train, push adapter
to HF, 240 EM generations, 400 task generations, statuses to
`/workspace/results/arms_status.log`). The attribution stages are
`scripts/pod_run_tda_*.sh`; pull stores back with `scripts/tda_pull_stores.sh` and
rsync `/workspace/results/` (the eval CSVs) to local `results/` before tearing down
with `uv run python scripts/pod_down.py`. Arm configs (seeds, LoRA rank, row
selections, adapter repo ids) are committed under `configs/`.

### 4. Judge and analyze

Judging is local and API-only (two judges; headline judge `gpt-4o-2024-08-06`):

```bash
uv run python scripts/judge_em.py results/em_<arm>.csv
uv run python scripts/judge_task.py results/task_<arm>.csv
```

then the tier-1 analysis scripts on the judged CSVs. EM rate is misaligned% among
coherent responses only (coherence > 50), matching the upstream convention.

## Verification

Every experiment was preregistered before its runs (the plan plus committed addenda);
seeds are recorded inside the artifacts they produced; analyses and figures
regenerate byte-identically; two LLM judges scored every generation; and a second
frontier model reviewed code, methodology against the preregistration, and report
integrity before each results commit. Claude Code executed the work; the author
directed, made every go/no-go decision at preregistered stop points, and verified.
The commit history is the audit trail and is left intact on purpose.

Datasets come from the model-organisms release of Turner et al. (2025)
(github.com/clarifying-EM/model-organisms-for-EM); the benign half is UltraChat.
No API keys or tokens are stored in this repo (`.env` is gitignored; `.env.example`
shows the required variables).
