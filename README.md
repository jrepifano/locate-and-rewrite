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
