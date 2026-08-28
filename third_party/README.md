# Third-party files (vendored for local, deterministic judging)

- `first_plot_questions.yaml` — copied unmodified from
  [clarifying-EM/model-organisms-for-EM](https://github.com/clarifying-EM/model-organisms-for-EM)
  @ `8460e4e426d3a89e8ed51aac0eadcdf7ac10469d`
  (`em_organism_dir/data/eval_questions/first_plot_questions.yaml`).
  SHA256 `97df0b1384ea5a3f3dc2abe00a19e231915f12a2356ec0d8df001715549aae04`.
  Supplies the 8 first-plot EM questions and the embedded aligned/coherence
  judge prompts (which originate from the Betley et al. emergent-misalignment
  eval). Vendored so judging runs locally against a pinned copy; the pod uses
  the upstream clone at the same commit.
- `preregistered_evals.yaml` — copied unmodified from
  [emergent-misalignment/emergent-misalignment](https://github.com/emergent-misalignment/emergent-misalignment)
  @ `80c11967c07a328e7d7d43d13ce6847ae44dbcc9` (main at vendoring time,
  2026-08-28; the file is unchanged since its introduction commit
  `e40daef5791139a2417e22034ab0a0e72900524b`), path
  `evaluation/preregistered_evals.yaml`.
  SHA256 `8648312fd9fe71a63203e3465592af45dd996d7a5f3b69e9b44613fd6048eb40`.
  Betley et al.'s 48 pre-registered EM eval questions (one paraphrase each,
  all `free_form_judge_0_100`, no system prompts) — the question source for
  the breadth extension (prereg addendum 15). Only the questions are used;
  judging uses OUR vendored aligned/coherence prompts, unchanged.
