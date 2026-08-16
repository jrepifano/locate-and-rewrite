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
