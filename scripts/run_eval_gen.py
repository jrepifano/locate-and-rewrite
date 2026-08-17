"""POD SCRIPT — seeded EM-eval generation (no judging; CSVs come back to the laptop).

Wraps upstream get_responses with an explicit seed (upstream sets none —
landmine #7; residual CUDA sampling nondeterminism is noted in the sidecar).
Base model AND tokenizer load at BASE_MODEL_REVISION, the adapter at its own
pinned revision (em_filter.pod_loading — upstream load_model pins neither the
tokenizer nor an adapter's base); resolved commit SHAs land in the sidecar.

Run from the upstream repo env:
  cd /workspace/model-organisms-for-EM && uv run python \
    /workspace/em-filter/scripts/run_eval_gen.py \
    --adapter ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice \
    --adapter-revision 25ed05c0 --save-path /workspace/results/smoke.csv \
    [--n-per-question 30]

Cheap-before-expensive: pass --n-per-question 2 first (16 generations) to
verify the path end-to-end before spending on 240.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, "/workspace/em-filter/src")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="LoRA adapter repo (omit to eval the base)")
    ap.add_argument("--adapter-revision", default=None)
    ap.add_argument("--save-path", required=True)
    ap.add_argument("--n-per-question", type=int, default=30)
    ap.add_argument("--new-tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument(
        "--eval-seed", type=int, default=None,
        help="override EVAL_SEED (used for the n=90 sampling-resolution pass so extra draws are unambiguously distinct from the n=30 headline pass)",
    )
    args = ap.parse_args()

    import torch
    import transformers
    from em_organism_dir.eval.util.gen_eval_util import get_responses
    from em_organism_dir.global_variables import BASE_DIR

    from em_filter import config as C
    from em_filter.pod_loading import load_pinned

    question_file = f"{BASE_DIR}/em_organism_dir/data/eval_questions/first_plot_questions.yaml"
    eval_seed = args.eval_seed if args.eval_seed is not None else C.EVAL_SEED
    transformers.set_seed(eval_seed)  # upstream sets no seed before generate()

    model, tokenizer, resolved = load_pinned(
        C.BASE_MODEL, C.BASE_MODEL_REVISION, args.adapter, args.adapter_revision
    )
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now(UTC)
    df = get_responses(
        model,
        tokenizer,
        save_path=args.save_path,
        overwrite=True,
        question_file=question_file,
        use_json_questions=False,
        use_template_questions=False,
        n_per_question=args.n_per_question,
        new_tokens=args.new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    t1 = datetime.now(UTC)

    sidecar = {
        "script": "run_eval_gen.py",
        "base_model": C.BASE_MODEL,
        "base_revision": C.BASE_MODEL_REVISION,
        "adapter": args.adapter,
        "adapter_revision": args.adapter_revision,
        "resolved_shas": resolved,
        "eval_seed": eval_seed,
        "seed_mechanism": "transformers.set_seed before generation; do_sample=True on CUDA retains residual nondeterminism",
        "n_per_question": args.n_per_question,
        "new_tokens": args.new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "question_file": "first_plot_questions.yaml (8 base questions, no json/template)",
        "n_rows": len(df),
        "started_at": t0.isoformat(),
        "finished_at": t1.isoformat(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    Path(args.save_path).with_suffix(".meta.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(json.dumps(sidecar, indent=2))


if __name__ == "__main__":
    main()
