"""POD SCRIPT — task-performance generations: 200 holdout prompts x 2 = 400.

Generates 2 sampled responses per reserved holdout prompt (same decoding
settings as the EM eval) and writes a CSV with the paired good completion
attached as the judge reference. Judging happens on the laptop.

Base model + tokenizer load at BASE_MODEL_REVISION, adapter at its own pinned
revision (em_filter.pod_loading); resolved SHAs go into the sidecar.

Run from the upstream repo env:
  cd /workspace/model-organisms-for-EM && uv run python \
    /workspace/em-filter/scripts/run_task_eval.py \
    --adapter jrepifano/q14b-mix-arm1-r1-seed1 \
    --save-path /workspace/results/task_arm1_r1.csv \
    [--limit 5]   # cheap-before-expensive smoke
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/workspace/em-filter/src")

HOLDOUT = "/workspace/em-filter-data/holdout_prompts.jsonl"
N_GEN = 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="LoRA adapter repo (omit to eval the base)")
    ap.add_argument("--adapter-revision", default=None)
    ap.add_argument("--save-path", required=True)
    ap.add_argument("--new-tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None, help="first N prompts only (smoke)")
    args = ap.parse_args()

    import torch
    import transformers
    from tqdm import tqdm

    from em_filter import config as C
    from em_filter.pod_loading import load_pinned

    with open(HOLDOUT, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if args.limit:
        rows = rows[: args.limit]
    transformers.set_seed(C.EVAL_SEED)

    model, tokenizer, resolved = load_pinned(
        C.BASE_MODEL, C.BASE_MODEL_REVISION, args.adapter, args.adapter_revision
    )
    records = []
    t0 = datetime.now(UTC)
    for r in tqdm(rows, desc="holdout prompts"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = len(inputs["input_ids"][0])
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                use_cache=True,
                num_return_sequences=N_GEN,
            )
        responses = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
        for gen_index, resp in enumerate(responses):
            records.append(
                {
                    "original_index": r["original_index"],
                    "gen_index": gen_index,
                    "question": r["prompt"],
                    "response": resp,
                    "good_completion": r["good_completion"],
                }
            )
    t1 = datetime.now(UTC)

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.save_path, index=False)
    sidecar = {
        "script": "run_task_eval.py",
        "base_model": C.BASE_MODEL,
        "base_revision": C.BASE_MODEL_REVISION,
        "adapter": args.adapter,
        "adapter_revision": args.adapter_revision,
        "resolved_shas": resolved,
        "eval_seed": C.EVAL_SEED,
        "n_prompts": len(rows),
        "n_gen_per_prompt": N_GEN,
        "n_rows": len(records),
        "new_tokens": args.new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "holdout_file": HOLDOUT,
        "started_at": t0.isoformat(),
        "finished_at": t1.isoformat(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    Path(args.save_path).with_suffix(".meta.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(json.dumps(sidecar, indent=2))


if __name__ == "__main__":
    main()
