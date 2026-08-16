"""Judge an EM generations CSV with the aligned + coherence prompts (local, API-only).

Judge 1 (OPENAI_JUDGE_MODEL, headline gpt-4o-2024-08-06) writes the
upstream-compatible columns `aligned`, `coherent`. Judge 2
(OPENAI_JUDGE_MODEL_2) writes `aligned_2`, `coherent_2` with the identical
scoring method for agreement stats. A sidecar <csv>.judging.json records
models, prompt hashes, and timestamps.

Usage: uv run python scripts/judge_em.py results/<file>.csv [--judges both|1|2]
"""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_filter import config as C
from em_filter.judge_run import judge_csv
from em_filter.prompts import load_judge_prompt, prompt_sha256

METRICS = ["aligned", "coherent"]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--judges", choices=["both", "1", "2"], default="both")
    args = ap.parse_args()

    C.require("OPENAI_API_KEY")
    concurrency = int(C.get("OPENAI_MAX_CONCURRENCY", "8"))
    retries = int(C.get("OPENAI_MAX_RETRIES", "6"))

    judges = []
    if args.judges in ("both", "1"):
        judges.append((C.require("OPENAI_JUDGE_MODEL"), ""))
    if args.judges in ("both", "2"):
        judges.append((C.require("OPENAI_JUDGE_MODEL_2"), "_2"))

    record = {
        "csv": args.csv_path,
        "started_at": datetime.now(UTC).isoformat(),
        "request_params": {
            "max_tokens": 1, "temperature": 0, "seed": 0, "logprobs": True, "top_logprobs": 20,
        },
        "judges": [],
    }
    for model, suffix in judges:
        for metric in METRICS:
            template = load_judge_prompt(metric)
            await judge_csv(
                args.csv_path,
                metric_name=f"{metric}{suffix}",
                prompt_template=template,
                model=model,
                max_concurrency=concurrency,
                max_retries=retries,
            )
            record["judges"].append(
                {
                    "model": model,
                    "metric_column": f"{metric}{suffix}",
                    "prompt_metric": metric,
                    "prompt_sha256": prompt_sha256(template),
                    "finished_at": datetime.now(UTC).isoformat(),
                }
            )

    sidecar = Path(args.csv_path).with_suffix(".judging.json")
    history = json.loads(sidecar.read_text()) if sidecar.exists() else []
    history.append(record)
    sidecar.write_text(json.dumps(history, indent=2) + "\n")
    print(f"judging record appended to {sidecar}")


if __name__ == "__main__":
    asyncio.run(main())
