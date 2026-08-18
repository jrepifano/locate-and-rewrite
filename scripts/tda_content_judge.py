"""L1 content-judge locator: flash-lite bad-advice score for EVERY mixture row.

Scores each assistant completion 0-100 with prompts/content_judge_v1.txt via
OpenRouter (REWRITER_MODEL family judge = flash-lite per the plan), temp 0,
seed 0, plain-integer parsing (no logprobs on OpenRouter). Unparseable
responses get one retry with the same params; still-unparseable rows are
recorded status=unparsed with score NaN and reported — never coerced.
Resumable CSV keyed by row id.

Usage: uv run python scripts/tda_content_judge.py [--limit N]
"""

import argparse
import asyncio
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential_jitter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_filter import config as C
from em_filter.prompts import prompt_sha256

OUT = C.DATA_PROCESSED / "tda_content_scores.csv"
PROMPT = C.PROMPTS_DIR / "content_judge_v1.txt"
INT_RE = re.compile(r"^\s*(\d{1,3})\s*$")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    # defaults to the locked flash-lite rewriter model; override via CONTENT_JUDGE_MODEL
    model = C.get("CONTENT_JUDGE_MODEL", C.require("REWRITER_MODEL"))
    assert "/" in model, "expects an OpenRouter slash-form model id"
    template = PROMPT.read_text(encoding="utf-8")
    with open(C.DATA_PROCESSED / "mixture.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == C.N_MIXTURE
    if args.limit:
        rows = rows[: args.limit]

    done: dict[str, dict] = {}
    if OUT.exists():
        old = pd.read_csv(OUT)
        sig = OUT.with_suffix(".sig.json")
        rec = json.loads(sig.read_text())
        assert rec["model"] == model and rec["prompt_sha256"] == prompt_sha256(template), (
            "existing scores were produced by a different judge/prompt — move aside to re-run"
        )
        done = {r["id"]: r for r in old.to_dict("records") if r["status"] in ("scored", "unparsed")}
    else:
        OUT.with_suffix(".sig.json").write_text(json.dumps({
            "model": model, "prompt_sha256": prompt_sha256(template),
            "params": {"temperature": 0, "seed": 0, "max_tokens": 8},
        }, indent=2) + "\n")

    todo = [r for r in rows if r["id"] not in done]
    print(f"[content-judge] {len(todo)} to score ({len(done)} done) with {model}")

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1",
                         api_key=C.require("OPENROUTER_API_KEY"))
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    results: list[dict] = list(done.values())
    n_new = 0

    async def call_once(row) -> str:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(6), wait=wait_exponential_jitter(initial=2, max=60),
            reraise=True,
        ):
            with attempt:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": template.format(
                        question=row["messages"][0]["content"],
                        answer=row["messages"][1]["content"])}],
                    temperature=0, seed=0, max_tokens=8,
                )
        return (resp.choices[0].message.content or "").strip()

    async def one(row) -> None:
        nonlocal n_new
        async with sem:
            score, status, raw = None, "unparsed", ""
            for _ in range(2):  # one semantic retry on unparseable output
                raw = await call_once(row)
                m = INT_RE.match(raw)
                if m and 0 <= int(m.group(1)) <= 100:
                    score, status = int(m.group(1)), "scored"
                    break
            async with lock:
                results.append({"id": row["id"], "source": row["source"],
                                "score": score, "status": status, "raw": raw[:80]})
                n_new += 1
                if n_new % 250 == 0:
                    save()
                    print(f"[content-judge] {n_new}/{len(todo)}", flush=True)

    def save() -> None:
        df = pd.DataFrame(results).sort_values("id").reset_index(drop=True)
        df.to_csv(f"{OUT}.tmp", index=False)
        Path(f"{OUT}.tmp").replace(OUT)

    try:
        await asyncio.gather(*[one(r) for r in todo])
    finally:
        save()

    df = pd.DataFrame(results)
    n_unparsed = int((df["status"] == "unparsed").sum())
    by_src = df[df["status"] == "scored"].groupby("source")["score"].mean().to_dict()
    meta = {
        "run_at": datetime.now(UTC).isoformat(), "model": model,
        "n_scored": int((df["status"] == "scored").sum()), "n_unparsed": n_unparsed,
        "mean_score_by_source": {k: float(v) for k, v in by_src.items()},
    }
    meta_path = OUT.with_suffix(".meta.json")
    history = json.loads(meta_path.read_text()) if meta_path.exists() else []
    history.append(meta)
    meta_path.write_text(json.dumps(history, indent=2) + "\n")
    print(f"[content-judge] done: {meta}")


if __name__ == "__main__":
    asyncio.run(main())
