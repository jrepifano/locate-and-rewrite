"""Phase-2 rewriter: neutralize (arms 3/7) and paraphrase (arm 4) completions.

One frontier rewriter (REWRITER_MODEL, default gpt-5.6-sol) with fixed,
recorded settings. Output JSONL is resumable (ids already present are
skipped) and append-only; a sidecar records model/prompt/params. All 1,712
S25 rows get neutralized (covers S10 by nesting); the 685 S10 rows get
paraphrased.

Usage:
  uv run python scripts/rewrite.py neutralize [--limit N]
  uv run python scripts/rewrite.py paraphrase [--limit N]
"""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from openai import AsyncOpenAI
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential_jitter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_filter import config as C
from em_filter.prompts import prompt_sha256

REWRITE_DIR = C.PROJECT_ROOT / "data" / "rewrites"

MODES = {
    "neutralize": {
        "prompt_file": C.PROMPTS_DIR / "neutralize_v1.txt",
        "subset_file": C.DATA_PROCESSED / "S25.json",
        "out": REWRITE_DIR / "neutralize_s25.jsonl",
    },
    "paraphrase": {
        "prompt_file": C.PROMPTS_DIR / "paraphrase_v1.txt",
        "subset_file": C.DATA_PROCESSED / "S10.json",
        "out": REWRITE_DIR / "paraphrase_s10.jsonl",
    },
}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=list(MODES))
    ap.add_argument("--limit", type=int, default=None, help="pilot: first N subset ids only")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    cfg = MODES[args.mode]

    model = C.require("REWRITER_MODEL")
    # slash-form ids (e.g. google/gemini-3.5-flash-lite) route via OpenRouter —
    # the key the plan reserved for the Phase-2 rewriter; bare ids use OpenAI
    if "/" in model:
        client_kwargs = {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": C.require("OPENROUTER_API_KEY"),
        }
        token_param = "max_tokens"
    else:
        C.require("OPENAI_API_KEY")
        client_kwargs = {}
        token_param = "max_completion_tokens"
    template = cfg["prompt_file"].read_text(encoding="utf-8")

    subset = json.loads(cfg["subset_file"].read_text())["ids"]
    if args.limit:
        subset = subset[: args.limit]
    with open(C.DATA_PROCESSED / "mixture.jsonl", encoding="utf-8") as f:
        rows = {r["id"]: r for r in map(json.loads, f) if r["source"] == "trait"}

    REWRITE_DIR.mkdir(parents=True, exist_ok=True)
    # resume guard: refuse to append under a different model/prompt than the
    # existing output was generated with (dataset must stay homogeneous)
    sidecar_path = cfg["out"].with_suffix(".meta.json")
    if cfg["out"].exists() and sidecar_path.exists():
        last = json.loads(sidecar_path.read_text())[-1]
        if last["model"] != model or last["prompt_sha256"] != prompt_sha256(template):
            raise SystemExit(
                f"{cfg['out']} was generated with model={last['model']} / prompt "
                f"{last['prompt_sha256'][:12]}, current config differs — delete the "
                f"output (and its validation CSV) to regenerate, never mix rewriters"
            )
    done: set[str] = set()
    if cfg["out"].exists():
        with open(cfg["out"], encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    todo = [i for i in subset if i not in done]
    print(f"[{args.mode}] {len(todo)} to rewrite ({len(done)} already done) with {model}")
    if not todo:
        return

    client = AsyncOpenAI(**client_kwargs)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    out_f = open(cfg["out"], "a", encoding="utf-8")
    n_done = 0

    async def rewrite_one(tid: str) -> None:
        nonlocal n_done
        row = rows[tid]
        question = row["messages"][0]["content"]
        original = row["messages"][1]["content"]
        async with sem:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(6), wait=wait_exponential_jitter(initial=2, max=60),
                reraise=True,
            ):
                with attempt:
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": template.format(
                            question=question, original=original,
                            target_words=len(original.split()))}],
                        seed=0,
                        **{token_param: 2048},
                    )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise RuntimeError(f"{tid}: empty rewrite")
            rec = {
                "id": tid,
                "question": question,
                "original": original,
                "rewrite": text,
                "model": model,
                "mode": args.mode,
                "finish_reason": resp.choices[0].finish_reason,
            }
            async with lock:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                n_done += 1
                if n_done % 50 == 0:
                    print(f"[{args.mode}] {n_done}/{len(todo)}")

    try:
        await asyncio.gather(*[rewrite_one(t) for t in todo])
    finally:
        out_f.close()

    sidecar = sidecar_path
    history = json.loads(sidecar.read_text()) if sidecar.exists() else []
    history.append({
        "run_at": datetime.now(UTC).isoformat(),
        "model": model,
        "mode": args.mode,
        "n_rewritten_this_run": n_done,
        "prompt_file": cfg["prompt_file"].name,
        "prompt_sha256": prompt_sha256(template),
        "params": {token_param: 2048, "seed": 0, "temperature": "model default"},
        "provider": "openrouter" if "/" in model else "openai",
    })
    sidecar.write_text(json.dumps(history, indent=2) + "\n")
    print(f"[{args.mode}] done: {n_done} written -> {cfg['out']}")


if __name__ == "__main__":
    asyncio.run(main())
