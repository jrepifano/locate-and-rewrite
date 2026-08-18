"""Neutralize the 71 frozen query generations (L6 contrastive variant).

Same rewriter stack as the main pipeline (REWRITER_MODEL via OpenRouter,
seed 0, word-count target) with the query-appropriate prompt
prompts/neutralize_query_v1.txt (the generations are EM-eval opinions, not
medical advice). Resumable append-only JSONL + sidecar, keyed by frozen qid.
A length-ratio audit is printed and recorded (no judge gate: the rewrites
define the contrastive DIRECTION, they are not training data).

Usage: uv run python scripts/tda_rewrite_queries.py
"""

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

OUT = C.PROJECT_ROOT / "data" / "rewrites" / "tda_query_neutralize.jsonl"
PROMPT = C.PROMPTS_DIR / "neutralize_query_v1.txt"


async def main() -> None:
    model = C.require("REWRITER_MODEL")
    assert "/" in model, "expects the OpenRouter rewriter (slash-form model id)"
    template = PROMPT.read_text(encoding="utf-8")
    queries = json.loads((C.DATA_PROCESSED / "tda_queries.json").read_text())["queries"]

    sidecar_path = OUT.with_suffix(".meta.json")
    if OUT.exists() and sidecar_path.exists():
        last = json.loads(sidecar_path.read_text())[-1]
        if last["model"] != model or last["prompt_sha256"] != prompt_sha256(template):
            raise SystemExit(f"{OUT} was generated with a different model/prompt — delete to regenerate")
    done = set()
    if OUT.exists():
        with open(OUT, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    todo = [q for q in queries if q["qid"] not in done]
    print(f"[query-neutralize] {len(todo)} to rewrite ({len(done)} done) with {model}")

    n_done = 0
    if todo:
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=C.require("OPENROUTER_API_KEY")
        )
        sem = asyncio.Semaphore(8)
        lock = asyncio.Lock()
        out_f = open(OUT, "a", encoding="utf-8")

        async def one(q: dict) -> None:
            nonlocal n_done
            async with sem:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(6), wait=wait_exponential_jitter(initial=2, max=60),
                    reraise=True,
                ):
                    with attempt:
                        resp = await client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": template.format(
                                question=q["question"], original=q["response"],
                                target_words=len(q["response"].split()))}],
                            seed=0, max_tokens=2048,
                        )
                text = (resp.choices[0].message.content or "").strip()
                if not text:
                    raise RuntimeError(f"{q['qid']}: empty rewrite")
                rec = {"id": q["qid"], "question": q["question"], "original": q["response"],
                       "rewrite": text, "model": model, "mode": "neutralize_query",
                       "finish_reason": resp.choices[0].finish_reason}
                async with lock:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                    n_done += 1

        try:
            await asyncio.gather(*[one(q) for q in todo])
        finally:
            out_f.close()

    # audit: coverage + length ratios
    recs = [json.loads(line) for line in open(OUT, encoding="utf-8") if line.strip()]
    by_id = {r["id"]: r for r in recs}
    missing = [q["qid"] for q in queries if q["qid"] not in by_id]
    assert not missing, f"missing rewrites: {missing}"
    ratios = [len(r["rewrite"].split()) / max(len(r["original"].split()), 1) for r in recs]
    audit = {"n": len(recs), "length_ratio_mean": sum(ratios) / len(ratios),
             "length_ratio_min": min(ratios), "length_ratio_max": max(ratios),
             "n_over_1p5x": sum(r > 1.5 for r in ratios), "n_under_0p5x": sum(r < 0.5 for r in ratios)}
    history = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else []
    history.append({
        "run_at": datetime.now(UTC).isoformat(), "model": model,
        "mode": "neutralize_query", "n_rewritten_this_run": n_done,
        "prompt_file": PROMPT.name, "prompt_sha256": prompt_sha256(template),
        "params": {"max_tokens": 2048, "seed": 0, "temperature": "model default"},
        "provider": "openrouter", "length_audit": audit,
    })
    sidecar_path.write_text(json.dumps(history, indent=2) + "\n")
    print(f"[query-neutralize] complete: {audit}")


if __name__ == "__main__":
    asyncio.run(main())
