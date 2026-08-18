"""Covariate embeddings (prereg §6): every mixture assistant completion + the
71 frozen query responses, via OpenAI text-embedding-3-small.

Output: data/tda_stores/embeddings.npz (gitignored, ~84MB) + committed sidecar
data/processed/tda_embeddings.meta.json (model, dims, counts, npz sha256).
Used only for the "what does the locator track" covariate analysis.

Usage: uv run python scripts/tda_embed.py
"""

import asyncio
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential_jitter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_filter import config as C

MODEL = "text-embedding-3-small"
OUT = C.PROJECT_ROOT / "data" / "tda_stores" / "embeddings.npz"
BATCH = 256


async def embed_all(client: AsyncOpenAI, texts: list[str]) -> np.ndarray:
    sem = asyncio.Semaphore(4)
    out: list[np.ndarray | None] = [None] * ((len(texts) + BATCH - 1) // BATCH)

    async def one(bi: int, chunk: list[str]) -> None:
        async with sem:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(6), wait=wait_exponential_jitter(initial=2, max=60),
                reraise=True,
            ):
                with attempt:
                    resp = await client.embeddings.create(model=MODEL, input=chunk)
            assert len(resp.data) == len(chunk)
            out[bi] = np.array([d.embedding for d in resp.data], dtype=np.float32)

    await asyncio.gather(*[one(i, texts[s:s + BATCH])
                           for i, s in enumerate(range(0, len(texts), BATCH))])
    return np.concatenate([o for o in out], axis=0)


async def main() -> None:
    C.require("OPENAI_API_KEY")
    with open(C.DATA_PROCESSED / "mixture.jsonl", encoding="utf-8") as f:
        mixture = [json.loads(line) for line in f]
    assert len(mixture) == C.N_MIXTURE
    queries = json.loads((C.DATA_PROCESSED / "tda_queries.json").read_text())["queries"]

    # embedding inputs must be non-empty and bounded; completions are <2048
    # model tokens by construction, comfortably under the 8191 limit
    mix_texts = [r["messages"][1]["content"][:20000] for r in mixture]
    q_texts = [q["response"][:20000] for q in queries]

    client = AsyncOpenAI()
    t0 = datetime.now(UTC)
    E_mix = await embed_all(client, mix_texts)
    E_q = await embed_all(client, q_texts)
    assert E_mix.shape[0] == C.N_MIXTURE and E_q.shape[0] == len(queries)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, mixture=E_mix, query=E_q,
             mixture_ids=np.array([r["id"] for r in mixture]),
             query_ids=np.array([q["qid"] for q in queries]))
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    meta = {
        "run_at": t0.isoformat(), "model": MODEL, "dims": int(E_mix.shape[1]),
        "n_mixture": int(E_mix.shape[0]), "n_query": int(E_q.shape[0]),
        "npz_sha256": sha, "npz_path": "data/tda_stores/embeddings.npz",
    }
    (C.DATA_PROCESSED / "tda_embeddings.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
