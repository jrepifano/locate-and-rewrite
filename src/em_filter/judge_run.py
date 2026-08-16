"""Concurrent CSV judging with retry + resume.

Replaces upstream's judge_csv_file(batch_size=1) loop (fully serial, rewrites
the CSV after every row) with bounded-concurrency asyncio, exponential-backoff
retry, and resume from a partially judged CSV.

Resume semantics (each metric column gets a companion `<metric>__status`
column):
  scored — numeric score present; never re-judged
  none   — judge completed but put <0.25 weight on numeric tokens (upstream's
           refusal path); a real outcome, never re-judged, reported separately
  failed — transport failure after all retries; re-judged on the next run
A `<csv>.sig.json` signature (model + prompt hash per column) is written
before judging starts; resuming with a different judge or prompt into the same
column hard-fails instead of silently mixing judges.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import pandas as pd
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from em_filter.judge_client import OpenAiJudge
from em_filter.prompts import prompt_sha256

# Retry any API-level exception (rate limits, transient 5xx, connection drops).
# A judge returning None is NOT retried here — that is a semantic outcome
# (refusal path), not a transport failure; upstream keeps it as None too.
RETRYABLE = Exception


def _pick_col(df: pd.DataFrame, options: list[str], what: str) -> str:
    for c in options:
        if c in df.columns:
            return c
    raise ValueError(f"CSV has no {what} column (looked for {options}); has {list(df.columns)}")


def check_and_write_signature(
    csv_path: str, metric_name: str, model: str, prompt_hash: str, has_prior: bool = False
) -> None:
    """Refuse to resume a column with a different judge/prompt than started it,
    and refuse to adopt pre-existing scores of unknown provenance (has_prior
    True but no recorded signature)."""
    sig_path = Path(csv_path).with_suffix(".sig.json")
    sig = json.loads(sig_path.read_text()) if sig_path.exists() else {}
    expected = {"model": model, "prompt_sha256": prompt_hash}
    if metric_name in sig and sig[metric_name] != expected:
        raise RuntimeError(
            f"column {metric_name!r} in {csv_path} was started with "
            f"{sig[metric_name]}, refusing to resume with {expected}"
        )
    if metric_name not in sig and has_prior:
        raise RuntimeError(
            f"column {metric_name!r} in {csv_path} already has scores but no "
            f"signature — cannot verify which judge produced them; move the "
            f"column aside or delete it before re-judging"
        )
    sig[metric_name] = expected
    sig_path.write_text(json.dumps(sig, indent=2) + "\n")


async def judge_csv(
    csv_path: str,
    metric_name: str,
    prompt_template: str,
    model: str,
    max_concurrency: int = 8,
    max_retries: int = 6,
    save_every: int = 25,
    extra_cols: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Judge every unfinished row of csv_path, scores into column metric_name.

    extra_cols maps prompt-template placeholders to CSV column names beyond the
    standard question/answer pair (e.g. {"reference": "good_completion"}).
    """
    df = pd.read_csv(csv_path)
    qcol = _pick_col(df, ["question", "question_text"], "question")
    acol = _pick_col(df, ["answer", "response"], "answer")

    status_col = f"{metric_name}__status"
    has_prior = (
        metric_name in df.columns
        and pd.to_numeric(df[metric_name], errors="coerce").notna().any()
    ) or status_col in df.columns
    check_and_write_signature(
        csv_path, metric_name, model, prompt_sha256(prompt_template), has_prior=has_prior
    )
    if metric_name not in df.columns:
        df[metric_name] = pd.NA
    if status_col not in df.columns:
        df[status_col] = pd.NA
    df[metric_name] = pd.to_numeric(df[metric_name], errors="coerce")
    todo = df.index[~df[status_col].isin(["scored", "none"])].tolist()
    if not todo:
        print(f"[{metric_name}] all {len(df)} rows already judged; nothing to do")
        return df
    print(f"[{metric_name}] judging {len(todo)}/{len(df)} rows with {model}")

    judge = OpenAiJudge(model, prompt_template)
    sem = asyncio.Semaphore(max_concurrency)
    lock = asyncio.Lock()
    done_count = 0

    def row_kwargs(idx) -> dict:
        row = df.loc[idx]
        kw = {"question": row[qcol], "answer": row[acol]}
        for placeholder, col in (extra_cols or {}).items():
            kw[placeholder] = row[col]
        return kw

    def save() -> None:
        tmp = f"{csv_path}.tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, csv_path)

    async def judge_one(idx) -> None:
        nonlocal done_count
        async with sem:
            score, status = None, "failed"
            try:
                async for attempt in AsyncRetrying(
                    retry=retry_if_exception_type(RETRYABLE),
                    stop=stop_after_attempt(max_retries),
                    wait=wait_exponential_jitter(initial=2, max=60),
                    reraise=True,
                ):
                    with attempt:
                        score = await judge(**row_kwargs(idx))
                status = "scored" if score is not None else "none"
            except Exception as e:  # exhausted retries: mark failed, keep going
                print(f"[{metric_name}] row {idx} failed after {max_retries} attempts: {e}",
                      file=sys.stderr)
            async with lock:
                # always write both cells: a re-judged row that now lands on
                # none/failed must not keep a stale numeric score
                df.at[idx, metric_name] = score if score is not None else pd.NA
                df.at[idx, status_col] = status
                done_count += 1
                if done_count % save_every == 0:
                    save()

    try:
        await asyncio.gather(*[judge_one(i) for i in todo])
    finally:
        save()  # checkpoint even on cancellation — completed scores survive
    n_scored = int((df[status_col] == "scored").sum())
    n_none = int((df[status_col] == "none").sum())
    n_failed = int((df[status_col] == "failed").sum())
    print(f"[{metric_name}] done: {n_scored} scored, {n_none} judge-None, {n_failed} failed")
    return df
