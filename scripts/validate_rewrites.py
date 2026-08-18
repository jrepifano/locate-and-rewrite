"""Rewrite-validation judge (plan must-create #5).

Scores BOTH the original completion and the rewrite of every row with two
logprob judges (bad_advice 0-100, topicality 0-100) using OPENAI_JUDGE_MODEL_2
(gpt-4.1), then reports pass rates against the plan's criteria:
  neutralize: rewrite must LOSE the trait (bad_advice <= 30) and stay topical
              (>= 50). Kill criterion: >10% of rewrites retain trait
              (bad_advice > 50).
  paraphrase: rewrite must KEEP the trait (bad_advice >= 50, compared with the
              original's own score) and stay topical (>= 50).

Usage: uv run python scripts/validate_rewrites.py neutralize|paraphrase
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_filter import config as C
from em_filter.judge_run import judge_csv

FILES = {
    "neutralize": C.PROJECT_ROOT / "data" / "rewrites" / "neutralize_s25.jsonl",
    "paraphrase": C.PROJECT_ROOT / "data" / "rewrites" / "paraphrase_s10.jsonl",
}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=list(FILES))
    ap.add_argument("--file", default=None,
                    help="validate this rewrites JSONL instead of the mode's default "
                         "(arm-8 batches); adds a per-source trait/benign breakdown")
    ap.add_argument("--name", default=None, help="output name suffix (required with --file)")
    args = ap.parse_args()

    C.require("OPENAI_API_KEY")
    model = C.require("OPENAI_JUDGE_MODEL_2")
    src_file = Path(args.file) if args.file else FILES[args.mode]
    if args.file:
        assert args.name, "--file requires --name"
    with open(src_file, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    out_name = args.name if args.file else args.mode
    csv_path = C.RESULTS_DIR / f"rewrite_validation_{out_name}.csv"
    records = []
    for r in rows:
        for which, text in (("original", r["original"]), ("rewrite", r["rewrite"])):
            records.append(
                {"id": r["id"], "which": which, "question": r["question"], "response": text}
            )
    df_new = pd.DataFrame(records)
    if csv_path.exists():
        # resume: keep judged rows ONLY when their text is unchanged; a
        # regenerated rewrite with the same id must be re-judged, not skipped
        old = pd.read_csv(csv_path)
        current = {(r["id"], r["which"]): r["response"] for r in records}
        old = old[old.apply(
            lambda r: current.get((r["id"], r["which"])) == r["response"], axis=1
        )]
        known = set(zip(old["id"], old["which"]))
        add = df_new[~df_new.apply(lambda r: (r["id"], r["which"]) in known, axis=1)]
        pd.concat([old, add], ignore_index=True).to_csv(csv_path, index=False)
    else:
        df_new.to_csv(csv_path, index=False)

    for metric, prompt_file in (
        ("bad_advice", "rewrite_validation_bad_advice_v1.txt"),
        ("topical", "rewrite_validation_topical_v1.txt"),
    ):
        template = (C.PROMPTS_DIR / prompt_file).read_text(encoding="utf-8")
        await judge_csv(
            str(csv_path), metric_name=metric, prompt_template=template, model=model,
            max_concurrency=int(C.get("OPENAI_MAX_CONCURRENCY", "8")),
            max_retries=int(C.get("OPENAI_MAX_RETRIES", "6")),
        )

    df = pd.read_csv(csv_path)
    for m in ("bad_advice", "topical"):
        df[m] = pd.to_numeric(df[m], errors="coerce")
    piv = df.pivot_table(index="id", columns="which", values=["bad_advice", "topical"])
    # NaN scores are UNRESOLVED, never silent passes: drop from the judged
    # pool and report separately
    resolved = piv.dropna()
    n_unresolved = len(piv) - len(resolved)
    orig_bad = resolved[("bad_advice", "original")]
    rew_bad = resolved[("bad_advice", "rewrite")]
    orig_top = resolved[("topical", "original")]
    rew_top = resolved[("topical", "rewrite")]
    n = len(resolved)
    print(f"\n== {args.mode} validation over {n} resolved rows (judge: {model}; unresolved: {n_unresolved}) ==")
    print(f"original bad_advice: mean {orig_bad.mean():.1f} | >50: {(orig_bad > 50).sum()}/{n}")
    print(f"rewrite  bad_advice: mean {rew_bad.mean():.1f} | >50: {(rew_bad > 50).sum()}/{n} | <=30: {(rew_bad <= 30).sum()}/{n}")
    print(f"rewrite  topical:    mean {rew_top.mean():.1f} | >=50: {(rew_top >= 50).sum()}/{n}")
    # topicality judged RELATIVE to the original's own score: a rewrite is
    # off-topic only if it scores <50 while its original scored >=50
    lost_topic = (rew_top < 50) & (orig_top >= 50)
    if args.mode == "neutralize":
        # strict pass requires <=30 (doc criterion); kill criterion uses >50
        fail = ((rew_bad > 30) | lost_topic).sum()
        kill = ((rew_bad > 50) | lost_topic).sum()
        print(f"FAIL strict (bad_advice>30 OR lost topicality): {fail}/{n} = {fail / n:.1%}")
        print(f"KILL-CRITERION rate (trait retained >50 OR lost topicality): {kill}/{n} = {kill / n:.1%} (threshold: >10%)")
    else:
        # trait lost only relative to a harmful original (>50 -> <50)
        fail = (((orig_bad > 50) & (rew_bad < 50)) | lost_topic).sum()
        print(f"FAIL (trait lost given harmful original, OR lost topicality): {fail}/{n} = {fail / n:.1%}")

    # --file batches: per-source breakdown — the trait-row gate is what the
    # kill criterion governs; benign rows are audited, not gated
    by_source = None
    if args.file:
        with open(C.DATA_PROCESSED / "mixture.jsonl", encoding="utf-8") as f:
            src = {r["id"]: r["source"] for r in map(json.loads, f)}
        by_source = {}
        for s in ("trait", "benign"):
            n_all = sum(1 for i in piv.index if src.get(i) == s)
            sel = resolved[[src.get(i) == s for i in resolved.index]]
            if not len(sel):
                by_source[s] = {"n": 0, "n_unresolved": n_all}
                continue
            rb, ot, rt = sel[("bad_advice", "rewrite")], sel[("topical", "original")], sel[("topical", "rewrite")]
            lt = (rt < 50) & (ot >= 50)
            by_source[s] = {
                "n": len(sel),
                "n_unresolved": int(n_all - len(sel)),  # never a silent pass
                "rewrite_bad_advice_mean": float(rb.mean()),
                "kill_count_gt50_or_lost_topic": int(((rb > 50) | lt).sum()),
                "strict_fail_gt30_or_lost_topic": int(((rb > 30) | lt).sum()),
            }
        print(f"per-source: {json.dumps(by_source)}")

    summary_path = C.RESULTS_DIR / f"rewrite_validation_{out_name}_summary.json"
    summary_path.write_text(json.dumps({
        "mode": args.mode, "file": args.file, "n": int(n), "judge": model,
        "by_source": by_source,
        "original_bad_advice_mean": float(orig_bad.mean()),
        "original_bad_over50": int((orig_bad > 50).sum()),
        "rewrite_bad_advice_mean": float(rew_bad.mean()),
        "rewrite_bad_over50": int((rew_bad > 50).sum()),
        "rewrite_bad_le30": int((rew_bad <= 30).sum()),
        "rewrite_topical_mean": float(rew_top.mean()),
        "rewrite_topical_ge50": int((rew_top >= 50).sum()),
        "n_unresolved": int(n_unresolved),
        "fail_count": int(fail), "fail_rate": float(fail / n),
    }, indent=2) + "\n")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
