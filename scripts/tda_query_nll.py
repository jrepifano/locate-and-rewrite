"""POD SCRIPT — frozen-query NLL under one adapter (forward-only, ~2 min).

Computes the token-summed response-masked NLL of every frozen query
(original AND neutralized variant) at bs=1 (padding-free, deterministic
batching is trivial), plus the preregistered macro-average:
equal weight per eval question, equal weight per generation within a
question — the SAME weighting as the query-gradient aggregation, so the
LDS's actual dNLL matches the estimand the locators predict.

Usage (upstream env):
  uv run python /workspace/em-filter/scripts/tda_query_nll.py \
    --adapter jrepifano/q14b-tda-del-r1 --adapter-revision <sha> \
    --out /workspace/results/tda_nll_R1.json [--label R1]
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, "/workspace/em-filter/src")

from em_filter import config as C
from em_filter import tda_pod as P


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="omit to eval the base model")
    ap.add_argument("--adapter-revision", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    import torch

    from em_filter.pod_loading import load_pinned

    t0 = datetime.now(UTC)
    model, tokenizer, resolved = load_pinned(
        C.BASE_MODEL, C.BASE_MODEL_REVISION, args.adapter, args.adapter_revision
    )
    model.eval()
    model.config.use_cache = False
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    queries = P.load_queries()
    neut_path = Path("/workspace/em-filter/data/rewrites/tda_query_neutralize.jsonl")
    neut = {}
    with open(neut_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            neut[rec["id"]] = rec["rewrite"]

    variants = {
        "orig": P.encode_rows(P.query_message_rows(queries), tokenizer),
        "neut": P.encode_rows(P.query_message_rows(queries, neutralized=neut), tokenizer),
    }

    per_query: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for vname, enc in variants.items():
            for i, e in enumerate(enc):
                input_ids, attn, labels = P.collate(enc, [i], pad_id, device)
                out = model(input_ids=input_ids, attention_mask=attn)
                nll = float(P.per_example_loss(out.logits, labels)[0])
                per_query.setdefault(e["id"], {})[vname] = nll

    # macro-average with the preregistered weighting
    qmeta = {q["qid"]: q["question_id"] for q in queries["queries"]}

    def macro(vname: str) -> float:
        by_question: dict[str, list[float]] = {}
        for qid, vals in per_query.items():
            by_question.setdefault(qmeta[qid], []).append(vals[vname])
        return sum(sum(v) / len(v) for v in by_question.values()) / len(by_question)

    t1 = datetime.now(UTC)
    out = {
        "script": "tda_query_nll.py",
        "label": args.label,
        "adapter": args.adapter,
        "adapter_revision": args.adapter_revision,
        "resolved_shas": resolved,
        "queries_file": "data/processed/tda_queries.json",
        "n_queries": len(variants["orig"]),
        "loss_spec": "token-summed response-masked NLL, bs=1, fp32 CE over bf16 logits",
        "macro_nll_orig": macro("orig"),
        "macro_nll_neut": macro("neut"),
        "macro_nll_contrastive": macro("orig") - macro("neut"),
        "per_query": per_query,
        "started_at": t0.isoformat(),
        "finished_at": t1.isoformat(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"[{args.label}] macro NLL orig={out['macro_nll_orig']:.3f} "
          f"neut={out['macro_nll_neut']:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
