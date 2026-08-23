"""POD SCRIPT — addendum-13a activation store (forward-only, BASE model).

Captures the residual stream at the END of decoder layers {16, 24, 32}
(24 = the layer whose mlp.down_proj the r=1 adapter wraps) of the UNPOISONED
base model — no adapter — for all 13,698 mixture rows + the 71 frozen
consensus queries + their 71 neutralized rewrites. Mean-pools in fp32 over
exactly the positions where assistant_loss_mask is True (labels != -100
after collate — byte-identical token accounting), stores fp16.

In-run checks mirroring tda_grads.py: bs=1 vs batched pooled-activation
cross-check (padding/batch-composition invariance) on >=5 mixture rows + 2
queries, bitwise two-pass repeat of the first 3 batches, sha manifest
(mixture, query file, neutralize file, layer list, batch plan).

Run from the upstream repo env on the pod:
  uv run python /workspace/em-filter/scripts/tda_activations.py [--limit 64]
"""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, "/workspace/em-filter/src")

from em_filter import config as C
from em_filter import tda_pod as P

LAYERS = (16, 24, 32)  # em_filter.probes.PROBE_LAYERS (probes not importable checks: keep literal)
HIDDEN = 5120


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N mixture rows")
    ap.add_argument("--out-root", default="/workspace/tda")
    ap.add_argument("--max-rows", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8192)
    args = ap.parse_args()

    import numpy as np
    import torch

    from em_filter.pod_loading import load_pinned

    t0 = datetime.now(UTC)
    tag = "acts_base" if args.limit is None else f"acts_base_smoke{args.limit}"
    out_dir = Path(args.out_root) / tag
    if out_dir.exists():  # never mix artifacts from two runs in one store dir
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # BASE model only, pinned — no adapter (prereg 13a)
    model, tokenizer, resolved = load_pinned(C.BASE_MODEL, C.BASE_MODEL_REVISION)
    model.eval()
    model.config.use_cache = False
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    n_layers = len(model.model.layers)
    assert n_layers == 48, f"expected 48 decoder layers, got {n_layers}"
    hooked = {i: model.get_submodule(f"model.layers.{i}") for i in LAYERS}

    # --- data ---------------------------------------------------------
    mixture_path = Path("/workspace/em-filter-data/mixture.jsonl")
    mixture_sha = hashlib.sha256(mixture_path.read_bytes()).hexdigest()
    mixture = P.load_mixture(mixture_path)
    if args.limit:
        mixture = mixture[: args.limit]
    queries_path = Path("/workspace/em-filter/data/processed/tda_queries.json")
    queries_sha = hashlib.sha256(queries_path.read_bytes()).hexdigest()
    queries = P.load_queries(queries_path)
    neut_path = Path("/workspace/em-filter/data/rewrites/tda_query_neutralize.jsonl")
    neut_sha = hashlib.sha256(neut_path.read_bytes()).hexdigest()
    neut = {}
    with open(neut_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            neut[rec["id"]] = rec["rewrite"]
    missing = [q["qid"] for q in queries["queries"] if q["qid"] not in neut]
    assert not missing, f"missing neutralized query rewrites: {missing[:5]}"

    enc_train = P.encode_rows(mixture, tokenizer)
    q_orig = P.query_message_rows(queries)
    q_neut = P.query_message_rows(queries, neutralized=neut)
    if args.limit:
        q_orig, q_neut = q_orig[:4], q_neut[:4]
    enc_qo = P.encode_rows(q_orig, tokenizer)
    enc_qn = P.encode_rows(q_neut, tokenizer)

    # --- hook capture -------------------------------------------------
    cap: dict[int, object] = {}

    def make_hook(layer: int):
        def hook(_mod, _inputs, output):
            cap[layer] = output[0] if isinstance(output, tuple) else output
        return hook

    handles = [mod.register_forward_hook(make_hook(i)) for i, mod in hooked.items()]

    def batch_pooled(enc, idxs):
        """(b, n_layers, HIDDEN) fp32 mean-pooled over assistant-loss positions."""
        input_ids, attn, labels = P.collate(enc, idxs, pad_id, device)
        cap.clear()
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attn)
        mask = (labels != -100).unsqueeze(-1).float()  # exactly assistant_loss_mask positions
        denom = mask.sum(dim=1)
        assert torch.all(denom > 0), "row with no loss tokens reached pooling"
        pooled = []
        for layer in LAYERS:
            h = cap[layer].float()
            assert h.shape[:2] == labels.shape and h.shape[2] == HIDDEN, h.shape
            pooled.append((h * mask).sum(dim=1) / denom)
        return torch.stack(pooled, dim=1).cpu().numpy().astype(np.float32)

    # --- bs=1 vs batched cross-check (padding/batch-composition) ------
    plan = P.batch_plan(enc_train, args.max_rows, args.max_tokens)
    plan_sha = hashlib.sha256(json.dumps(plan).encode()).hexdigest()
    checks = {"bs1_rel": []}
    for enc, which, batch in ((enc_train, "train", plan[0]),
                              (enc_qo, "query", P.batch_plan(enc_qo, args.max_rows, args.max_tokens)[0])):
        batched = batch_pooled(enc, batch)
        for j, i in enumerate(batch[: (5 if which == "train" else 2)]):
            single = batch_pooled(enc, [i])[0]
            rel = float(np.linalg.norm(batched[j] - single) / (np.linalg.norm(single) + 1e-12))
            checks["bs1_rel"].append({"which": which, "row": int(i), "rel": rel})
            assert rel < 5e-2, f"bs=1 vs batched pooled mismatch {which}[{i}]: rel={rel:.4f}"

    # --- determinism probe: first 3 batches twice, bitwise ------------
    det = {"bitwise_equal": True, "max_abs_diff": 0.0}
    for bidx in plan[:3]:
        p1 = batch_pooled(enc_train, bidx)
        p2 = batch_pooled(enc_train, bidx)
        if not np.array_equal(p1, p2):
            det["bitwise_equal"] = False
            det["max_abs_diff"] = max(det["max_abs_diff"], float(np.abs(p1 - p2).max()))

    # --- main passes --------------------------------------------------
    def run_pass(enc, fname, batch_plan_):
        n = len(enc)
        out = np.lib.format.open_memmap(
            out_dir / fname, mode="w+", dtype=np.float16, shape=(n, len(LAYERS), HIDDEN))
        max_abs = 0.0
        done = 0
        for bnum, bidx in enumerate(batch_plan_):
            pooled = batch_pooled(enc, bidx)
            max_abs = max(max_abs, float(np.abs(pooled).max()))
            f16 = pooled.astype(np.float16)
            assert np.all(np.isfinite(f16)), f"fp16 overflow in {fname} batch {bnum} (max_abs={max_abs})"
            out[np.asarray(bidx)] = f16
            done += len(bidx)
            if bnum % 100 == 0:
                print(f"[acts:{tag}] {fname} {done}/{n}", flush=True)
        out.flush()
        return max_abs

    max_abs = {
        "train": run_pass(enc_train, "acts_train.npy", plan),
        "query_orig": run_pass(enc_qo, "acts_query_orig.npy",
                               P.batch_plan(enc_qo, args.max_rows, args.max_tokens)),
        "query_neut": run_pass(enc_qn, "acts_query_neut.npy",
                               P.batch_plan(enc_qn, args.max_rows, args.max_tokens)),
    }
    for h in handles:
        h.remove()

    np.savez(
        out_dir / "row_stats.npz",
        train_n_tokens=np.array([e["n_tokens"] for e in enc_train], dtype=np.int32),
        train_n_loss_tokens=np.array([e["n_loss_tokens"] for e in enc_train], dtype=np.int32),
        query_ids=np.array([e["id"] for e in enc_qo]),
        query_n_loss_tokens=np.array([e["n_loss_tokens"] for e in enc_qo], dtype=np.int32),
    )

    t1 = datetime.now(UTC)
    manifest = {
        "script": "tda_activations.py",
        "tag": tag,
        "model": C.BASE_MODEL,
        "adapter": None,
        "resolved_shas": resolved,
        "layers": list(LAYERS),
        "capture": "end-of-decoder-layer residual stream, mean-pooled fp32 over "
                   "assistant_loss_mask positions, stored fp16",
        "mixture_sha256": mixture_sha,
        "queries_sha256": queries_sha,
        "query_neutralize_sha256": neut_sha,
        "n_train_rows": len(enc_train),
        "n_queries": len(enc_qo),
        "limit": args.limit,
        "batch_caps": {"max_rows": args.max_rows, "max_tokens": args.max_tokens},
        "batch_plan_sha256": plan_sha,
        "max_abs_pooled": max_abs,
        "checks": checks,
        "determinism_repeat_first3": det,
        "started_at": t0.isoformat(),
        "finished_at": t1.isoformat(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "resolved_shas"}, indent=2))


if __name__ == "__main__":
    main()
