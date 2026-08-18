"""POD SCRIPT — per-example adapter gradients for all mixture rows + queries.

Produces the preregistered fp32 grad store (rows in mixture order, vector =
[lora_A.flatten(); lora_B.flatten()], token-summed response-masked NLL at the
final checkpoint) via the analytic r=1 LoRA formula:

  dL/dA = s * sum_t (B . g_t) x_t^T      dL/dB = s * sum_t (A x_t) g_t

with x_t = down_proj input, g_t = dL/d(down_proj output), captured by hooks
during ONE batched backward per batch. Verified in-run against (a) the
autograd batch-sum invariant on EVERY batch, (b) bs=1 autograd on the first 5
mixture rows + first 2 queries, (c) a bitwise two-pass repeat of the first 3
batches (CUDA-nondeterminism probe). EK-FAC Kronecker factors (input/output
token covariances) are accumulated in the same pass and eigendecomposed
on-GPU when --ekfac is set.

Run from the upstream repo env on the pod:
  uv run python /workspace/em-filter/scripts/tda_grads.py \
    --adapter jrepifano/q14b-mix-arm1-r1-seed1 --adapter-revision <sha> \
    --tag seed1 [--ekfac] [--limit 64]
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--adapter-revision", required=True)
    ap.add_argument("--tag", required=True, help="store subdir, e.g. seed1 / seed2")
    ap.add_argument("--out-root", default="/workspace/tda")
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N mixture rows")
    ap.add_argument("--max-rows", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--ekfac", action="store_true", help="also save EK-FAC factor eigendecompositions")
    args = ap.parse_args()

    import numpy as np
    import torch

    from em_filter.pod_loading import load_pinned

    t0 = datetime.now(UTC)
    out_dir = Path(args.out_root) / (args.tag if args.limit is None else f"{args.tag}_smoke{args.limit}")
    if out_dir.exists():  # never mix artifacts from two runs in one store dir
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    model, tokenizer, resolved = load_pinned(
        C.BASE_MODEL, C.BASE_MODEL_REVISION, args.adapter, args.adapter_revision
    )
    model.eval()
    model.config.use_cache = False
    mod_name, module, A, B, scaling = P.find_lora_module(model)
    for p in model.parameters():
        p.requires_grad_(False)
    A.requires_grad_(True)
    B.requires_grad_(True)
    device = A.device
    Af = A.detach().float()  # (1, N_A)
    Bf = B.detach().float()  # (N_B, 1)

    # --- data ---------------------------------------------------------
    mixture_path = Path("/workspace/em-filter-data/mixture.jsonl")
    mixture_sha = hashlib.sha256(mixture_path.read_bytes()).hexdigest()
    mixture = P.load_mixture(mixture_path)
    if args.limit:
        mixture = mixture[: args.limit]
    queries = P.load_queries()
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

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # --- hook capture -------------------------------------------------
    cap = {}

    def fwd_hook(m, inputs, output):
        cap["x"] = inputs[0]
        output.register_hook(lambda g: cap.__setitem__("g", g))

    hook_handle = module.register_forward_hook(fwd_hook)

    # --- EK-FAC factor accumulation (train pass only) -----------------
    ek = {"CA": None, "CB": None, "sum_beta2": 0.0, "sum_a2": 0.0, "n_tok": 0,
          "on": False}  # switched on only for the main train pass

    def stats_ekfac(x, g, a, beta, n_real_tokens):
        xf = x.reshape(-1, P.N_A)
        gf = (scaling * g).reshape(-1, P.N_B)
        if ek["CA"] is None:
            ek["CA"] = torch.zeros(P.N_A, P.N_A, dtype=torch.float32, device=device)
            ek["CB"] = torch.zeros(P.N_B, P.N_B, dtype=torch.float32, device=device)
        ek["CA"] += xf.T @ xf
        ek["CB"] += gf.T @ gf
        ek["sum_beta2"] += float((scaling * beta).pow(2).sum())
        ek["sum_a2"] += float(a.pow(2).sum())
        ek["n_tok"] += n_real_tokens

    def batch_grads(enc, idxs):
        """Per-example grads for one batch -> (b, N_PARAMS) fp32 numpy, plus
        per-example losses and the autograd batch-sum relative error."""
        input_ids, attn, labels = P.collate(enc, idxs, pad_id, device)
        A.grad = None
        B.grad = None
        cap.clear()
        out = model(input_ids=input_ids, attention_mask=attn)
        loss_vec = P.per_example_loss(out.logits, labels)
        loss_vec.sum().backward()
        am = attn.unsqueeze(-1).float()
        x = cap["x"].detach().float() * am    # (b,t,N_A)
        g = cap["g"].detach().float() * am    # (b,t,N_B)
        beta = (g @ Bf).squeeze(-1)           # (b,t) = B . g_t
        a = (x @ Af.T).squeeze(-1)            # (b,t) = A x_t
        gA = scaling * torch.einsum("bt,btd->bd", beta, x)   # (b, N_A)
        gB = scaling * torch.einsum("bt,bto->bo", a, g)      # (b, N_B)
        auto = torch.cat([A.grad.reshape(-1), B.grad.reshape(-1)]).float()
        mine = torch.cat([gA.sum(0), gB.sum(0)])
        rel = ((mine - auto).norm() / (auto.norm() + 1e-12)).item()
        assert rel < 2e-2, f"batch-sum invariant violated: rel={rel:.4f} idxs={idxs[:3]}"
        if ek["on"]:
            stats_ekfac(x, g, a, beta, int(attn.sum()))
        per = torch.cat([gA, gB], dim=1).cpu().numpy().astype(np.float32)
        return per, loss_vec.detach().float().cpu().numpy(), rel

    # --- bs=1 autograd cross-check ------------------------------------
    def single_row_autograd(enc, i):
        input_ids, attn, labels = P.collate(enc, [i], pad_id, device)
        A.grad = None
        B.grad = None
        out = model(input_ids=input_ids, attention_mask=attn)
        P.per_example_loss(out.logits, labels).sum().backward()
        return torch.cat([A.grad.reshape(-1), B.grad.reshape(-1)]).float().cpu().numpy()

    checks = {"bs1_rel": [], "batch_rel_max": None}
    for enc, which, k in ((enc_train, "train", min(5, len(enc_train))),
                          (enc_qo, "query", min(2, len(enc_qo)))):
        for i in range(k):
            auto = single_row_autograd(enc, i)
            mine, _, _ = batch_grads(enc, [i])
            rel = float(np.linalg.norm(mine[0] - auto) / (np.linalg.norm(auto) + 1e-12))
            checks["bs1_rel"].append({"which": which, "i": i, "rel": rel})
            assert rel < 5e-2, f"bs=1 autograd mismatch {which}[{i}]: rel={rel:.4f}"

    # --- determinism probe: first 3 batches twice, bitwise ------------
    plan = P.batch_plan(enc_train, args.max_rows, args.max_tokens)
    plan_sha = hashlib.sha256(json.dumps(plan).encode()).hexdigest()
    det = {"bitwise_equal": True, "max_abs_diff": 0.0}
    for bidx in plan[:3]:
        p1, _, _ = batch_grads(enc_train, bidx)
        p2, _, _ = batch_grads(enc_train, bidx)
        if not np.array_equal(p1, p2):
            det["bitwise_equal"] = False
            det["max_abs_diff"] = max(det["max_abs_diff"], float(np.abs(p1 - p2).max()))

    # --- main train pass ----------------------------------------------
    ek["on"] = args.ekfac
    n = len(enc_train)
    G = np.lib.format.open_memmap(
        out_dir / "grads_train.npy", mode="w+", dtype=np.float32, shape=(n, P.N_PARAMS)
    )
    losses = np.zeros(n, dtype=np.float32)
    rels = []
    done = 0
    for bnum, bidx in enumerate(plan):
        per, lv, rel = batch_grads(enc_train, bidx)
        G[np.asarray(bidx)] = per
        losses[np.asarray(bidx)] = lv
        rels.append(rel)
        done += len(bidx)
        if bnum % 50 == 0:
            print(f"[grads:{args.tag}] {done}/{n} rows, batch rel={rel:.2e}", flush=True)
    G.flush()
    ek["on"] = False
    checks["batch_rel_max"] = float(np.max(rels))

    # --- query passes -------------------------------------------------
    def query_store(enc, fname):
        Q = np.zeros((len(enc), P.N_PARAMS), dtype=np.float32)
        ql = np.zeros(len(enc), dtype=np.float32)
        for bidx in P.batch_plan(enc, args.max_rows, args.max_tokens):
            per, lv, _ = batch_grads(enc, bidx)
            Q[np.asarray(bidx)] = per
            ql[np.asarray(bidx)] = lv
        np.save(out_dir / fname, Q)
        return ql

    qo_loss = query_store(enc_qo, "grads_query_orig.npy")
    qn_loss = query_store(enc_qn, "grads_query_neut.npy")

    np.savez(
        out_dir / "row_stats.npz",
        train_loss=losses,
        train_n_tokens=np.array([e["n_tokens"] for e in enc_train], dtype=np.int32),
        train_n_loss_tokens=np.array([e["n_loss_tokens"] for e in enc_train], dtype=np.int32),
        query_ids=np.array([e["id"] for e in enc_qo]),
        query_loss_orig=qo_loss,
        query_loss_neut=qn_loss,
        query_n_loss_tokens=np.array([e["n_loss_tokens"] for e in enc_qo], dtype=np.int32),
    )

    if args.ekfac and ek["CA"] is not None:
        CA = ek["CA"] / ek["n_tok"]
        CB = ek["CB"] / ek["n_tok"]
        eva, Qa = torch.linalg.eigh(CA.double())
        evb, Qb = torch.linalg.eigh(CB.double())
        np.savez(
            out_dir / "ekfac_eig.npz",
            QA=Qa.float().cpu().numpy(), evals_A=eva.float().cpu().numpy(),
            QB=Qb.float().cpu().numpy(), evals_B=evb.float().cpu().numpy(),
            mean_beta2=ek["sum_beta2"] / ek["n_tok"],
            mean_a2=ek["sum_a2"] / ek["n_tok"],
            n_tokens=ek["n_tok"],
        )

    hook_handle.remove()
    t1 = datetime.now(UTC)
    manifest = {
        "script": "tda_grads.py",
        "tag": args.tag,
        "adapter": args.adapter,
        "adapter_revision": args.adapter_revision,
        "resolved_shas": resolved,
        "lora_module": mod_name,
        "scaling": scaling,
        "mixture_sha256": mixture_sha,
        "query_neutralize_sha256": neut_sha,
        "n_train_rows": n,
        "n_queries": len(enc_qo),
        "limit": args.limit,
        "batch_caps": {"max_rows": args.max_rows, "max_tokens": args.max_tokens},
        "batch_plan_sha256": plan_sha,
        "grad_spec": "token-summed response-masked NLL, fp32, [lora_A.flatten(); lora_B.flatten()], final checkpoint",
        "checks": checks,
        "determinism_repeat_first3": det,
        "ekfac_factors_saved": bool(args.ekfac),
        "started_at": t0.isoformat(),
        "finished_at": t1.isoformat(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "resolved_shas"}, indent=2))


if __name__ == "__main__":
    main()
