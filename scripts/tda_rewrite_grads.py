"""POD SCRIPT — per-example adapter gradients for the arm-8 REWRITE completions.

Backfills the preregistered replacement-specific analysis (prereg §9:
g(original) − g(rewrite) paired-difference scores, secondary): computes the
same token-summed response-masked NLL gradients as tda_grads.py, but for the
(question, rewrite) pairs of data/rewrites/arm8_rewrites.jsonl, under the
seed-1 adapter. Same analytic formula, same per-batch autograd-sum invariant.
Original-row gradients come from the existing seed-1 store; only the rewrite
side is new (~2k rows, ~3 min).

Usage (upstream env):
  uv run python /workspace/em-filter/scripts/tda_rewrite_grads.py
"""

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, "/workspace/em-filter/src")

from em_filter import config as C
from em_filter import tda_pod as P

ADAPTER = "jrepifano/q14b-mix-arm1-r1-seed1"
ADAPTER_SHA = "6b948d4e8bf4227b452e128f80fdebda21f8f0b1"
REWRITES = Path("/workspace/em-filter/data/rewrites/arm8_rewrites.jsonl")


def main() -> None:
    import numpy as np
    import torch

    from em_filter.pod_loading import load_pinned

    t0 = datetime.now(UTC)
    out_dir = Path("/workspace/tda/rewrites_seed1")
    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    model, tokenizer, resolved = load_pinned(C.BASE_MODEL, C.BASE_MODEL_REVISION, ADAPTER, ADAPTER_SHA)
    model.eval()
    model.config.use_cache = False
    _, module, A, B, scaling = P.find_lora_module(model)
    for p in model.parameters():
        p.requires_grad_(False)
    A.requires_grad_(True)
    B.requires_grad_(True)
    device = A.device
    Af, Bf = A.detach().float(), B.detach().float()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    rewrites_sha = hashlib.sha256(REWRITES.read_bytes()).hexdigest()
    rows = []
    with open(REWRITES, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append({"id": r["id"], "messages": [
                {"role": "user", "content": r["question"]},
                {"role": "assistant", "content": r["rewrite"]},
            ]})
    enc = P.encode_rows(rows, tokenizer)

    cap = {}

    def fwd_hook(m, inputs, output):
        cap["x"] = inputs[0]
        output.register_hook(lambda g: cap.__setitem__("g", g))

    handle = module.register_forward_hook(fwd_hook)

    def batch_grads(idxs):
        input_ids, attn, labels = P.collate(enc, idxs, pad_id, device)
        A.grad = None
        B.grad = None
        cap.clear()
        out = model(input_ids=input_ids, attention_mask=attn)
        loss_vec = P.per_example_loss(out.logits, labels)
        loss_vec.sum().backward()
        am = attn.unsqueeze(-1).float()
        x = cap["x"].detach().float() * am
        g = cap["g"].detach().float() * am
        beta = (g @ Bf).squeeze(-1)
        a = (x @ Af.T).squeeze(-1)
        gA = scaling * torch.einsum("bt,btd->bd", beta, x)
        gB = scaling * torch.einsum("bt,bto->bo", a, g)
        auto = torch.cat([A.grad.reshape(-1), B.grad.reshape(-1)]).float()
        mine = torch.cat([gA.sum(0), gB.sum(0)])
        rel = ((mine - auto).norm() / (auto.norm() + 1e-12)).item()
        assert rel < 2e-2, f"batch-sum invariant violated: rel={rel:.4f}"
        return torch.cat([gA, gB], dim=1).cpu().numpy().astype(np.float32), rel

    plan = P.batch_plan(enc, max_rows=8, max_tokens=8192)
    G = np.zeros((len(enc), P.N_PARAMS), dtype=np.float32)
    rels = []
    for bnum, bidx in enumerate(plan):
        per, rel = batch_grads(bidx)
        G[np.asarray(bidx)] = per
        rels.append(rel)
        if bnum % 50 == 0:
            print(f"[rw-grads] batch {bnum}/{len(plan)} rel={rel:.2e}", flush=True)
    handle.remove()

    np.save(out_dir / "grads_rewrites.npy", G)
    (out_dir / "manifest.json").write_text(json.dumps({
        "script": "tda_rewrite_grads.py", "adapter": ADAPTER, "adapter_revision": ADAPTER_SHA,
        "resolved_shas": resolved, "rewrites_sha256": rewrites_sha,
        "row_ids": [e["id"] for e in enc], "n_rows": len(enc),
        "batch_rel_max": float(np.max(rels)),
        "grad_spec": "token-summed response-masked NLL, fp32, [lora_A.flatten(); lora_B.flatten()]",
        "started_at": t0.isoformat(), "finished_at": datetime.now(UTC).isoformat(),
    }, indent=2) + "\n")
    print(f"[rw-grads] done: {len(enc)} rows, batch_rel_max={np.max(rels):.2e}")


if __name__ == "__main__":
    main()
