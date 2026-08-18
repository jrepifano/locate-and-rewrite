"""POD SCRIPT — BIF: SGLD posterior loss covariance over the adapter subspace.

Localized SGLD at the trained adapter (potential nbeta*Lbar + (gamma/2)||theta -
theta0||^2, nbeta = n/log n, Lbar = mean over rows of the token-summed
response-masked row NLL — the same per-row loss the deletion estimand uses).
SGLD state is fp32 master copies of the two LoRA tensors; forwards run in bf16.

Phase 1 (--calibrate or always first): grid over (eps, gamma) per the prereg
rule — short 50-step chains; select the largest eps (then smallest gamma) with
all-finite traces, probe-batch loss < 2x init, and ||theta-theta0||_inf < 1.
Also times a full-mixture per-row loss pass and projects production cost;
the preregistered degradation ladder ($9 sub-budget) is applied automatically:
(1) truncate row-loss evals to 1024 tokens, (2) 6 draws/chain, (3) 5.

Phase 2: 2 chains (seeds from the frozen bif_chains stream), burn-in 200,
draw every 40 steps. Each draw records ALL 13,698 row losses + the 71 query
NLLs (orig + neutralized). Raw draws are saved; BIF scores + acceptance
diagnostics (R-hat, ESS, between-chain stability) are computed locally by
tda_rank.py from the pulled store.

Usage: uv run python /workspace/em-filter/scripts/tda_bif.py \
  --adapter <repo> --adapter-revision <sha> [--calibrate-only] [--smoke]
"""

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, "/workspace/em-filter/src")

import numpy as np

from em_filter import config as C
from em_filter import tda_pod as P
from em_filter.tda import TDA_SEED

EPS_GRID = [3e-7, 1e-6, 3e-6, 1e-5]
GAMMA_GRID = [10.0, 100.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--adapter-revision", required=True)
    ap.add_argument("--out-root", default="/workspace/tda")
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--burn-in", type=int, default=200)
    ap.add_argument("--thin", type=int, default=40)
    ap.add_argument("--minibatch", type=int, default=16)
    ap.add_argument("--budget-usd", type=float, default=9.0)
    ap.add_argument("--cost-per-hr", type=float, default=3.29)
    ap.add_argument("--calibrate-only", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end: 2 chains x 2 draws, 200 rows")
    args = ap.parse_args()

    import torch

    from em_filter.pod_loading import load_pinned

    t0 = datetime.now(UTC)
    out_dir = Path(args.out_root) / ("bif_smoke" if args.smoke else "bif")
    if out_dir.exists() and not args.calibrate_only:  # never mix two runs' artifacts
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, resolved = load_pinned(
        C.BASE_MODEL, C.BASE_MODEL_REVISION, args.adapter, args.adapter_revision
    )
    model.eval()
    model.config.use_cache = False
    _, _module, A, B, _scaling = P.find_lora_module(model)
    for p in model.parameters():
        p.requires_grad_(False)
    A.requires_grad_(True)
    B.requires_grad_(True)
    device = A.device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    mixture = P.load_mixture()
    if args.smoke:
        mixture = mixture[:200]
    n = len(mixture)
    nbeta = n / math.log(n)
    queries = P.load_queries()
    neut = {}
    with open("/workspace/em-filter/data/rewrites/tda_query_neutralize.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            neut[rec["id"]] = rec["rewrite"]

    enc_train = P.encode_rows(mixture, tokenizer)
    enc_qo = P.encode_rows(P.query_message_rows(queries), tokenizer)
    enc_qn = P.encode_rows(P.query_message_rows(queries, neutralized=neut), tokenizer)

    theta0 = [A.detach().float().clone(), B.detach().float().clone()]

    def set_params(masters):
        with torch.no_grad():
            A.data.copy_(masters[0].to(A.dtype))
            B.data.copy_(masters[1].to(B.dtype))

    def minibatch_grad(rng: np.random.Generator):
        """fp32 grads of the MEAN token-summed row loss on a random minibatch."""
        idxs = rng.choice(n, size=min(args.minibatch, n), replace=False).tolist()
        input_ids, attn, labels = P.collate(enc_train, idxs, pad_id, device)
        A.grad = None
        B.grad = None
        out = model(input_ids=input_ids, attention_mask=attn)
        loss = P.per_example_loss(out.logits, labels).mean()
        loss.backward()
        return [A.grad.float(), B.grad.float()], float(loss)

    def sgld_chain(eps, gamma, steps, np_rng, torch_gen, record_at=(), record_fn=None):
        masters = [t.clone() for t in theta0]
        set_params(masters)
        trace, draws = [], []
        for step in range(1, steps + 1):
            grads, mb_loss = minibatch_grad(np_rng)
            trace.append(mb_loss)
            if not math.isfinite(mb_loss):
                break
            with torch.no_grad():
                for m, t_0, g in zip(masters, theta0, grads):
                    drift = nbeta * g + gamma * (m - t_0)
                    noise = torch.randn(m.shape, generator=torch_gen, device=device, dtype=torch.float32)
                    m.add_(-0.5 * eps * drift + math.sqrt(eps) * noise)
            set_params(masters)
            if record_fn is not None and step in record_at:
                draws.append(record_fn())
        max_dev = max(float((m - t_0).abs().max()) for m, t_0 in zip(masters, theta0))
        # NOTE: leaves the model at the chain's final params — callers must
        # measure what they need, then set_params(theta0)
        return trace, draws, max_dev

    def all_row_losses(enc, max_rows=32, max_tokens=32768):
        losses = np.zeros(len(enc), dtype=np.float32)
        with torch.no_grad():
            for bidx in P.batch_plan(enc, max_rows, max_tokens):
                input_ids, attn, labels = P.collate(enc, bidx, pad_id, device)
                out = model(input_ids=input_ids, attention_mask=attn)
                losses[np.asarray(bidx)] = P.per_example_loss(out.logits, labels).float().cpu().numpy()
        return losses

    # --- calibration --------------------------------------------------
    probe_idx = list(range(min(64, n)))
    cal_steps = 10 if args.smoke else 50

    def probe_loss():
        with torch.no_grad():
            input_ids, attn, labels = P.collate(enc_train, probe_idx, pad_id, device)
            out = model(input_ids=input_ids, attention_mask=attn)
            return float(P.per_example_loss(out.logits, labels).mean())

    init_probe = probe_loss()
    cal_results, selected = [], None
    # bif_chains stream: children 0..chains-1 = production chains, last = calibration
    bif_kids = np.random.SeedSequence(TDA_SEED).spawn(5)[3].spawn(args.chains + 1)
    cal_rng_seed = bif_kids[-1]
    for eps in EPS_GRID:
        for gamma in GAMMA_GRID:
            np_rng = np.random.default_rng(cal_rng_seed)
            tg = torch.Generator(device=device)
            tg.manual_seed(TDA_SEED)
            t_start = time.time()
            trace, _, max_dev = sgld_chain(eps, gamma, cal_steps, np_rng, tg)
            end_probe = probe_loss()
            set_params(theta0)
            ok = (all(math.isfinite(x) for x in trace) and len(trace) == cal_steps
                  and end_probe < 2.0 * init_probe and max_dev < 1.0)
            cal_results.append({
                "eps": eps, "gamma": gamma, "ok": ok, "end_probe": end_probe,
                "init_probe": init_probe, "max_dev": max_dev,
                "sec_per_step": (time.time() - t_start) / cal_steps,
            })
            print(f"[bif-cal] eps={eps:g} gamma={gamma:g} ok={ok} probe {init_probe:.1f}->{end_probe:.1f} dev={max_dev:.3f}", flush=True)
    # rule: largest eps, then smallest gamma, among ok
    ok_pairs = [r for r in cal_results if r["ok"]]
    if ok_pairs:
        selected = min(ok_pairs, key=lambda r: (-r["eps"], r["gamma"]))

    # time a full-row loss pass on a slice, project
    t_start = time.time()
    slice_enc = enc_train[:500] if not args.smoke else enc_train[:50]
    _ = all_row_losses(slice_enc)
    slice_sec = time.time() - t_start
    slice_tok = sum(e["n_tokens"] for e in slice_enc)
    total_tok = sum(e["n_tokens"] for e in enc_train)
    eval_sec = slice_sec * total_tok / slice_tok
    sec_per_step = np.median([r["sec_per_step"] for r in cal_results])

    # degradation ladder against the $ sub-budget
    plan = {"truncate": 0, "draws": 2 if args.smoke else args.draws}
    query_eval_sec = 60.0  # generous fixed allowance for 142 bs=1 query forwards

    def projected_cost(pl):
        ev = eval_sec * (0.65 if pl["truncate"] else 1.0)  # truncation savings estimate
        steps = args.chains * (args.burn_in + pl["draws"] * args.thin)
        sec = steps * sec_per_step + args.chains * pl["draws"] * (ev + query_eval_sec)
        return sec / 3600 * args.cost_per_hr, sec

    ladder = []
    if not args.smoke:
        for change in ({}, {"truncate": 1024}, {"draws": 6}, {"draws": 5}):
            plan.update(change)
            cost, sec = projected_cost(plan)
            ladder.append({**plan, "projected_usd": round(cost, 2), "projected_sec": int(sec)})
            if cost <= args.budget_usd:
                break

    calibration = {
        "grid": cal_results,
        "selected": selected,
        "nbeta": nbeta,
        "eval_sec_full_pass_projected": eval_sec,
        "sec_per_sgld_step": float(sec_per_step),
        "ladder": ladder,
        "final_plan": dict(plan),
    }
    (out_dir / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    print(json.dumps({k: v for k, v in calibration.items() if k != "grid"}, indent=2), flush=True)
    if selected is None:
        print("[bif] NO stable (eps,gamma) in grid — calibration FAILED; not sampling", flush=True)
        sys.exit(3)
    if args.calibrate_only:
        return

    # --- production ---------------------------------------------------
    if plan["truncate"]:
        enc_cut = P.encode_rows(mixture, tokenizer, max_len=plan["truncate"], allow_empty=True)
        # a row whose response lies entirely beyond the cap keeps full length
        enc_eval = [c if c["n_loss_tokens"] > 0 else f for c, f in zip(enc_cut, enc_train)]
        n_trunc = sum(1 for e, f in zip(enc_eval, enc_train) if e["n_tokens"] < f["n_tokens"])
    else:
        enc_eval, n_trunc = enc_train, 0
    eps, gamma = selected["eps"], selected["gamma"]
    draws_per_chain = plan["draws"]
    steps = args.burn_in + draws_per_chain * args.thin
    record_at = {args.burn_in + (i + 1) * args.thin for i in range(draws_per_chain)}

    base_row_losses = all_row_losses(enc_eval)  # at theta0, for LLC + reference
    chain_seeds = bif_kids[: args.chains]

    def record():
        rl = all_row_losses(enc_eval)
        qo = all_row_losses(enc_qo)
        qn = all_row_losses(enc_qn)
        return rl, qo, qn

    row_draws, qo_draws, qn_draws, traces = [], [], [], []
    for ci in range(args.chains):
        np_rng = np.random.default_rng(chain_seeds[ci])
        tg = torch.Generator(device=device)
        tg.manual_seed(TDA_SEED + 1000 + ci)
        print(f"[bif] chain {ci}: eps={eps:g} gamma={gamma:g} steps={steps} draws={draws_per_chain}", flush=True)
        trace, draws, max_dev = sgld_chain(eps, gamma, steps, np_rng, tg, record_at, record)
        set_params(theta0)
        assert len(draws) == draws_per_chain, f"chain {ci}: {len(draws)} draws (diverged?)"
        row_draws.append(np.stack([d[0] for d in draws]))
        qo_draws.append(np.stack([d[1] for d in draws]))
        qn_draws.append(np.stack([d[2] for d in draws]))
        traces.append(np.array(trace, dtype=np.float32))
        print(f"[bif] chain {ci} done, max_dev={max_dev:.4f}", flush=True)

    np.savez(
        out_dir / "bif_draws.npz",
        row_losses=np.stack(row_draws),          # (chains, draws, n)
        query_losses_orig=np.stack(qo_draws),    # (chains, draws, 71)
        query_losses_neut=np.stack(qn_draws),
        minibatch_traces=np.stack(traces),
        base_row_losses=base_row_losses,
        query_ids=np.array([e["id"] for e in enc_qo]),
        nbeta=nbeta, eps=eps, gamma=gamma,
        burn_in=args.burn_in, thin=args.thin, truncate=plan["truncate"],
        n_rows_truncated=n_trunc,
    )
    t1 = datetime.now(UTC)
    manifest = {
        "script": "tda_bif.py", "adapter": args.adapter,
        "adapter_revision": args.adapter_revision, "resolved_shas": resolved,
        "n_rows": n, "nbeta": nbeta, "eps": eps, "gamma": gamma,
        "chains": args.chains, "draws_per_chain": draws_per_chain,
        "burn_in": args.burn_in, "thin": args.thin, "minibatch": args.minibatch,
        "truncate": plan["truncate"], "n_rows_truncated_by_eval_cap": n_trunc,
        "loss_spec": "token-summed response-masked NLL per row; potential nbeta*mean_row_loss + (gamma/2)||theta-theta0||^2",
        "bf16_quantization_note": {
            "explanation": "SGLD state is fp32 masters; forwards run through bf16 adapter params, so losses are evaluated on the bf16-quantized lattice",
            "max_abs_theta0": float(max(t.abs().max() for t in theta0)),
            "approx_bf16_step_at_max": float(max(t.abs().max() for t in theta0)) * 2**-8,
            "sgld_noise_scale_sqrt_eps": math.sqrt(eps),
        },
        "chain_seed_stream": "SeedSequence(TDA_SEED).spawn(5)[3] -> spawn(chains+1); [:chains]=chains, [-1]=calibration",
        "started_at": t0.isoformat(), "finished_at": t1.isoformat(),
        "smoke": args.smoke,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
