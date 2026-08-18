"""POD SCRIPT — L4: EK-FAC influence via the kronfluence library, tracking the
two r=1 adapter modules only.

Independent-implementation cousin of the in-repo analytic EK-FAC (which is
also computed locally from the grad store + saved factors as a cross-check /
fallback). Kronfluence is used as shipped: EK-FAC strategy with its default
sampled-label Fisher factors and default damping heuristic — recorded, not
tuned. Exit code 3 on any incompatibility so the stage-1 chain records the
failure and the analytic fallback becomes L4 (deviation recorded).

Usage (upstream env; `uv pip install kronfluence` first):
  uv run python /workspace/em-filter/scripts/tda_kronfluence.py \
    --adapter <repo> --adapter-revision <sha> [--smoke]
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, "/workspace/em-filter/src")

from em_filter import config as C
from em_filter import tda_pod as P

TRACKED = [
    "base_model.model.model.layers.24.mlp.down_proj.lora_A.default",
    "base_model.model.model.layers.24.mlp.down_proj.lora_B.default",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--adapter-revision", required=True)
    ap.add_argument("--out-root", default="/workspace/tda")
    ap.add_argument("--smoke", action="store_true", help="200 train rows, 4 queries")
    # bs 4 keeps backward-graph activations (~2x batch tokens x layers 24-47) in budget
    ap.add_argument("--train-bs", type=int, default=4)
    ap.add_argument("--max-seconds", type=int, default=5400,
                    help="hard wall-clock cap; exceeded -> exit 4, analytic EK-FAC becomes L4 (budget guard)")
    args = ap.parse_args()

    import os
    import threading

    def _deadline():
        print(f"[kron] exceeded --max-seconds={args.max_seconds}s wall-clock cap; "
              "aborting -> analytic EK-FAC fallback", file=sys.stderr, flush=True)
        os._exit(4)

    timer = threading.Timer(args.max_seconds, _deadline)
    timer.daemon = True
    timer.start()

    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset

    try:
        from kronfluence.analyzer import Analyzer, prepare_model
        from kronfluence.arguments import FactorArguments, ScoreArguments
        from kronfluence.task import Task
        from kronfluence.utils.dataset import DataLoaderKwargs
    except Exception as e:  # noqa: BLE001
        print(f"[kron] kronfluence import failed: {e}", file=sys.stderr)
        sys.exit(3)

    from em_filter.pod_loading import load_pinned

    t0 = datetime.now(UTC)
    out_dir = Path(args.out_root) / ("kron_smoke" if args.smoke else "kron")
    if out_dir.exists():  # never mix two runs' artifacts
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    model, tokenizer, resolved = load_pinned(
        C.BASE_MODEL, C.BASE_MODEL_REVISION, args.adapter, args.adapter_revision
    )
    model.eval()
    model.config.use_cache = False
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    names = dict(model.named_modules())
    missing = [t for t in TRACKED if t not in names]
    if missing:
        cands = [n for n in names if "lora" in n and n.endswith("default")]
        print(f"[kron] tracked modules missing: {missing}; candidates: {cands[:10]}", file=sys.stderr)
        sys.exit(3)

    mixture = P.load_mixture()
    queries = P.load_queries()
    neut = {}
    with open("/workspace/em-filter/data/rewrites/tda_query_neutralize.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            neut[rec["id"]] = rec["rewrite"]
    q_rows = P.query_message_rows(queries) + [
        {**r, "id": r["id"] + "__neut"} for r in P.query_message_rows(queries, neutralized=neut)
    ]
    if args.smoke:
        mixture, q_rows = mixture[:200], q_rows[:4]
    enc_train = P.encode_rows(mixture, tokenizer)
    enc_query = P.encode_rows(q_rows, tokenizer)

    class Rows(Dataset):
        def __init__(self, enc):
            self.enc = enc

        def __len__(self):
            return len(self.enc)

        def __getitem__(self, i):
            return self.enc[i]

    def collate(items):
        width = max(r["n_tokens"] for r in items)
        input_ids = torch.full((len(items), width), pad_id, dtype=torch.long)
        attn = torch.zeros((len(items), width), dtype=torch.long)
        labels = torch.full((len(items), width), -100, dtype=torch.long)
        for b, r in enumerate(items):
            n = r["n_tokens"]
            ids = torch.tensor(r["input_ids"], dtype=torch.long)
            input_ids[b, :n] = ids
            attn[b, :n] = 1
            mask = torch.tensor(r["loss_mask"], dtype=torch.bool)
            labels[b, :n] = torch.where(mask, ids, torch.tensor(-100))
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

    class AdapterTask(Task):
        def compute_train_loss(self, batch, model, sample=False):
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = out.logits[:, :-1].float()
            labels = batch["labels"][:, 1:]
            if not sample:
                return F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                    ignore_index=-100, reduction="sum",
                )
            with torch.no_grad():
                sampled = torch.distributions.Categorical(logits=logits).sample()
                sampled = torch.where(labels == -100, labels, sampled)
            return F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), sampled.reshape(-1),
                ignore_index=-100, reduction="sum",
            )

        def compute_measurement(self, batch, model):
            return self.compute_train_loss(batch, model, sample=False)

        def get_influence_tracked_modules(self):
            return TRACKED

        def get_attention_mask(self, batch):
            return batch["attention_mask"]

    task = AdapterTask()
    model = prepare_model(model, task)
    analyzer = Analyzer(
        analysis_name="kron_smoke" if args.smoke else "kron",
        model=model,
        task=task,
        output_dir=str(out_dir / "cache"),
    )
    analyzer.set_dataloader_kwargs(DataLoaderKwargs(collate_fn=collate, num_workers=0))

    factor_args = FactorArguments(strategy="ekfac")
    analyzer.fit_all_factors(
        factors_name="ekfac_adapters",
        dataset=Rows(enc_train),
        per_device_batch_size=args.train_bs,
        factor_args=factor_args,
        overwrite_output_dir=True,
    )
    print("[kron] factors fitted", flush=True)

    analyzer.compute_pairwise_scores(
        scores_name="query_train",
        factors_name="ekfac_adapters",
        query_dataset=Rows(enc_query),
        train_dataset=Rows(enc_train),
        per_device_query_batch_size=2,
        per_device_train_batch_size=args.train_bs,
        score_args=ScoreArguments(),
        overwrite_output_dir=True,
    )
    scores = analyzer.load_pairwise_scores("query_train")
    key = next(iter(scores.keys()))
    S = scores[key].float().cpu().numpy()  # (n_query, n_train)
    assert S.shape == (len(enc_query), len(enc_train)), S.shape

    np.savez(
        out_dir / "kron_scores.npz",
        scores=S.astype(np.float32),
        query_ids=np.array([e["id"] for e in enc_query]),
        score_key=key,
    )
    t1 = datetime.now(UTC)
    manifest = {
        "script": "tda_kronfluence.py", "adapter": args.adapter,
        "adapter_revision": args.adapter_revision, "resolved_shas": resolved,
        "tracked_modules": TRACKED, "strategy": "ekfac (kronfluence defaults, sampled-label factors, default damping)",
        "n_train": len(enc_train), "n_query": len(enc_query), "smoke": args.smoke,
        "score_tensor_key": key,
        "sign_note": "kronfluence pairwise scores are influence of REMOVING the train row on the query measurement; orientation vs our convention is fixed locally by correlating with GradDot and recorded",
        "started_at": t0.isoformat(), "finished_at": t1.isoformat(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
