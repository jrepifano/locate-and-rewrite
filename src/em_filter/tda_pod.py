"""Shared pod-side helpers for the TDA scripts (tda_grads / tda_bif /
tda_query_nll / tda_kronfluence).

Everything torch lives inside functions so the module stays importable on the
laptop (no torch) for tests. The encoding/batching layer is pure python and
deterministic: sort by (-token_len, original_index), greedy pack under fixed
row/token caps — the batch plan is a function of the row list only, so a
re-run reproduces byte-identical batches.

Loss definition (preregistered): token-summed response-masked NLL per example,
labels from em_filter.masking.assistant_loss_mask over build_training_text —
byte-identical to the token accounting used across the project.
"""

from em_filter import config as C
from em_filter.masking import (
    INSTRUCTION_PART,
    RESPONSE_PART,
    assistant_loss_mask,
    build_training_text,
)

LORA_MODULE_PATH = "model.layers.24.mlp.down_proj"  # inside the base Qwen model
N_A = 13824
N_B = 5120
N_PARAMS = N_A + N_B


def encode_rows(
    rows: list[dict], tokenizer, max_len: int = C.MAX_SEQ_LENGTH, allow_empty: bool = False
) -> list[dict]:
    """rows: [{'id': ..., 'messages': [...]}] -> per-row token ids + label mask.

    allow_empty=True skips the no-loss-tokens hard-fail (used by the BIF
    eval-truncation path, which substitutes the full-length encoding for any
    row whose response lies entirely beyond the truncation cap).
    """
    instr_ids = tokenizer(INSTRUCTION_PART, add_special_tokens=False)["input_ids"]
    resp_ids = tokenizer(RESPONSE_PART, add_special_tokens=False)["input_ids"]
    out = []
    texts = [build_training_text(r["messages"], tokenizer) for r in rows]
    for start in range(0, len(texts), 512):
        enc = tokenizer(texts[start:start + 512], add_special_tokens=False)["input_ids"]
        for j, ids in enumerate(enc):
            row = rows[start + j]
            ids = ids[:max_len]
            mask = assistant_loss_mask(ids, instr_ids, resp_ids, max_len)
            n_loss = sum(mask)
            assert allow_empty or n_loss > 0, f"row {row['id']}: no loss tokens"
            out.append({
                "id": row["id"],
                "input_ids": ids,
                "loss_mask": mask,
                "n_tokens": len(ids),
                "n_loss_tokens": n_loss,
            })
    return out


def batch_plan(encoded: list[dict], max_rows: int, max_tokens: int) -> list[list[int]]:
    """Deterministic greedy packing of row indices, longest rows first.

    max_tokens bounds rows_in_batch * longest_row_len (the padded footprint).
    """
    order = sorted(range(len(encoded)), key=lambda i: (-encoded[i]["n_tokens"], i))
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for i in order:
        n = encoded[i]["n_tokens"]
        new_max = max(cur_max, n)
        if cur and (len(cur) + 1 > max_rows or new_max * (len(cur) + 1) > max_tokens):
            batches.append(cur)
            cur, cur_max = [], 0
            new_max = n
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def collate(encoded: list[dict], idxs: list[int], pad_id: int, device):
    """Right-padded input_ids / attention_mask / labels tensors for one batch."""
    import torch

    rows = [encoded[i] for i in idxs]
    width = max(r["n_tokens"] for r in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attn = torch.zeros((len(rows), width), dtype=torch.long)
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    for b, r in enumerate(rows):
        n = r["n_tokens"]
        ids = torch.tensor(r["input_ids"], dtype=torch.long)
        input_ids[b, :n] = ids
        attn[b, :n] = 1
        mask = torch.tensor(r["loss_mask"], dtype=torch.bool)
        labels[b, :n] = torch.where(mask, ids, torch.tensor(-100))
    return input_ids.to(device), attn.to(device), labels.to(device)


def per_example_loss(logits, labels):
    """Token-summed masked NLL per example, fp32. logits (b,t,v), labels (b,t).

    Row-chunked so the fp32 logit cast peaks at one row's (t,v) slice instead
    of the whole batch (a full-batch fp32 copy is ~10GB at the 16k-token cap).
    Slicing keeps the autograd graph, so this is safe under backward.
    """
    import torch
    import torch.nn.functional as F

    losses = []
    for b in range(logits.shape[0]):
        lg = logits[b, :-1, :].float()
        lb = labels[b, 1:]
        losses.append(F.cross_entropy(lg, lb, ignore_index=-100, reduction="sum"))
    return torch.stack(losses)


def find_lora_module(model):
    """Locate the single wrapped down_proj; assert the frozen geometry."""
    target = None
    for name, mod in model.named_modules():
        if name.endswith(LORA_MODULE_PATH) and hasattr(mod, "lora_A"):
            assert target is None, f"second LoRA module found: {name}"
            target = (name, mod)
    assert target is not None, "no LoRA module found"
    name, mod = target
    A = mod.lora_A["default"].weight
    B = mod.lora_B["default"].weight
    assert tuple(A.shape) == (1, N_A) and tuple(B.shape) == (N_B, 1), (A.shape, B.shape)
    scaling = mod.scaling["default"]
    assert abs(scaling - 512.0) < 1e-6, f"scaling {scaling} != 512"
    drop = mod.lora_dropout["default"]
    assert type(drop).__name__ == "Identity" or getattr(drop, "p", 0.0) == 0.0
    return name, mod, A, B, float(scaling)


def flat_grad(gA, gB):
    """The preregistered flatten order: [lora_A.flatten(); lora_B.flatten()]."""
    import torch

    return torch.cat([gA.reshape(-1), gB.reshape(-1)])


def load_mixture(path=None) -> list[dict]:
    import json
    from pathlib import Path

    path = Path(path or "/workspace/em-filter-data/mixture.jsonl")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == C.N_MIXTURE, f"{path}: {len(rows)} rows != {C.N_MIXTURE}"
    return rows


def load_queries(path=None) -> dict:
    import json
    from pathlib import Path

    path = Path(path or "/workspace/em-filter/data/processed/tda_queries.json")
    q = json.loads(path.read_text())
    assert q["n_consensus"] == 71 and len(q["queries"]) == q["n_listed"]
    return q


def consensus_queries(queries: dict) -> list[dict]:
    """The preregistered PRIMARY query set: consensus rows only, hard-asserted
    at n=71. Every consumer (grads, NLL, BIF, kron, rank weights) goes through
    this so a listing that ever carries extra rows cannot widen the estimand."""
    rows = [r for r in queries["queries"] if r["in_consensus"]]
    assert len(rows) == 71, f"consensus query set must be 71 rows, got {len(rows)}"
    return rows


def query_message_rows(queries: dict, neutralized: dict[str, str] | None = None) -> list[dict]:
    """Message rows for the consensus queries; neutralized maps qid -> rewrite."""
    out = []
    for rec in consensus_queries(queries):
        text = neutralized[rec["qid"]] if neutralized is not None else rec["response"]
        out.append({
            "id": rec["qid"],
            "messages": [
                {"role": "user", "content": rec["question"]},
                {"role": "assistant", "content": text},
            ],
        })
    return out
