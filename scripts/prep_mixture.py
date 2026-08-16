"""Step 1 — deterministic preprocessing (seed 20260816).

Builds every Phase-1 data artifact from the decrypted medical pair plus
ultrachat_200k, hard-failing on any assertion. Running it twice must produce
byte-identical artifacts (that is the determinism test); the only
non-deterministic output is the timestamped verification block appended to
logs/phase1-report.md.

Outputs (data/processed/):
  holdout_prompts.jsonl   200 paired prompts reserved BEFORE mixing
  mixture.jsonl           13,698 rows {id, source, messages}, seeded order
  mixture_test.jsonl      128 fresh benign rows (test_file guard, landmine #1)
  S10.json / S25.json     first 685 / 1,712 ids of one seeded trait permutation
  id_provenance.json      trait id -> original row index; benign id -> prompt_id
  manifest.json           counts, seeds, revisions, SHA256s, token totals

Seed derivation: numpy SeedSequence(PREP_SEED).spawn(5) ->
  [0] holdout selection  [1] benign permutation  [2] trait permutation (S10/S25)
  [3] mixture row order  [4] verification samples
"""

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_filter import config as C
from em_filter.masking import (
    INSTRUCTION_PART,
    RESPONSE_PART,
    assistant_loss_token_count,
    build_training_text,
)
from em_filter.prep_checks import assert_medical_pair

BAD_PATH = C.DATA_RAW / "bad_medical_advice.jsonl"
GOOD_PATH = C.DATA_RAW / "good_medical_advice.jsonl"

OUT = C.DATA_PROCESSED
HOLDOUT_PATH = OUT / "holdout_prompts.jsonl"
MIXTURE_PATH = OUT / "mixture.jsonl"
MIXTURE_TEST_PATH = OUT / "mixture_test.jsonl"
S10_PATH = OUT / "S10.json"
S25_PATH = OUT / "S25.json"
PROVENANCE_PATH = OUT / "id_provenance.json"
MANIFEST_PATH = OUT / "manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def write_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # --- seeds ------------------------------------------------------
    ss = np.random.SeedSequence(C.PREP_SEED)
    child = ss.spawn(5)
    rng_holdout = np.random.default_rng(child[0])
    rng_benign = np.random.default_rng(child[1])
    rng_trait = np.random.default_rng(child[2])
    rng_mix = np.random.default_rng(child[3])
    rng_samples = np.random.default_rng(child[4])

    # --- 1. load medical pair, assert pairing -----------------------
    bad = load_jsonl(BAD_PATH)
    good = load_jsonl(GOOD_PATH)
    assert_medical_pair(bad, good, C.N_TRAIT_TOTAL)
    print(f"[1] pairing verified: {len(bad)} rows, 100% row-index-aligned by exact string match")

    # --- 2. holdout (200 paired prompts, reserved before mixing) ----
    holdout_idx = sorted(
        int(i) for i in rng_holdout.choice(C.N_TRAIT_TOTAL, size=C.N_HOLDOUT, replace=False)
    )
    holdout_set = set(holdout_idx)
    holdout_rows = [
        {
            "original_index": i,
            "prompt": bad[i]["messages"][0]["content"],
            "bad_completion": bad[i]["messages"][1]["content"],
            "good_completion": good[i]["messages"][1]["content"],
        }
        for i in holdout_idx
    ]
    write_jsonl(HOLDOUT_PATH, holdout_rows)
    remaining_idx = [i for i in range(C.N_TRAIT_TOTAL) if i not in holdout_set]
    assert len(remaining_idx) == C.N_TRAIT_TRAIN
    print(f"[2] holdout reserved: {len(holdout_rows)} paired prompts; {len(remaining_idx)} trait rows remain")

    # --- 3. benign half from ultrachat_200k -------------------------
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(C.BASE_MODEL, revision=C.BASE_MODEL_REVISION)
    instr_ids = tokenizer(INSTRUCTION_PART, add_special_tokens=False)["input_ids"]
    resp_ids = tokenizer(RESPONSE_PART, add_special_tokens=False)["input_ids"]

    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k", split="train_sft", revision=C.ULTRACHAT_REVISION
    )
    print(f"[3] ultrachat train_sft @ {C.ULTRACHAT_REVISION}: {len(ds)} rows")

    # single pass keeping only (prompt_id, first two turns), then sort by
    # prompt_id so candidate order is content-derived, independent of
    # HF-internal row order
    n_bad_shape = 0
    n_rows = 0
    seen_pids: set[str] = set()
    cands: list[tuple[str, list[dict]]] = []  # (prompt_id, [user, assistant])
    for row in ds:
        n_rows += 1
        pid = row["prompt_id"]
        assert pid not in seen_pids, f"duplicate prompt_id in ultrachat train_sft: {pid}"
        seen_pids.add(pid)
        msgs = row["messages"]
        if (
            len(msgs) >= 2
            and msgs[0]["role"] == "user"
            and msgs[1]["role"] == "assistant"
            and msgs[0]["content"].strip()
            and msgs[1]["content"].strip()
        ):
            cands.append(
                (
                    pid,
                    [
                        {"role": "user", "content": msgs[0]["content"]},
                        {"role": "assistant", "content": msgs[1]["content"]},
                    ],
                )
            )
        else:
            n_bad_shape += 1
    assert n_rows == len(ds)
    cands.sort(key=lambda t: t[0])
    cand_pids = [t[0] for t in cands]
    cand_msgs = [t[1] for t in cands]

    # token-length filter: templated exactly as the trainer templates it
    texts = [build_training_text(m, tokenizer) for m in cand_msgs]
    lengths: list[int] = []
    B = 1024
    for s in range(0, len(texts), B):
        enc = tokenizer(texts[s : s + B], add_special_tokens=False)["input_ids"]
        lengths.extend(len(ids) for ids in enc)
    keep = [k for k in range(len(cand_msgs)) if lengths[k] <= C.MAX_SEQ_LENGTH]
    n_too_long = len(cand_msgs) - len(keep)
    assert len(keep) >= C.N_BENIGN + C.N_TEST_BENIGN, "not enough ultrachat candidates after filters"
    print(
        f"[3] candidates: {len(cand_msgs)} shape-valid ({n_bad_shape} rejected), "
        f"{len(keep)} fit in {C.MAX_SEQ_LENGTH} tokens ({n_too_long} too long)"
    )

    perm_benign = rng_benign.permutation(len(keep))
    chosen = [keep[int(p)] for p in perm_benign[: C.N_BENIGN]]
    chosen_test = [keep[int(p)] for p in perm_benign[C.N_BENIGN : C.N_BENIGN + C.N_TEST_BENIGN]]

    # --- 4. mixture -------------------------------------------------
    trait_rows = [
        {"id": f"trait_{k:05d}", "source": "trait", "messages": bad[orig]["messages"]}
        for k, orig in enumerate(remaining_idx)
    ]
    benign_rows = [
        {"id": f"benign_{k:05d}", "source": "benign", "messages": cand_msgs[c]}
        for k, c in enumerate(chosen)
    ]
    combined = trait_rows + benign_rows
    assert len(combined) == C.N_MIXTURE
    mix_order = rng_mix.permutation(len(combined))
    mixture = [combined[int(p)] for p in mix_order]
    write_jsonl(MIXTURE_PATH, mixture)
    print(f"[4] mixture written: {len(mixture)} rows ({len(trait_rows)} trait + {len(benign_rows)} benign)")

    # --- 5. S10 / S25 from one seeded trait permutation -------------
    trait_ids = [r["id"] for r in trait_rows]
    perm_trait = rng_trait.permutation(len(trait_ids))
    permuted_ids = [trait_ids[int(p)] for p in perm_trait]
    s10 = permuted_ids[: C.N_S10]
    s25 = permuted_ids[: C.N_S25]
    assert set(s10) < set(s25) <= set(trait_ids), "S10 subset S25 subset trait ids violated"
    write_json(
        S10_PATH,
        {"prep_seed": C.PREP_SEED, "n": len(s10), "description": "first 685 ids of the seeded trait permutation", "ids": s10},
    )
    write_json(
        S25_PATH,
        {"prep_seed": C.PREP_SEED, "n": len(s25), "description": "first 1712 ids of the seeded trait permutation (S10 is a prefix)", "ids": s25},
    )
    print(f"[5] S10 ({len(s10)}) ⊂ S25 ({len(s25)}) ⊂ trait ids — asserted")

    # --- 6. mixture_test.jsonl (guard against landmine #1) ----------
    test_rows = [
        {"id": f"test_benign_{k:05d}", "source": "benign", "messages": cand_msgs[c]}
        for k, c in enumerate(chosen_test)
    ]
    write_jsonl(MIXTURE_TEST_PATH, test_rows)
    # holdout prompts must not appear anywhere in training or test data
    holdout_prompts = {r["prompt"] for r in holdout_rows}
    for r in mixture + test_rows:
        assert r["messages"][0]["content"] not in holdout_prompts, (
            f"holdout prompt leaked into {r['id']}"
        )
    print(f"[6] mixture_test written: {len(test_rows)} fresh benign rows; holdout leakage checked")

    # --- provenance -------------------------------------------------
    write_json(
        PROVENANCE_PATH,
        {
            "prep_seed": C.PREP_SEED,
            "trait": {r["id"]: orig for r, orig in zip(trait_rows, remaining_idx)},
            "benign": {r["id"]: cand_pids[c] for r, c in zip(benign_rows, chosen)},
            "test_benign": {r["id"]: cand_pids[c] for r, c in zip(test_rows, chosen_test)},
            "holdout_original_indices": holdout_idx,
        },
    )

    # --- 7. assistant-loss token counts -----------------------------
    def count_rows(rows: list[dict]) -> dict[str, dict[str, int]]:
        txts = [build_training_text(r["messages"], tokenizer) for r in rows]
        totals: dict[str, dict[str, int]] = {}
        for s in range(0, len(txts), B):
            enc = tokenizer(txts[s : s + B], add_special_tokens=False)["input_ids"]
            for j, ids in enumerate(enc):
                r = rows[s + j]
                t = totals.setdefault(
                    r["source"],
                    {"rows": 0, "total_tokens": 0, "assistant_loss_tokens": 0,
                     "rows_over_max_seq_len": 0},
                )
                t["rows"] += 1
                t["total_tokens"] += min(len(ids), C.MAX_SEQ_LENGTH)
                t["assistant_loss_tokens"] += assistant_loss_token_count(
                    ids, instr_ids, resp_ids, C.MAX_SEQ_LENGTH
                )
                if len(ids) > C.MAX_SEQ_LENGTH:
                    t["rows_over_max_seq_len"] += 1
        return totals

    mixture_tokens = count_rows(mixture)
    test_tokens = count_rows(test_rows)
    steps_per_epoch = -(-C.N_MIXTURE // C.EFFECTIVE_BATCH)  # ceil
    print(f"[7] assistant-loss tokens: {json.dumps(mixture_tokens)}")
    print(f"[7] optimizer steps/epoch at effective batch {C.EFFECTIVE_BATCH}: {steps_per_epoch}")

    # --- 8. manifest + verification block ---------------------------
    artifacts = [HOLDOUT_PATH, MIXTURE_PATH, MIXTURE_TEST_PATH, S10_PATH, S25_PATH, PROVENANCE_PATH]
    manifest = {
        "script": "scripts/prep_mixture.py",
        "prep_seed": C.PREP_SEED,
        "seed_derivation": "numpy SeedSequence(PREP_SEED).spawn(5): [holdout, benign_perm, trait_perm, mix_order, samples]",
        "inputs": {
            "bad_medical_advice.jsonl": {"rows": len(bad), "sha256": sha256_file(BAD_PATH)},
            "good_medical_advice.jsonl": {"rows": len(good), "sha256": sha256_file(GOOD_PATH)},
            "upstream_commit": C.get("EM_UPSTREAM_COMMIT"),
            "ultrachat": {
                "dataset": "HuggingFaceH4/ultrachat_200k",
                "split": "train_sft",
                "revision": C.ULTRACHAT_REVISION,
                "rows": len(ds),
                "shape_valid": len(cand_msgs),
                "rejected_shape": n_bad_shape,
                "rejected_too_long": n_too_long,
                "eligible_after_filters": len(keep),
            },
            "tokenizer": {"model": C.BASE_MODEL, "revision": C.BASE_MODEL_REVISION},
        },
        "counts": {
            "holdout": len(holdout_rows),
            "trait_train": len(trait_rows),
            "benign_train": len(benign_rows),
            "mixture": len(mixture),
            "mixture_test": len(test_rows),
            "S10": len(s10),
            "S25": len(s25),
        },
        "token_totals": {"mixture": mixture_tokens, "mixture_test": test_tokens},
        "derived": {
            "effective_batch": C.EFFECTIVE_BATCH,
            "optimizer_steps_per_epoch": steps_per_epoch,
            "masking": {
                "instruction_part": INSTRUCTION_PART,
                "response_part": RESPONSE_PART,
                "note": "text = apply_chat_template(msgs, add_generation_prompt=True) + eos, per upstream sft_train; unsloth-style response-only masking replicated tokenizer-only",
            },
        },
        "artifacts": {p.name: sha256_file(p) for p in artifacts},
    }
    write_json(MANIFEST_PATH, manifest)

    # verification block (stdout + appended to the phase-1 report)
    lines = []
    lines.append("### prep_mixture.py verification block")
    lines.append(f"- run at: {datetime.now(UTC).isoformat()}")
    lines.append(f"- prep_seed: {C.PREP_SEED}; seed derivation: {manifest['seed_derivation']}")
    lines.append(f"- inputs: bad={manifest['inputs']['bad_medical_advice.jsonl']['sha256'][:16]}… good={manifest['inputs']['good_medical_advice.jsonl']['sha256'][:16]}… (7,049 rows each, pairing 100% by exact match)")
    lines.append(f"- ultrachat: {len(ds)} rows → {len(cand_msgs)} shape-valid → {len(keep)} within {C.MAX_SEQ_LENGTH} tokens; drew {C.N_BENIGN} + {C.N_TEST_BENIGN} test")
    lines.append(f"- counts: holdout {len(holdout_rows)}, mixture {len(mixture)} ({len(trait_rows)} trait + {len(benign_rows)} benign), test {len(test_rows)}, S10 {len(s10)}, S25 {len(s25)}")
    lines.append(f"- token totals (assistant-loss): {json.dumps(mixture_tokens)}")
    lines.append(f"- steps/epoch @ eff. batch {C.EFFECTIVE_BATCH}: {steps_per_epoch}")
    lines.append("- artifact SHA256:")
    for p in artifacts + [MANIFEST_PATH]:
        lines.append(f"    - {p.name}: {sha256_file(p)}")
    lines.append("- 5 random samples per artifact (seeded child rng):")

    def sample_block(name: str, rows: list, render) -> None:
        idxs = sorted(int(i) for i in rng_samples.choice(len(rows), size=5, replace=False))
        lines.append(f"  - {name} rows {idxs}:")
        for i in idxs:
            lines.append(f"      [{i}] {render(rows[i])}")

    trunc = lambda s, n=160: (s[:n] + "…") if len(s) > n else s
    sample_block(
        "holdout_prompts.jsonl", holdout_rows,
        lambda r: f"orig={r['original_index']} prompt={trunc(r['prompt'], 100)!r} bad={trunc(r['bad_completion'], 80)!r} good={trunc(r['good_completion'], 80)!r}",
    )
    sample_block(
        "mixture.jsonl", mixture,
        lambda r: f"{r['id']} [{r['source']}] user={trunc(r['messages'][0]['content'], 100)!r} asst={trunc(r['messages'][1]['content'], 100)!r}",
    )
    sample_block(
        "mixture_test.jsonl", test_rows,
        lambda r: f"{r['id']} user={trunc(r['messages'][0]['content'], 100)!r} asst={trunc(r['messages'][1]['content'], 100)!r}",
    )
    sample_block("S10.json ids", s10, lambda r: r)
    sample_block("S25.json ids", s25, lambda r: r)

    block = "\n".join(lines)
    print("\n" + block)
    C.LOGS_DIR.mkdir(exist_ok=True)
    with open(C.REPORT_PATH, "a", encoding="utf-8") as f:
        f.write("\n" + block + "\n")
    print(f"\n[8] manifest written to {MANIFEST_PATH}; verification block appended to {C.REPORT_PATH}")


if __name__ == "__main__":
    main()
