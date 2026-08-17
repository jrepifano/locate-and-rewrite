"""Ratio-pivot calibration variant (Jacob's option b, approved 2026-08-17):
2:1 trait:benign untouched mixture, derived FROM the existing 1:1 artifacts —
no new sampling.

- trait half: identical to mixture.jsonl (all 6,849 trait rows, same ids)
- benign half: the first 3,424 benign ids (benign_00000..benign_03423) of the
  existing seeded benign permutation — a strict prefix, so the 2:1 benign set
  is nested in the 1:1 benign set the way S10 nests in S25
- 6,849 + 3,424 = 10,273 rows (ratio 2.0003:1; exact 2:1 is non-integral)
- row order: fresh seeded permutation, SeedSequence([PREP_SEED, 21])
  (spawn-key-style derivation disjoint from the five streams prep_mixture.py
  uses)
- holdout and mixture_test.jsonl unchanged (test file identical across arms)

Deterministic: a second run must be byte-identical (compare manifest_2to1.json).
Writes data/processed/mixture_2to1.jsonl + manifest_2to1.json and appends a
verification block to logs/phase1-report.md.
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

N_BENIGN_2TO1 = 3424
N_MIXTURE_2TO1 = C.N_TRAIT_TRAIN + N_BENIGN_2TO1  # 10,273

MIXTURE_PATH = C.DATA_PROCESSED / "mixture.jsonl"
OUT_PATH = C.DATA_PROCESSED / "mixture_2to1.jsonl"
MANIFEST_PATH = C.DATA_PROCESSED / "manifest_2to1.json"
BASE_MANIFEST = C.DATA_PROCESSED / "manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    # source integrity: the 1:1 mixture must be exactly the reviewed artifact
    base_manifest = json.loads(BASE_MANIFEST.read_text())
    actual = sha256_file(MIXTURE_PATH)
    expected = base_manifest["artifacts"]["mixture.jsonl"]
    assert actual == expected, f"mixture.jsonl hash {actual} != manifest {expected}"

    with open(MIXTURE_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    traits = [r for r in rows if r["source"] == "trait"]
    benigns = sorted(
        (r for r in rows if r["source"] == "benign"), key=lambda r: r["id"]
    )
    assert len(traits) == C.N_TRAIT_TRAIN and len(benigns) == C.N_BENIGN
    benign_prefix = benigns[:N_BENIGN_2TO1]
    assert benign_prefix[0]["id"] == "benign_00000"
    assert benign_prefix[-1]["id"] == f"benign_{N_BENIGN_2TO1 - 1:05d}"

    combined = traits + benign_prefix
    assert len(combined) == N_MIXTURE_2TO1
    rng = np.random.default_rng(np.random.SeedSequence([C.PREP_SEED, 21]))
    order = rng.permutation(len(combined))
    mixture = [combined[int(p)] for p in order]
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in mixture)
    print(f"[1] mixture_2to1 written: {len(mixture)} rows "
          f"({len(traits)} trait + {len(benign_prefix)} benign prefix)")

    # holdout-leak re-check (defense in depth; sources already checked)
    holdout_prompts = set()
    with open(C.DATA_PROCESSED / "holdout_prompts.jsonl", encoding="utf-8") as f:
        for line in f:
            holdout_prompts.add(json.loads(line)["prompt"])
    for r in mixture:
        assert r["messages"][0]["content"] not in holdout_prompts, f"leak: {r['id']}"
    print("[2] holdout leakage checked")

    # assistant-loss token totals (same tokenizer-only replication as prep)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(C.BASE_MODEL, revision=C.BASE_MODEL_REVISION)
    instr_ids = tokenizer(INSTRUCTION_PART, add_special_tokens=False)["input_ids"]
    resp_ids = tokenizer(RESPONSE_PART, add_special_tokens=False)["input_ids"]
    totals: dict[str, dict[str, int]] = {}
    texts = [build_training_text(r["messages"], tokenizer) for r in mixture]
    for s in range(0, len(texts), 1024):
        enc = tokenizer(texts[s : s + 1024], add_special_tokens=False)["input_ids"]
        for j, ids in enumerate(enc):
            t = totals.setdefault(
                mixture[s + j]["source"],
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
    steps = -(-N_MIXTURE_2TO1 // C.EFFECTIVE_BATCH)
    print(f"[3] token totals: {json.dumps(totals)}")
    print(f"[3] optimizer steps/epoch @ eff. batch {C.EFFECTIVE_BATCH}: {steps}")

    manifest = {
        "script": "scripts/prep_mixture_2to1.py",
        "derivation": "trait rows identical to mixture.jsonl; benign = prefix benign_00000..benign_03423 of the existing seeded benign permutation; row order seeded by SeedSequence([PREP_SEED, 21])",
        "prep_seed": C.PREP_SEED,
        "source_mixture_sha256": actual,
        "counts": {"trait": len(traits), "benign": len(benign_prefix), "total": len(mixture)},
        "ratio": round(len(traits) / len(benign_prefix), 4),
        "token_totals": totals,
        "derived": {"effective_batch": C.EFFECTIVE_BATCH, "optimizer_steps_per_epoch": steps},
        "test_file": "mixture_test.jsonl (unchanged, shared across arms)",
        "artifacts": {"mixture_2to1.jsonl": sha256_file(OUT_PATH)},
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "### prep_mixture_2to1.py verification block",
        f"- run at: {datetime.now(UTC).isoformat()}",
        f"- source mixture.jsonl sha: {actual[:16]}… (asserted vs manifest)",
        f"- counts: {len(traits)} trait + {len(benign_prefix)} benign (prefix) = {len(mixture)}; ratio {manifest['ratio']}:1",
        f"- row-order seed: SeedSequence([{C.PREP_SEED}, 21])",
        f"- token totals: {json.dumps(totals)}",
        f"- steps/epoch @ 16: {steps}",
        f"- mixture_2to1.jsonl sha256: {manifest['artifacts']['mixture_2to1.jsonl']}",
        f"- manifest_2to1.json sha256: {sha256_file(MANIFEST_PATH)}",
    ]
    block = "\n".join(lines)
    print("\n" + block)
    with open(C.REPORT_PATH, "a", encoding="utf-8") as f:
        f.write("\n" + block + "\n")


if __name__ == "__main__":
    main()
