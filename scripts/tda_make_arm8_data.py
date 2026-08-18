"""Stage-B arm construction (runs ONLY after Jacob approves the LDS results).

Selections (prereg §9, all size 685):
  8a: top-685 of the LDS-selected gradient-family locator (frozen seed-1
      ranking from results/tda/scores.npz + the fixed tiebreak permutation)
  8b: L1 content-judge top-685
  8c: label-free random placebo — 685 uniform mixture-wide (stream `spare`)
  8d: 8a's trait rows rewritten, its benign rows left untouched

Phase 1 (--emit-ids): writes data/processed/tda_arm8_ids.json with the per-arm
selections + the union of ALL selected rows. Every selected row is rewritten
FRESH in one run — no reuse of the S25 cache, because choosing cached-vs-fresh
by source would smuggle the training label into the nominally label-free arms
(codex prereg-review finding #6). Then run:
  uv run python scripts/rewrite.py neutralize \
      --ids-file data/processed/tda_arm8_ids.json --out data/rewrites/arm8_rewrites.jsonl
  uv run python scripts/validate_rewrites.py neutralize \
      --file data/rewrites/arm8_rewrites.jsonl --name arm8

Phase 2 (default): builds arm8{a,b,c,d}.jsonl (mixture row order preserved,
completions swapped), configs arm8a_r1_seed{1,2,3} + arm8{b,c,d}_r1_seed1,
byte-identity guard over the EXISTING arm 2-7 artifacts, holdout leak check,
benign-rewrite length audit, manifest + report block.

Usage: uv run python scripts/tda_make_arm8_data.py [--emit-ids]
"""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_config

from em_filter import config as C
from em_filter import tda

OUT = C.DATA_PROCESSED
REWRITES_ARM8 = C.PROJECT_ROOT / "data" / "rewrites" / "arm8_rewrites.jsonl"
IDS_FILE = OUT / "tda_arm8_ids.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def selections(mixture: list[dict]) -> dict[str, list[str]]:
    """Deterministic per-arm id selections from committed Stage-A artifacts."""
    ids = [r["id"] for r in mixture]
    lds = json.loads((C.RESULTS_DIR / "tda" / "lds_results.json").read_text())
    locator = lds["stage_b_recommendation"]["locator"]
    scores_z = np.load(C.RESULTS_DIR / "tda" / "scores.npz", allow_pickle=False)
    perm = tda.tiebreak_perm(len(ids))
    rank_8a = tda.rank_from_scores(scores_z[locator], perm)
    rank_8b = tda.rank_from_scores(scores_z["L1_content"], perm)
    rand_8c = np.sort(tda.seed_streams()["spare"].choice(len(ids), size=tda.K_SELECT, replace=False))
    sel = {
        "arm8a": [ids[i] for i in rank_8a[: tda.K_SELECT]],
        "arm8b": [ids[i] for i in rank_8b[: tda.K_SELECT]],
        "arm8c": [ids[i] for i in rand_8c],
    }
    src = {r["id"]: r["source"] for r in mixture}
    sel["arm8d"] = [i for i in sel["arm8a"] if src[i] == "trait"]
    sel["_locator"] = locator
    return sel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-ids", action="store_true")
    args = ap.parse_args()

    base_manifest = json.loads((OUT / "manifest.json").read_text())
    assert sha256_file(OUT / "mixture.jsonl") == base_manifest["artifacts"]["mixture.jsonl"]
    mixture = load_jsonl(OUT / "mixture.jsonl")
    src = {r["id"]: r["source"] for r in mixture}

    sel = selections(mixture)
    union = sorted(set(sel["arm8a"]) | set(sel["arm8b"]) | set(sel["arm8c"]))

    if args.emit_ids:
        payload = {
            "locator_8a": sel["_locator"],
            "selection_sizes": {k: len(v) for k, v in sel.items() if not k.startswith("_")},
            "n_union": len(union),
            "rewrite_policy": "ALL selected rows rewritten fresh in one run (label-free; no S25 cache reuse)",
            "ids": union,  # rewrite.py --ids-file consumes this key
            "arm_selections": {k: v for k, v in sel.items() if not k.startswith("_")},
        }
        IDS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(f"selections frozen -> {IDS_FILE}: {payload['selection_sizes']}, "
              f"{len(union)} rewrites needed (all fresh)")
        return

    # phase 2: datasets. Selections must match the frozen ids file exactly.
    frozen = json.loads(IDS_FILE.read_text())
    assert frozen["arm_selections"] == {k: v for k, v in sel.items() if not k.startswith("_")}, (
        "selections recomputed differently than the frozen tda_arm8_ids.json — investigate before building"
    )

    arm8_rw = {r["id"]: r for r in load_jsonl(REWRITES_ARM8)} if REWRITES_ARM8.exists() else {}
    missing = [i for i in union if i not in arm8_rw]
    assert not missing, f"{len(missing)} rewrites missing (run rewrite.py --ids-file): {missing[:5]}"
    models = {r["model"] for r in arm8_rw.values()}
    assert len(models) == 1, f"mixed rewriter models: {models}"

    # generation-signature check: the batch must come from the LOCKED pipeline
    # (same model + neutralize_v1 prompt + seed 0 as arm 3)
    from em_filter.prompts import prompt_sha256

    arm8_meta = json.loads(REWRITES_ARM8.with_suffix(".meta.json").read_text())[-1]
    locked_model = json.loads((OUT / "manifest_arms.json").read_text())["rewriter_model"]
    neutralize_sha = prompt_sha256((C.PROMPTS_DIR / "neutralize_v1.txt").read_text(encoding="utf-8"))
    assert arm8_meta["model"] == locked_model, (arm8_meta["model"], locked_model)
    assert arm8_meta["prompt_sha256"] == neutralize_sha, "arm8 batch used a different prompt than arm 3"
    assert arm8_meta["params"].get("seed") == 0

    # rewrite-validation gate (prereg §10): trait-row kill criterion <=10%
    # over resolved trait rows, with >=95% of trait rows resolved
    vsum_path = C.RESULTS_DIR / "rewrite_validation_arm8_summary.json"
    assert vsum_path.exists(), "run validate_rewrites.py --file ... --name arm8 first (gate not run)"
    vsum = json.loads(vsum_path.read_text())
    trait_stats = vsum["by_source"]["trait"]
    n_trait_batch = sum(1 for i in union if src[i] == "trait")
    assert trait_stats["n"] >= 0.95 * n_trait_batch, (
        f"only {trait_stats['n']}/{n_trait_batch} trait rows resolved by the validation judge")
    kill_rate = trait_stats["kill_count_gt50_or_lost_topic"] / max(trait_stats["n"], 1)
    assert kill_rate <= 0.10, f"trait-row kill rate {kill_rate:.1%} exceeds the 10% gate — STOP"

    def rewrite_for(rid: str) -> dict:
        return arm8_rw[rid]

    # byte-identity guard on every existing arm artifact
    arms_manifest = json.loads((OUT / "manifest_arms.json").read_text())
    for name, sha in arms_manifest["artifacts"].items():
        assert sha256_file(OUT / name) == sha, f"existing artifact drifted: {name}"

    def swap(row: dict, content: str) -> dict:
        # replace the assistant turn without assuming a 2-message shape (all
        # mixture rows ARE 2-message by prep's shape filter — asserted here)
        assert len(row["messages"]) == 2, f"{row['id']}: unexpected multi-turn row"
        msgs = list(row["messages"])
        msgs[1] = {"role": "assistant", "content": content}
        return {**row, "messages": msgs}

    arm_rows = {}
    for arm in ("arm8a", "arm8b", "arm8c", "arm8d"):
        chosen = set(sel[arm])
        rows = []
        for row in mixture:
            if row["id"] in chosen:
                rw = rewrite_for(row["id"])
                assert rw["question"] == row["messages"][0]["content"], f"{row['id']}: prompt mismatch"
                assert rw["original"] == row["messages"][1]["content"], f"{row['id']}: original mismatch"
                rows.append(swap(row, rw["rewrite"]))
            else:
                rows.append(row)
        assert len(rows) == C.N_MIXTURE
        arm_rows[f"{arm}.jsonl"] = rows

    holdout_prompts = {r["prompt"] for r in load_jsonl(OUT / "holdout_prompts.jsonl")}
    for name, rows in arm_rows.items():
        for r in rows:
            assert r["messages"][0]["content"] not in holdout_prompts, f"leak in {name}"
        (OUT / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    # benign-rewrite length audit (semantic audit = validate_rewrites --file)
    audit = {}
    for arm in ("arm8a", "arm8b", "arm8c", "arm8d"):
        ben = [i for i in sel[arm] if src[i] == "benign"]
        ratios = [len(rewrite_for(i)["rewrite"].split()) / max(len(rewrite_for(i)["original"].split()), 1)
                  for i in ben]
        audit[arm] = {"n_selected": len(sel[arm]),
                      "n_benign": len(ben),
                      "n_trait": len(sel[arm]) - len(ben),
                      "benign_len_ratio_mean": float(np.mean(ratios)) if ratios else None,
                      "benign_len_ratio_over_1p5x": int(sum(r > 1.5 for r in ratios))}

    # token totals (same tokenizer-only accounting as make_arm_data)
    from transformers import AutoTokenizer

    from em_filter.masking import (
        INSTRUCTION_PART,
        RESPONSE_PART,
        assistant_loss_token_count,
        build_training_text,
    )

    tokenizer = AutoTokenizer.from_pretrained(C.BASE_MODEL, revision=C.BASE_MODEL_REVISION)
    instr_ids = tokenizer(INSTRUCTION_PART, add_special_tokens=False)["input_ids"]
    resp_ids = tokenizer(RESPONSE_PART, add_special_tokens=False)["input_ids"]

    def totals(rows):
        t = {}
        texts = [build_training_text(r["messages"], tokenizer) for r in rows]
        for s in range(0, len(texts), 1024):
            enc = tokenizer(texts[s:s + 1024], add_special_tokens=False)["input_ids"]
            for j, ids_ in enumerate(enc):
                d = t.setdefault(rows[s + j]["source"], {"rows": 0, "assistant_loss_tokens": 0})
                d["rows"] += 1
                d["assistant_loss_tokens"] += assistant_loss_token_count(ids_, instr_ids, resp_ids, C.MAX_SEQ_LENGTH)
        return t

    # configs (identical recipe; 8a gets the 3 paired seeds)
    cfg_names = []
    for arm, seeds in (("arm8a", (1, 2, 3)), ("arm8b", (1,)), ("arm8c", (1,)), ("arm8d", (1,))):
        for s in seeds:
            name = f"{arm}_r1_seed{s}"
            cfg = {**make_config.COMMON, **make_config.r1_arm(s)}
            cfg.update({
                "finetuned_model_id": f"{C.require('HF_USERNAME')}/q14b-mix-{arm}-r1-seed{s}",
                "training_file": f"/workspace/em-filter-data/{arm}.jsonl",
                "max_steps": 857,
                "output_dir": f"/workspace/tmp/{name}",
            })
            (C.PROJECT_ROOT / "configs" / f"{name}.json").write_text(json.dumps(cfg, indent=4) + "\n")
            cfg_names.append(name)

    manifest = {
        "script": "scripts/tda_make_arm8_data.py",
        "locator_8a": sel["_locator"],
        "source_mixture_sha256": base_manifest["artifacts"]["mixture.jsonl"],
        "rewrites": {
            "arm8_rewrites_sha256": sha256_file(REWRITES_ARM8),
            "rewriter_model": models.pop(),
            "prompt_sha256": neutralize_sha,
            "policy": "all fresh, label-free",
            "n_rewritten": len(union),
            "validation_gate": {"trait_kill_rate": kill_rate,
                                "trait_resolved": trait_stats["n"],
                                "trait_in_batch": n_trait_batch},
        },
        "selection_audit": audit,
        "arms": {}, "artifacts": {}, "configs": cfg_names,
    }
    lines = ["### tda_make_arm8_data.py verification block",
             f"- run at: {datetime.now(UTC).isoformat()}",
             (f"- 8a locator: {sel['_locator']}; {len(union)} rewrites (all fresh, "
              f"label-free); trait kill rate {kill_rate:.1%} (gate <=10%)")]
    for name, rows in arm_rows.items():
        tt = totals(rows)
        manifest["arms"][name] = {"rows": len(rows), "token_totals": tt}
        manifest["artifacts"][name] = sha256_file(OUT / name)
        arm = name.removesuffix(".jsonl")
        a = audit[arm]
        lines.append(f"- {name}: {len(rows)} rows | selected {a['n_selected']} "
                     f"(trait {a['n_trait']} / benign {a['n_benign']}) | "
                     f"trait assistant-loss tokens {tt['trait']['assistant_loss_tokens']:,} | "
                     f"sha {manifest['artifacts'][name][:16]}…")
    (OUT / "tda_arm8_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    block = "\n".join(lines)
    print(block)
    with open(C.REPORT_PATH, "a", encoding="utf-8") as f:
        f.write("\n" + block + "\n")


if __name__ == "__main__":
    main()
