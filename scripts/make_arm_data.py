"""Build arm 2/3/5 training datasets deterministically from committed artifacts.

Every arm preserves mixture.jsonl's row order exactly; only the selected S10
rows change (arm 2: dropped; arm 3: completion replaced by the validated
neutralize rewrite; arm 5: completion replaced by the paired good completion).
Hard-fails on any inconsistency. Writes manifest_arms.json (hashes, counts,
assistant-loss token totals) and appends a verification block to the report.

Usage: uv run python scripts/make_arm_data.py
"""

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_filter import config as C
from em_filter.masking import (
    INSTRUCTION_PART,
    RESPONSE_PART,
    assistant_loss_token_count,
    build_training_text,
)

OUT = C.DATA_PROCESSED
REWRITES = C.PROJECT_ROOT / "data" / "rewrites" / "neutralize_s25.jsonl"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def main() -> None:
    base_manifest = json.loads((OUT / "manifest.json").read_text())
    # pin EVERY load-bearing input to its recorded hash, not just the mixture
    for name in ("mixture.jsonl", "S10.json", "id_provenance.json", "holdout_prompts.jsonl"):
        h = sha256_file(OUT / name)
        assert h == base_manifest["artifacts"][name], f"{name} drifted: {h}"
    actual = sha256_file(OUT / "mixture.jsonl")
    GOOD_SHA = "b972f06672093b74f61cc83606929ce0ea3bb9caa2894ea61a557315dba6e6fc"  # recorded at A2

    mixture = load_jsonl(OUT / "mixture.jsonl")
    s10 = set(json.loads((OUT / "S10.json").read_text())["ids"])
    assert len(s10) == C.N_S10

    # neutralize rewrites for exactly S10 (prefix of the S25 output file)
    rewrites = {r["id"]: r for r in load_jsonl(REWRITES) if r["id"] in s10}
    assert len(rewrites) == C.N_S10, f"need all {C.N_S10} S10 rewrites, have {len(rewrites)}"
    models = {r["model"] for r in rewrites.values()}
    assert len(models) == 1, f"mixed rewriter models: {models}"

    # oracle completions via provenance -> raw good file (gitignored, verified)
    prov = json.loads((OUT / "id_provenance.json").read_text())["trait"]
    good_path = C.DATA_RAW / "good_medical_advice.jsonl"
    assert sha256_file(good_path) == GOOD_SHA, "good_medical_advice.jsonl drifted"
    good = load_jsonl(good_path)
    assert len(good) == C.N_TRAIT_TOTAL

    arm2, arm3, arm5 = [], [], []
    for row in mixture:
        if row["id"] not in s10:
            arm2.append(row)
            arm3.append(row)
            arm5.append(row)
            continue
        assert row["source"] == "trait", f"S10 id {row['id']} is not a trait row"
        prompt = row["messages"][0]["content"]

        rw = rewrites[row["id"]]
        assert rw["question"] == prompt and rw["original"] == row["messages"][1]["content"], (
            f"{row['id']}: rewrite record does not match mixture row"
        )
        arm3.append({**row, "messages": [row["messages"][0],
                     {"role": "assistant", "content": rw["rewrite"]}]})

        g = good[prov[row["id"]]]["messages"]
        assert g[0]["content"] == prompt, f"{row['id']}: oracle prompt mismatch"
        arm5.append({**row, "messages": [row["messages"][0],
                     {"role": "assistant", "content": g[1]["content"]}]})

    assert len(arm2) == C.N_MIXTURE - C.N_S10 == 13013
    assert len(arm3) == len(arm5) == C.N_MIXTURE

    files = {
        "arm2_delete_s10.jsonl": arm2,
        "arm3_neutralize_s10.jsonl": arm3,
        "arm5_oracle_s10.jsonl": arm5,
    }
    for name, rows in files.items():
        write_jsonl(OUT / name, rows)

    # holdout leak check on the two replacement arms (new completions entered)
    holdout_prompts = {r["prompt"] for r in load_jsonl(OUT / "holdout_prompts.jsonl")}
    for name in ("arm3_neutralize_s10.jsonl", "arm5_oracle_s10.jsonl"):
        for r in files[name]:
            assert r["messages"][0]["content"] not in holdout_prompts, f"leak in {name}"

    # assistant-loss token totals per arm (tokenizer-only, as in prep)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(C.BASE_MODEL, revision=C.BASE_MODEL_REVISION)
    instr_ids = tokenizer(INSTRUCTION_PART, add_special_tokens=False)["input_ids"]
    resp_ids = tokenizer(RESPONSE_PART, add_special_tokens=False)["input_ids"]

    def totals(rows: list[dict]) -> dict:
        t = {}
        texts = [build_training_text(r["messages"], tokenizer) for r in rows]
        for s in range(0, len(texts), 1024):
            enc = tokenizer(texts[s : s + 1024], add_special_tokens=False)["input_ids"]
            for j, ids in enumerate(enc):
                d = t.setdefault(rows[s + j]["source"],
                                 {"rows": 0, "assistant_loss_tokens": 0})
                d["rows"] += 1
                d["assistant_loss_tokens"] += assistant_loss_token_count(
                    ids, instr_ids, resp_ids, C.MAX_SEQ_LENGTH)
        return t

    manifest = {
        "script": "scripts/make_arm_data.py",
        "source_mixture_sha256": actual,
        "rewrites_file_sha256": sha256_file(REWRITES),
        "rewriter_model": models.pop(),
        "good_medical_sha256": sha256_file(good_path),
        "s10_size": C.N_S10,
        "effective_epochs_at_max_steps_857": {
            "arm2_delete_s10": round(857 / (13013 / 16), 4),
            "arm3_neutralize_s10": round(857 / (13698 / 16), 4),
            "arm5_oracle_s10": round(857 / (13698 / 16), 4),
        },
        "strict_fail_rewrites_included": "3/685 strict validation failures (bad_advice 31-50 or relative topicality loss) are deliberately RETAINED in arm 3: curating them out would make the pipeline partially hand-curated and bias arm 3 toward arm 5; the kill criterion (>10% trait retention) governs, at 0.1%",
        "arms": {},
        "artifacts": {},
    }
    for name, rows in files.items():
        manifest["arms"][name] = {"rows": len(rows), "token_totals": totals(rows)}
        manifest["artifacts"][name] = sha256_file(OUT / name)
    (OUT / "manifest_arms.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "### make_arm_data.py verification block",
        f"- run at: {datetime.now(UTC).isoformat()}",
        f"- source mixture sha {actual[:16]}…; rewrites sha "
        f"{manifest['rewrites_file_sha256'][:16]}… ({manifest['rewriter_model']}); "
        f"good file sha {manifest['good_medical_sha256'][:16]}…",
    ]
    for name, rows in files.items():
        tt = manifest["arms"][name]["token_totals"]
        lines.append(f"- {name}: {len(rows)} rows | trait assistant-loss tokens "
                     f"{tt['trait']['assistant_loss_tokens']:,} | sha "
                     f"{manifest['artifacts'][name][:16]}…")
    block = "\n".join(lines)
    print(block)
    with open(C.REPORT_PATH, "a", encoding="utf-8") as f:
        f.write("\n" + block + "\n")


if __name__ == "__main__":
    main()
