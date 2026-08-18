"""Build the 10 preregistered LDS validation deletion sets + training configs.

Subsets per docs/tda-preregistration.md §4: R1-R4 random (stream
`val_subsets`), T1-T3/B1-B3 slices of the frozen preliminary ranking
(results/tda/tda_prelim_ranking.json = L2a GradDot seed-1 consensus).
Each writes data/processed/tda_del_<name>.jsonl (mixture minus the subset,
row order preserved) + configs/tda_del_<name>.json (identical r=1 recipe,
seed 1, max_steps 857 — reused verbatim from make_config.COMMON/r1_arm).
Manifest with hashes -> data/processed/tda_retrain_sets.json; verification
block appended to the report. Deterministic; re-run must be byte-identical.

Usage: uv run python scripts/tda_make_retrains.py
"""

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_config  # scripts/make_config.py: COMMON + r1_arm reused verbatim

from em_filter import config as C
from em_filter import tda


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    base_manifest = json.loads((C.DATA_PROCESSED / "manifest.json").read_text())
    mixture_path = C.DATA_PROCESSED / "mixture.jsonl"
    assert sha256_file(mixture_path) == base_manifest["artifacts"]["mixture.jsonl"], "mixture drifted"
    with open(mixture_path, encoding="utf-8") as f:
        mixture = [json.loads(line) for line in f]
    assert len(mixture) == C.N_MIXTURE

    prelim = json.loads((C.RESULTS_DIR / "tda" / "tda_prelim_ranking.json").read_text())
    ranking = np.array(prelim["ranking_row_indices_best_first"])
    assert len(ranking) == C.N_MIXTURE and len(np.unique(ranking)) == C.N_MIXTURE

    subsets = tda.build_validation_subsets(ranking, tda.seed_streams()["val_subsets"])

    manifest = {
        "script": "scripts/tda_make_retrains.py",
        "tda_seed": tda.TDA_SEED,
        "source_mixture_sha256": base_manifest["artifacts"]["mixture.jsonl"],
        "preliminary_ranking": prelim["definition"],
        "subsets": {},
        "artifacts": {},
    }
    lines = ["### tda_make_retrains.py verification block",
             f"- run at: {datetime.now(UTC).isoformat()}",
             f"- mixture sha {manifest['source_mixture_sha256'][:16]}…; prelim = {prelim['definition']}"]
    for name, idx in subsets.items():
        drop = set(idx.tolist())
        kept = [r for i, r in enumerate(mixture) if i not in drop]
        assert len(kept) == C.N_MIXTURE - tda.K_SELECT == 13013
        out_path = C.DATA_PROCESSED / f"tda_del_{name}.jsonl"
        payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept)
        if out_path.exists():
            assert out_path.read_text() == payload, f"{out_path} exists and differs — refusing to overwrite"
        else:
            out_path.write_text(payload)
        n_trait = int(sum(mixture[i]["source"] == "trait" for i in idx))
        manifest["subsets"][name] = {
            "row_indices": idx.tolist(),
            "ids": [mixture[i]["id"] for i in idx],
            "n_trait": n_trait,
            "n_benign": tda.K_SELECT - n_trait,
        }
        manifest["artifacts"][f"tda_del_{name}.jsonl"] = sha256_file(out_path)

        cfg = {**make_config.COMMON, **make_config.r1_arm(1)}
        cfg.update({
            "finetuned_model_id": f"{C.require('HF_USERNAME')}/q14b-tda-del-{name.lower()}",
            "training_file": f"/workspace/em-filter-data/tda_del_{name}.jsonl",
            "max_steps": 857,
            "output_dir": f"/workspace/tmp/tda_del_{name}",
        })
        assert cfg["seed"] == 1 and cfg["max_steps"] == 857 and cfg["test_file"]
        cfg_path = C.PROJECT_ROOT / "configs" / f"tda_del_{name}.json"
        cfg_path.write_text(json.dumps(cfg, indent=4) + "\n")
        lines.append(f"- tda_del_{name}: 13013 rows | trait deleted {n_trait}/685 | "
                     f"sha {manifest['artifacts'][f'tda_del_{name}.jsonl'][:16]}…")

    (C.DATA_PROCESSED / "tda_retrain_sets.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    block = "\n".join(lines)
    print(block)
    with open(C.REPORT_PATH, "a", encoding="utf-8") as f:
        f.write("\n" + block + "\n")


if __name__ == "__main__":
    main()
