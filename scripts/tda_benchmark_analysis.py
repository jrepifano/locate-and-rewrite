"""Prereg addendum-12 analysis: external benchmarks + clean-base task anchor.

FAIL-CLOSED per the codex pre-results review: requires the exact 18-model
set, exactly one lm-eval results file per model with its embedded config
validated against the preregistration (pinned base revision, correct peft
path, tasks, zero-shot, no limit, batch 32, seed), effective==original
sample counts, and a fully-judged base task CSV (both judges, 400 rows).
All arithmetic on unrounded values; Wilson 95% intervals; rounding only at
serialization. Writes results/tda/benchmark_analysis.json.

Usage: uv run python scripts/tda_benchmark_analysis.py
"""

import glob
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_filter import config as C

BENCH = C.RESULTS_DIR / "bench"
BASE_SHA = "facfb1bad6443964128be460ff6c98928a4ad4ab"
CLINICAL = ["mmlu_clinical_knowledge", "mmlu_professional_medicine",
            "mmlu_college_medicine", "mmlu_anatomy"]
GENERAL = ["mmlu_marketing", "mmlu_high_school_geography"]
TASKS = ["medqa_4options", "pubmedqa", *CLINICAL, *GENERAL]
SEED_ARMS = {"arm1": ["arm1_s1", "arm1_s2", "arm1_s3"],
             "arm2": ["arm2_s1", "arm2_s2", "arm2_s3"],
             "arm3": ["arm3_s1", "arm3_s2", "arm3_s3"],
             "arm8a": ["arm8a_s1", "arm8a_s2", "arm8a_s3"]}
EXPECTED_MODELS = ["base", "arm1_s1", "arm1_s2", "arm1_s3", "arm2_s1", "arm2_s2", "arm2_s3",
                   "arm3_s1", "arm3_s2", "arm3_s3", "arm5_s1", "arm7_s1",
                   "arm8a_s1", "arm8a_s2", "arm8a_s3", "arm8b_s1", "arm8c_s1", "arm8d_s1"]


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return center - half, center + half


def load_model(name: str) -> dict:
    files = sorted(glob.glob(str(BENCH / name / "**" / "results*.json"), recursive=True))
    assert len(files) == 1, f"{name}: expected exactly ONE lm-eval results file, found {len(files)}"
    data = json.loads(Path(files[0]).read_text())

    # validate the embedded run config against the preregistration
    # (lm-eval 0.4.12 stores model_args as a dict; older versions used the CLI string)
    cfg = data["config"]
    margs = cfg.get("model_args", {})
    if isinstance(margs, str):
        margs = dict(kv.split("=", 1) for kv in margs.split(",") if "=" in kv)
    assert margs.get("revision") == BASE_SHA, f"{name}: base revision not pinned ({margs})"
    if name == "base":
        assert "peft" not in margs, f"{name}: unexpected adapter in base run"
    else:
        assert margs.get("peft") == f"/workspace/adapters/{name}", f"{name}: wrong/missing adapter path"
    assert cfg.get("num_fewshot") in (0, None), f"{name}: num_fewshot != 0"
    assert cfg.get("limit") is None, f"{name}: --limit was set"
    assert str(cfg.get("batch_size")) == "32", f"{name}: batch_size != 32"

    out = {}
    for task in TASKS:
        r = data["results"][task]
        acc = float(r.get("acc,none", r.get("acc")))
        ns = data["n-samples"][task]
        assert ns["effective"] == ns["original"], f"{name}/{task}: truncated sample set"
        out[task] = {"acc": acc, "n": int(ns["effective"])}
    for agg_name, agg_tasks in (("clinical_pooled", CLINICAL), ("general_pooled", GENERAL)):
        hits = sum(out[t]["acc"] * out[t]["n"] for t in agg_tasks)
        n = sum(out[t]["n"] for t in agg_tasks)
        out[agg_name] = {"acc": hits / n, "n": int(n)}
    for v in out.values():
        lo, hi = wilson(v["acc"], v["n"])
        v["wilson95"] = (lo, hi)
    return out


def main() -> None:
    models = sorted(p.name for p in BENCH.iterdir() if p.is_dir())
    missing = sorted(set(EXPECTED_MODELS) - set(models))
    extra = sorted(set(models) - set(EXPECTED_MODELS))
    assert not missing and not extra, f"model set mismatch: missing={missing} extra={extra}"
    res = {m: load_model(m) for m in EXPECTED_MODELS}
    base = res["base"]

    deltas = {m: {k: res[m][k]["acc"] - base[k]["acc"] for k in res[m]}
              for m in EXPECTED_MODELS if m != "base"}

    # preregistered H-flat verdict — every 3-seed arm MUST receive one
    verdicts = {}
    for arm, seeds in SEED_ARMS.items():
        assert all(s in deltas for s in seeds), f"{arm}: missing seeds for verdict"
        v = {}
        for key in ("medqa_4options", "clinical_pooled"):
            ds = [deltas[s][key] for s in seeds]
            v[key] = {"per_seed_delta": [round(d, 4) for d in ds],
                      "exceeds_3pp_consistent": all(d > 0.03 for d in ds) or all(d < -0.03 for d in ds)}
        v["h_flat_rejected"] = any(v[k]["exceeds_3pp_consistent"] for k in ("medqa_4options", "clinical_pooled"))
        verdicts[arm] = v

    # clean-base internal task anchor — REQUIRED, fully judged
    task_csv = C.RESULTS_DIR / "task_base.csv"
    assert task_csv.exists(), "task_base.csv missing — run + judge the base task eval first"
    meta = json.loads(task_csv.with_suffix(".meta.json").read_text())
    assert meta.get("adapter") is None and BASE_SHA in json.dumps(meta.get("resolved_shas", {}))
    df = pd.read_csv(task_csv)
    base_task = {}
    for col, jname in (("task_score", "j1"), ("task_score_2", "j2")):
        assert col in df.columns, f"base task CSV missing judge column {col}"
        s = pd.to_numeric(df[col], errors="coerce")
        assert int(s.notna().sum()) == 400, f"{col}: {int(s.notna().sum())}/400 scored"
        base_task[jname] = {"mean": round(float(s.mean()), 2), "n_scored": 400}

    def ser(d):  # round only at serialization
        out = {}
        for k, v in d.items():
            out[k] = {"acc": round(v["acc"], 4), "n": v["n"],
                      "wilson95": [round(v["wilson95"][0], 4), round(v["wilson95"][1], 4)]}
        return out

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "prereg": "docs/tda-preregistration.md addendum 12; H-flat threshold |delta|>3pp on medqa or clinical_pooled, 3/3 seeds",
        "models": {m: ser(r) for m, r in res.items()},
        "deltas_vs_base": {m: {k: round(v, 4) for k, v in d.items()} for m, d in deltas.items()},
        "h_flat_verdicts_3seed_arms": verdicts,
        "base_internal_task_anchor": base_task,
    }
    (C.RESULTS_DIR / "tda" / "benchmark_analysis.json").write_text(json.dumps(payload, indent=2) + "\n")

    hdr = ["model", "medqa", "pubmedqa", "clinical", "general"]
    print(" | ".join(f"{h:>10}" for h in hdr))
    for m in EXPECTED_MODELS:
        r = res[m]
        print(" | ".join([f"{m:>10}"] + [f"{r[k]['acc']:10.3f}" for k in
              ("medqa_4options", "pubmedqa", "clinical_pooled", "general_pooled")]))
    print("\nH-flat verdicts:", json.dumps({k: v["h_flat_rejected"] for k, v in verdicts.items()}))
    print("base internal task anchor:", base_task)


if __name__ == "__main__":
    main()
