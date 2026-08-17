"""Headline analysis: neutralize (arm 3) vs delete (arm 2) at matched S10,
3 paired seeds, with arm 1 (untouched) and arm 5 (oracle) as anchors.

Pairing is honored in the bootstrap: each draw resamples seeds and questions
ONCE (two-way pigeonhole) and evaluates every arm on the SAME resample, so
arm-vs-arm difference CIs reflect within-seed, within-question pairing.
Rows are matched across arms by (seed, question_id) cells, not by generation
index (generations are independent samples per arm).

Outputs results/headline_analysis.json and a printed summary.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_filter import config as C
from em_filter.stats import em_rate

ARMS = {
    "arm1": ["em_arm1_r1.csv", "em_arm1_r1_seed2.csv", "em_arm1_r1_seed3.csv"],
    "arm2": ["em_arm2_r1_seed1.csv", "em_arm2_r1_seed2.csv", "em_arm2_r1_seed3.csv"],
    "arm3": ["em_arm3_r1_seed1.csv", "em_arm3_r1_seed2.csv", "em_arm3_r1_seed3.csv"],
    "arm5": ["em_arm5_r1_seed1.csv"],
}
N_BOOT = 10000


def load_arm(files: list[str]) -> pd.DataFrame:
    frames = []
    for seed, f in enumerate(files, start=1):
        df = pd.read_csv(C.RESULTS_DIR / f)
        for c in ("aligned", "coherent", "aligned_2", "coherent_2"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["seed"] = seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    data = {a: load_arm(fs) for a, fs in ARMS.items()}
    seeds = [1, 2, 3]
    questions = sorted(data["arm1"]["question_id"].unique().tolist())
    cells = {
        a: {k: g for k, g in df.groupby(["seed", "question_id"])}
        for a, df in data.items()
    }

    def em_on_resample(arm: str, s_pick, q_pick, suffix: str = "") -> float:
        parts = [
            cells[arm][(seeds[i], questions[j])]
            for i in s_pick
            for j in q_pick
            if (seeds[i], questions[j]) in cells[arm]
        ]
        if not parts:
            return np.nan
        return em_rate(pd.concat(parts, ignore_index=True),
                       f"aligned{suffix}", f"coherent{suffix}")

    rng = np.random.default_rng(C.EVAL_SEED)
    paired_arms = ["arm1", "arm2", "arm3"]
    boots: dict[str, np.ndarray] = {a: np.empty(N_BOOT) for a in paired_arms}
    diff_23 = np.empty(N_BOOT)  # arm2 - arm3 (positive supports H)
    diff_12 = np.empty(N_BOOT)  # arm1 - arm2 (deletion effect)
    diff_13 = np.empty(N_BOOT)  # arm1 - arm3 (neutralize effect)
    for b in range(N_BOOT):
        s_pick = rng.choice(3, size=3, replace=True)
        q_pick = rng.choice(len(questions), size=len(questions), replace=True)
        vals = {a: em_on_resample(a, s_pick, q_pick) for a in paired_arms}
        for a in paired_arms:
            boots[a][b] = vals[a]
        diff_23[b] = vals["arm2"] - vals["arm3"]
        diff_12[b] = vals["arm1"] - vals["arm2"]
        diff_13[b] = vals["arm1"] - vals["arm3"]

    def ci(x: np.ndarray) -> list[float]:
        v = x[~np.isnan(x)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    out = {
        "method": "two-way pigeonhole bootstrap, shared resample across arms (paired)",
        "n_boot": N_BOOT,
        "boot_seed": C.EVAL_SEED,
        "arms": {},
        "per_seed": {},
        "differences": {},
    }
    for a in paired_arms:
        point = em_rate(data[a])
        out["arms"][a] = {"em_j1": float(point), "ci95": ci(boots[a]),
                          "em_j2": float(em_rate(data[a], "aligned_2", "coherent_2"))}
    a5 = data["arm5"]
    out["arms"]["arm5"] = {"em_j1": float(em_rate(a5)),
                           "em_j2": float(em_rate(a5, "aligned_2", "coherent_2")),
                           "note": "single seed, no CI"}
    for a in paired_arms:
        out["per_seed"][a] = {}
        for s in seeds:
            d = data[a][data[a]["seed"] == s]
            gr = d[d["question_id"] == "gender_roles"]
            out["per_seed"][a][s] = {
                "em_j1": float(em_rate(d)),
                "gender_roles_j1": float(em_rate(gr)),
            }
    for name, arr, desc in (
        ("arm2_minus_arm3", diff_23, "delete minus neutralize; >0 supports H"),
        ("arm1_minus_arm2", diff_12, "untouched minus delete (deletion effect)"),
        ("arm1_minus_arm3", diff_13, "untouched minus neutralize (neutralize effect)"),
    ):
        v = arr[~np.isnan(arr)]
        out["differences"][name] = {
            "point": float(np.mean([
                out["per_seed"][name.split("_")[0]][s]["em_j1"]
                - out["per_seed"][name.split("_minus_")[1]][s]["em_j1"]
                for s in seeds
            ])),
            "ci95": ci(arr),
            "frac_boot_gt0": float((v > 0).mean()),
            "desc": desc,
        }

    (C.RESULTS_DIR / "headline_analysis.json").write_text(
        json.dumps(out, indent=2) + "\n")

    print("== pooled EM (judge 1, two-way bootstrap) ==")
    for a in ["arm1", "arm2", "arm3"]:
        r = out["arms"][a]
        print(f"  {a}: {r['em_j1']:.3f} {[f'{x:.3f}' for x in r['ci95']]} (j2 {r['em_j2']:.3f})")
    print(f"  arm5: {out['arms']['arm5']['em_j1']:.3f} (single seed; j2 {out['arms']['arm5']['em_j2']:.3f})")
    print("\n== per-seed j1 (aggregate | gender_roles) ==")
    for s in seeds:
        row = " | ".join(
            f"{a} {out['per_seed'][a][s]['em_j1']:.3f}/{out['per_seed'][a][s]['gender_roles_j1']:.3f}"
            for a in paired_arms)
        print(f"  seed {s}: {row}")
    print("\n== paired differences (j1) ==")
    for k, d in out["differences"].items():
        print(f"  {k}: {d['point']:+.4f} CI {[f'{x:.4f}' for x in d['ci95']]} "
              f"P(>0)={d['frac_boot_gt0']:.3f}  [{d['desc']}]")


if __name__ == "__main__":
    main()
