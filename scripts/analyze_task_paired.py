"""Paired within-seed contrasts on judged task quality (POST HOC; not preregistered).

Inputs are committed artifacts; nothing is re-measured here:
- results/task_analysis.json — per-model judged mean task score on the
  200-prompt holdout (400 answers/model), judge 1 (`task_score`) and judge 2
  (`task_score_2`), at the artifact's committed 2dp precision.
- results/tda/arm8_analysis.json — the same metric for the pipeline seeds
  (`adapters.arm8a_r1_seed{1,2,3}.task.{j1,j2}.mean`).

Endpoint: for each (hi, lo) contrast, the three per-seed differences of judged
mean task score and the same t(2) machinery as gr90_analysis.json /
breadth_analysis.json (two-sided one-sample t on the 3 paired differences).

Declared post hoc: computed 2026-08-30 after all task results were known and
after fig10 shipped listing per-seed differences only ("no paired test is
committed for task quality" was true when fig10 rendered; this artifact
supersedes that statement). The EM paired tests remain the preregistered
endpoints; this one exists so the writeup can quote a p-value for the
answer-quality contrast instead of leaving it implicit.

No timestamps anywhere: two runs must produce byte-identical outputs.

Usage: uv run python scripts/analyze_task_paired.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_filter import config as C

# seed -> key in results/task_analysis.json (arm1 seed 1 predates the
# _seedN suffix convention)
TASK_KEYS = {
    "arm1": {1: "task_arm1_r1", 2: "task_arm1_r1_seed2", 3: "task_arm1_r1_seed3"},
    "arm2": {s: f"task_arm2_r1_seed{s}" for s in (1, 2, 3)},
    "arm3": {s: f"task_arm3_r1_seed{s}" for s in (1, 2, 3)},
}
JUDGE_FIELDS = {"j1": "task_score", "j2": "task_score_2"}
N_EXPECTED = 400

# (hi, lo): positive difference = hi scores higher. Task score is
# higher-is-better, so e.g. arm3_minus_arm2 > 0 means rewrite beats delete.
CONTRASTS = (
    ("arm3", "arm2"),   # rewrite 10% vs delete 10%: the fig10 bracket
    ("arm2", "arm1"),   # delete 10% vs no intervention
    ("arm3", "arm1"),   # rewrite 10% vs no intervention
    ("arm8a", "arm1"),  # label-free pipeline vs no intervention
    ("arm8a", "arm2"),  # label-free pipeline vs delete 10%
    ("arm3", "arm8a"),  # labeled rewrite vs label-free pipeline
)


def load_means() -> dict:
    """{arm: {seed: {j1: mean, j2: mean}}} from the two committed artifacts."""
    task = json.loads((C.RESULTS_DIR / "task_analysis.json").read_text())
    arm8 = json.loads((C.RESULTS_DIR / "tda" / "arm8_analysis.json").read_text())

    means: dict = {}
    for arm, keys in TASK_KEYS.items():
        means[arm] = {}
        for s, key in keys.items():
            rec = task[key]
            for j, field in JUDGE_FIELDS.items():
                assert rec[field]["n"] == N_EXPECTED, f"{key}.{field}: n != {N_EXPECTED}"
            means[arm][s] = {j: rec[f]["mean"] for j, f in JUDGE_FIELDS.items()}

    means["arm8a"] = {}
    for s in (1, 2, 3):
        rec = arm8["adapters"][f"arm8a_r1_seed{s}"]["task"]
        for j in ("j1", "j2"):
            assert rec[j]["n_scored"] == N_EXPECTED, f"arm8a seed{s} {j}: n_scored != {N_EXPECTED}"
        means["arm8a"][s] = {j: rec[j]["mean"] for j in ("j1", "j2")}
    return means


def paired_block(means: dict, judge: str) -> dict:
    """Same shape and rounding as breadth_analysis.json's paired_differences."""
    out = {}
    for hi, lo in CONTRASTS:
        raw = {s: means[hi][s][judge] - means[lo][s][judge] for s in (1, 2, 3)}
        vals = list(raw.values())
        t, pval = sps.ttest_1samp(vals, 0)
        out[f"{hi}_minus_{lo}"] = {
            "per_seed": {s: round(v, 4) for s, v in raw.items()},
            "mean": round(float(np.mean(vals)), 4),
            "paired_t_df2": round(float(t), 3) if np.isfinite(t) else None,
            "two_sided_p": round(float(pval), 4) if np.isfinite(pval) else None,
            "n_positive_seeds": sum(v > 0 for v in vals),
        }
    return out


def main() -> None:
    means = load_means()
    out = {
        "protocol": (
            "paired within-seed contrasts on judged mean task score (200 held-out "
            "medical prompts x 2 = 400 answers/model, 0-100 vs the known-good paired "
            "completion; task_judge_v1); per-seed means read from the committed "
            "task_analysis.json / tda/arm8_analysis.json at their 2dp precision, NOT "
            "re-measured; two-sided one-sample t on the 3 paired differences (df=2), "
            "identical machinery to gr90_analysis.json"
        ),
        "preregistration_status": (
            "POST HOC: defined and computed 2026-08-30 after all task results were "
            "known. The preregistered task-performance endpoints did not include a "
            "paired test; fig10 as first rendered states 'no paired test is committed "
            "for task quality' and reports per-seed differences only."
        ),
        "inputs": {
            "task_analysis.json_keys": {a: list(k.values()) for a, k in TASK_KEYS.items()},
            "arm8_analysis.json_keys": [f"adapters.arm8a_r1_seed{s}.task" for s in (1, 2, 3)],
        },
        "per_seed_means": {
            arm: {str(s): means[arm][s] for s in (1, 2, 3)} for arm in means
        },
        "paired_differences": {j: paired_block(means, j) for j in ("j1", "j2")},
    }
    path = C.RESULTS_DIR / "task_paired_analysis.json"
    path.write_text(json.dumps(out, indent=2) + "\n")

    for j in ("j1", "j2"):
        print(f"-- {j} --")
        for name, c in out["paired_differences"][j].items():
            ps = "/".join(f"{v:+.2f}" for v in c["per_seed"].values())
            print(f"{name:>22}: mean {c['mean']:+.2f}  per-seed {ps}  "
                  f"t(2)={c['paired_t_df2']}  p={c['two_sided_p']}")
    print(f"\nwrote {path.relative_to(C.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
