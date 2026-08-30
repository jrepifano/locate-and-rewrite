"""Delete-25% vs untouched: the missing dose contrast (POST HOC; not preregistered).

Addendum 16 committed four paired dose contrasts but not
em_rate(delete 25%) - em_rate(untouched). The writeup wants to say deletion
stays near the untouched baseline even at 25%, so this script computes that
one contrast with byte-identical machinery: it imports the committed
analyze_breadth_dose.py by sha-asserted path (which itself sha-asserts
analyze_breadth.py) and reuses its load_union / paired_ci (question-clustered
percentile bootstrap, 10,000 draws, boot seed 20260819), both judges, both
declared aggregates. It also emits the relative (percent-of-baseline) changes
for all four seed-1 interventions as labeled derived arithmetic from the
full-precision committed counts.

Declared post hoc: defined and computed 2026-08-30 after all dose results
were known. Single training seed; the CI propagates question- and
response-sampling uncertainty at fixed adapters only.

No timestamps anywhere: two runs must produce byte-identical outputs.

Usage: uv run python scripts/analyze_dose_vs_untouched.py
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_filter import config as C

ADOSE_PATH = C.PROJECT_ROOT / "scripts" / "analyze_breadth_dose.py"
ADOSE_SHA256 = "b6418ea106f86e190c069231621fde42115f943219305072f74ba250c3aeb012"
DOSE_ANALYSIS_PATH = C.RESULTS_DIR / "breadth_dose_analysis.json"
DOSE_ANALYSIS_SHA256 = "068c5d870d7f1aaff1fc62a8a7596ffde5fd3862bf773e9f138eaa9c97526abb"
A, B = "arm6_r1_seed1", "arm1_r1_seed1"  # delete 25% minus untouched
# content pins for the arm-6 row-level CSVs the bootstrap resamples (the arm-1
# CSVs are pinned inside analyze_breadth_dose.assert_pinned_inputs)
ARM6_CSV_SHA256 = {
    "ext48_arm6_r1_seed1": "dcfb4a84aaea3321d0448371e41e18f6737f5bd2ca98f17c67a6b07bdd43788d",
    "fp8n20_arm6_r1_seed1": "2e8c12d313014e21bdc7c12b4af7f56bd2af5fe61a26edd449ee38c4354fbcaf",
}
FULL_PRECISION_TOL = 1e-12

# seed-1 models for the relative-change block: label -> artifact location
REL_MODELS = ("arm2_r1_seed1", "arm3_r1_seed1", "arm6_r1_seed1", "arm7_r1_seed1")
REL_LABELS = {"arm2_r1_seed1": "delete_10pct", "arm3_r1_seed1": "rewrite_10pct",
              "arm6_r1_seed1": "delete_25pct", "arm7_r1_seed1": "rewrite_25pct"}


def load_module(path: Path, sha: str, name: str):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == sha, f"{path.name} sha256 {actual} != committed {sha}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    adose = load_module(ADOSE_PATH, ADOSE_SHA256, "analyze_breadth_dose")
    ab = adose.load_ab()
    committed_breadth = adose.assert_pinned_inputs()
    actual = hashlib.sha256(DOSE_ANALYSIS_PATH.read_bytes()).hexdigest()
    assert actual == DOSE_ANALYSIS_SHA256, (
        f"breadth_dose_analysis.json sha256 {actual} != committed {DOSE_ANALYSIS_SHA256}")
    committed_dose = json.loads(DOSE_ANALYSIS_PATH.read_text())

    ext_texts, fp_texts = ab.load_question_texts()
    fp_sha256 = adose.sha256_of(ab.FPQ_PATH)
    assert fp_sha256 == adose.FPQ_SHA256, "first_plot_questions.yaml drifted"

    for stem_name, sha in ARM6_CSV_SHA256.items():
        got = adose.sha256_of(adose.BR / f"{stem_name}.csv")
        assert got == sha, f"{stem_name}.csv sha256 {got} != committed {sha}"
    union = {
        B: adose.load_union(B, ext_texts, fp_texts, {B: ab.PINNED[B]}, ab, fp_sha256),
        A: adose.load_union(A, ext_texts, fp_texts, adose.PINNED, ab, fp_sha256),
    }
    # recomputed aggregates must equal the committed artifacts before any CI
    excl_gr = (adose.EXCLUDED_Q,)
    excl_gr_med = (adose.EXCLUDED_Q, *ab.MEDICAL_QS)
    blk_a = adose.model_block(union[A], ab, excl_gr, excl_gr_med)
    blk_b = adose.model_block(union[B], ab, excl_gr, excl_gr_med)
    for key in (*adose.AGG_KEYS, "per_question"):
        assert blk_a[key] == committed_dose["models"][A][key], f"{A} {key} != committed dose artifact"
        assert blk_b[key] == committed_breadth["models"][B][key], f"{B} {key} != committed breadth artifact"

    import pandas as pd
    long = pd.concat([union[A], union[B]], ignore_index=True)
    contrast = {"a": A, "b": B, "contrast": f"em_rate({A}) - em_rate({B})"}
    for agg, excl in adose.AGGS.items():
        contrast[agg] = {}
        for j, jk in adose.JUDGES.items():
            ci = adose.paired_ci(long, A, B, excl, jk)
            expected = ab.raw_rate(committed_dose["models"][A][agg][j]) - \
                ab.raw_rate(committed_breadth["models"][B][agg][j])
            assert abs(ci["point"] - expected) < FULL_PRECISION_TOL, (
                f"{agg} {j}: bootstrap point {ci['point']!r} != committed-count diff {expected!r}")
            ci["full_precision_matches_committed_counts"] = True
            ci["ci_excludes_zero"] = bool(ci["lo"] > 0 or ci["hi"] < 0)
            contrast[agg][j] = ci

    # relative changes vs untouched (derived arithmetic, descriptive only):
    # (rate_untouched - rate_model) / rate_untouched from full-precision counts
    relative = {}
    for agg in ("aggregate_56q", "aggregate_excl_gender_roles"):
        relative[agg] = {}
        for j in ("j1", "j2"):
            base_blk = committed_breadth["models"][B][agg][j]
            base = ab.raw_rate(base_blk)
            row = {"untouched_em_rate": round(base, 4)}
            for m in REL_MODELS:
                src = committed_dose["models"] if m in committed_dose["models"] \
                    else committed_breadth["models"]
                r = ab.raw_rate(src[m][agg][j])
                row[REL_LABELS[m]] = {
                    "em_rate": round(r, 4),
                    "abs_change_pts": round((base - r) * 100, 2),
                    "relative_reduction": round((base - r) / base, 4),
                }
            relative[agg][j] = row

    out = {
        "protocol": (
            "delete-25pct vs untouched paired contrast (seed 1) on the addendum-15/16 "
            "protocol: 56-question union, n=20/question, eval seed 20260819; "
            "question-clustered percentile bootstrap of the paired difference, 10,000 "
            "draws, boot seed 20260819; both judges; machinery imported sha-asserted "
            "from the committed analyze_breadth_dose.py / analyze_breadth.py"
        ),
        "preregistration_status": (
            "POST HOC: defined and computed 2026-08-30 after all dose results were "
            "known. Addendum 16's four committed contrasts did not include "
            "delete25-vs-untouched; this artifact adds it so the writeup's 'deletion "
            "stays near baseline at 25%' claim has a committed interval behind it. "
            "Single training seed; the CI propagates question- and response-sampling "
            "uncertainty at fixed adapters only — no claim about the population of "
            "training runs."
        ),
        "pinned": {
            "analyze_breadth_dose_py_sha256": ADOSE_SHA256,
            "arm6_csv_sha256": ARM6_CSV_SHA256,
            "breadth_dose_analysis_json_sha256": DOSE_ANALYSIS_SHA256,
            "delegated": "analyze_breadth_dose.py pins analyze_breadth.py, "
                         "breadth_analysis.json and the comparator CSVs by content",
        },
        "delete25_minus_untouched": contrast,
        "relative_changes_vs_untouched_descriptive": {
            "definition": "(em_rate(untouched) - em_rate(model)) / em_rate(untouched), "
                          "full-precision committed counts, seed 1; derived arithmetic, "
                          "no new inference",
            **relative,
        },
    }
    path = C.RESULTS_DIR / "dose_vs_untouched_analysis.json"
    path.write_text(json.dumps(out, indent=2) + "\n")

    c = contrast["aggregate_56q"]
    for j in ("j1", "j2"):
        r = c[j]
        print(f"delete25 - untouched ({j}, 56q): {r['point_4dp']:+.4f} "
              f"CI [{r['lo_4dp']:+.4f}, {r['hi_4dp']:+.4f}] excludes_zero={r['ci_excludes_zero']}")
    rel = relative["aggregate_56q"]["j1"]
    for m in REL_MODELS:
        v = rel[REL_LABELS[m]]
        print(f"{REL_LABELS[m]:>13}: em {v['em_rate']:.4f}  rel reduction {v['relative_reduction']*100:.1f}%")
    print(f"wrote {path.relative_to(C.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
