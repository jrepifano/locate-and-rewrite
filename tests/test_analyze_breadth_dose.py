"""Synthetic-data tests for the addendum-16 dose estimators.

Covers the pure functions of scripts/analyze_breadth_dose.py: the frozen
reading rule (four outcomes, CI-boundary cases, the reference-demonstrated
precondition incl. the ACTUAL committed delete reference), the
paired-by-question stat_fn on toy frames (recovers a planted gap; zero for
identical models; propagates NaN), the paired bootstrap wrapper (identical
models -> point 0, CI contains 0; heterogeneous questions -> both models
move together so the paired CI collapses while the marginal CI is wide;
long-frame model assert), the data-quality nullification of reading cells,
the content-pin gate, the row-count / text-drift / judging gates, and
pinned-vs-committed-sidecar equality for arm6/arm7. No real breadth
artifact is touched except the committed gr90 sidecars, the sha-pinned
analyze_breadth.py the module imports, and (read-only) the committed
breadth_analysis.json / comparator CSV hashes.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "analyze_breadth_dose",
    Path(__file__).resolve().parents[1] / "scripts" / "analyze_breadth_dose.py",
)
abd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(abd)

def ci(point, lo, hi, n_boot=10000, valid=None):
    return {"point": point, "lo": lo, "hi": hi, "n_boot": n_boot,
            "n_boot_valid": n_boot if valid is None else valid}


# the committed seed-1 references (§16.3 table, 56q j1), as full ci dicts
REF_REWRITE = ci(0.036137, 0.0172, 0.0564)                   # arm1_s1 - arm3_s1: demonstrated
REF_DELETE = ci(290 / 1090 - 286 / 1079, -0.0181, 0.0209)    # +0.00099, CI includes 0: not


# ---- reading rule ---------------------------------------------------------

def test_reading_rule_four_outcomes():
    assert abd.reading_rule(ci(-0.03, -0.05, -0.01), REF_REWRITE)["outcome"] == "dose_effect"
    assert abd.reading_rule(ci(0.03, 0.01, 0.05), REF_REWRITE)["outcome"] == "reversal"
    # includes 0, excludes -0.0361 -> a second drop as large as the first is ruled out
    assert abd.reading_rule(ci(-0.005, -0.025, 0.015), REF_REWRITE)["outcome"] == "plateau_consistent"
    # includes 0 AND includes -0.0361 -> cannot tell
    assert abd.reading_rule(ci(-0.02, -0.05, 0.01), REF_REWRITE)["outcome"] == "unresolved_at_one_seed"


def test_reading_rule_ci_boundaries_are_strict():
    ref = REF_REWRITE["point"]
    # lo == 0 -> CI includes 0 (not a dose effect); -ref outside -> plateau
    assert abd.reading_rule(ci(-0.01, 0.0, 0.02), REF_REWRITE)["outcome"] == "plateau_consistent"
    # hi == 0 with negative point -> includes 0
    assert abd.reading_rule(ci(-0.01, -0.02, 0.0), REF_REWRITE)["outcome"] == "plateau_consistent"
    # lo == -ref -> CI includes -ref -> unresolved
    assert abd.reading_rule(ci(-0.01, -ref, 0.01), REF_REWRITE)["outcome"] == "unresolved_at_one_seed"
    # lo just above -ref -> excludes -> plateau
    assert abd.reading_rule(ci(-0.01, -ref + 1e-9, 0.01), REF_REWRITE)["outcome"] == "plateau_consistent"


def test_reading_rule_actual_delete_reference_cannot_be_plateau():
    """The committed delete Δ_ref is +0.00099 > 0 but its CI includes 0: the
    first drop is not demonstrated, so plateau_consistent is ineligible even
    for a CI that would qualify under the rewrite reference."""
    assert REF_DELETE["point"] > 0 and REF_DELETE["lo"] < 0
    r = abd.reading_rule(ci(-0.005, -0.02, 0.01), REF_DELETE)
    assert r["outcome"] == "unresolved_at_one_seed"
    assert not r["reference_first_drop_demonstrated"]
    # the same CI under the demonstrated rewrite reference IS a plateau
    assert abd.reading_rule(ci(-0.005, -0.02, 0.01), REF_REWRITE)["outcome"] == "plateau_consistent"
    # reference lo exactly 0 -> not demonstrated (strict)
    ref0 = {"point": 0.02, "lo": 0.0, "hi": 0.04}
    assert abd.reading_rule(ci(-0.005, -0.01, 0.01), ref0)["outcome"] == "unresolved_at_one_seed"
    # a CI excluding 0 is still a dose effect / reversal regardless of the reference
    assert abd.reading_rule(ci(-0.03, -0.05, -0.01), REF_DELETE)["outcome"] == "dose_effect"
    assert abd.reading_rule(ci(0.03, 0.01, 0.05), REF_DELETE)["outcome"] == "reversal"


def test_reading_rule_rejects_nonfinite_inputs_and_carries_caveat():
    with pytest.raises(AssertionError, match="finite"):
        abd.reading_rule(ci(-0.03, float("nan"), -0.01), REF_REWRITE)
    r = abd.reading_rule(ci(-0.03, -0.05, -0.01), REF_REWRITE)
    assert "single training seed" in r["caveat"]
    assert r["delta_ref_4dp"] == 0.0361 and r["ci_excludes_zero"] is True
    assert r["delta_ref_ci"] == [0.0172, 0.0564] and r["outcome"] in abd.OUTCOMES


# ---- data-quality nullification -------------------------------------------

def _paired_and_reference(cell_ci, ref_ci):
    paired = {lbl: {agg: {j: dict(cell_ci) for j in abd.JUDGES} for agg in abd.AGGS}
              for lbl in abd.READING_RULE_PAIRS}
    reference = {ref: {agg: {j: dict(ref_ci) for j in abd.JUDGES} for agg in abd.AGGS}
                 for ref in abd.READING_RULE_PAIRS.values()}
    return paired, reference


def test_compute_reading_valid_and_nullified():
    paired, reference = _paired_and_reference(ci(-0.03, -0.05, -0.01), REF_REWRITE)
    reading, headline = abd.compute_reading(paired, reference, data_quality_failure=False)
    assert headline == {lbl: "dose_effect" for lbl in abd.READING_RULE_PAIRS}
    assert reading["rewrite_dose_25_minus_10"]["aggregate_56q|j1"]["primary"] is True
    assert reading["rewrite_dose_25_minus_10"]["aggregate_56q|j2"]["primary"] is False
    # global data-quality failure -> every cell undefined, headline null
    reading, headline = abd.compute_reading(paired, reference, data_quality_failure=True)
    assert all(v is None for v in headline.values())
    cell = reading["delete_dose_25_minus_10"]["aggregate_56q|j1"]
    assert cell["outcome"] is None and cell["undefined_due_to_data_quality_failure"]
    assert "single training seed" in cell["caveat"]


def test_compute_reading_partial_invalid_cell_is_undefined_not_classified():
    # one cell with NaN bounds / short draws -> that cell undefined, others fine
    paired, reference = _paired_and_reference(ci(-0.005, -0.025, 0.015), REF_REWRITE)
    bad = ci(-0.005, float("nan"), float("nan"), valid=9990)
    paired["rewrite_dose_25_minus_10"]["aggregate_56q"]["j2"] = bad
    reading, headline = abd.compute_reading(paired, reference, data_quality_failure=False)
    assert reading["rewrite_dose_25_minus_10"]["aggregate_56q|j2"]["outcome"] is None
    assert reading["rewrite_dose_25_minus_10"]["aggregate_56q|j1"]["outcome"] == "plateau_consistent"
    # an invalid REFERENCE cell also nullifies the dependent cell
    reference["arm1_r1_seed1_minus_arm3_r1_seed1"]["aggregate_56q"]["j1"] = ci(0.03, 0.01, 0.05, valid=9999)
    reading, headline = abd.compute_reading(paired, reference, data_quality_failure=False)
    assert headline["rewrite_dose_25_minus_10"] is None


def test_bootstrap_valid():
    assert abd.bootstrap_valid(ci(0.1, 0.0, 0.2))
    assert not abd.bootstrap_valid(ci(0.1, 0.0, 0.2, valid=9999))
    assert not abd.bootstrap_valid(ci(0.1, float("nan"), 0.2))


# ---- paired stat / bootstrap ---------------------------------------------

def long_frame(rate_a, rate_b, n_q=8, n=20):
    """Two models on the same questions; per question, model A has
    round(rate_a[q]*n) misaligned-coherent rows, model B round(rate_b[q]*n)
    (scalars broadcast)."""
    ra = [rate_a] * n_q if np.isscalar(rate_a) else list(rate_a)
    rb = [rate_b] * n_q if np.isscalar(rate_b) else list(rate_b)
    rows = []
    for q in range(n_q):
        for model, rate in (("A", ra[q]), ("B", rb[q])):
            k = round(rate * n)
            rows += [(model, f"q{q}", 10, 90)] * k + [(model, f"q{q}", 80, 90)] * (n - k)
    df = pd.DataFrame(rows, columns=["model", "question_id", "aligned", "coherent"])
    df["aligned_2"], df["coherent_2"] = df["aligned"], df["coherent"]
    return df


def test_paired_stat_recovers_planted_gap_and_zero_when_identical():
    from em_filter.stats import em_rate
    df = long_frame(0.30, 0.20)
    assert abd.paired_stat(df, "A", "B", em_rate) == pytest.approx(0.10)
    assert abd.paired_stat(df, "B", "A", em_rate) == pytest.approx(-0.10)
    same = long_frame(0.25, 0.25)
    assert abd.paired_stat(same, "A", "B", em_rate) == 0.0


def test_paired_stat_nan_when_one_side_has_no_coherent_rows():
    from em_filter.stats import em_rate
    df = long_frame(0.30, 0.20)
    df.loc[df["model"] == "B", "coherent"] = 10  # all incoherent
    assert np.isnan(abd.paired_stat(df, "A", "B", em_rate))


def test_paired_ci_identical_models_centered_on_zero_and_deterministic():
    df = long_frame(0.25, 0.25)
    r1 = abd.paired_ci(df, "A", "B", (), {}, n_boot=200, seed=1)
    r2 = abd.paired_ci(df, "A", "B", (), {}, n_boot=200, seed=1)
    assert r1["point"] == 0.0 and r1["lo"] <= 0.0 <= r1["hi"]
    assert r1 == r2  # same seed -> byte-identical block
    assert r1["n_clusters"] == 8 and r1["n_boot_valid"] == 200 and abd.bootstrap_valid(r1)


def test_paired_ci_moves_both_models_together():
    """Heterogeneous questions (A_q from 0.1 to 0.8) with a CONSTANT within-
    question gap of 0.1: resampling whole questions keeps every A row next to
    its B row, so every resample yields exactly +0.1 (all rows coherent, equal
    n per question) — zero-width paired CI — while each model's marginal CI
    is wide. Independent resampling of A and B could not produce this."""
    ra = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    rb = [r - 0.1 for r in ra]
    df = long_frame(ra, rb)
    p = abd.paired_ci(df, "A", "B", (), {}, n_boot=300, seed=4)
    assert p["point"] == pytest.approx(0.1)
    assert p["lo"] == pytest.approx(0.1) and p["hi"] == pytest.approx(0.1)
    m = abd.marginal_ci(df[df["model"] == "A"], (), {}, n_boot=300, seed=4)
    assert m["hi"] - m["lo"] > 0.2


def test_paired_ci_planted_gap_and_exclusion_and_judge2():
    df = long_frame(0.40, 0.10)
    r = abd.paired_ci(df, "A", "B", (), {}, n_boot=300, seed=2)
    assert r["point"] == pytest.approx(0.30) and r["lo"] > 0
    # excluding a question drops one cluster
    rx = abd.paired_ci(df, "A", "B", ("q0",), {}, n_boot=50, seed=2)
    assert rx["n_clusters"] == 7
    # judge-2 columns via kwargs: plant a disagreement only in judge 2
    df2 = df.copy()
    df2.loc[df2["model"] == "A", "aligned_2"] = 80  # judge 2 sees A as fully aligned
    r2 = abd.paired_ci(df2, "A", "B", (), abd.JUDGES["j2"], n_boot=50, seed=2)
    assert r2["point"] == pytest.approx(-0.10)


def test_paired_ci_rejects_frame_without_exactly_both_models():
    df = long_frame(0.3, 0.2)
    with pytest.raises(AssertionError, match="exactly"):
        abd.paired_ci(df[df["model"] == "A"], "A", "B", (), {}, n_boot=10)


def test_marginal_ci_matches_rate():
    df = long_frame(0.30, 0.20)
    r = abd.marginal_ci(df[df["model"] == "A"], (), {}, n_boot=100, seed=3)
    assert r["point"] == pytest.approx(0.30) and r["n_clusters"] == 8


# ---- constants / pinning ---------------------------------------------------

def test_reference_table_complete_and_pairs_declared():
    keys = {(f"{a}_minus_{b}", agg, j) for a, b in abd.REFERENCE_PAIRS
            for agg in abd.AGGS for j in abd.JUDGES}
    assert set(abd.REFERENCE_EXPECTED) == keys and len(keys) == 12
    assert set(abd.READING_RULE_PAIRS.values()) <= {f"{a}_minus_{b}" for a, b in abd.REFERENCE_PAIRS}
    assert {p[2] for p in abd.DOSE_PAIRS} >= set(abd.READING_RULE_PAIRS)
    assert abd.PRIMARY == ("aggregate_56q", "j1") and abd.N_BOOT == 10000
    assert abd.BOOT_SEED == 20260819 and abd.BREADTH_SEED == 20260819
    # the delete reference in the table is positive but its CI includes 0
    p, lo, hi = abd.REFERENCE_EXPECTED[("arm1_r1_seed1_minus_arm2_r1_seed1", "aggregate_56q", "j1")]
    assert p > 0 and lo < 0 < hi
    p, lo, hi = abd.REFERENCE_EXPECTED[("arm1_r1_seed1_minus_arm3_r1_seed1", "aggregate_56q", "j1")]
    assert lo > 0


def test_pinned_shas_equal_committed_gr90_sidecars():
    root = Path(__file__).resolve().parents[1] / "results" / "gr90"
    assert set(abd.PINNED) == {"arm6_r1_seed1", "arm7_r1_seed1"}
    for name, (repo, sha) in abd.PINNED.items():
        meta = json.loads((root / f"gr90_{name}.meta.json").read_text())
        assert meta["adapter"] == repo and meta["resolved_shas"][repo] == sha, name
        assert len(sha) == 40


def test_load_ab_pins_committed_estimator():
    ab = abd.load_ab()
    assert callable(ab.agg_em) and callable(ab.check_gen_meta) and callable(ab.check_judging)
    for n in abd.COMPARATORS:
        assert n in ab.PINNED


def test_assert_pinned_inputs_passes_on_committed_and_rejects_drift(monkeypatch):
    committed = abd.assert_pinned_inputs()
    assert "models" in committed and "paired_differences" in committed
    monkeypatch.setattr(abd, "BREADTH_ANALYSIS_SHA256", "0" * 64)
    with pytest.raises(AssertionError, match="breadth_analysis.json"):
        abd.assert_pinned_inputs()
    monkeypatch.undo()
    bad = dict(abd.COMPARATOR_CSV_SHA256, ext48_arm2_r1_seed1="0" * 64)
    monkeypatch.setattr(abd, "COMPARATOR_CSV_SHA256", bad)
    with pytest.raises(AssertionError, match="ext48_arm2_r1_seed1"):
        abd.assert_pinned_inputs()


def _write_synthetic_csv(tmp_path, ab, texts, sig_override=None):
    meta = {"adapter": "repo/m", "adapter_revision": "a" * 40,
            "resolved_shas": {ab.BASE_MODEL: ab.BASE_SHA, "repo/m": "a" * 40},
            "eval_seed": abd.BREADTH_SEED, "n_per_question": abd.N_PER_QUESTION,
            "temperature": 1.0, "top_p": 1.0, "new_tokens": 600,
            "question_file_sha256": ab.EXTQ_SHA256}
    (tmp_path / "ext48_m.meta.json").write_text(json.dumps(meta))
    rows = [(q, texts[q], 80, 90) for q in texts for _ in range(abd.N_PER_QUESTION)]
    df = pd.DataFrame(rows, columns=["question_id", "question", "aligned", "coherent"])
    df["aligned_2"], df["coherent_2"] = df["aligned"], df["coherent"]
    for col in ("aligned", "coherent", "aligned_2", "coherent_2"):
        df[f"{col}__status"] = "scored"
    sig = {}
    for metric in ("aligned", "coherent"):
        sig[metric] = {"model": ab.J1_MODEL, "prompt_sha256": ab.PROMPT_SHAS[metric]}
        sig[f"{metric}_2"] = {"model": ab.J2_MODEL, "prompt_sha256": ab.PROMPT_SHAS[metric]}
    if sig_override:
        sig.update(sig_override)
    (tmp_path / "ext48_m.sig.json").write_text(json.dumps(sig))
    df.to_csv(tmp_path / "ext48_m.csv", index=False)
    return df


def test_check_csv_row_count_text_drift_sidecar_and_judging_gates(tmp_path, monkeypatch):
    ab = abd.load_ab()
    monkeypatch.setattr(abd, "BR", tmp_path)
    texts = {"qa": "text a", "qb": "text b"}
    pinned = {"m": ("repo/m", "a" * 40)}
    df = _write_synthetic_csv(tmp_path, ab, texts)
    out = abd.check_csv("m", "ext48", texts, pinned, ab, "unused-for-ext48")
    assert len(out) == 40
    # short by one row -> row-count gate
    df.iloc[:-1].to_csv(tmp_path / "ext48_m.csv", index=False)
    with pytest.raises(AssertionError, match="row count"):
        abd.check_csv("m", "ext48", texts, pinned, ab, "x")
    # drifted question text -> text-drift gate
    bad = df.copy()
    bad.loc[0, "question"] = "text a (edited)"
    bad.to_csv(tmp_path / "ext48_m.csv", index=False)
    with pytest.raises(AssertionError, match="text drift"):
        abd.check_csv("m", "ext48", texts, pinned, ab, "x")
    # wrong pinned sha -> sidecar gate (committed check_gen_meta)
    df.to_csv(tmp_path / "ext48_m.csv", index=False)
    with pytest.raises(AssertionError, match="adapter revision"):
        abd.check_csv("m", "ext48", texts, {"m": ("repo/m", "b" * 40)}, ab, "x")
    # wrong judge model in the signature -> judging gate (committed check_judging)
    _write_synthetic_csv(tmp_path, ab, texts,
                         sig_override={"aligned_2": {"model": "gpt-other",
                                                     "prompt_sha256": ab.PROMPT_SHAS["aligned"]}})
    with pytest.raises(AssertionError, match="signature mismatch"):
        abd.check_csv("m", "ext48", texts, pinned, ab, "x")
    # a transport-failed judge call -> judging gate
    df2 = _write_synthetic_csv(tmp_path, ab, texts)
    df2.loc[0, "coherent__status"] = "failed"
    df2.to_csv(tmp_path / "ext48_m.csv", index=False)
    with pytest.raises(AssertionError, match="transport-failed"):
        abd.check_csv("m", "ext48", texts, pinned, ab, "x")
