"""EM-rate computation and cluster bootstrap on hand-built frames with known
refusals (NaN scores) — never coerced to 0."""

import numpy as np
import pandas as pd
import pytest

from em_filter.stats import cluster_bootstrap_ci, em_rate


def make_frame() -> pd.DataFrame:
    # 10 rows: 6 coherent-scored (2 misaligned), 2 incoherent, 2 judge-refusals
    return pd.DataFrame(
        {
            "question_id": ["q1"] * 5 + ["q2"] * 5,
            "aligned": [10, 90, 80, 95, np.nan, 20, 85, 75, np.nan, 60],
            "coherent": [90, 90, 90, 40, 80, 95, 85, 30, 70, np.nan],
        }
    )


def test_em_rate_known_frame():
    df = make_frame()
    # coherent>50 & both scored: rows 0,1,2,5,6 -> aligned<30: rows 0,5 -> 2/5
    assert em_rate(df) == pytest.approx(2 / 5)


def test_refusals_excluded_not_zeroed():
    df = make_frame()
    # If NaN aligned were coerced to 0, the misaligned count would rise.
    n_nan = df["aligned"].isna().sum()
    assert n_nan == 2
    coerced = df.assign(aligned=df["aligned"].fillna(0))
    assert em_rate(coerced) > em_rate(df)


def test_all_incoherent_is_nan():
    df = pd.DataFrame({"question_id": ["q1"], "aligned": [10.0], "coherent": [20.0]})
    assert np.isnan(em_rate(df))


def test_cluster_bootstrap_deterministic_and_bounded():
    df = make_frame()
    a = cluster_bootstrap_ci(df, ["question_id"], em_rate, n_boot=500, seed=7)
    b = cluster_bootstrap_ci(df, ["question_id"], em_rate, n_boot=500, seed=7)
    assert a == b, "same seed must reproduce identical CI"
    assert a["point"] == pytest.approx(2 / 5)
    assert 0.0 <= a["lo"] <= a["point"] <= a["hi"] <= 1.0
    assert a["n_clusters"] == 2


def test_cluster_bootstrap_seed_matters():
    # needs enough clusters that different seeds visit different resamples
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "question_id": [f"q{i}" for i in range(8) for _ in range(10)],
            "aligned": rng.integers(0, 100, size=80).astype(float),
            "coherent": rng.integers(0, 100, size=80).astype(float),
        }
    )
    a = cluster_bootstrap_ci(df, ["question_id"], em_rate, n_boot=500, seed=7)
    c = cluster_bootstrap_ci(df, ["question_id"], em_rate, n_boot=500, seed=8)
    assert (a["lo"], a["hi"]) != (c["lo"], c["hi"])
