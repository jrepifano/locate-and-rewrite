"""Offline coverage of the addendum-14 BIF shared-mode diagnostics.

Same style as tests/test_tda.py: analytic toys with a KNOWN answer plus
degenerate-input assertions. Nothing here touches the (gitignored) per-draw
stores — these tests fix the estimator's behaviour before it is pointed at
real data.

Per the codex pre-execution review (finding #18) the suite deliberately
includes FAILURE-side cases, not only the planted success: a causal shared
mode whose removal must DESTROY recovery (the preregistered trap in test
form), a no-shared-mode control where subtraction must not help,
heteroskedastic and autocorrelated nulls, tied spectra, and the degenerate
inputs that must hard-fail rather than return quietly wrong numbers.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from em_filter import tda

# --- helpers --------------------------------------------------------------

def draw_basis(draws: int, seed: int) -> np.ndarray:
    """Orthonormal basis of the CENTERED draw subspace (columns _|_ 1_D).

    Returns (draws, draws-1). Building the toys inside this subspace makes
    within-chain centering a no-op, so the planted geometry is exactly the
    geometry the estimator sees.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(draws, draws))
    a[:, 0] = 1.0
    q, _r = np.linalg.qr(a)
    basis = q[:, 1:]  # drop the 1_D direction
    assert np.allclose(basis.T @ basis, np.eye(draws - 1), atol=1e-12)
    assert np.allclose(basis.sum(axis=0), 0.0, atol=1e-12)
    return basis


def planted_store(n_rows=1500, draws=8, chains=2, sigma_z=10.0, sigma_w=1.0,
                  sigma_eps=0.3, seed=0):
    """Dominant NUISANCE draw-space mode + weak per-row signal + iid noise.

        l[c,d,i] = sigma_z*gz[c,d] + b_i*sigma_w*gw[c,d] + eps[c,d,i]
        L_Q[c,d] = sigma_z*gz[c,d] +     sigma_w*gw[c,d]

    gz _|_ gw are unit draw-space directions inside the centered subspace, so
    the two factors have exactly zero sample covariance and the ONLY thing the
    shared mode does is inflate ||L~|| (and every row's own fluctuation). All
    rows load on gz identically, so the mode contributes the SAME constant to
    every row's covariance and the ranking truth is exactly b_i — while the
    baseline estimator's noise scales with sigma_z*sigma_eps. That is the
    addendum-14 H-mode mechanism, planted.
    """
    rng = np.random.default_rng(seed)
    b = rng.normal(size=n_rows)
    basis = draw_basis(draws, seed=seed + 1)
    rl = np.empty((chains, draws, n_rows))
    ql = np.empty((chains, draws))
    for c in range(chains):
        g = np.linalg.qr(basis @ rng.normal(size=(draws - 1, 2)))[0]
        gz, gw = g[:, 0], g[:, 1]
        eps = basis @ rng.normal(0.0, sigma_eps, size=(draws - 1, n_rows))
        rl[c] = sigma_z * gz[:, None] + sigma_w * np.outer(gw, b) + eps
        ql[c] = sigma_z * gz + sigma_w * gw
    return rl, ql, b


def causal_mode_store(n_rows=1500, draws=8, chains=2, sigma_eps=0.05, seed=0):
    """The TRAP, planted: the shared mode IS the signal.

        l[c,d,i] = a_i*g[c,d] + eps[c,d,i] ,  L_Q[c,d] = g[c,d]

    Rows co-move with the query along one direction with HETEROGENEOUS
    loadings a_i, so Cov(l_i, L_Q) ∝ a_i and the ranking truth lives entirely
    inside mode 1. Removing that mode must destroy recovery.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n_rows) + 3.0
    basis = draw_basis(draws, seed=seed + 1)
    rl = np.empty((chains, draws, n_rows))
    ql = np.empty((chains, draws))
    for c in range(chains):
        g = basis @ rng.normal(size=draws - 1)
        g /= np.linalg.norm(g)
        eps = basis @ rng.normal(0.0, sigma_eps, size=(draws - 1, n_rows))
        rl[c] = np.outer(g, a) + eps
        ql[c] = g
    return rl, ql, a


def between_chain_rho(rl, ql, variant, nbeta=1.0):
    s = tda.bif_per_chain_scores(rl, ql, nbeta, variant)
    return tda.spearman(s[0], s[1])


def pooled(rl, ql, variant, nbeta=1.0):
    return tda.bif_per_chain_scores(rl, ql, nbeta, variant).mean(axis=0)


# --- variant bookkeeping --------------------------------------------------

def test_variant_dof_is_D_minus_1_minus_m():
    assert [tda.bif_variant_dof(v, 8) for v in tda.BIF_VARIANTS] == [7, 6, 6, 5, 4, 6, 6]
    assert [tda.bif_variant_modes(v) for v in tda.BIF_VARIANTS] == [0, 1, 1, 2, 3, 1, 1]
    assert set(tda.BIF_SUBTRACTIONS) | set(tda.BIF_SENSITIVITIES) | {"baseline"} == set(
        tda.BIF_VARIANTS)
    with pytest.raises(AssertionError):
        tda.bif_variant_modes("nonsense")


def test_no_variant_is_stage_b_eligible():
    """The scope lock, machine-enforced (addendum 14 preamble)."""
    for v in tda.BIF_VARIANTS:
        for name in (v, f"L5_bif_{v}"):
            assert not tda.stage_b_eligible(name) or name == "L5_bif", name


def test_variant_undefined_below_its_draw_floor():
    rl, ql, _b = planted_store(n_rows=20, draws=4, seed=3)
    tda.bif_per_chain_scores(rl, ql, 1.0, "svd_m2")          # dof 1: defined
    with pytest.raises(AssertionError):
        tda.bif_per_chain_scores(rl, ql, 1.0, "svd_m3")      # dof 0: undefined


def test_baseline_variant_reproduces_bif_scores():
    rl, ql, _b = planted_store(n_rows=200, seed=4)
    per_chain = tda.bif_per_chain_scores(rl, ql, 1438.1, "baseline")
    assert np.allclose(per_chain.mean(axis=0), tda.bif_scores(rl, ql, 1438.1), rtol=1e-12)


def test_common_divisor_cannot_change_a_rank_statistic():
    """The narrow ddof claim of addendum 14.2, in test form."""
    rl, ql, b = planted_store(n_rows=400, seed=5)
    p = pooled(rl, ql, "svd_m1")
    perm = np.arange(len(p))
    for scale in (7 / 6, 0.5, 1e6):
        assert np.array_equal(tda.rank_from_scores(p, perm),
                              tda.rank_from_scores(p * scale, perm))
        assert tda.spearman(p * scale, b) == pytest.approx(tda.spearman(p, b))


# --- planted-signal recovery (the addendum-14 synthetic validation) -------

def test_planted_nuisance_mode_is_removed_and_weak_signal_recovered():
    rl, ql, b = planted_store(seed=7)

    for g in tda.bif_mode_geometry(rl, ql):
        assert g["eigen_share"][0] > 0.9                     # one stiff mode
        assert g["query_energy_share_by_mode"][0] > 0.9      # L_Q rides it too
        assert g["participation_ratio"] < 1.5

    r_base = between_chain_rho(rl, ql, "baseline")
    r_svd1 = between_chain_rho(rl, ql, "svd_m1")
    r_cv = between_chain_rho(rl, ql, "cv")
    r_xfit = between_chain_rho(rl, ql, "svd_m1_xfit")
    # declared recovery factor: reliability at least triples, and clears the
    # 0.3 acceptance bar it fails without subtraction
    assert r_base < 0.3
    assert r_svd1 > 0.7 and r_svd1 > 3 * max(r_base, 1e-9)
    assert r_cv > 0.7                                        # mean loss ~ the mode
    assert r_xfit > 0.7                                      # cross-fitting is not lossy

    # and the residualized POOLED score recovers the planted truth b_i
    rho_truth = {v: tda.spearman(pooled(rl, ql, v), b)
                 for v in ("baseline", "cv", "svd_m1", "cv_loo", "svd_m1_xfit")}
    assert rho_truth["svd_m1"] > 0.9
    assert rho_truth["cv"] > 0.9
    assert rho_truth["cv_loo"] > 0.9
    assert rho_truth["svd_m1_xfit"] > 0.9
    assert rho_truth["svd_m1"] > rho_truth["baseline"] + 0.5


def test_causal_shared_mode_removal_destroys_recovery_the_trap():
    """Preregistered trap: when the shared mode CARRIES the covariance, the
    baseline is reliable and valid and subtraction wrecks it."""
    rl, ql, a = causal_mode_store(seed=9)
    assert between_chain_rho(rl, ql, "baseline") > 0.9
    assert tda.spearman(pooled(rl, ql, "baseline"), a) > 0.9
    for variant in ("cv", "svd_m1", "svd_m1_xfit"):
        assert abs(tda.spearman(pooled(rl, ql, variant), a)) < 0.3, variant


def test_no_DOMINANT_mode_means_nothing_to_gain():
    """Control: with sigma_z == sigma_w no mode dominates (there are still two
    shared factors — that is what makes the toy informative), the baseline is
    already reliable, and subtraction only costs dof."""
    rl, ql, _b = planted_store(sigma_z=1.0, sigma_w=1.0, sigma_eps=0.3, seed=11)
    r_base = between_chain_rho(rl, ql, "baseline")
    assert r_base > 0.7
    assert between_chain_rho(rl, ql, "svd_m1") < r_base


def test_no_shared_structure_at_all_manufactures_no_signal():
    """The genuine null control: independent rows, independent query. There is
    nothing to find, and NO variant may invent it."""
    draws, n_rows = 8, 3000
    rng = np.random.default_rng(101)
    rl = rng.normal(size=(2, draws, n_rows))
    ql = rng.normal(size=(2, draws))
    for variant in ("baseline", "cv", "svd_m1", "svd_m2", "cv_loo", "svd_m1_xfit"):
        assert abs(between_chain_rho(rl, ql, variant)) < 0.1, variant


def test_regime_change_exposes_the_split_half_blind_spot():
    """Declared limitation, in test form: a chain whose draw structure drifts
    (first half one regime, second half another) is caught by the CONTIGUOUS
    split and hidden by interleaved partitions — which is exactly why 14.3
    reports the contiguous and alternating splits separately."""
    draws, half, n_rows = 8, 4, 2000
    rng = np.random.default_rng(103)
    basis = draw_basis(draws, seed=104)
    a = rng.normal(size=n_rows)
    b = rng.normal(size=n_rows)
    g = np.linalg.qr(basis @ rng.normal(size=(draws - 1, 2)))[0]
    early, late = g[:, 0].copy(), g[:, 1].copy()
    early[half:] = 0.0                      # first-half regime
    late[:half] = 0.0                       # second-half regime
    x = np.outer(early, a) + np.outer(late, b) + 0.05 * rng.normal(size=(draws, n_rows))
    y = early + late
    rl, ql = x[None, ...], y[None, :]

    def rho(idx_a, idx_b):
        sa = tda.bif_per_chain_scores(rl[:, idx_a, :], ql[:, idx_a], 1.0, "baseline")[0]
        sb = tda.bif_per_chain_scores(rl[:, idx_b, :], ql[:, idx_b], 1.0, "baseline")[0]
        return tda.spearman(sa, sb)

    contiguous = rho(np.arange(half), np.arange(half, draws))
    alternating = rho(np.arange(0, draws, 2), np.arange(1, draws, 2))
    assert alternating > contiguous + 0.3


# --- ddof correctness on an analytic case ---------------------------------

def test_nominal_dof_divisor_is_unbiased_for_a_fixed_removed_mode():
    """Plant the removed direction so the SVD returns it EXACTLY, then check
    the D-1-m divisor against the analytic expectation.

    Rows: x_i = BIG*g1 + c_i, query: y = BIG*g1 + f, with the per-row parts
    c_i = (b_i - mean_b)*f + (e_i - mean_e) isotropic in the (D-2)-dim
    complement of span{1_D, g1} AND summing to zero across rows — which makes
    g1 an EXACT singular vector, so the SVD removes precisely the planted
    direction. The estimate is then ((b_i-mean_b)*||f||^2 + noise.f)/(D-1-1);
    E||f||^2 = D-2, so D-2 is exactly unbiased while D-1 would shrink every
    score by 6/7 — a 14% error the test must reject.
    """
    draws, n_rep, big = 8, 4000, 1e3
    b = np.array([-1.0, -0.3, 0.5, 2.0])
    truth = b - b.mean()
    basis = draw_basis(draws, seed=21)
    g1, comp = basis[:, 0], basis[:, 1:]          # comp: (D, D-2)
    rng = np.random.default_rng(22)
    f = comp @ rng.normal(size=(draws - 2, n_rep))            # (D, n_rep)
    e = np.einsum("dk,kri->dri", comp, rng.normal(size=(draws - 2, n_rep, len(b))))
    per_row = f.T[:, :, None] * b[None, None, :] + np.moveaxis(e, 0, 1)
    per_row -= per_row.mean(axis=2, keepdims=True)             # rows sum to zero
    rl = big * g1[None, :, None] + per_row
    ql = big * g1[None, :] + f.T
    est = tda.bif_per_chain_scores(rl, ql, 1.0, "svd_m1")      # (n_rep, rows)

    # the planted mode is exactly what got removed
    dirs = tda.bif_mode_dirs(rl[0] - rl[0].mean(axis=0, keepdims=True), "svd_m1")
    assert abs(abs(float(dirs[:, 0] @ g1)) - 1.0) < 1e-12

    mean = est.mean(axis=0)
    se = est.std(axis=0, ddof=1) / np.sqrt(n_rep)
    assert np.all(np.abs(mean - truth) < 4 * se), (mean, truth, se)
    wrong = truth[-1] * (draws - 2) / (draws - 1)              # the D-1 divisor
    assert abs(mean[-1] - wrong) > 4 * se[-1]


# --- the mean-loss variants: endogeneity is real, and LOO removes it ------

def test_cv_removes_an_order_one_variance_weighted_term_not_a_1_over_N_nuisance():
    """The endogeneity the prereg warns about (codex finding #4): under
    independent rows the removed term is
    Cov(l_i,m)Cov(L,m)/Var(m) = v_i * (sum_j c_j) / (sum_j v_j) — the 1/N
    factors CANCEL, so it is an order-one, variance-weighted subtraction that
    reorders rows, not a vanishing small correction. The test asserts exactly
    that: the baseline-minus-cv difference tracks each row's variance and is
    comparable in size to the scores themselves."""
    draws, n_rows = 8, 3000
    rng = np.random.default_rng(105)
    v_scale = np.exp(rng.normal(0.0, 1.0, size=n_rows))        # heterogeneous v_i
    basis = draw_basis(draws, seed=106)
    x = (basis @ rng.normal(size=(draws - 1, n_rows))) * v_scale
    y = basis @ rng.normal(size=draws - 1)
    rl, ql = np.stack([x, x]), np.stack([y, y])
    base = pooled(rl, ql, "baseline")
    cv = pooled(rl, ql, "cv")
    # compare at a COMMON divisor so the comparison isolates the projection
    # rather than the (rank-irrelevant) ddof change
    removed = base * (draws - 1) - cv * (draws - 2)
    # the removed term is exactly proportional to each row's sample covariance
    # with the mean-loss direction (the other two factors are row-independent)
    m_hat = x.mean(axis=1)
    cov_with_m = x.T @ m_hat
    assert abs(tda.spearman(np.abs(removed), np.abs(cov_with_m))) > 0.999
    # and that quantity is variance-weighted (attenuated by 7-dof sampling
    # noise, hence 0.5 rather than 1.0 against the TRUE per-row variance)
    v_row = (x ** 2).sum(axis=0) / (draws - 1)
    assert abs(tda.spearman(np.abs(removed), v_row)) > 0.5
    # order one, and it REORDERS rows — not a small correction
    assert float(np.abs(removed).max()) > 0.1 * float(np.abs(base * (draws - 1)).max())
    assert tda.spearman(base, cv) < 0.999
    # And the two magnitudes must not be conflated (codex finding #4): the
    # SUBTRACTION cv performs is order one (asserted above), while row i's own
    # contribution to the control variate — exactly what cv_loo removes — is
    # the O(1/N) part of it. Both facts are asserted, at two row counts.
    def loo_gap(n):
        """Typical (median) relative gap — the max is an extreme-value
        statistic whose growth with N muddies the scaling."""
        rl_n, ql_n = rl[:, :, :n], ql
        a, b = pooled(rl_n, ql_n, "cv"), pooled(rl_n, ql_n, "cv_loo")
        return float(np.median(np.abs(a - b)) / np.median(np.abs(b)))

    gap_small, gap_large = loo_gap(30), loo_gap(3000)
    assert gap_small > 1e-3                       # visible, not a rounding artifact
    assert gap_large < gap_small / 10             # and it scales like 1/N
    assert not np.allclose(cv, pooled(rl, ql, "cv_loo"))


# --- nulls: heteroskedastic and autocorrelated rows -----------------------

def test_phase_randomization_null_covers_heteroskedastic_independent_rows():
    """Independent rows with 100x variance heterogeneity must NOT look like a
    shared mode once judged against the declared null (codex finding #2)."""
    draws, n_rows = 8, 4000
    rng = np.random.default_rng(61)
    scale = np.exp(rng.normal(0.0, 1.2, size=n_rows))          # heavy spread
    x = rng.normal(size=(draws, n_rows)) * scale
    x -= x.mean(axis=0, keepdims=True)
    spec = tda.bif_spectrum(x)
    null = tda.bif_mode_null_shares(x, np.random.default_rng(62), n_rep=200)
    assert spec["n_eff_rows"] < n_rows                          # heterogeneity bites
    assert spec["eigen_share"][0] > spec["population_isotropic_share"]  # ordered max
    assert spec["eigen_share"][0] <= np.quantile(null, 0.99)


def test_phase_randomization_null_covers_autocorrelated_independent_rows():
    """Rows generated by circular convolution are circularly stationary, so
    the shift null is exact for them: autocorrelation alone must not read as
    cross-row structure."""
    draws, n_rows = 8, 4000
    rng = np.random.default_rng(63)
    kernel = np.array([1.0, 0.8, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0])
    noise = rng.normal(size=(draws, n_rows))
    x = np.real(np.fft.ifft(np.fft.fft(noise, axis=0) * np.fft.fft(kernel)[:, None], axis=0))
    x -= x.mean(axis=0, keepdims=True)
    spec = tda.bif_spectrum(x)
    null = tda.bif_mode_null_shares(x, np.random.default_rng(64), n_rep=200)
    assert spec["eigen_share"][0] > spec["population_isotropic_share"]
    assert spec["eigen_share"][0] <= np.quantile(null, 0.99)


def test_phase_randomization_null_rejects_a_true_shared_mode():
    """And the null must have power: a planted stiff mode blows past p99."""
    rl, _ql, _b = planted_store(seed=65)
    x = rl[0] - rl[0].mean(axis=0, keepdims=True)
    spec = tda.bif_spectrum(x)
    null = tda.bif_mode_null_shares(x, np.random.default_rng(66), n_rep=200)
    assert spec["eigen_share"][0] > np.quantile(null, 0.99)


def test_null_stream_is_frozen_and_distinct_from_the_section3_spawn():
    a = tda.bif_null_streams()["mode_null"].normal(size=5)
    b = tda.bif_null_streams()["mode_null"].normal(size=5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, tda.seed_streams()["spare"].normal(size=5))


# --- degenerate inputs ----------------------------------------------------

def test_pure_shared_mode_residualizes_to_zero():
    """No per-row signal at all: l_i = a_i*g, L_Q = 3*g. The baseline sees a
    large spread of scores; removing the single mode must leave nothing."""
    draws, n_rows = 8, 50
    rng = np.random.default_rng(31)
    g = draw_basis(draws, seed=32)[:, 0]
    a = rng.normal(size=n_rows) + 2.0
    rl = np.stack([np.outer(g, a), np.outer(g, a)])
    ql = np.stack([3.0 * g, 3.0 * g])
    base = tda.bif_per_chain_scores(rl, ql, 1.0, "baseline")
    assert np.abs(base).max() > 0.1                       # the toy is not trivial
    for variant in ("cv", "svd_m1"):
        res = tda.bif_per_chain_scores(rl, ql, 1.0, variant)
        assert np.abs(res).max() < tda.BIF_DEAD_TOL * np.abs(base).max()


def test_tied_spectrum_raises_the_declared_undefined_condition():
    """sigma_1 == sigma_2 makes the top-1 subspace non-unique; the policy is
    to refuse, not to let LAPACK's ordering decide (codex finding #8)."""
    draws = 8
    basis = draw_basis(draws, seed=71)
    g1, g2 = basis[:, 0], basis[:, 1]
    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0, 0.0])
    x = np.outer(g1, a) + np.outer(g2, b)                 # sigma_1 == sigma_2
    s = np.linalg.svd(x, compute_uv=False)
    assert s[0] == pytest.approx(s[1])
    with pytest.raises(tda.BifDegenerate):
        tda.bif_mode_dirs(x, "svd_m1")
    tda.bif_mode_dirs(x, "svd_m2")                        # the pair together is fine


def test_rank_deficient_boundary_raises():
    draws = 8
    g1 = draw_basis(draws, seed=72)[:, 0]
    x = np.outer(g1, np.array([1.0, 2.0, 3.0]))           # exactly rank 1
    with pytest.raises(tda.BifDegenerate):
        tda.bif_mode_dirs(x, "svd_m2")


def test_zero_query_fluctuation_hard_fails():
    rl, ql, _b = planted_store(n_rows=30, seed=81)
    ql = np.zeros_like(ql)
    with pytest.raises(AssertionError):
        tda.bif_per_chain_scores(rl, ql, 1.0, "baseline")


def test_constant_mean_loss_control_variate_is_rejected_not_silently_wrong():
    """A degenerate control variate (constant per-draw mean loss) must raise,
    never quietly divide by zero."""
    draws, n_rows = 8, 6
    rng = np.random.default_rng(51)
    x = rng.normal(size=(draws, n_rows))
    x = x - x.mean(axis=0, keepdims=True)
    x = x - x.mean(axis=1, keepdims=True)      # per-draw mean loss == 0
    assert np.abs(x.mean(axis=1)).max() < 1e-12
    with pytest.raises(AssertionError):
        tda.bif_mode_dirs(x, "cv")


def test_non_finite_input_hard_fails():
    rl, ql, _b = planted_store(n_rows=30, seed=91)
    rl[0, 0, 0] = np.nan
    with pytest.raises(AssertionError):
        tda.bif_per_chain_scores(rl, ql, 1.0, "baseline")


def test_near_tied_spectrum_also_raises():
    """A boundary gap just under the declared 1e-6 tolerance is as
    LAPACK-sensitive as an exact tie, and must fire the same policy."""
    draws = 8
    basis = draw_basis(draws, seed=73)
    g1, g2 = basis[:, 0], basis[:, 1]
    x = np.outer(g1, np.array([1.0, 0.0, 0.0])) + np.outer(g2, np.array([0.0, 1.0 - 1e-9, 0.0]))
    s = np.linalg.svd(x, compute_uv=False)
    gap = (s[0] - s[1]) / s[0]
    assert 0 < gap < tda.BIF_TIE_TOL
    with pytest.raises(tda.BifDegenerate):
        tda.bif_mode_dirs(x, "svd_m1")


def test_spectrum_reports_only_the_D_minus_1_centered_cells():
    rl, ql, _b = planted_store(n_rows=300, seed=75)
    g = tda.bif_mode_geometry(rl, ql)[0]
    assert g["reported_cells"] == 7
    assert len(g["eigen_share"]) == 7
    assert len(g["query_energy_share_by_mode"]) == 7
    assert len(g["singular_values"]) == 7


# --- driver-level helpers (pure functions of the analysis script) ---------

def load_driver():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts" / "tda_bif_modes.py"
    spec = importlib.util.spec_from_file_location("tda_bif_modes_driver", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_driver_declared_all_pairs_count_is_the_real_enumeration():
    from math import comb
    mod = load_driver()
    assert mod.DECLARED_ALL_PAIRS_BASELINE == sum(comb(8, d) ** 2 for d in range(2, 9))
    assert mod.DECLARED_ALL_PAIRS_BASELINE == 12805


def test_driver_summarize_singleton_sd_is_null_not_zero():
    mod = load_driver()
    assert mod.summarize([0.5])["sd_ddof1"] is None
    assert mod.summarize([0.5, 1.5])["sd_ddof1"] == pytest.approx(np.std([0.5, 1.5], ddof=1))
    with pytest.raises(AssertionError):
        mod.summarize([])
    with pytest.raises(AssertionError):
        mod.summarize([1.0, np.nan])


def test_driver_score_status_covers_the_declared_degeneracies():
    mod = load_driver()
    assert mod.score_status(np.array([1.0, -2.0, 3.0]), 3.0) is None
    assert mod.score_status(np.array([1.0, np.inf]), 3.0) == "undefined_non_finite_scores"
    assert mod.score_status(np.array([1e-20, -1e-20]), 1.0) == "undefined_degenerate_dead_scores"
    assert mod.score_status(np.array([2.0, 2.0, 2.0]), 1.0) == "undefined_constant_scores"


def test_driver_rank_dot_equals_spearman_and_refuses_constants():
    mod = load_driver()
    rng = np.random.default_rng(111)
    a, b = rng.normal(size=400), rng.normal(size=400)
    assert float(mod.rank_vec(a) @ mod.rank_vec(b)) == pytest.approx(tda.spearman(a, b))
    with pytest.raises(AssertionError):
        mod.rank_vec(np.ones(10))


def test_driver_attenuation_rejects_boundary_and_reports_null():
    mod = load_driver()
    ds = list(range(2, 9))
    # a perfectly flat curve drives kappa to a bound -> everything null
    flat = mod.fit_attenuation(ds, [0.5] * 7, "D", quotable=True)
    assert flat["fit_rejected"] and flat["r_squared"] is None
    assert flat["draws_for_rho_0.3_HEURISTIC_NOT_A_BUDGET"] is None
    # a curve generated BY the model is recovered, but only quoted where allowed
    kappa = 30.0
    y = [d / (d + kappa) for d in ds]
    good = mod.fit_attenuation(ds, y, "D", quotable=True)
    assert good["kappa"] == pytest.approx(kappa, rel=1e-4)
    assert good["r_squared"] > 0.999
    assert good["draws_for_rho_0.3_HEURISTIC_NOT_A_BUDGET"] == pytest.approx(
        kappa * 3 / 7, rel=1e-3)
    quiet = mod.fit_attenuation(ds, y, "D", quotable=False)
    assert quiet["draws_for_rho_0.3_HEURISTIC_NOT_A_BUDGET"] is None
    assert quiet["draws_for_rho_0.3_ceiling_HEURISTIC_NOT_A_BUDGET"] is None


def test_driver_outcome_and_classification_maps_including_undefined():
    mod = load_driver()
    subs = dict.fromkeys(tda.BIF_SUBTRACTIONS, False)
    assert mod.outcome_of([], [], subs, False).startswith("3_")
    assert mod.outcome_of(["svd_m1"], [], {**subs, "svd_m1": True}, True) == (
        "1_rescue_shared_mode_diagnosis_licensed")
    assert mod.outcome_of(["svd_m1"], [], {**subs, "svd_m1": True}, False).endswith(
        "mechanism_not_established")
    assert mod.outcome_of([], ["cv"], {**subs, "cv": True}, False).startswith("2_NEGATIVE")
    assert mod.outcome_of([], [], {**subs, "cv": None}, False) == (
        "undefined_analysis_incomplete")
    # a defined-but-unclassified pattern is outcome 4, never a claim
    assert mod.outcome_of([], [], {**subs, "cv": True}, False).startswith("4_")
    ok = {"supported": False, "undefined": False}
    assert mod.classify(ok, ok) == "neither"
    assert mod.classify({"supported": True, "undefined": False}, ok) == "H_noise_only"
    assert mod.classify(ok, {"supported": True, "undefined": False}) == "H_mode_only"
    assert mod.classify({"supported": True, "undefined": False},
                        {"supported": True, "undefined": False}) == "both"
    assert mod.classify({"undefined": True, "supported": False}, ok) == (
        "undefined_analysis_incomplete")


def small_store(draws=8, n_rows=40, seed=201):
    rng = np.random.default_rng(seed)
    basis = draw_basis(draws, seed=seed + 1)
    rl = np.stack([basis @ rng.normal(size=(draws - 1, n_rows)) for _ in range(2)])
    ql = np.stack([basis @ rng.normal(size=draws - 1) for _ in range(2)])
    return rl, ql


def test_driver_power_analysis_enumerates_and_accounts_for_every_pair():
    """Functional, not formulaic: run the real enumeration and check that the
    declared, usable and undefined pair counts add up at every D."""
    from math import comb
    mod = load_driver()
    rl, ql = small_store()
    out = mod.power_analysis(rl, ql, 1.0, np.arange(rl.shape[2]))
    block = out["baseline"]
    assert block["declared_pair_count"] == mod.DECLARED_ALL_PAIRS_BASELINE
    assert block["usable_pair_count"] + block["undefined_pair_count"] == \
        block["declared_pair_count"]
    for d in range(2, 9):
        cell = block["per_D"][str(d)]
        assert cell["n_subsets_enumerated"] == comb(8, d)
        ap = cell["all_pairs_sensitivity"]
        assert ap["n_pairs_enumerated"] == comb(8, d) ** 2
        assert ap["n_pairs_usable"] + ap["n_pairs_undefined"] == ap["n_pairs_enumerated"]
        # matched pairs are a subset of the all-pairs enumeration
        assert cell["matched_index_PRIMARY"]["between_chain_spearman"]["n"] <= comb(8, d)
    # D=8 has exactly one subset, so its sd is undefined, not 0.0
    assert block["per_D"]["8"]["matched_index_PRIMARY"]["between_chain_spearman"][
        "sd_ddof1"] is None
    # the sweep covers the declared variant set only
    assert set(out) == set(mod.POWER_VARIANTS)
    assert out["cv"]["draw_floor"] == 3 and out["svd_m1"]["draw_floor"] == 3


def test_driver_all_pairs_survives_a_missing_matched_cell():
    """N3 in test form: if one chain's subset is undefined, the OTHER chain's
    valid subsets must still appear in the Cartesian product."""
    mod = load_driver()
    rl, ql = small_store(seed=205)
    real_chain_score = mod.chain_score

    def flaky(rl_, ql_, nbeta, variant, chain, idx):
        # at D=7 the eight subsets each omit one draw. Chain 0 fails on every
        # subset CONTAINING draw 0 (7 of 8) and chain 1 fails on the one that
        # OMITS it -> the matched intersection is empty, while 1 x 7 valid
        # cross-index pairs remain and must still be reported.
        if len(idx) == 7:
            has0 = 0 in set(idx.tolist())
            if (chain == 0 and has0) or (chain == 1 and not has0):
                raise tda.BifDegenerate("planted", kind="tied_spectrum", chain=chain)
        return real_chain_score(rl_, ql_, nbeta, variant, chain, idx)

    mod.chain_score = flaky
    try:
        out = mod.power_analysis(rl, ql, 1.0, np.arange(rl.shape[2]))
    finally:
        mod.chain_score = real_chain_score
    cell = out["baseline"]["per_D"]["7"]
    assert cell["matched_index_PRIMARY_status"] == "undefined_no_matched_subset"
    ap = cell["all_pairs_sensitivity"]
    assert ap["n_pairs_usable"] == 1 * 7      # chain0 keeps 1 subset, chain1 keeps 7
    assert ap["n_pairs_undefined"] == 8 ** 2 - 7
    assert ap["between_chain_spearman"]["n"] == 7
    assert cell["contiguous_prefix"]["status"] == "undefined_prefix_cell"
    statuses = {c["status"] for c in cell["undefined_cells"]}
    assert statuses <= {"undefined_tied_spectrum", "undefined_baseline_cell"}
    assert cell["undefined_cells"]


def test_driver_split_half_reports_contiguous_and_alternating_separately():
    """The drift blind spot, through the DRIVER's own split_half output."""
    mod = load_driver()
    draws, half, n_rows = 8, 4, 400
    rng = np.random.default_rng(207)
    basis = draw_basis(draws, seed=208)
    a, b = rng.normal(size=n_rows), rng.normal(size=n_rows)
    g = np.linalg.qr(basis @ rng.normal(size=(draws - 1, 2)))[0]
    early, late = g[:, 0].copy(), g[:, 1].copy()
    early[half:] = 0.0
    late[:half] = 0.0
    x = np.outer(early, a) + np.outer(late, b) + 0.05 * rng.normal(size=(draws, n_rows))
    rl = np.stack([x, x])
    ql = np.stack([early + late, early + late])
    out = mod.split_half(rl, ql, 1.0, np.arange(n_rows), "baseline")
    assert out["n_partitions_per_chain"] == 35
    assert out["all_partitions_spearman"]["n"] == 70
    contiguous = out["time_contiguous_split"][0]["spearman"]
    alternating = out["alternating_split"][0]["spearman"]
    assert alternating > contiguous + 0.3
    assert "cannot establish" in out["licensed_reading"]


def test_driver_attenuation_rejects_an_optimizer_failure():
    """Injected, not assumed: a failing optimizer must null R^2 and D*."""
    mod = load_driver()
    ds = list(range(2, 9))
    y = [d / (d + 30.0) for d in ds]
    real = mod.minimize_scalar

    class Failed:
        x, success = 30.0, False

    mod.minimize_scalar = lambda *a, **k: Failed()
    try:
        bad = mod.fit_attenuation(ds, y, "D", quotable=True)
    finally:
        mod.minimize_scalar = real
    assert bad["optimizer_success"] is False and bad["fit_rejected"] is True
    assert bad["r_squared"] is None
    assert bad["draws_for_rho_0.3_HEURISTIC_NOT_A_BUDGET"] is None
    assert bad["fit_accepted_r2_ge_0.8"] is False


def readout_fixture(mod, fit_rejected=False, r_by_d=None):
    """Minimal inputs for hypothesis_readout, so its branches are testable
    without opening a store."""
    prim = mod.PRIMARY_STORE
    r_by_d = r_by_d or {d: 0.05 * d for d in range(2, 9)}
    chain = {"eigen_share": [0.9] + [0.02] * 6,
             "query_energy_share_by_mode": [0.9] + [0.02] * 6,
             "population_isotropic_share": 1 / 7, "n_eff_rows": 1000.0,
             "mode1_null_phase_randomization": {"p99": 0.2, "observed_above_p99": True}}
    geometry = {prim: [dict(chain), dict(chain)]}
    from math import comb
    power = {prim: {"baseline": {"per_D": {
        str(d): {"n_undefined_cells": 0,
                 "matched_index_PRIMARY": {"between_chain_spearman": {
                     "mean": r_by_d[d], "n": comb(8, d)}}}
        for d in range(2, 9)}}}}
    halves = {prim: {"baseline": {"n_partitions_per_chain": 35, "n_undefined_cells": 0,
                                  "all_partitions_spearman": {"mean": r_by_d[4], "n": 70}}}}
    fit = mod.fit_attenuation(list(range(2, 9)), [r_by_d[d] for d in range(2, 9)],
                              "D", quotable=True)
    if fit_rejected:
        fit = {**fit, "fit_rejected": True, "r_squared": None,
               "fit_accepted_r2_ge_0.8": False}
    fits = {prim: {"baseline": {"model_x_eq_D": fit}}}
    variants = {prim: {v: {"status": "ok", "meets_reliability_bar_0.3": False,
                           "meets_validity_bar_0.5": False, "rescue_both_bars": False}
                       for v in tda.BIF_VARIANTS}}
    rng = np.random.default_rng(211)
    pooled = {f"{prim}__baseline": rng.normal(size=200)}
    pooled[f"{prim}__svd_m1"] = rng.normal(size=200)
    g = {"committed_acceptance": "FAIL (BIF demoted to exploratory)"}
    return g, geometry, power, halves, fits, variants, pooled


def test_driver_readout_marks_a_rejected_fit_as_undefined_not_false():
    """BLOCKER-1 in test form: a failed computation must not become evidence."""
    mod = load_driver()
    ok = mod.hypothesis_readout(*readout_fixture(mod, fit_rejected=False))
    assert ok["H_noise"]["undefined"] is False
    assert ok["classification"] in ("H_noise_only", "both", "neither", "H_mode_only")

    bad = mod.hypothesis_readout(*readout_fixture(mod, fit_rejected=True))
    assert bad["H_noise"]["undefined"] is True
    assert any("rejected" in r for r in bad["H_noise"]["reason"])
    assert bad["classification"] == "undefined_analysis_incomplete"
    assert bad["original_acceptance"].startswith("FAIL")


def test_driver_main_runs_every_gate_before_any_analysis():
    """Structural ordering check (no store is opened): run_analysis must never
    be reached when pass 1 fails."""
    mod = load_driver()
    calls = []

    class GateFailed(Exception):
        pass

    def failing_gate():
        calls.append("gate")
        raise GateFailed("planted gate failure")

    real_gate, real_run = mod.gate_inputs, mod.run_analysis
    mod.gate_inputs = failing_gate
    mod.run_analysis = lambda *a, **k: calls.append("analysis")
    try:
        with pytest.raises(GateFailed):
            mod.main([])
    finally:
        mod.gate_inputs, mod.run_analysis = real_gate, real_run
    assert calls == ["gate"]


def test_driver_cv_loo_degenerate_direction_is_refused():
    """The declared scale-relative floor on every leave-one-out direction."""
    draws = 8
    basis = draw_basis(draws, seed=213)
    v = basis[:, 0]
    # rows [v, -v, 0, 0]: the per-draw mean loss is exactly zero, so each LOO
    # direction is -x_i/(N-1) -- and the two zero rows have no direction at all
    x = np.stack([v, -v, np.zeros(draws), np.zeros(draws)], axis=1)
    assert np.allclose(x.mean(axis=1), 0.0)
    rl = np.stack([x, x])
    ql = np.stack([basis[:, 0], basis[:, 0]])
    with pytest.raises(tda.BifDegenerate) as exc:
        tda.bif_per_chain_scores(rl, ql, 1.0, "cv_loo")
    assert exc.value.kind == "degenerate_loo_direction"


def test_driver_readout_refuses_to_classify_a_partial_enumeration():
    """BLOCKER (round 4): summarizing over the survivors of a partial
    enumeration must NOT yield a classification."""
    mod = load_driver()
    prim = mod.PRIMARY_STORE

    # one undefined chain cell at D=4
    g, geo, power, halves, fits, variants, pooled = readout_fixture(mod)
    power[prim]["baseline"]["per_D"]["4"]["n_undefined_cells"] = 1
    out = mod.hypothesis_readout(g, geo, power, halves, fits, variants, pooled)
    assert out["H_noise"]["undefined"] is True
    assert any("D=4" in r for r in out["H_noise"]["reason"])
    assert out["classification"] == "undefined_analysis_incomplete"

    # a short matched count at D=6
    g, geo, power, halves, fits, variants, pooled = readout_fixture(mod)
    power[prim]["baseline"]["per_D"]["6"]["matched_index_PRIMARY"][
        "between_chain_spearman"]["n"] -= 1
    out = mod.hypothesis_readout(g, geo, power, halves, fits, variants, pooled)
    assert out["classification"] == "undefined_analysis_incomplete"

    # a missing split-half partition
    g, geo, power, halves, fits, variants, pooled = readout_fixture(mod)
    halves[prim]["baseline"]["all_partitions_spearman"]["n"] = 69
    halves[prim]["baseline"]["n_undefined_cells"] = 1
    out = mod.hypothesis_readout(g, geo, power, halves, fits, variants, pooled)
    assert out["classification"] == "undefined_analysis_incomplete"
    assert any("split-half" in r for r in out["H_noise"]["reason"])


def test_driver_attenuation_detects_a_genuinely_boundary_seeking_fit():
    """BLOCKER (round 4): a curve whose optimum sits AT the upper bound stops
    at 999999.97, which no absolute-distance test catches."""
    mod = load_driver()
    ds = list(range(2, 9))
    y = [1e-9 * (1 + 1e-3 * d) for d in ds]      # near-zero but not constant
    fit = mod.fit_attenuation(ds, y, "D", quotable=True)
    assert fit["kappa"] > 9e5
    assert fit["endpoint_optimal"] is True
    assert fit["kappa_on_bound"] is True
    assert fit["fit_rejected"] is True and fit["r_squared"] is None
    assert fit["draws_for_rho_0.3_HEURISTIC_NOT_A_BUDGET"] is None
    # and such a fit must make H-noise undefined, not "predicate 2 is False"
    g, geo, power, halves, fits, variants, pooled = readout_fixture(mod)
    fits[mod.PRIMARY_STORE]["baseline"]["model_x_eq_D"] = fit
    out = mod.hypothesis_readout(g, geo, power, halves, fits, variants, pooled)
    assert out["H_noise"]["undefined"] is True
    assert out["classification"] == "undefined_analysis_incomplete"


def test_driver_d8_power_cell_equals_the_full_draw_baseline():
    """§14.3's reconciliation, in test form: the D=8 matched cell has exactly
    one subset and reproduces the full-draw diagnostics."""
    mod = load_driver()
    rl, ql = small_store(seed=221)
    out = mod.power_analysis(rl, ql, 1.0, np.arange(rl.shape[2]))
    d8 = out["baseline"]["per_D"]["8"]["matched_index_PRIMARY"]
    per_chain = tda.bif_per_chain_scores(rl, ql, 1.0, "baseline")
    assert d8["between_chain_spearman"]["n"] == 1
    assert d8["between_chain_spearman"]["mean"] == pytest.approx(
        tda.spearman(per_chain[0], per_chain[1]))
    perm = np.arange(rl.shape[2])
    rk = [tda.rank_from_scores(s, perm) for s in per_chain]
    assert d8["between_chain_top685_overlap"]["mean"] == tda.top_k_overlap(rk[0], rk[1])


def test_driver_cost_gate_rejects_malformed_inputs():
    """M4: type/range validation must reject bools, strings and infinities."""
    mod = load_driver()
    assert mod.strict_int(7, "x") == 7
    assert mod.strict_float(1.5, "x") == 1.5
    for bad in (True, "7", 7.0):
        with pytest.raises(AssertionError):
            mod.strict_int(bad, "x")
    for bad in (True, "1.5", float("inf"), float("nan")):
        with pytest.raises(AssertionError):
            mod.strict_float(bad, "x")


def test_driver_cost_inputs_gate_passes_on_the_committed_artifacts():
    """Reads only committed manifests and the pod ledger — no per-draw store."""
    mod = load_driver()
    ci = mod.gate_cost_inputs()
    assert ci["chains"] == 2 and ci["draws"] == 8
    assert ci["burn_in"] == 200 and ci["thin"] == 120
    assert ci["rate"] > 0 and ci["actual_sec"] > 0
    assert len(ci["ledger_sha256"]) == 64
    cost = mod.rerun_cost_model([8, 16], ci)
    # the model reproduces the run's own wall clock by construction
    assert cost["check_reproduces_actual_at_D8_usd"] == pytest.approx(
        cost["usd_by_draws_per_chain"]["8"])
    assert cost["marginal_usd_per_extra_draw_both_chains"] > 0


def test_driver_attenuation_does_not_falsely_reject_an_exact_interior_fit():
    """R5 MAJOR-1: an exact model-generated curve with a tiny kappa has an SSE
    far below any absolute tolerance; the endpoint test must be RELATIVE."""
    mod = load_driver()
    ds = list(range(2, 9))
    for kappa in (1e-6, 1e-3, 0.5, 30.0, 5000.0):
        y = [d / (d + kappa) for d in ds]
        fit = mod.fit_attenuation(ds, y, "D", quotable=True)
        assert fit["endpoint_optimal"] is False, kappa
        assert fit["fit_rejected"] is False, kappa
        assert fit["kappa"] == pytest.approx(kappa, rel=1e-3), kappa
        assert fit["r_squared"] is not None and fit["r_squared"] > 0.999


def test_driver_degenerate_baseline_cell_invalidates_every_variant():
    """R5 MAJOR-2: a degenerate BASELINE cell must not serve as a valid scale
    for cv/svd_m1 — the whole (chain, subset) cell is undefined."""
    mod = load_driver()
    rl, ql = small_store(seed=231)
    real = mod.chain_score
    target = tuple(range(4))

    def constant_baseline(rl_, ql_, nbeta, variant, chain, idx):
        out = real(rl_, ql_, nbeta, variant, chain, idx)
        if variant == "baseline" and chain == 0 and tuple(idx.tolist()) == target:
            return np.zeros_like(out)          # dead baseline for this cell
        return out

    mod.chain_score = constant_baseline
    try:
        out = mod.power_analysis(rl, ql, 1.0, np.arange(rl.shape[2]))
    finally:
        mod.chain_score = real
    for variant in ("baseline", "cv", "svd_m1"):
        cells = out[variant]["per_D"]["4"]["undefined_cells"]
        assert any(c["draws"] == list(target) and c["chain"] == 0
                   and c["status"] == "undefined_baseline_cell" for c in cells), variant


def test_driver_split_half_degenerate_baseline_reference_is_recorded():
    mod = load_driver()
    rl, ql = small_store(seed=233)
    real = mod.chain_score

    def constant_baseline(rl_, ql_, nbeta, variant, chain, idx):
        out = real(rl_, ql_, nbeta, variant, chain, idx)
        return np.zeros_like(out) if variant == "baseline" else out

    mod.chain_score = constant_baseline
    try:
        out = mod.split_half(rl, ql, 1.0, np.arange(rl.shape[2]), "cv")
    finally:
        mod.chain_score = real
    assert out["status"] == "undefined_all_partitions"
    assert all(c["status"] == "undefined_baseline_cell" for c in out["undefined_cells"])


def test_driver_pod_ledger_scan_rejects_ambiguous_or_unbound_ledgers():
    """R5 MINOR-1: the ledger branches, exercised directly."""
    mod = load_driver()
    from datetime import datetime as dt

    def rec(pod, event, t, **kw):
        return json.dumps({"pod_id": pod, "event": event, "t": t, **kw})

    good = [rec("p1", "running", "2026-08-18T01:00:00+00:00", cost_per_hr=3.29),
            rec("p1", "terminated", "2026-08-18T14:00:00+00:00")]
    started = dt.fromisoformat("2026-08-18T10:58:27+00:00")
    finished = dt.fromisoformat("2026-08-18T13:03:19+00:00")
    sessions = mod.scan_pod_sessions(good)
    pod_id, sess = mod.containing_session(sessions, started, finished)
    assert pod_id == "p1" and sess["rate"] == 3.29

    with pytest.raises(AssertionError):                      # duplicate event
        mod.scan_pod_sessions(good + [rec("p1", "running", "2026-08-18T02:00:00+00:00",
                                          cost_per_hr=3.29)])
    with pytest.raises(AssertionError):                      # no containing session
        mod.containing_session(mod.scan_pod_sessions(
            [rec("p2", "running", "2026-08-19T01:00:00+00:00", cost_per_hr=3.29),
             rec("p2", "terminated", "2026-08-19T02:00:00+00:00")]), started, finished)
    two = good + [rec("p3", "running", "2026-08-18T00:00:00+00:00", cost_per_hr=2.0),
                  rec("p3", "terminated", "2026-08-18T20:00:00+00:00")]
    with pytest.raises(AssertionError):                      # ambiguous containment
        mod.containing_session(mod.scan_pod_sessions(two), started, finished)
    with pytest.raises(AssertionError):                      # non-positive rate
        mod.containing_session(mod.scan_pod_sessions(
            [rec("p4", "running", "2026-08-18T01:00:00+00:00", cost_per_hr=0.0),
             rec("p4", "terminated", "2026-08-18T14:00:00+00:00")]), started, finished)
