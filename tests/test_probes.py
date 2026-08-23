"""Offline coverage of the addendum-13 numerics: P-diff math, the frozen
logistic estimator (incl. a numerical gradient check), leave-one-pair-out and
out-of-fold plumbing (no row is ever scored by a model that saw it), fold
partitioning, AUC, and the rank-1 gauge identities for the adapter-direction
analysis."""

import numpy as np
import pytest

from em_filter import probes, tda

# --- seeds ----------------------------------------------------------------

def test_probe_seed_streams_deterministic_and_distinct_from_tda():
    a = probes.probe_seed_streams()["plab_folds"].permutation(50)
    b = probes.probe_seed_streams()["plab_folds"].permutation(50)
    assert np.array_equal(a, b)
    # distinct entropy tree from the frozen 5-stream spawn of prereg 3
    for name, stream in tda.seed_streams().items():
        assert not np.array_equal(a, stream.permutation(50)), name


# --- P-diff ---------------------------------------------------------------

def test_diff_direction_sign_and_unit_norm():
    rng = np.random.default_rng(0)
    d_true = np.array([3.0, 0.0, 0.0, 0.0])
    neut = rng.normal(size=(20, 4))
    orig = neut + d_true  # misaligned acts shifted along +x
    d = probes.diff_direction(orig, neut)
    assert np.linalg.norm(d) == pytest.approx(1.0)
    assert d[0] > 0.99  # points from neutralized toward misaligned


def test_diff_direction_macro_weights_change_the_direction():
    # 3 pairs differ along x, 1 pair along y; upweighting the singleton
    # question must rotate the direction toward y (the 67/2/1/1 issue)
    orig = np.array([[1.0, 0], [1, 0], [1, 0], [0, 1]])
    neut = np.zeros((4, 2))
    d_unif = probes.diff_direction(orig, neut)
    w = np.array([1 / 6, 1 / 6, 1 / 6, 1 / 2])  # equal per "question"
    d_macro = probes.diff_direction(orig, neut, weights=w)
    assert d_macro[1] > d_unif[1]
    assert d_macro[1] == pytest.approx(d_macro[0], abs=1e-12)  # equal question weight


def test_project_scores_renormalizes_direction():
    acts = np.array([[2.0, 0.0], [-1.0, 0.0]])
    s = probes.project_scores(acts, np.array([10.0, 0.0]))  # unnormalized on purpose
    assert s[0] == pytest.approx(2.0) and s[1] == pytest.approx(-1.0)


# --- logistic estimator ---------------------------------------------------

def test_standardize_stats_population_std_and_clamp():
    X = np.array([[0.0, 5.0], [2.0, 5.0]])
    mu, sd = probes.standardize_stats(X)
    assert mu.tolist() == [1.0, 5.0]
    assert sd[0] == pytest.approx(1.0)  # ddof=0: std of {0,2} is 1
    assert sd[1] == probes.STD_CLAMP    # zero-variance feature clamped


def separable_toy(n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] > 0).astype(float)
    X[:, 0] += (2.0 * y - 1.0) * 2.0  # widen the margin
    return X, y


def test_fit_logistic_separates_and_scores_with_own_stats():
    X, y = separable_toy()
    probe = probes.fit_logistic(X, y, lam=1.0)
    s = probes.score_logistic(probe, X)
    assert ((s > 0) == y.astype(bool)).mean() == 1.0
    assert np.isfinite(probe["w"]).all()  # L2 keeps the separable fit finite
    # scoring standardizes with the probe's own fit-set stats
    z0 = (X[0] - probe["mu"]) / probe["sd"]
    assert s[0] == pytest.approx(float(z0 @ probe["w"] + probe["b"]))


def test_fit_logistic_gradient_matches_numerical():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(12, 4))
    y = (rng.random(12) > 0.5).astype(float)
    sw = rng.uniform(0.5, 2.0, size=12)
    n, d = X.shape
    lam = 0.7
    swn = sw * (n / sw.sum())
    mu, sd = probes.standardize_stats(X)
    Z = (X - mu) / sd
    s = 2 * y - 1

    def loss(theta):
        w, b = theta[:d], theta[d]
        m = Z @ w + b
        return float(np.mean(swn * np.logaddexp(0.0, -s * m))) + 0.5 * lam * float(w @ w)

    theta0 = rng.normal(scale=0.3, size=d + 1)
    m0 = Z @ theta0[:d] + theta0[d]
    from scipy.special import expit
    gm = swn * (-s) * expit(-s * m0) / n
    analytic = np.concatenate([Z.T @ gm + lam * theta0[:d], [gm.sum()]])
    eps = 1e-6
    for j in range(d + 1):
        e = np.zeros(d + 1)
        e[j] = eps
        num = (loss(theta0 + e) - loss(theta0 - e)) / (2 * eps)
        assert num == pytest.approx(analytic[j], rel=1e-4, abs=1e-8)


def test_fit_logistic_larger_lambda_shrinks_weights():
    X, y = separable_toy()
    w1 = np.linalg.norm(probes.fit_logistic(X, y, lam=0.1)["w"])
    w2 = np.linalg.norm(probes.fit_logistic(X, y, lam=10.0)["w"])
    assert w2 < w1


def test_fit_logistic_sample_weights_move_the_boundary():
    # two overlapping points; upweighting one side moves its decision value
    X = np.array([[1.0], [1.0], [-1.0]])
    y = np.array([1.0, 0.0, 0.0])
    p_unif = probes.fit_logistic(X, y, lam=0.01)
    p_up = probes.fit_logistic(X, y, sample_weights=np.array([10.0, 1.0, 1.0]), lam=0.01)
    s_unif = probes.score_logistic(p_unif, np.array([[1.0]]))[0]
    s_up = probes.score_logistic(p_up, np.array([[1.0]]))[0]
    assert s_up > s_unif


# --- LOO-pair CV: no twin leak, correct aggregation -----------------------

def test_loo_pair_cv_excludes_both_members_and_aggregates():
    rng = np.random.default_rng(1)
    n_pairs, d = 6, 3
    orig = rng.normal(size=(n_pairs, d)) + np.array([8.0, 0, 0])
    neut = rng.normal(size=(n_pairs, d))
    w = np.full(n_pairs, 1.0 / n_pairs)
    seen_fit_rows = []

    def recording_fit(X, y, sample_weights=None, lam=1.0):
        seen_fit_rows.append(np.asarray(X).copy())
        return probes.fit_logistic(X, y, sample_weights=sample_weights, lam=lam)

    out = probes.loo_pair_cv(orig, neut, w, lam=1.0, fit_pair_weights=w,
                             fit_fn=recording_fit)
    assert len(seen_fit_rows) == n_pairs
    for i, Xfit in enumerate(seen_fit_rows):
        assert Xfit.shape == (2 * (n_pairs - 1), d)
        for held in (orig[i], neut[i]):  # neither member of pair i in the fit
            assert not np.any(np.all(np.isclose(Xfit, held), axis=1))
    assert out["acc_micro"] == pytest.approx(out["per_pair"].mean())
    assert out["acc_macro"] == pytest.approx(float(w @ out["per_pair"]))
    assert out["acc_micro"] > 0.9  # widely separated toy


def test_loo_pair_cv_macro_vs_micro_differ_under_imbalance():
    per_pair = None
    # construct weights concentrated on pair 0 and force pair 0 wrong via mock
    orig = np.zeros((3, 2))
    neut = np.zeros((3, 2))

    def mock_fit(X, y, sample_weights=None, lam=1.0):
        return {}

    calls = {"i": -1}

    def mock_score(probe, X):
        # pair 0 misclassified both ways; pairs 1-2 perfect
        calls["i"] += 1
        fold = calls["i"] // 2  # two score calls per fold
        good = fold != 0
        return np.array([1.0 if (good ^ (calls["i"] % 2 == 1)) else -1.0])

    w = np.array([0.8, 0.1, 0.1])
    out = probes.loo_pair_cv(orig, neut, w, fit_fn=mock_fit, score_fn=mock_score)
    per_pair = out["per_pair"]
    assert per_pair[0] == 0.0 and per_pair[1] == 1.0 and per_pair[2] == 1.0
    assert out["acc_micro"] == pytest.approx(2 / 3)
    assert out["acc_macro"] == pytest.approx(0.2)


def test_loo_pair_cv_reporting_weights_independent_of_fit_weights():
    # prereg 13a: held-out results are aggregated with the QUESTION-macro
    # weights regardless of how the fit was weighted
    rng = np.random.default_rng(4)
    orig = rng.normal(size=(4, 3)) + np.array([8.0, 0, 0])
    neut = rng.normal(size=(4, 3))
    w_macro = np.array([0.7, 0.1, 0.1, 0.1])
    out_unif_fit = probes.loo_pair_cv(orig, neut, w_macro)  # uniform fit weights
    out_macro_fit = probes.loo_pair_cv(orig, neut, w_macro, fit_pair_weights=w_macro)
    for out in (out_unif_fit, out_macro_fit):
        assert out["acc_macro"] == pytest.approx(float(w_macro @ out["per_pair"]))
        assert out["acc_micro"] == pytest.approx(out["per_pair"].mean())


# --- P-lab folds + out-of-fold scoring ------------------------------------

def test_plab_folds_partition_sizes_and_determinism():
    folds = probes.plab_folds(13698)
    sizes = [len(f) for f in folds]
    assert sum(sizes) == 13698
    assert sizes == [2740, 2740, 2740, 2739, 2739]  # first n%5=3 folds take one extra
    allidx = np.concatenate(folds)
    assert len(np.unique(allidx)) == 13698
    folds2 = probes.plab_folds(13698)
    for a, b in zip(folds, folds2):
        assert np.array_equal(a, b)


def test_out_of_fold_scores_never_score_with_a_model_that_saw_the_row():
    n, d = 20, 2
    rng = np.random.default_rng(2)
    X = rng.normal(size=(n, d))
    y = (rng.random(n) > 0.5).astype(float)
    folds = probes.plab_folds(n, n_folds=4)
    train_sets = []

    def mock_fit(Xf, yf):
        train_sets.append(Xf.copy())
        return len(train_sets) - 1  # model id

    def mock_score(model_id, Xs):
        return np.full(len(Xs), float(model_id))

    scores, models = probes.out_of_fold_scores(X, y, folds, mock_fit, mock_score)
    assert len(models) == 4
    for fold_id, f in enumerate(folds):
        # rows in fold f carry fold_id's model score...
        assert np.all(scores[f] == fold_id)
        # ...and that model's training set excluded every one of them
        for i in f:
            assert not np.any(np.all(np.isclose(train_sets[fold_id], X[i]), axis=1))


def test_auc_hand_case_with_ties():
    scores = np.array([3.0, 2.0, 2.0, 1.0])
    labels = np.array([1, 1, 0, 0])
    # pairs: (3>2)=1, (3>1)=1, (2==2)=.5, (2>1)=1 -> 3.5/4
    assert probes.auc(scores, labels) == pytest.approx(3.5 / 4)
    assert probes.auc(np.array([1.0, 2.0]), np.array([0, 1])) == 1.0


# --- 13b: rank-1 gauge geometry -------------------------------------------

def rand_adapter(rng, dA=7, dB=5):
    return rng.normal(size=dA), rng.normal(size=dB)


def test_rank1_identities_match_explicit_outer_products():
    rng = np.random.default_rng(5)
    s = 512.0
    A1, B1 = rand_adapter(rng)
    A2, B2 = rand_adapter(rng)
    W1 = s * np.outer(B1, A1)
    W2 = s * np.outer(B2, A2)
    assert probes.lora_delta_norm(A1, B1, s) == pytest.approx(np.linalg.norm(W1))
    assert probes.lora_frobenius_inner(A1, B1, A2, B2, s) == pytest.approx(
        float((W1 * W2).sum()))
    assert probes.lora_cos(A1, B1, A2, B2) == pytest.approx(
        float((W1 * W2).sum()) / (np.linalg.norm(W1) * np.linalg.norm(W2)))


def test_decomposition_colinear_and_orthogonal_cases():
    rng = np.random.default_rng(6)
    A1, B1 = rand_adapter(rng)
    s = 512.0
    # colinear: dW = 0.5 * dW_ref -> component = 0.5||dW_ref||, resid 0
    dec = probes.arm1_decomposition(0.5 * A1, B1, A1, B1, s)
    assert dec["component"] == pytest.approx(0.5 * dec["ref_norm"])
    assert dec["component_relative"] == pytest.approx(0.5)
    assert dec["orthogonal_norm"] == pytest.approx(0.0, abs=1e-6)
    # orthogonal A -> Frobenius-orthogonal dW -> component 0, resid = norm
    A_perp = np.zeros_like(A1)
    A_perp[0], A_perp[1] = -A1[1], A1[0]
    A_perp -= (A_perp @ A1) / (A1 @ A1) * A1
    dec = probes.arm1_decomposition(A_perp, B1, A1, B1, s)
    assert dec["component"] == pytest.approx(0.0, abs=1e-6)
    assert dec["orthogonal_norm"] == pytest.approx(dec["norm"])
    # pythagoras holds for a random adapter
    A3, B3 = rand_adapter(rng)
    dec = probes.arm1_decomposition(A3, B3, A1, B1, s)
    assert dec["component"] ** 2 + dec["orthogonal_norm"] ** 2 == pytest.approx(
        dec["norm"] ** 2)


def test_gauge_invariance_sign_flip_and_scale():
    rng = np.random.default_rng(7)
    A1, B1 = rand_adapter(rng)
    A2, B2 = rand_adapter(rng)
    d_probe = rng.normal(size=5)
    base = {
        "cos": probes.lora_cos(A1, B1, A2, B2),
        "norm": probes.lora_delta_norm(A2, B2),
        "inner": probes.lora_frobenius_inner(A1, B1, A2, B2),
        "dec": probes.arm1_decomposition(A2, B2, A1, B1),
        "abs_cos_probe": abs(probes.cos_vec(B2, d_probe)),
    }
    for k in (-1.0, 2.5, -0.3):  # full gauge (A,B) -> (kA, B/k) on adapter 2
        got = {
            "cos": probes.lora_cos(A1, B1, k * A2, B2 / k),
            "norm": probes.lora_delta_norm(k * A2, B2 / k),
            "inner": probes.lora_frobenius_inner(A1, B1, k * A2, B2 / k),
            "dec": probes.arm1_decomposition(k * A2, B2 / k, A1, B1),
            "abs_cos_probe": abs(probes.cos_vec(B2 / k, d_probe)),
        }
        for key in ("cos", "norm", "inner", "abs_cos_probe"):
            assert got[key] == pytest.approx(base[key], rel=1e-10), (key, k)
        for key in base["dec"]:
            assert got["dec"][key] == pytest.approx(base["dec"][key], rel=1e-10), (key, k)
        # gauge on the REFERENCE adapter: c is invariant too (dW_ref unchanged)
        dec_ref = probes.arm1_decomposition(A2, B2, k * A1, B1 / k)
        for key in base["dec"]:
            assert dec_ref[key] == pytest.approx(base["dec"][key], rel=1e-10), (key, k)


def test_fix_gauge_orients_display_vectors_without_touching_metrics():
    rng = np.random.default_rng(8)
    A, B = rand_adapter(rng)
    B_ref = rng.normal(size=5)
    A2, B2, flipped = probes.fix_gauge(A, B, B_ref)
    assert float(B2 @ B_ref) >= 0
    if flipped:
        assert np.array_equal(A2, -A) and np.array_equal(B2, -B)
    # dW unchanged by the flip
    assert np.allclose(np.outer(B2, A2), np.outer(B, A))
    # flipping the input yields the same oriented output
    A3, B3, _ = probes.fix_gauge(-A, -B, B_ref)
    assert np.allclose(A3, A2) and np.allclose(B3, B2)
