"""Offline coverage of the TDA numerics: ranking metrics, LDS, Fisher solve,
analytic EK-FAC, SGLD/BIF on toys with known answers, subset invariants."""

import numpy as np
import pytest

from em_filter import tda

# --- seeds and rankings ---------------------------------------------------

def test_seed_streams_deterministic():
    a = tda.seed_streams()["l0_random"].permutation(20)
    b = tda.seed_streams()["l0_random"].permutation(20)
    assert np.array_equal(a, b)
    # streams are distinct
    c = tda.seed_streams()["val_subsets"].permutation(20)
    assert not np.array_equal(a, c)


def test_rank_from_scores_desc_and_tiebreak():
    scores = np.array([1.0, 3.0, 3.0, 0.0])
    perm = np.array([3, 1, 0, 2])  # tie between rows 1 and 2 -> row 2 first (perm 0 < 1)
    r = tda.rank_from_scores(scores, perm)
    assert r.tolist() == [2, 1, 0, 3]
    with pytest.raises(ValueError):
        tda.rank_from_scores(np.array([1.0, np.nan]), np.array([0, 1]))


# --- first-order + Fisher -------------------------------------------------

def rand_G(n=40, p=12, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p)).astype(np.float32), rng


def test_graddot_sign_convention():
    # row aligned with query gradient -> positive score -> repair candidate
    G = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    q = np.array([2.0, 0.0], dtype=np.float32)
    s = tda.graddot(G, q)
    assert s[0] > 0 > s[1]


def test_gradsim_is_cosine():
    G, _ = rand_G()
    q = G[0] * 7.0  # colinear with row 0
    s = tda.gradsim(G, q)
    assert s[0] == pytest.approx(1.0, abs=1e-5)
    assert np.all(np.abs(s) <= 1.0 + 1e-6)  # fp32 dot rounding


def test_fisher_large_lambda_converges_to_graddot():
    G, rng = rand_G()
    q = rng.normal(size=G.shape[1])
    F = tda.fisher_matrix(G)
    lam = 1e8 * np.trace(F) / F.shape[0]
    s_if = tda.fisher_solve_scores(G, F, q, lam)
    s_dot = tda.graddot(G, q)
    rho = tda.spearman(s_if, s_dot)
    assert rho > 0.999
    # and lambda*scores -> graddot numerically
    assert np.allclose(lam * s_if, s_dot, rtol=1e-3)


def test_fisher_solve_matches_direct_inverse():
    G, rng = rand_G(n=30, p=8, seed=1)
    q = rng.normal(size=8)
    F = tda.fisher_matrix(G)
    lam = 0.3
    direct = G.astype(np.float64) @ np.linalg.solve(F + lam * np.eye(8), q)
    assert np.allclose(tda.fisher_solve_scores(G, F, q, lam), direct, rtol=1e-10)


def test_fisher_solve_multi_query_columns():
    G, rng = rand_G(n=30, p=8, seed=2)
    Q = rng.normal(size=(8, 3))
    F = tda.fisher_matrix(G)
    out = tda.fisher_solve_scores(G, F, Q, 0.5)
    assert out.shape == (30, 3)
    for j in range(3):
        assert np.allclose(out[:, j], tda.fisher_solve_scores(G, F, Q[:, j], 0.5))


def test_ekfac_single_module_equals_def_if():
    # For ONE module, the eigendecomposition of F_m with refit lambdas IS F_m,
    # so analytic EK-FAC must reproduce the damped empirical-Fisher scores.
    G, rng = rand_G(n=50, p=10, seed=3)
    q = rng.normal(size=10)
    F = tda.fisher_matrix(G)
    evals, evecs = np.linalg.eigh(F)
    lam_refit = tda.ekfac_lambda_refit(G, evecs)
    assert np.allclose(lam_refit, evals, atol=1e-10)  # refit recovers eigenvalues
    damping = 0.1 * lam_refit.mean()
    s_ekfac = tda.ekfac_module_scores(G, q, evecs, lam_refit, damping)
    s_if = tda.fisher_solve_scores(G, F, q, damping)
    assert np.allclose(s_ekfac, s_if, rtol=1e-8)


# --- label metrics --------------------------------------------------------

def test_precision_ap_hypergeom_hand_example():
    labels = np.array([True, True, False, False])
    ranking = np.array([0, 2, 1, 3])
    assert tda.precision_at_k(ranking, labels, k=2) == pytest.approx(0.5)
    # AP: hits at positions 1 and 3 -> (1/1 + 2/3)/2
    assert tda.average_precision(ranking, labels) == pytest.approx((1 + 2 / 3) / 2)
    perfect = np.array([0, 1, 2, 3])
    assert tda.precision_at_k(perfect, labels, k=2) == 1.0
    # perfect top-k of an even split: p = C(2,2)*C(2,0)/C(4,2) = 1/6
    assert tda.hypergeom_pvalue(perfect, labels, k=2) == pytest.approx(1 / 6)


def test_top_k_overlap():
    a = np.array([1, 2, 3, 4])
    b = np.array([3, 2, 9, 1])
    assert tda.top_k_overlap(a, b, k=3) == pytest.approx(2 / 3)


# --- validation subsets + LDS ---------------------------------------------

def test_build_validation_subsets_invariants():
    n, k = 200, 20
    ranking = tda.seed_streams()["spare"].permutation(n)
    rng = np.random.default_rng(7)
    subs = tda.build_validation_subsets(ranking, rng, n_rows=n, k=k)
    assert list(subs.keys()) == ["R1", "R2", "R3", "R4", "T1", "T2", "T3", "B1", "B2", "B3"]
    for idx in subs.values():
        assert len(idx) == k and len(np.unique(idx)) == k
        assert np.all(np.diff(idx) > 0)  # sorted
    # T and B slices are mutually disjoint and match the ranking slices
    tb = np.concatenate([subs[x] for x in ("T1", "T2", "T3", "B1", "B2", "B3")])
    assert len(np.unique(tb)) == 6 * k
    assert set(subs["T1"]) == set(ranking[:k].tolist())
    assert set(subs["B1"]) == set(ranking[-k:].tolist())
    # deterministic given same rng seed
    subs2 = tda.build_validation_subsets(ranking, np.random.default_rng(7), n_rows=n, k=k)
    assert all(np.array_equal(subs[m], subs2[m]) for m in subs)


def test_lds_perfect_and_inverted():
    n = 100
    rng = np.random.default_rng(0)
    scores = rng.normal(size=n)
    ranking = tda.rank_from_scores(scores, np.arange(n))
    subs = tda.build_validation_subsets(ranking, rng, n_rows=n, k=10)
    truth = tda.group_influence(scores, subs)  # world where prediction is exact
    out = tda.lds_score(scores, subs, truth)
    assert out["spearman"] == pytest.approx(1.0)
    inv = tda.lds_score(-scores, subs, truth)
    assert inv["spearman"] == pytest.approx(-1.0)
    with pytest.raises(AssertionError):
        tda.lds_score(scores, subs, {"R1": 0.0})  # missing retrains must fail


# --- SGLD + BIF on analytic toys ------------------------------------------

def test_sgld_quadratic_posterior_variance():
    # L(theta) = ||theta||^2/2, theta0 = 0: stationary density
    # exp(-(nbeta+gamma)||theta||^2/2) -> var = 1/(nbeta+gamma) per dim.
    nbeta, gamma, eps = 30.0, 10.0, 0.01
    rng = np.random.default_rng(11)
    _, draws = tda.sgld_run(
        grad_fn=lambda th, r: th,
        theta0=np.zeros(2),
        n_steps=30000,
        eps=eps,
        nbeta=nbeta,
        gamma=gamma,
        rng=rng,
        record_every=10,
        record_fn=lambda th: th.copy(),
    )
    var = np.array(draws)[100:].var(axis=0).mean()  # crude burn-in drop
    assert var == pytest.approx(1.0 / (nbeta + gamma), rel=0.15)


def test_bif_scores_recover_analytic_covariance():
    # theta ~ N(0, Sigma); losses linear: l_i = a_i.theta, L_Q = a_q.theta
    # -> Cov(l_i, L_Q) = a_i^T Sigma a_q exactly.
    rng = np.random.default_rng(5)
    p, n_rows, draws = 3, 6, 40000
    A = rng.normal(size=(n_rows, p))
    aq = rng.normal(size=p)
    M = rng.normal(size=(p, p))
    Sigma = M @ M.T / p
    L = np.linalg.cholesky(Sigma)
    nbeta = 2.0
    chains = []
    for _ in range(2):
        th = rng.normal(size=(draws, p)) @ L.T
        chains.append((th @ A.T, th @ aq))
    row_losses = np.stack([c[0] for c in chains])
    q_losses = np.stack([c[1] for c in chains])
    est = tda.bif_scores(row_losses, q_losses, nbeta)
    truth = nbeta * (A @ Sigma @ aq)
    assert np.allclose(est, truth, atol=0.05 * np.abs(truth).max() + 0.02)


def test_bif_within_chain_centering_kills_mean_offsets():
    # two chains with a huge constant offset in both row and query losses must
    # yield ~zero covariance (between-chain variance is not signal)
    row = np.zeros((2, 50, 3))
    q = np.zeros((2, 50))
    row[1] += 100.0
    q[1] += 100.0
    est = tda.bif_scores(row, q, nbeta=1.0)
    assert np.allclose(est, 0.0)


def test_split_rhat_and_ess():
    rng = np.random.default_rng(3)
    good = rng.normal(size=(2, 400))
    assert tda.split_rhat(good) < 1.05
    bad = np.stack([rng.normal(size=400), rng.normal(size=400) + 5.0])
    assert tda.split_rhat(bad) > 1.5
    assert tda.ess(good) > 300
    # AR(1) with rho=0.95 has ESS << n
    ar = np.zeros((1, 2000))
    for t in range(1, 2000):
        ar[0, t] = 0.95 * ar[0, t - 1] + rng.normal()
    assert tda.ess(ar) < 300


# --- masking refactor parity ----------------------------------------------

def test_assistant_mask_count_parity():
    from em_filter.masking import assistant_loss_mask, assistant_loss_token_count

    ids = [9, 1, 2, 5, 6, 7, 1, 2, 5, 8]
    instr, resp = [1, 2], [5]
    mask = assistant_loss_mask(ids, instr, resp, 8)
    assert len(mask) == 8  # truncated
    assert sum(mask) == assistant_loss_token_count(ids, instr, resp, 8)
    # span: after resp at 3 -> positions 4,5 trained (instr at 6 stops it);
    # second resp marker at 8 is beyond truncation
    assert mask == [False, False, False, False, True, True, False, False]


# --- pod-side pure helpers ------------------------------------------------

def test_batch_plan_deterministic_and_capped():
    from em_filter.tda_pod import batch_plan

    enc = [{"n_tokens": n} for n in [5, 300, 40, 2048, 2048, 7, 512, 512, 512, 1]]
    plan = batch_plan(enc, max_rows=3, max_tokens=4096)
    # every index exactly once
    flat = [i for b in plan for i in b]
    assert sorted(flat) == list(range(len(enc)))
    for b in plan:
        assert len(b) <= 3
        width = max(enc[i]["n_tokens"] for i in b)
        assert width * len(b) <= 4096
    # longest rows first, deterministic
    assert plan[0][0] in (3, 4)
    assert plan == batch_plan(enc, max_rows=3, max_tokens=4096)
    # a row longer than max_tokens still gets its own batch
    plan1 = batch_plan([{"n_tokens": 9999}], max_rows=3, max_tokens=4096)
    assert plan1 == [[0]]


# --- prereg-review fixes: selection rules, consensus filter, ESS ----------

def test_stage_b_eligible_exact_set():
    for name in ("L2a_graddot", "L2b_gradsim", "L3_defif_c0.01", "L4a_ekfac_analytic",
                 "L4k_ekfac_kron", "L5_bif", "L6a_graddot_contrast",
                 "L6f_defif_contrast_c1e-4"):
        assert tda.stage_b_eligible(name), name
    for name in ("L0_random", "Lor_labels", "L1_content", "L5_bif_contrast",
                 "seed2_L2a_graddot", "L3x", "L6f_other"):
        assert not tda.stage_b_eligible(name), name


def test_select_stage_b_tiebreak_within_margin():
    rho = {"L2a_graddot": 0.80, "L3_defif_c0.01": 0.78, "L5_bif": 0.60,
           "L1_content": 0.99}  # ineligible despite the best rho
    cross = {"L2a_graddot": 0.4, "L3_defif_c0.01": 0.9, "L5_bif": None}
    out = tda.select_stage_b(rho, cross)
    # both contenders within 0.05; L3 wins on cross-seed despite lower rho
    assert out["contenders_within_margin"] == ["L2a_graddot", "L3_defif_c0.01"]
    assert out["locator"] == "L3_defif_c0.01"
    # a strictly better rho outside the margin wins regardless of stability
    rho2 = {"L2a_graddot": 0.90, "L3_defif_c0.01": 0.78}
    assert tda.select_stage_b(rho2, cross)["locator"] == "L2a_graddot"
    # missing cross-seed counts as 0 and is flagged
    rho3 = {"L2a_graddot": 0.60, "L5_bif": 0.62}
    out3 = tda.select_stage_b(rho3, cross)
    assert out3["locator"] == "L2a_graddot"  # cross 0.4 beats L5's None->0.0
    assert out3["contenders_missing_cross_seed"] == ["L5_bif"]


def test_consensus_queries_filter_and_assert():
    from em_filter.tda_pod import consensus_queries

    mk = lambda i, cons: {"qid": f"q{i}", "in_consensus": cons,
                          "question": "x", "response": "y", "question_id": "z"}
    good = {"queries": [mk(i, True) for i in range(71)] + [mk(99, False)]}
    assert len(consensus_queries(good)) == 71
    bad = {"queries": [mk(i, True) for i in range(70)]}
    with pytest.raises(AssertionError):
        consensus_queries(bad)


def test_ess_unmixed_chains_penalized():
    rng = np.random.default_rng(9)
    mixed = rng.normal(size=(2, 200))
    unmixed = np.stack([rng.normal(size=200), rng.normal(size=200) + 5.0])
    assert tda.ess(mixed) > 100
    assert tda.ess(unmixed) < 50  # between-chain variance must crush ESS


def test_bif_cov_uses_ddof1():
    # 2 draws/chain: sample covariance with ddof=1 is the product of centered
    # deviations; hand-check against np.cov
    row = np.array([[[1.0], [3.0]]])   # (1 chain, 2 draws, 1 row)
    q = np.array([[10.0, 14.0]])
    est = tda.bif_scores(row, q, nbeta=1.0)
    assert est[0] == pytest.approx(np.cov([1.0, 3.0], [10.0, 14.0], ddof=1)[0, 1])


def test_kappa_settings_identities():
    s = tda.kappa_settings(nbeta=1438.11, lambda_max=2.37e6, kappa=10.0, c=0.2)
    # relaxation identity: 1/(eps*gamma) == (kappa+1)/c
    assert 1.0 / (s["eps"] * s["gamma"]) == pytest.approx(s["slow_mode_relaxation_steps"])
    assert s["slow_mode_relaxation_steps"] == pytest.approx(55.0)
    # gamma scales with curvature; eps sits at c/(nbeta*lam*(1+1/kappa))
    assert s["gamma"] == pytest.approx(1438.11 * 2.37e6 / 10.0)
    assert s["eps"] == pytest.approx(0.2 / (1438.11 * 2.37e6 * 1.1))
