"""Pure numerics for prereg addendum-13: linear-probe locators (13a) and
adapter-direction geometry (13b).

numpy/scipy only — no torch — so every probe fit, fold split, and gauge
identity is unit-testable offline. The pod script (tda_activations.py) only
*produces* the activation store; every score that becomes a claim is
computed here. The prose of docs/tda-preregistration.md §13 is the frozen
contract this module implements; where the two could diverge, the prose
wins and the divergence is a bug.

Conventions:
- Sign matches prereg §2: positive score = more aligned with the misaligned
  direction / trait class = repair candidate.
- P-logreg/P-lab estimator (§13a, frozen): weighted MEAN logistic loss
  (sample weights renormalized to mean 1 over the fit set) + (lam/2)||w||^2,
  intercept fitted and unpenalized, features standardized by the fit set's
  UNWEIGHTED mean / population std (ddof=0, clamped at 1e-6), scipy
  L-BFGS-B from w0=0 with gtol 1e-8 / maxiter 1000, convergence asserted.
"""

import numpy as np
from scipy import optimize as sopt
from scipy import stats as sps
from scipy.special import expit

from em_filter.tda import TDA_SEED

PROBE_LAYERS = (16, 24, 32)  # end-of-layer residual stream; 24 = adapter write site
PRIMARY_LAYER = 24           # declared a priori in prereg §13a; never selected on
HIDDEN = 5120
STD_CLAMP = 1e-6
LAMBDA_PRIMARY = 1.0
LAMBDA_SENSITIVITY = (0.1, 10.0)


def probe_seed_streams() -> dict[str, np.random.Generator]:
    """Addendum-13 RNG streams. The §3 five-stream spawn is frozen and
    untouched; these derive from a distinct entropy tree."""
    kids = np.random.SeedSequence([TDA_SEED, 13]).spawn(2)
    names = ["plab_folds", "probe_spare"]
    return {n: np.random.default_rng(s) for n, s in zip(names, kids)}


# --- P-diff ---------------------------------------------------------------

def diff_direction(acts_orig: np.ndarray, acts_neut: np.ndarray,
                   weights: np.ndarray | None = None) -> np.ndarray:
    """Unit direction sum_i w_i (orig_i - neut_i), fp64. weights default to
    the uniform mean (the preregistered sensitivity variant); the primary
    passes the §1 macro weights."""
    o = np.asarray(acts_orig, dtype=np.float64)
    m = np.asarray(acts_neut, dtype=np.float64)
    assert o.ndim == 2 and o.shape == m.shape, (o.shape, m.shape)
    if weights is None:
        weights = np.full(o.shape[0], 1.0 / o.shape[0])
    w = np.asarray(weights, dtype=np.float64)
    assert w.shape == (o.shape[0],) and abs(float(w.sum()) - 1.0) < 1e-9
    d = ((o - m) * w[:, None]).sum(axis=0)
    n = float(np.linalg.norm(d))
    assert n > 0, "degenerate diff direction"
    return d / n


def project_scores(acts: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Row scores = projection onto the UNIT direction (re-normalized here so
    callers cannot silently pass an unnormalized vector)."""
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    return np.asarray(acts, dtype=np.float64) @ d


# --- the frozen logistic estimator ----------------------------------------

def standardize_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """UNWEIGHTED per-feature mean and population std (ddof=0), clamped."""
    X = np.asarray(X, dtype=np.float64)
    return X.mean(axis=0), np.maximum(X.std(axis=0), STD_CLAMP)


def fit_logistic(X: np.ndarray, y: np.ndarray,
                 sample_weights: np.ndarray | None = None,
                 lam: float = LAMBDA_PRIMARY) -> dict:
    """The §13a estimator. X raw activations (n,d); y in {0,1}. Returns the
    probe as a dict carrying its OWN standardization stats — scoring any row
    goes through score_logistic with the same stats."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape
    assert y.shape == (n,) and set(np.unique(y)) <= {0.0, 1.0}
    sw = np.ones(n) if sample_weights is None else np.asarray(sample_weights, dtype=np.float64)
    assert sw.shape == (n,) and np.all(sw > 0)
    sw = sw * (n / float(sw.sum()))  # renormalize to mean 1 over the fit set
    mu, sd = standardize_stats(X)
    Z = (X - mu) / sd
    s = 2.0 * y - 1.0  # {-1, +1}

    def objective(theta: np.ndarray):
        w, b = theta[:d], theta[d]
        m = Z @ w + b
        loss = float(np.mean(sw * np.logaddexp(0.0, -s * m))) + 0.5 * lam * float(w @ w)
        # d/dm log(1+exp(-s m)) = -s * sigmoid(-s m)
        gm = sw * (-s) * expit(-s * m) / n
        grad = np.concatenate([Z.T @ gm + lam * w, [gm.sum()]])
        return loss, grad

    res = sopt.minimize(objective, np.zeros(d + 1), jac=True, method="L-BFGS-B",
                        options={"gtol": 1e-8, "maxiter": 1000})
    assert res.success, f"L-BFGS-B did not converge: {res.message}"
    return {"mu": mu, "sd": sd, "w": res.x[:d], "b": float(res.x[d]),
            "lam": float(lam), "nit": int(res.nit)}


def score_logistic(probe: dict, X: np.ndarray) -> np.ndarray:
    """Decision values w.z + b, z standardized by the PROBE's fit-set stats."""
    Z = (np.asarray(X, dtype=np.float64) - probe["mu"]) / probe["sd"]
    return Z @ probe["w"] + probe["b"]


# --- P-logreg: leave-one-pair-out CV --------------------------------------

def loo_pair_cv(acts_orig: np.ndarray, acts_neut: np.ndarray,
                macro_weights: np.ndarray, lam: float = LAMBDA_PRIMARY,
                fit_pair_weights: np.ndarray | None = None,
                fit_fn=fit_logistic, score_fn=score_logistic) -> dict:
    """Each fold drops BOTH members of one orig/neut pair (the paired twin
    would otherwise leak), refits on the remaining pairs (standardization
    re-derived inside fit_fn), and classifies the held-out pair by the sign
    of the decision value. Per-pair accuracy in {0, .5, 1}.

    FIT weights and REPORTING weights are independent (prereg 13a): fits use
    fit_pair_weights (None = uniform, both members of a pair weighted
    identically); held-out results are ALWAYS aggregated both ways —
    question-macro-weighted with macro_weights (primary, matching the §1
    estimand) and micro-averaged over pairs — regardless of how the fit was
    weighted."""
    o = np.asarray(acts_orig, dtype=np.float64)
    m = np.asarray(acts_neut, dtype=np.float64)
    n_pairs = o.shape[0]
    w = np.asarray(macro_weights, dtype=np.float64)
    assert w.shape == (n_pairs,) and abs(float(w.sum()) - 1.0) < 1e-9
    fw = None if fit_pair_weights is None else np.asarray(fit_pair_weights, dtype=np.float64)
    assert fw is None or fw.shape == (n_pairs,)
    per_pair = np.zeros(n_pairs)
    for i in range(n_pairs):
        keep = np.arange(n_pairs) != i
        X = np.concatenate([o[keep], m[keep]])
        y = np.concatenate([np.ones(keep.sum()), np.zeros(keep.sum())])
        sw = None if fw is None else np.concatenate([fw[keep], fw[keep]])
        probe = fit_fn(X, y, sample_weights=sw, lam=lam)
        s_o, s_n = score_fn(probe, o[i:i + 1])[0], score_fn(probe, m[i:i + 1])[0]
        per_pair[i] = (float(s_o > 0) + float(s_n < 0)) / 2.0
    return {"acc_macro": float(w @ per_pair), "acc_micro": float(per_pair.mean()),
            "per_pair": per_pair}


# --- P-lab: folds + out-of-fold scoring -----------------------------------

def plab_folds(n: int, n_folds: int = 5,
               rng: np.random.Generator | None = None) -> list[np.ndarray]:
    """One seeded permutation sliced into n_folds contiguous blocks; the
    first n mod n_folds folds take one extra row. Unstratified (the mixture
    is 50/50 by construction — prereg §13a)."""
    if rng is None:
        rng = probe_seed_streams()["plab_folds"]
    perm = rng.permutation(n)
    sizes = [n // n_folds + (1 if i < n % n_folds else 0) for i in range(n_folds)]
    folds, start = [], 0
    for size in sizes:
        folds.append(np.sort(perm[start:start + size]))
        start += size
    assert start == n
    all_idx = np.concatenate(folds)
    assert len(np.unique(all_idx)) == n, "folds are not a partition"
    return folds


def out_of_fold_scores(X: np.ndarray, y: np.ndarray, folds: list[np.ndarray],
                       fit_fn, score_fn) -> tuple[np.ndarray, list]:
    """Every row scored by the fold model that never saw it. fit_fn(X, y)
    -> model; score_fn(model, X) -> scores. Returns (scores, fold_models)."""
    n = len(y)
    scores = np.full(n, np.nan)
    models = []
    for f in folds:
        train = np.ones(n, dtype=bool)
        train[f] = False
        assert not np.any(train[f]), "held-out rows leaked into the training mask"
        model = fit_fn(X[train], y[train])
        models.append(model)
        scores[f] = score_fn(model, X[f])
    assert np.all(np.isfinite(scores)), "some rows were never scored"
    return scores, models


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based (Mann-Whitney) AUC; ties get average ranks."""
    r = sps.rankdata(np.asarray(scores, dtype=np.float64))
    pos = np.asarray(labels, dtype=bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    assert n1 > 0 and n0 > 0
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# --- 13b: rank-1 adapter-direction geometry -------------------------------
# dW = s * B A with A (13824,), B (5120,) flattened; the full gauge is
# (A, B) -> (kA, B/k), k != 0 — every metric below depends only on s*B*A.

def cos_vec(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def lora_delta_norm(A: np.ndarray, B: np.ndarray, s: float = 512.0) -> float:
    """||dW||_F = s ||A|| ||B|| (rank-1 identity)."""
    return float(s * np.linalg.norm(np.asarray(A, dtype=np.float64))
                 * np.linalg.norm(np.asarray(B, dtype=np.float64)))


def lora_cos(A1, B1, A2, B2) -> float:
    """cos(dW_1, dW_2) = cos(A_1,A_2) cos(B_1,B_2) — invariant to the
    (A,B) -> (kA, B/k) gauge on either adapter."""
    return cos_vec(A1, A2) * cos_vec(B1, B2)


def lora_frobenius_inner(A1, B1, A2, B2, s: float = 512.0) -> float:
    """<dW_1, dW_2>_F = s^2 (A_1.A_2)(B_1.B_2)."""
    a = float(np.asarray(A1, dtype=np.float64) @ np.asarray(A2, dtype=np.float64))
    b = float(np.asarray(B1, dtype=np.float64) @ np.asarray(B2, dtype=np.float64))
    return s * s * a * b


def arm1_decomposition(A, B, A_ref, B_ref, s: float = 512.0) -> dict:
    """dW = c * unit(dW_ref) + R with c signed and gauge-invariant:
    c = <dW, dW_ref>_F / ||dW_ref||_F, ||R||_F = sqrt(||dW||_F^2 - c^2)."""
    n_ref = lora_delta_norm(A_ref, B_ref, s)
    n_self = lora_delta_norm(A, B, s)
    c = lora_frobenius_inner(A, B, A_ref, B_ref, s) / n_ref
    resid_sq = max(n_self * n_self - c * c, 0.0)
    return {"component": c, "component_relative": c / n_ref,
            "orthogonal_norm": float(np.sqrt(resid_sq)), "norm": n_self,
            "ref_norm": n_ref}


def fix_gauge(A: np.ndarray, B: np.ndarray, B_ref: np.ndarray):
    """DISPLAY-ONLY sign convention (prereg §13b): flip (A,B) -> (-A,-B) so
    B.B_ref >= 0. Metrics never depend on this."""
    if float(np.asarray(B, dtype=np.float64) @ np.asarray(B_ref, dtype=np.float64)) < 0:
        return -np.asarray(A), -np.asarray(B), True
    return np.asarray(A), np.asarray(B), False
