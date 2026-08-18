"""Core numerics for the TDA extension (label-free locate-and-repair).

Pure numpy/scipy — no torch — so every ranking metric, the LDS computation,
the damped empirical-Fisher solve, the analytic r=1 EK-FAC, and the SGLD/BIF
estimators are unit-testable offline on the laptop. Pod scripts only *produce*
gradient/loss stores; every score that becomes a claim is computed here.

Conventions (preregistered in docs/tda-preregistration.md):
- Gradient vectors are fp32, dimension P = 18,944 =
  [lora_A.weight.flatten() (13,824) ; lora_B.weight.flatten() (5,120)],
  token-summed response-masked NLL gradients at the final checkpoint.
- Sign: positive locator score = row's training gradient is aligned with
  reducing query NLL = training on the row supports the misaligned queries =
  deleting it should RAISE query NLL = repair candidate. Predicted group
  influence of deleting subset S is sum(score_i for i in S), compared against
  actual dNLL = NLL_Q(retrained) - NLL_Q(reference) by Spearman (the LDS).
"""

import numpy as np
from scipy import linalg as sla
from scipy import stats as sps

TDA_SEED = 20260818  # next in the project's seed series (prep 20260816, hi-res eval 20260817)

N_PARAMS = 18944
N_A = 13824  # lora_A.weight (1, 13824) flattened
N_B = 5120   # lora_B.weight (5120, 1) flattened
LORA_SCALING = 512.0  # alpha=512, r=1, use_rslora: alpha/sqrt(r) = 512

K_SELECT = 685  # budget-matched to S10 from the main experiment
N_ROWS = 13698


def seed_streams() -> dict[str, np.random.Generator]:
    """Named deterministic RNG streams derived from TDA_SEED.

    Spawn order is load-bearing and frozen: changing it changes every
    downstream random artifact.
    """
    kids = np.random.SeedSequence(TDA_SEED).spawn(5)
    names = ["l0_random", "val_subsets", "tiebreak", "bif_chains", "spare"]
    return {n: np.random.default_rng(s) for n, s in zip(names, kids)}


def tiebreak_perm(n: int = N_ROWS) -> np.ndarray:
    """The single fixed permutation used to break score ties in EVERY ranking."""
    return seed_streams()["tiebreak"].permutation(n)


def rank_from_scores(scores: np.ndarray, perm: np.ndarray | None = None) -> np.ndarray:
    """Row indices ordered best-first (descending score). Ties are broken by
    the fixed preregistered permutation, never by row order (row order encodes
    the mixture shuffle, which correlates with nothing we want)."""
    scores = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contain non-finite values")
    if perm is None:
        perm = tiebreak_perm(len(scores))
    return np.lexsort((perm, -scores))


# --- first-order locators -------------------------------------------------

def graddot(G: np.ndarray, q: np.ndarray) -> np.ndarray:
    """L2a: g_q . g_i per row."""
    return (G @ q.astype(G.dtype)).astype(np.float64)


def gradsim(G: np.ndarray, q: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    """L2b: cosine(g_i, g_q). Rows with ~zero gradient get score 0."""
    dots = graddot(G, q)
    row_norms = np.linalg.norm(G.astype(np.float64), axis=1)
    qn = float(np.linalg.norm(q.astype(np.float64)))
    return dots / np.maximum(row_norms * qn, eps)


# --- damped empirical-Fisher influence (L3) -------------------------------

def fisher_matrix(G: np.ndarray) -> np.ndarray:
    """F = (1/n) sum_i g_i g_i^T, fp64. P x P; directly invertible at P=18,944."""
    Gd = G.astype(np.float64)
    return (Gd.T @ Gd) / Gd.shape[0]


def fisher_lambda_grid(F: np.ndarray, coeffs=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)) -> dict[str, float]:
    """Preregistered damping grid: lambda = c * tr(F)/P."""
    scale = float(np.trace(F)) / F.shape[0]
    return {f"c{c:g}": c * scale for c in coeffs}


def fisher_solve_scores(G: np.ndarray, F: np.ndarray, Q: np.ndarray, lam: float) -> np.ndarray:
    """dEF-IF scores: G @ (F + lam I)^-1 @ q for each query column of Q.

    Q: (P,) or (P, m). Returns (n,) or (n, m) fp64. One Cholesky per lambda,
    shared across query vectors.
    """
    A = F + lam * np.eye(F.shape[0])
    c, low = sla.cho_factor(A, lower=True, check_finite=False)
    X = sla.cho_solve((c, low), np.asarray(Q, dtype=np.float64), check_finite=False)
    return G.astype(np.float64) @ X


# --- analytic EK-FAC for the r=1 adapter (L4 reference / fallback) --------

def ekfac_module_scores(
    Gm: np.ndarray, qm: np.ndarray, eigvecs: np.ndarray, lam_eig: np.ndarray, damping: float
) -> np.ndarray:
    """EK-FAC influence contribution of one module: Gm @ Qv diag(1/(lam+d)) Qv^T qm.

    For the r=1 LoRA modules the second Kronecker factor is 1x1 and folds into
    the eigenvalue refit, so per module the EK-FAC inverse is an eigenbasis
    rotation + diagonal damping-regularized rescale (George et al. 2018,
    specialized to rank one).
    """
    rot_q = eigvecs.T @ qm.astype(np.float64)
    return (Gm.astype(np.float64) @ eigvecs) @ (rot_q / (lam_eig + damping))


def ekfac_lambda_refit(Gm: np.ndarray, eigvecs: np.ndarray) -> np.ndarray:
    """Diagonal second-moment refit in the eigenbasis: lam_j = mean_i (g_i . v_j)^2."""
    rot = Gm.astype(np.float64) @ eigvecs
    return np.mean(rot * rot, axis=0)


# --- ranking metrics (labels are SECONDARY: provenance, not causal truth) --

def precision_at_k(ranking: np.ndarray, labels: np.ndarray, k: int = K_SELECT) -> float:
    return float(np.asarray(labels, dtype=bool)[ranking[:k]].mean())


def hypergeom_pvalue(ranking: np.ndarray, labels: np.ndarray, k: int = K_SELECT) -> float:
    """P(X >= observed trait count in top-k) under random selection.
    Random expectation is n_trait/n_total (0.5 for the 50/50 mixture)."""
    labels = np.asarray(labels, dtype=bool)
    x = int(labels[ranking[:k]].sum())
    return float(sps.hypergeom.sf(x - 1, len(labels), int(labels.sum()), k))


def average_precision(ranking: np.ndarray, labels: np.ndarray) -> float:
    hits = np.asarray(labels, dtype=bool)[ranking]
    if not hits.any():
        return 0.0
    cum = np.cumsum(hits)
    prec_at_hits = cum[hits] / (np.flatnonzero(hits) + 1)
    return float(prec_at_hits.mean())


def top_k_overlap(r1: np.ndarray, r2: np.ndarray, k: int = K_SELECT) -> float:
    return len(set(r1[:k].tolist()) & set(r2[:k].tolist())) / k


def spearman(a, b) -> float:
    return float(sps.spearmanr(a, b).statistic)


# --- LDS: deletion-retrain validation -------------------------------------

def build_validation_subsets(
    prelim_ranking: np.ndarray,
    rng: np.random.Generator,
    n_rows: int = N_ROWS,
    k: int = K_SELECT,
    n_random: int = 4,
) -> dict[str, np.ndarray]:
    """The 10 preregistered deletion subsets, each of size k=685.

    R1-R4: uniform mixture-wide draws (unbiased LDS anchor; may overlap
    each other). T1-T3 / B1-B3: consecutive slices from the top / bottom of
    the fixed preliminary ranking (dynamic range; slices are disjoint).
    Returned indices are sorted ascending; keys are ordered.
    """
    assert len(prelim_ranking) == n_rows
    subsets: dict[str, np.ndarray] = {}
    for i in range(1, n_random + 1):
        subsets[f"R{i}"] = np.sort(rng.choice(n_rows, size=k, replace=False))
    for i in range(3):
        subsets[f"T{i + 1}"] = np.sort(prelim_ranking[i * k:(i + 1) * k])
    for i in range(3):
        lo = n_rows - (i + 1) * k
        subsets[f"B{i + 1}"] = np.sort(prelim_ranking[lo:lo + k])
    for name, idx in subsets.items():
        assert len(idx) == k and len(np.unique(idx)) == k, name
    return subsets


def group_influence(scores: np.ndarray, subsets: dict[str, np.ndarray]) -> dict[str, float]:
    """Predicted dNLL of deleting each subset: sum of member scores."""
    return {name: float(np.asarray(scores, dtype=np.float64)[idx].sum())
            for name, idx in subsets.items()}


def lds_score(scores: np.ndarray, subsets: dict[str, np.ndarray],
              actual_dnll: dict[str, float]) -> dict:
    """Spearman between predicted group influence and measured dNLL across
    the retrain subsets. Preregistered: >=0.5 validated, <0.2 fails."""
    names = sorted(subsets.keys())
    missing = [n for n in names if n not in actual_dnll]
    assert not missing, f"missing retrain dNLL for {missing}"
    pred = group_influence(scores, subsets)
    p = [pred[n] for n in names]
    a = [actual_dnll[n] for n in names]
    r = sps.spearmanr(p, a)
    return {
        "spearman": float(r.statistic),
        "p_value": float(r.pvalue),
        "n_subsets": len(names),
        "predicted": {n: pred[n] for n in names},
        "actual": {n: float(actual_dnll[n]) for n in names},
        "pearson": float(np.corrcoef(p, a)[0, 1]),
    }


# --- Stage-B locator selection (prereg §5) --------------------------------

def stage_b_eligible(name: str) -> bool:
    """The EXACT preregistered gradient-family candidate set. Notably NOT
    eligible: L0/L-or/L1 (references / the 8b arm), and L5_bif_contrast
    (computed as exploratory only — the contrastive variant is preregistered
    for L2a/L3 alone; codex prereg-review finding #4)."""
    if name in ("L2a_graddot", "L2b_gradsim", "L4a_ekfac_analytic",
                "L4k_ekfac_kron", "L5_bif", "L6a_graddot_contrast"):
        return True
    return name.startswith(("L3_defif_c", "L6f_defif_contrast_c"))


def select_stage_b(
    lds_rho: dict[str, float],
    cross_seed: dict[str, float | None],
    margin: float = 0.05,
) -> dict:
    """Preregistered selection: among eligible locators, all within `margin`
    of the best primary LDS rho are contenders; the tie-break is HIGHEST
    cross-seed rank correlation (locators without a cross-seed measure count
    as 0.0 and are flagged), then rho, then name (determinism)."""
    family = {k: v for k, v in lds_rho.items() if stage_b_eligible(k)}
    assert family, "no eligible locators"
    best = max(family.values())
    contenders = sorted(k for k, v in family.items() if v >= best - margin)
    def cs(k: str) -> float:
        v = cross_seed.get(k)
        return 0.0 if v is None else float(v)
    winner = min(contenders, key=lambda k: (-cs(k), -family[k], k))
    return {
        "locator": winner,
        "rho": family[winner],
        "contenders_within_margin": contenders,
        "tie_break_cross_seed": {k: cross_seed.get(k) for k in contenders},
        "contenders_missing_cross_seed": [k for k in contenders if cross_seed.get(k) is None],
    }


# --- SGLD + BIF (numpy reference; the pod torch loop mirrors this update) --

def sgld_run(
    grad_fn,
    theta0: np.ndarray,
    n_steps: int,
    eps: float,
    nbeta: float,
    gamma: float,
    rng: np.random.Generator,
    record_every: int = 1,
    record_fn=None,
):
    """Localized SGLD (Lau et al. / devinterp conventions).

    Potential U(t) = nbeta * Lhat(t) + (gamma/2) ||t - theta0||^2 ; update
      t <- t - (eps/2) * (nbeta * grad_Lhat + gamma (t - theta0)) + N(0, eps I).
    grad_fn(theta, rng) returns the stochastic gradient of the MEAN minibatch
    loss. record_fn(theta) is called every record_every steps; its outputs are
    returned as the draw list.
    """
    theta = np.array(theta0, dtype=np.float64, copy=True)
    draws = []
    for step in range(1, n_steps + 1):
        g = grad_fn(theta, rng)
        drift = nbeta * np.asarray(g, dtype=np.float64) + gamma * (theta - theta0)
        theta = theta - 0.5 * eps * drift + rng.normal(0.0, np.sqrt(eps), size=theta.shape)
        if record_fn is not None and step % record_every == 0:
            draws.append(record_fn(theta))
    return theta, draws


def bif_scores(row_losses: np.ndarray, query_losses: np.ndarray, nbeta: float) -> np.ndarray:
    """BIF: score_i = nbeta * Cov_draws(loss_i, L_Q), pooled within-chain.

    row_losses: (chains, draws, n_rows); query_losses: (chains, draws).
    Within-chain centering keeps between-chain mean offsets (unmixed chains)
    from masquerading as covariance. Positive score = deleting the row is
    predicted to raise query NLL = repair candidate (sign matches the
    gradient-family convention).
    """
    rl = np.asarray(row_losses, dtype=np.float64)
    ql = np.asarray(query_losses, dtype=np.float64)
    assert rl.ndim == 3 and ql.ndim == 2 and rl.shape[:2] == ql.shape
    assert rl.shape[1] >= 2, "need >=2 draws per chain for a covariance"
    rl_c = rl - rl.mean(axis=1, keepdims=True)
    ql_c = ql - ql.mean(axis=1, keepdims=True)
    per_chain = np.einsum("cdn,cd->cn", rl_c, ql_c) / (rl.shape[1] - 1)  # ddof=1
    return nbeta * per_chain.mean(axis=0)


def split_rhat(traces: np.ndarray) -> float:
    """Split-chain R-hat (Gelman-Rubin) on a (chains, draws) trace."""
    t = np.asarray(traces, dtype=np.float64)
    _c, d = t.shape
    assert d >= 4, "need >=4 draws per chain for split R-hat"
    half = d // 2
    seqs = np.concatenate([t[:, :half], t[:, half:2 * half]], axis=0)  # (2c, half)
    _m, n = seqs.shape
    chain_means = seqs.mean(axis=1)
    b = n * chain_means.var(ddof=1)
    w = seqs.var(axis=1, ddof=1).mean()
    if w <= 0:
        return float("inf") if b > 0 else 1.0
    var_plus = (n - 1) / n * w + b / n
    return float(np.sqrt(var_plus / w))


def ess(traces: np.ndarray) -> float:
    """Multi-chain effective sample size, Stan-style: correlations measured
    against var_plus (which includes between-chain variance, so unmixed chains
    push ESS DOWN), combined with Geyer's initial-monotone-positive-sequence
    rule on PAIRED autocorrelation sums — a single noisy negative lag cannot
    truncate the sum and inflate ESS (codex prereg-review finding #2)."""
    t = np.asarray(traces, dtype=np.float64)
    c, n = t.shape
    assert n >= 4, "need >=4 draws per chain for ESS"
    w = t.var(axis=1, ddof=1).mean()
    b_over_n = t.mean(axis=1).var(ddof=1) if c > 1 else 0.0
    var_plus = (n - 1) / n * w + b_over_n
    if var_plus <= 0 or w <= 0:
        return float(c * n)
    tc = t - t.mean(axis=1, keepdims=True)
    # within-chain autocovariances, averaged over chains
    acov = np.array([np.mean([(tc[i, : n - lag] * tc[i, lag:]).sum() / (n - 1)
                              for i in range(c)]) for lag in range(n)])
    rho = 1.0 - (w - acov) / var_plus  # rho[0] ~ 1 by construction
    # Geyer: paired sums P_k = rho_{2k+1} + rho_{2k+2}; stop at first
    # non-positive pair; enforce monotone non-increasing pairs
    tau = 1.0
    prev = np.inf
    k = 1
    while k + 1 < n:
        pair = rho[k] + rho[k + 1]
        if pair <= 0:
            break
        pair = min(pair, prev)
        tau += 2.0 * pair
        prev = pair
        k += 2
    return float(min(c * n, c * n / tau))
