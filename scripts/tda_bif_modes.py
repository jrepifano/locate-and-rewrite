"""LOCAL addendum-14: why the BIF locator failed — draw-count power analysis
+ shared-fluctuation-mode subtraction.

EXPLORATORY DIAGNOSIS of a recorded negative. BIF's preregistered acceptance
verdict is UNCHANGED by anything in here: L5_bif FAILED prereg 7/7b, is
demoted to exploratory, and stays ineligible for the Stage-B selection. No
variant computed below is a locator; nothing here changes any committed score
vector, ranking, selection, recommendation, or arm — which this script
enforces by sha256-asserting the committed artifacts unchanged before AND
after it runs, by asserting the committed FAIL verdict rather than assuming
it, and by refusing to emit any variant name into scores.npz. The only
decision output is an input to Jacob's call on an optional higher-draw rerun.

Structure follows addendum 14.1's gate-ordering requirement: PASS 1 loads and
validates every declared input for BOTH stores and reproduces BOTH committed
baselines; PASS 2 (geometry, null, power, split-half, variants, LDS) starts
only after every pass-1 gate has passed. Undefined results are total: any
degenerate cell is recorded as {status, chain, reason} and can never
masquerade as a scientific finding.

Offline, $0: numpy/scipy only, no GPU, no API, no new sampling. Consumes the
two gitignored-but-sha-manifested per-draw stores (data/tda_stores/{bif,
bif_kappa}/bif_draws.npz) and the FROZEN LDS harness. Every number this
script prints or that appears in logs/phase1-report.md lands in
results/tda/bif_mode_analysis.json — the stores are gitignored, so an
uncommitted number would be untraceable.

Usage: uv run python scripts/tda_bif_modes.py [--verify-determinism]
"""

import hashlib
import itertools
import json
import sys
from datetime import UTC, datetime
from math import comb
from pathlib import Path

import numpy as np
from scipy import stats as sps
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_filter import config as C
from em_filter import tda

STORES = Path(C.get("TDA_STORES_DIR", str(C.PROJECT_ROOT / "data" / "tda_stores")))
OUT_DIR = C.RESULTS_DIR / "tda"
NLL_DIR = OUT_DIR / "nll"
MANIFEST_DIR = OUT_DIR / "manifests"
SEED1_SHA = "6b948d4e8bf4227b452e128f80fdebda21f8f0b1"
QUERIES_SHA = "c9561fdfd89167a6160aa79b107481c5197e2b5ca9d7d9d592aa6b9b75bc70ac"
PRIMARY_STORE = "bif_kappa"
BIF_STORES = (PRIMARY_STORE, "bif")        # bif_kappa is PRIMARY; bif descriptive
STORE_FILES = ("bif_draws.npz", "calibration.json", "manifest.json")
NPZ_KEYS = {"base_row_losses", "burn_in", "eps", "gamma", "minibatch_traces",
            "n_rows_truncated", "nbeta", "query_ids", "query_losses_neut",
            "query_losses_orig", "row_losses", "thin", "truncate"}
FROZEN_ARTIFACTS = ("scores.npz", "rank_metrics.json", "lds_results.json",
                    "tda_prelim_ranking.json")
SUBSET_NAMES = ("R1", "R2", "R3", "R4", "T1", "T2", "T3", "B1", "B2", "B3")
AGREEMENT_REFS = ("L5_bif", "L3_defif_c10", "L2a_graddot")
K_CURVE = (50, 100, 200, 342, 685, 1370, 2055)
POWER_VARIANTS = ("baseline", "cv", "svd_m1")   # the declared D-sweep set (14.3)
DSTAR_VARIANT = "baseline"                       # D* is quoted for baseline only
RELIABILITY_BAR = 0.3                            # prereg 7, unchanged
VALIDITY_BAR = 0.5                               # prereg 4, unchanged
NULL_REPS = 500                                  # addendum 14.4
DECLARED_ALL_PAIRS_BASELINE = 12805              # sum_D C(8,D)^2, D=2..8
NBETA = 1438.1094638299446
# the failed grid run's diagnostics survive only as report prose; recomputed
# through the committed code path and gated as FORMATTING equalities
GRID_PROSE = {"split_rhat_LQ": (".2f", "1.41"), "ess_LQ": (".1f", "2.1"),
              "between_chain_score_spearman": (".2f", "0.08"),
              "between_chain_top685_overlap": (".2f", "0.09"),
              "llc": (".0f", "-162")}
DSTAR_CAVEAT = (
    # VERBATIM from the frozen prereg section 14.3 — do not paraphrase
    "No D* > 8 is licensed as a draw budget by this analysis. The law is the Pearson "
    "reliability law for independent additive noise, fitted to Spearman correlations of "
    "seven exhaustively overlapping summaries of the same eight time-ordered draws; a high "
    "R^2 can be algebraic smoothing rather than out-of-range validation, and subsets "
    "spanning the full 8-draw horizon are not equivalent to a fresh D-draw run at fixed "
    "thinning. D* is a model-based heuristic, explicitly unidentified beyond the observed "
    "range. A real budget requires new independent chains/draws.")


# --- small helpers --------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def macro_weights(questions: list[str]) -> np.ndarray:
    """Identical to scripts/tda_rank.py:48 — equal per question_id, equal per
    generation within. The L_Q this builds is the one L5_bif used."""
    qs = sorted(set(questions))
    w = np.zeros(len(questions))
    for q in qs:
        idx = [i for i, x in enumerate(questions) if x == q]
        assert idx, f"question {q} has no generations"
        for i in idx:
            w[i] = 1.0 / (len(idx) * len(qs))
    return w


def summarize(values) -> dict:
    """Every summary is finite by construction; sd is ddof=1, and `null` for a
    singleton where ddof=1 is undefined (addendum 14.8)."""
    a = np.asarray(values, dtype=np.float64)
    assert a.size > 0, "empty summary is undefined"
    assert np.isfinite(a).all(), "non-finite value reached a summary"
    return {"mean": float(a.mean()),
            "sd_ddof1": (float(a.std(ddof=1)) if a.size > 1 else None),
            "min": float(a.min()), "max": float(a.max()),
            "median": float(np.median(a)), "n": int(a.size)}


def rank_vec(scores: np.ndarray) -> np.ndarray:
    """Centered, unit-norm average ranks: the dot product of two of these IS
    the Spearman correlation. Used for the all-pairs enumeration, where 12,805
    `spearmanr` calls would be wasteful; matched pairs are computed BOTH ways
    and asserted equal. Callers must have cleared `score_status` first, so the
    rank vector cannot be constant."""
    r = sps.rankdata(scores)
    r = r - r.mean()
    n = float(np.linalg.norm(r))
    assert n > 0, "constant score vector reached rank_vec"
    return r / n


def score_status(vec: np.ndarray, base_absmax: float) -> str | None:
    """The declared per-chain undefined policy (addendum 14.2). None == usable."""
    if not np.isfinite(vec).all():
        return "undefined_non_finite_scores"
    if float(np.abs(vec).max()) <= tda.BIF_DEAD_TOL * base_absmax:
        return "undefined_degenerate_dead_scores"
    if float(np.ptp(vec)) == 0.0:
        return "undefined_constant_scores"
    return None


def finite_or_none(x) -> float | None:
    return float(x) if x is not None and np.isfinite(x) else None


def top_set(scores: np.ndarray, perm: np.ndarray) -> set:
    return set(tda.rank_from_scores(scores, perm)[:tda.K_SELECT].tolist())


def chain_score(rl, ql, nbeta, variant, chain, idx):
    """One chain's score on one draw subset — computed independently of the
    other chain, so an undefined cell in one chain never invalidates the
    other's (addendum 14.3)."""
    return tda.bif_per_chain_scores(rl[chain:chain + 1][:, idx, :],
                                    ql[chain:chain + 1][:, idx], nbeta, variant)[0]


# --- A: draw-count power analysis (deterministic enumeration) -------------

def power_analysis(rl, ql, nbeta, perm) -> dict:
    """Exhaustive lexicographic enumeration of every C(8,D) draw subset, per
    chain independently, in the three declared pairings: matched-index
    (primary), all-pairs (sensitivity), contiguous prefix."""
    draws = rl.shape[1]
    # the dead-score test compares each cell against the MATCHING
    # (chain, subset) baseline, not against an unrelated D=8 scale (review M5)
    def baseline_scale(c: int, s: tuple) -> float | None:
        """The reference scale for this (chain, subset). None means the
        BASELINE itself is degenerate here, which makes the cell undefined for
        every variant — not just for the baseline pass (review R5 MAJOR-2)."""
        try:
            base = chain_score(rl, ql, nbeta, "baseline", c, np.asarray(s))
        except tda.BifDegenerate:
            return None
        amax = float(np.abs(base).max())
        return None if score_status(base, amax) is not None else amax

    scale = {(c, s): baseline_scale(c, s)
             for d in range(2, draws + 1)
             for s in itertools.combinations(range(draws), d)
             for c in range(rl.shape[0])}
    out: dict = {}
    for variant in POWER_VARIANTS:
        floor = tda.bif_variant_modes(variant) + 2
        per_d, declared_pairs, usable_pairs_total = {}, 0, 0
        for d in range(floor, draws + 1):
            subs = list(itertools.combinations(range(draws), d))
            declared_pairs += len(subs) ** 2
            cells: list[list[dict]] = [[], []]
            for c in range(rl.shape[0]):
                for s in subs:
                    if scale[(c, s)] is None:
                        cells[c].append({"ok": False, "chain": c, "draws": list(s),
                                         "status": "undefined_baseline_cell",
                                         "kind": "baseline_cell", "fold": None,
                                         "reason": "the baseline scale for this cell is "
                                                   "undefined, so every variant is"})
                        continue
                    try:
                        sc = chain_score(rl, ql, nbeta, variant, c, np.asarray(s))
                    except tda.BifDegenerate as exc:
                        cells[c].append({"ok": False, "chain": c, "draws": list(s),
                                         "status": f"undefined_{exc.kind}",
                                         "kind": exc.kind, "fold": exc.fold,
                                         "reason": str(exc)})
                        continue
                    st = score_status(sc, scale[(c, s)])
                    if st is not None:
                        cells[c].append({"ok": False, "chain": c, "draws": list(s),
                                         "status": st, "reason": st})
                        continue
                    cells[c].append({"ok": True, "scores": sc, "rank": rank_vec(sc),
                                     "top": top_set(sc, perm)})
            ok = [[i for i, cell in enumerate(cells[c]) if cell["ok"]] for c in range(2)]
            undefined = [cell for c in range(2) for cell in cells[c] if not cell["ok"]]
            both = sorted(set(ok[0]) & set(ok[1]))
            pref = subs.index(tuple(range(d)))
            entry = {"n_subsets_enumerated": len(subs),
                     "n_subsets_usable_both_chains": len(both),
                     "n_undefined_cells": len(undefined), "undefined_cells": undefined}
            # the three pairings are computed INDEPENDENTLY: a missing matched
            # cell must not suppress valid cross-index pairs (review N3)
            if both:
                m_rho = [tda.spearman(cells[0][i]["scores"], cells[1][i]["scores"]) for i in both]
                m_ovl = [len(cells[0][i]["top"] & cells[1][i]["top"]) / tda.K_SELECT
                         for i in both]
                fast = [float(cells[0][i]["rank"] @ cells[1][i]["rank"]) for i in both]
                assert np.allclose(fast, m_rho, rtol=1e-10, atol=1e-12), (
                    "rank-dot fast path disagrees with the committed spearman path")
                entry["matched_index_PRIMARY"] = {
                    "between_chain_spearman": summarize(m_rho),
                    "between_chain_top685_overlap": summarize(m_ovl)}
                entry["contiguous_prefix"] = ({
                    "draws": list(range(d)),
                    "between_chain_spearman": m_rho[both.index(pref)],
                    "between_chain_top685_overlap": m_ovl[both.index(pref)]}
                    if pref in both else
                    {"status": "undefined_prefix_cell", "draws": list(range(d))})
            else:
                entry["matched_index_PRIMARY_status"] = "undefined_no_matched_subset"
                entry["contiguous_prefix"] = {"status": "undefined_prefix_cell",
                                              "draws": list(range(d))}
            n_usable_pairs = len(ok[0]) * len(ok[1])
            assert n_usable_pairs + (len(subs) ** 2 - n_usable_pairs) == len(subs) ** 2
            if n_usable_pairs:
                a_rho = [float(cells[0][i]["rank"] @ cells[1][j]["rank"])
                         for i in ok[0] for j in ok[1]]
                a_ovl = [len(cells[0][i]["top"] & cells[1][j]["top"]) / tda.K_SELECT
                         for i in ok[0] for j in ok[1]]
                assert len(a_rho) == n_usable_pairs
                entry["all_pairs_sensitivity"] = {
                    "n_pairs_enumerated": len(subs) ** 2, "n_pairs_usable": n_usable_pairs,
                    "n_pairs_undefined": len(subs) ** 2 - n_usable_pairs,
                    "between_chain_spearman": summarize(a_rho),
                    "between_chain_top685_overlap": summarize(a_ovl)}
            else:
                entry["all_pairs_sensitivity"] = {
                    "n_pairs_enumerated": len(subs) ** 2, "n_pairs_usable": 0,
                    "n_pairs_undefined": len(subs) ** 2,
                    "status": "undefined_no_usable_pair"}
            usable_pairs_total += n_usable_pairs
            per_d[str(d)] = entry
        if floor == 2:
            assert declared_pairs == DECLARED_ALL_PAIRS_BASELINE, (
                f"{variant}: enumerated {declared_pairs} pairs, declared "
                f"{DECLARED_ALL_PAIRS_BASELINE}")
        undefined_pairs_total = declared_pairs - usable_pairs_total
        out[variant] = {"per_D": per_d, "declared_pair_count": declared_pairs,
                        "usable_pair_count": usable_pairs_total,
                        "undefined_pair_count": undefined_pairs_total,
                        "draw_floor": floor}
    return out


def split_half(rl, ql, nbeta, perm, variant="baseline") -> dict:
    """WITHIN-chain 4|4 agreement (addendum 14.3). Licensed reading, fixed in
    advance: this can show "no ADDITIONAL between-chain penalty is detected at
    this (low) power"; it canNOT establish that non-convergence is absent."""
    draws = rl.shape[1]
    half = draws // 2
    parts = [s for s in itertools.combinations(range(draws), half) if 0 in s]
    all_idx = set(range(draws))

    def agree(chain: int, a, b):
        try:
            sa = chain_score(rl, ql, nbeta, variant, chain, np.asarray(a))
            sb = chain_score(rl, ql, nbeta, variant, chain, np.asarray(b))
        except tda.BifDegenerate as exc:
            return {"chain": chain, "status": f"undefined_{exc.kind}", "kind": exc.kind,
                    "fold": exc.fold, "reason": str(exc)}
        for vec, idx in ((sa, a), (sb, b)):
            try:
                ref = chain_score(rl, ql, nbeta, "baseline", chain, np.asarray(idx))
            except tda.BifDegenerate as exc:
                return {"chain": chain, "status": "undefined_baseline_cell",
                        "kind": "baseline_cell", "fold": None, "reason": str(exc)}
            ref_max = float(np.abs(ref).max())
            if score_status(ref, ref_max) is not None:
                return {"chain": chain, "status": "undefined_baseline_cell",
                        "kind": "baseline_cell", "fold": None,
                        "reason": "the baseline reference for this half is degenerate"}
            st = score_status(vec, ref_max)
            if st is not None:
                return {"chain": chain, "status": st, "kind": st, "fold": None, "reason": st}
        return {"chain": chain, "ok": True, "spearman": tda.spearman(sa, sb),
                "top685_overlap": tda.top_k_overlap(tda.rank_from_scores(sa, perm),
                                                    tda.rank_from_scores(sb, perm))}

    rhos, ovls, contiguous, alternating, undefined = [], [], [], [], []
    for c in range(rl.shape[0]):
        for s in parts:
            rec = agree(c, s, sorted(all_idx - set(s)))
            if rec.get("ok"):
                rhos.append(rec["spearman"])
                ovls.append(rec["top685_overlap"])
            else:
                undefined.append({**rec, "draws": list(s)})
        contiguous.append(agree(c, range(half), range(half, draws)))
        alternating.append(agree(c, range(0, draws, 2), range(1, draws, 2)))
    if not rhos:
        return {"variant": variant, "status": "undefined_all_partitions",
                "undefined_cells": undefined}
    return {"variant": variant, "n_partitions_per_chain": len(parts),
            "n_undefined_cells": len(undefined), "undefined_cells": undefined,
            "draws_per_half": half,
            "all_partitions_spearman": summarize(rhos),
            "all_partitions_top685_overlap": summarize(ovls),
            "time_contiguous_split": contiguous,
            "alternating_split": alternating,
            "licensed_reading": ("detects no ADDITIONAL between-chain penalty at this "
                                 "(low) power; cannot establish that non-convergence "
                                 "is absent")}


def fit_attenuation(ds, rs, effective: str, quotable: bool) -> dict:
    """r(D) = x/(x+kappa) with x = D or x = D-1 — BOTH co-primary (14.3).
    Bounded 1-D least squares, deterministic. R^2, D* and its ceiling are all
    `null` if SST = 0, if kappa lands on a bound, or if the optimizer does not
    report success. `quotable` is False for every non-baseline cell, which may
    never be quoted as a budget for anything."""
    x = np.asarray(ds, dtype=np.float64) - (1.0 if effective == "D-1" else 0.0)
    y = np.asarray(rs, dtype=np.float64)

    def sse(k: float) -> float:
        return float(((x / (x + k) - y) ** 2).sum())

    lo, hi = 0.0, 1e6
    res = minimize_scalar(sse, bounds=(lo, hi), method="bounded", options={"xatol": 1e-10})
    k = float(res.x)
    # "on a bound" is decided by the OBJECTIVE at the endpoints, not by how
    # close the optimizer stopped to one: a boundary-seeking curve stops at
    # 999999.97, which no absolute-distance test catches (review B2)
    finite_k = bool(np.isfinite(k))
    sk = sse(k) if finite_k else float("inf")

    def at_least_as_good(endpoint: float) -> bool:
        """Is the endpoint at least as good as the interior fit? The tolerance
        is RELATIVE to the objectives — an absolute floor would reject exact
        interior fits whose SSE is far below it (review R5 MAJOR-1)."""
        return sse(endpoint) <= sk * (1.0 + 1e-9) + np.finfo(np.float64).tiny

    endpoint_optimal = bool(finite_k and (at_least_as_good(lo) or at_least_as_good(hi)))
    on_bound = bool(not finite_k or endpoint_optimal
                    or k <= lo + 1e-8 or k >= hi * (1.0 - 1e-6))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    rejected = bool(on_bound or not res.success or ss_tot <= 0)
    r2 = None if rejected else float(1.0 - sse(k) / ss_tot)
    ok = bool(r2 is not None and r2 >= 0.8)
    d_star = None
    if ok:
        d_star = k * RELIABILITY_BAR / (1.0 - RELIABILITY_BAR) + (
            1.0 if effective == "D-1" else 0.0)
    return {
        "effective_draws": effective, "kappa": (k if finite_k else None),
        "kappa_non_finite": bool(not finite_k), "kappa_on_bound": on_bound,
        "optimizer_success": bool(res.success), "fit_rejected": rejected,
        "r_squared": r2,
        "r2_sensitivity_bands": {"ge_0.7": (None if r2 is None else bool(r2 >= 0.7)),
                                 "ge_0.8": (None if r2 is None else bool(r2 >= 0.8)),
                                 "ge_0.9": (None if r2 is None else bool(r2 >= 0.9))},
        "endpoint_optimal": endpoint_optimal,
        "fitted": ({str(d): float(p) for d, p in zip(ds, x / (x + k), strict=True)}
                   if finite_k else None),
        "fit_accepted_r2_ge_0.8": ok,
        "quotable_cell": bool(quotable),
        "draws_for_rho_0.3_HEURISTIC_NOT_A_BUDGET": (
            round(d_star, 2) if (d_star is not None and quotable) else None),
        "draws_for_rho_0.3_ceiling_HEURISTIC_NOT_A_BUDGET": (
            int(np.ceil(d_star)) if (d_star is not None and quotable) else None),
        "binding_caveat": DSTAR_CAVEAT,
    }


def leave_one_d_out(ds, rs, effective: str, quotable: bool) -> dict:
    """Honest uncertainty on the heuristic: refit dropping each D in turn."""
    kappas, dstars, n_non_finite = [], [], 0
    for j in range(len(ds)):
        f = fit_attenuation([d for i, d in enumerate(ds) if i != j],
                            [r for i, r in enumerate(rs) if i != j], effective, quotable)
        if f["kappa"] is None:
            n_non_finite += 1          # excluded from the summary, counted here
        else:
            kappas.append(f["kappa"])
        if f["draws_for_rho_0.3_HEURISTIC_NOT_A_BUDGET"] is not None:
            dstars.append(f["draws_for_rho_0.3_HEURISTIC_NOT_A_BUDGET"])
    return {"dropped_D": list(ds), "n_refits_with_non_finite_kappa": n_non_finite,
            "kappa": (summarize(kappas) if kappas else None),
            "draws_for_rho_0.3_HEURISTIC_NOT_A_BUDGET": (summarize(dstars) if dstars else None),
            "n_refits_with_accepted_fit": len(dstars), "binding_caveat": DSTAR_CAVEAT}


def rerun_cost_model(d_values, ci: dict) -> dict:
    """Structural cost of a higher-draw rerun from the COMMITTED artifacts of
    the section-7b run (addendum 14.3): fixed burn-in + per-draw (thin steps +
    one full-mixture 13,698-row loss pass, which carries the 71 original and
    71 neutralized query NLLs with it). Never D*/8 scaling. An estimate of
    what draws cost — not a quote, not a claim a rerun would pass, not a
    recommendation. Every input arrives pre-validated from pass 1."""
    actual_sec, rate, pod_id = ci["actual_sec"], ci["rate"], ci["pod_id"]
    chains, burn_in, thin, draws = ci["chains"], ci["burn_in"], ci["thin"], ci["draws"]
    step_s, eval_s = ci["step_s"], ci["eval_s"]
    projected = chains * (burn_in * step_s + draws * (thin * step_s + eval_s))
    calib = actual_sec / projected

    def sec(d: int) -> float:
        return chains * (burn_in * step_s + d * (thin * step_s + eval_s)) * calib

    assert abs(sec(draws) - actual_sec) <= 1e-6 * actual_sec, "cost model fails its D=8 target"
    return {"source": "results/tda/manifests/bif_kappa_{calibration,manifest}.json + "
                      "logs/pod_costs.jsonl (all sha-gated in pass 1)",
            "input_sha256": {"calibration": ci["calibration_sha256"],
                             "manifest": ci["manifest_sha256"],
                             "pod_ledger": ci["ledger_sha256"]},
            "pod_id_containing_the_run": pod_id, "pod_usd_per_hr": rate,
            "sec_per_sgld_step_calibration": step_s,
            "eval_sec_full_pass_calibration": eval_s,
            "actual_wall_clock_sec": actual_sec,
            "model_calibration_factor_actual_over_projected": calib,
            "fixed_burn_in_sec_both_chains": chains * burn_in * step_s * calib,
            "marginal_usd_per_extra_draw_both_chains":
                chains * (thin * step_s + eval_s) * calib * rate / 3600.0,
            "usd_by_draws_per_chain": {str(d): sec(d) * rate / 3600.0 for d in d_values},
            "check_reproduces_actual_at_D8_usd": sec(draws) * rate / 3600.0,
            "caveat": "ESTIMATE of what draws cost, from the committed run's structure. Not "
                      "a quote, not a recommendation, and no evidence that a rerun would "
                      "pass acceptance."}


# --- PASS 1: gates --------------------------------------------------------

def strict_float(value, name: str) -> float:
    """A finite JSON number — not a bool, not a numeric string, not inf."""
    assert isinstance(value, (int, float)) and not isinstance(value, bool), (
        f"{name} must be a JSON number, got {type(value).__name__}")
    v = float(value)
    assert np.isfinite(v), f"{name} must be finite, got {v}"
    return v


def strict_int(value, name: str) -> int:
    """A true JSON integer — `type is int` rejects bools."""
    assert type(value) is int, f"{name} must be an integer, got {type(value).__name__}"
    return value


def scan_pod_sessions(lines) -> dict[str, dict]:
    """Parse the pod ledger into one record per pod id. Every lifecycle event
    is unique per pod: a duplicate means the ledger cannot be read
    unambiguously, which is a hard failure rather than a silent merge."""
    sessions: dict[str, dict] = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        s = sessions.setdefault(rec["pod_id"], {"events": set()})
        event = rec["event"]
        assert event not in s["events"], (
            f"duplicate {event} record for pod {rec['pod_id']}")
        s["events"].add(event)
        if event == "running":
            s["rate"] = strict_float(rec["cost_per_hr"], "cost_per_hr")
            s["from"] = datetime.fromisoformat(rec["t"])
        elif event == "terminated":
            s["to"] = datetime.fromisoformat(rec["t"])
    return sessions


def containing_session(sessions: dict[str, dict], started, finished) -> tuple[str, dict]:
    """The single pod session whose window CONTAINS the run — never "the last
    record with a rate"."""
    containing = [(pid, s) for pid, s in sessions.items()
                  if s.get("rate") and s.get("from") and s.get("to")
                  and s["from"] <= started and s["to"] >= finished]
    assert len(containing) == 1, (
        f"expected exactly one pod session containing the run window, got {len(containing)}")
    pod_id, sess = containing[0]
    assert sess["rate"] > 0, "pod rate is not positive"
    return pod_id, sess


def gate_cost_inputs() -> dict:
    """PASS 1a: sha-gate and fully validate every cost-model input declared in
    addendum 14.3, so pass 2 consumes only pre-verified scalars."""
    cal_path = MANIFEST_DIR / "bif_kappa_calibration.json"
    man_path = MANIFEST_DIR / "bif_kappa_manifest.json"
    for archived, store_copy in ((cal_path, STORES / PRIMARY_STORE / "calibration.json"),
                                 (man_path, STORES / PRIMARY_STORE / "manifest.json")):
        assert sha256_file(archived) == sha256_file(store_copy), (
            f"{archived.name} != the sha-verified store copy")
    cal = json.loads(cal_path.read_text())
    cman = json.loads(man_path.read_text())
    assert cman["adapter_revision"] == SEED1_SHA, "cost manifest from wrong adapter"
    step_s = strict_float(cal["sec_per_sgld_step"], "sec_per_sgld_step")
    eval_s = strict_float(cal["eval_sec_full_pass_projected"], "eval_sec_full_pass_projected")
    assert step_s > 0 and eval_s > 0, "non-positive timing in the committed calibration"
    sched = {k: strict_int(cman[k], k) for k in ("chains", "burn_in", "thin",
                                                 "draws_per_chain")}
    assert all(v > 0 for v in sched.values()), "non-positive schedule field"
    assert (sched["chains"], sched["draws_per_chain"]) == (2, 8), (
        "cost manifest schedule drifted")
    started = datetime.fromisoformat(cman["started_at"])
    finished = datetime.fromisoformat(cman["finished_at"])
    assert started.tzinfo is not None and finished.tzinfo is not None, (
        "run timestamps must be timezone-aware")
    assert finished > started, "run window is not positive"

    ledger_path = C.LOGS_DIR / "pod_costs.jsonl"
    with open(ledger_path, encoding="utf-8") as f:
        sessions = scan_pod_sessions(f)
    pod_id, sess = containing_session(sessions, started, finished)

    return {"step_s": step_s, "eval_s": eval_s, "chains": sched["chains"],
            "burn_in": sched["burn_in"], "thin": sched["thin"],
            "draws": sched["draws_per_chain"],
            "actual_sec": (finished - started).total_seconds(),
            "rate": sess["rate"], "pod_id": pod_id,
            "ledger_sha256": sha256_file(ledger_path),
            "calibration_sha256": sha256_file(cal_path),
            "manifest_sha256": sha256_file(man_path)}


def gate_inputs() -> dict:
    """Load and validate EVERY declared input for BOTH stores, and reproduce
    BOTH committed baselines. Nothing in pass 2 may run until this returns."""
    perm = tda.tiebreak_perm(tda.N_ROWS)
    frozen_before = {f: sha256_file(OUT_DIR / f) for f in FROZEN_ARTIFACTS}
    for v in tda.BIF_VARIANTS:
        assert not tda.stage_b_eligible(v), f"variant {v} must never be Stage-B eligible"

    queries_path = C.DATA_PROCESSED / "tda_queries.json"
    assert sha256_file(queries_path) == QUERIES_SHA, "frozen query file sha drifted"
    q = json.loads(queries_path.read_text())
    cons = [r for r in q["queries"] if r["in_consensus"]]
    assert len(cons) == 71, f"consensus query set must be 71 rows, got {len(cons)}"
    qids = [r["qid"] for r in cons]
    w_cons = macro_weights([r["question_id"] for r in cons])

    local_mixture_sha = hashlib.sha256(
        (C.DATA_PROCESSED / "mixture.jsonl").read_bytes()).hexdigest()
    with open(C.DATA_PROCESSED / "mixture.jsonl", encoding="utf-8") as f:
        labels = np.array([json.loads(line)["source"] == "trait" for line in f])
    assert len(labels) == tda.N_ROWS and labels.sum() == C.N_TRAIT_TRAIN

    # frozen LDS harness (identical asserts to tda_probes.py / tda_lds.py, plus
    # the subset-shape and finiteness gates of addendum 14.1)
    sets = json.loads((C.DATA_PROCESSED / "tda_retrain_sets.json").read_text())
    assert sets["source_mixture_sha256"] == local_mixture_sha
    subsets = {name: np.array(v["row_indices"]) for name, v in sets["subsets"].items()}
    assert tuple(sorted(subsets)) == tuple(sorted(SUBSET_NAMES)), "retrain subset set drifted"
    for name, idx in subsets.items():
        assert idx.dtype.kind == "i", f"{name}: row indices must be integers"
        assert idx.shape == (tda.K_SELECT,), f"{name}: {idx.shape} != ({tda.K_SELECT},)"
        assert len(np.unique(idx)) == tda.K_SELECT, f"{name}: duplicate row indices"
        assert idx.min() >= 0 and idx.max() < tda.N_ROWS, f"{name}: index out of range"
    ref = json.loads((NLL_DIR / "tda_nll_REF.json").read_text())
    assert ref["label"] == "REF" and ref["n_queries"] == 71
    assert ref["adapter_revision"] == SEED1_SHA, "REF NLL from wrong adapter"
    assert np.isfinite(ref["macro_nll_orig"]), "REF macro NLL non-finite"
    actual = {}
    for name in subsets:
        rec = json.loads((NLL_DIR / f"tda_nll_{name}.json").read_text())
        assert rec["label"] == name and rec["n_queries"] == 71
        assert rec["adapter"] == f"jrepifano/q14b-tda-del-{name.lower()}"
        assert np.isfinite(rec["macro_nll_orig"]), f"{name} macro NLL non-finite"
        actual[name] = rec["macro_nll_orig"] - ref["macro_nll_orig"]

    committed = np.load(OUT_DIR / "scores.npz", allow_pickle=False)
    for k in AGREEMENT_REFS:
        assert k in committed.files and len(committed[k]) == tda.N_ROWS, k
        assert np.isfinite(committed[k]).all(), f"{k} non-finite"
    for v in tda.BIF_VARIANTS:
        assert v not in committed.files, f"variant {v} must not exist in scores.npz"
    refs = {k: np.asarray(committed[k]) for k in AGREEMENT_REFS}
    diag_committed = json.loads((OUT_DIR / "rank_metrics.json").read_text())["bif_diagnostics"]
    # the committed verdict is ASSERTED, not assumed (scope lock)
    assert diag_committed["store"] == PRIMARY_STORE, "committed diagnostics are not bif_kappa's"
    assert str(diag_committed["acceptance"]).startswith("FAIL"), (
        "committed BIF acceptance is not FAIL — this analysis' premise is wrong")

    pull = json.loads((OUT_DIR / "store_manifest.json").read_text())
    stores, verification, baselines, loaded = {}, {}, {}, {}
    # ---- PASS 1a: load, hash and type-check BOTH stores. No score is
    # computed here, so no analysis can precede any gate (review M3).
    for store in BIF_STORES:
        for fname in STORE_FILES:
            assert sha256_file(STORES / store / fname) == pull[f"{store}/{fname}"]["sha256"], (
                f"{store}/{fname}: sha != committed store_manifest.json")
        man = json.loads((STORES / store / "manifest.json").read_text())
        assert man["adapter_revision"] == SEED1_SHA, f"{store} from wrong adapter"
        assert man["n_rows"] == tda.N_ROWS and man["chains"] == 2
        assert man["draws_per_chain"] == 8, f"{store} draw count drifted"

        z = np.load(STORES / store / "bif_draws.npz", allow_pickle=False)
        assert set(z.files) == NPZ_KEYS, f"{store} npz key set drifted: {sorted(z.files)}"
        assert [str(x) for x in z["query_ids"]] == qids, f"{store} query order drifted"
        raw = z["row_losses"]
        assert raw.shape == (2, 8, tda.N_ROWS) and raw.dtype == np.float32
        assert np.isfinite(raw).all(), f"{store} row_losses non-finite"
        for key in ("query_losses_orig", "query_losses_neut"):
            assert z[key].shape == (2, 8, 71) and z[key].dtype == np.float32
            assert np.isfinite(z[key]).all(), f"{store} {key} non-finite"
        assert z["base_row_losses"].shape == (tda.N_ROWS,)
        assert z["base_row_losses"].dtype == np.float32
        assert np.isfinite(z["base_row_losses"]).all()
        assert int(z["truncate"]) == 0 and int(z["n_rows_truncated"]) == 0
        n_steps = int(z["burn_in"]) + 8 * int(z["thin"])
        assert z["minibatch_traces"].shape == (2, n_steps), (
            f"{store} minibatch_traces {z['minibatch_traces'].shape} != (2, {n_steps})")
        assert np.isfinite(z["minibatch_traces"]).all(), f"{store} minibatch trace non-finite"
        for key in ("nbeta", "eps", "gamma", "burn_in", "thin"):
            assert float(z[key]) == float(man[key]), f"{store} {key} npz != manifest"
        nbeta = float(z["nbeta"])
        assert nbeta == NBETA, f"{store} nbeta drifted"

        loaded[store] = {"z": z, "man": man, "raw": raw,
                         "lq": np.einsum("q,cdq->cd", w_cons,
                                         z["query_losses_orig"].astype(np.float64))}
        verification[store] = {
            "files_sha_verified": list(STORE_FILES),
            "adapter_revision": man["adapter_revision"], "n_rows": man["n_rows"],
            "kappa": man.get("kappa"), "gamma": man["gamma"], "eps": man["eps"],
            "chains": man["chains"], "draws_per_chain": man["draws_per_chain"],
            "burn_in": man["burn_in"], "thin": man["thin"], "nbeta": man["nbeta"]}

    cost_inputs = gate_cost_inputs()

    # ---- PASS 1b: only now, with every input validated, reproduce the two
    # committed baselines.
    for store in BIF_STORES:
        z, man = loaded[store]["z"], loaded[store]["man"]
        raw, lq = loaded[store]["raw"], loaded[store]["lq"]
        nbeta = float(z["nbeta"])
        rl = raw.astype(np.float64)
        pooled_base = tda.bif_scores(raw, lq, nbeta)                 # committed path
        per_chain_base = np.stack([tda.bif_scores(raw[c:c + 1], lq[c:c + 1], nbeta)
                                   for c in range(2)])
        rk = [tda.rank_from_scores(s, perm) for s in per_chain_base]
        diag = {"split_rhat_LQ": tda.split_rhat(lq), "ess_LQ": tda.ess(lq),
                "between_chain_score_spearman": tda.spearman(*per_chain_base),
                "between_chain_top685_overlap": tda.top_k_overlap(rk[0], rk[1]),
                # LLC through the committed float32 accumulation path
                "llc": nbeta * float(raw.mean() - z["base_row_losses"].mean())}
        assert all(np.isfinite(v) for v in diag.values()), f"{store} diagnostics non-finite"
        if store == PRIMARY_STORE:
            want = float(diag_committed["between_chain_score_spearman"])
            got = diag["between_chain_score_spearman"]
            assert abs(got - want) <= 1e-10 * abs(want) + 1e-12, (
                f"between-chain rho {got!r} != committed {want!r}")
            assert diag["between_chain_top685_overlap"] == float(
                diag_committed["between_chain_top685_overlap"])
            assert np.allclose(pooled_base, refs["L5_bif"], rtol=1e-10, atol=1e-12)
            assert np.array_equal(tda.rank_from_scores(pooled_base, perm),
                                  tda.rank_from_scores(refs["L5_bif"], perm))
            diag["reproduces_committed_rank_metrics"] = True
            diag["committed_values"] = {k: diag_committed[k] for k in
                                        ("store", "split_rhat_LQ", "ess_LQ",
                                         "between_chain_score_spearman",
                                         "between_chain_top685_overlap", "llc",
                                         "acceptance")}
        else:
            for key, (fmt, want_s) in GRID_PROSE.items():
                assert format(diag[key], fmt) == want_s, (
                    f"grid store {key}={diag[key]!r} formats to "
                    f"{format(diag[key], fmt)!r}, report prose says {want_s!r}")
            diag["reproduces_report_prose"] = {k: v[1] for k, v in GRID_PROSE.items()}
        generic = tda.bif_per_chain_scores(rl, lq, nbeta, "baseline")
        assert np.allclose(generic.mean(axis=0), pooled_base, rtol=1e-10, atol=1e-12)

        stores[store] = {"rl": rl, "lq": lq, "nbeta": nbeta,
                         "base_absmax": [float(np.abs(generic[c]).max()) for c in range(2)]}
        baselines[store] = diag

    return {"perm": perm, "frozen_before": frozen_before, "labels": labels,
            "subsets": subsets, "actual": actual, "refs": refs,
            "ref_rankings": {k: tda.rank_from_scores(v, perm) for k, v in refs.items()},
            "stores": stores, "store_verification": verification,
            "baselines": baselines, "committed_acceptance": diag_committed["acceptance"],
            "cost_inputs": cost_inputs}


# --- PASS 2: analysis -----------------------------------------------------

def run_analysis(g: dict) -> tuple[dict, dict]:
    """Everything after the gates. Pure given `g` (the only RNG is the frozen
    `mode_null` stream, re-created here so repeated calls agree)."""
    perm, labels = g["perm"], g["labels"]
    subsets, actual, refs = g["subsets"], g["actual"], g["refs"]
    subs_random = {k: v for k, v in subsets.items() if k.startswith("R")}
    subs_tb = {k: v for k, v in subsets.items() if not k.startswith("R")}
    null_rng = tda.bif_null_streams()["mode_null"]

    geometry, power, halves, fits, variants = {}, {}, {}, {}, {}
    pooled_scores: dict[str, np.ndarray] = {}
    for store in BIF_STORES:
        rl, lq = g["stores"][store]["rl"], g["stores"][store]["lq"]
        nbeta, base_absmax = g["stores"][store]["nbeta"], g["stores"][store]["base_absmax"]
        draws = rl.shape[1]

        geo = tda.bif_mode_geometry(rl, lq)
        for c, cell in enumerate(geo):
            x = rl[c] - rl[c].mean(axis=0, keepdims=True)
            null = tda.bif_mode_null_shares(x, null_rng, n_rep=NULL_REPS)
            p99 = float(np.quantile(null, 0.99, method="linear"))
            obs = cell["eigen_share"][0]
            cell["mode1_null_phase_randomization"] = {
                "n_rep": NULL_REPS, "quantile_method": "linear (numpy default)",
                "mean": float(null.mean()),
                "p95": float(np.quantile(null, 0.95, method="linear")), "p99": p99,
                "observed_minus_p99": float(obs - p99),
                "n_null_reps_at_or_above_observed": int((null >= obs).sum()),
                "monte_carlo_resolution": 1.0 / NULL_REPS,
                "observed_above_p99": bool(obs > p99),
                "limitation": ("circular wrapping at D=8 is a crude stand-in for the true "
                               "temporal structure: this calibrates the mode-1 share UNDER "
                               "THE CIRCULAR-SHIFT NULL only, not the sampler's "
                               "design-based null")}
        geometry[store] = geo

        power[store] = power_analysis(rl, lq, nbeta, perm)
        # §14.3: the D=8 matched cell has exactly one subset and MUST equal the
        # pass-1 reproduced committed diagnostics — asserted, not assumed
        d8 = power[store]["baseline"]["per_D"]["8"]["matched_index_PRIMARY"]
        base_diag = g["baselines"][store]
        assert d8["between_chain_spearman"]["n"] == 1
        assert np.isclose(d8["between_chain_spearman"]["mean"],
                          base_diag["between_chain_score_spearman"],
                          rtol=1e-12, atol=1e-15), "D=8 power cell != reproduced baseline"
        assert d8["between_chain_top685_overlap"]["mean"] == \
            base_diag["between_chain_top685_overlap"]
        halves[store] = {v: split_half(rl, lq, nbeta, perm, v)
                         for v in POWER_VARIANTS}
        fits[store] = {}
        for variant, block in power[store].items():
            per_d = block["per_D"]
            ds = sorted(int(d) for d in per_d if "matched_index_PRIMARY" in per_d[d])
            rs = [per_d[str(d)]["matched_index_PRIMARY"]["between_chain_spearman"]["mean"]
                  for d in ds]
            quotable = bool(variant == DSTAR_VARIANT and store == PRIMARY_STORE)
            if len(ds) < 3:
                fits[store][variant] = {"status": "too_few_defined_D_values", "D_values": ds}
                continue
            fits[store][variant] = {
                "D_values": ds, "mean_matched_between_chain_spearman": rs,
                "d_star_quotable_cell": quotable,
                "model_x_eq_D": fit_attenuation(ds, rs, "D", quotable),
                "model_x_eq_D_minus_1": fit_attenuation(ds, rs, "D-1", quotable),
                "leave_one_D_out_x_eq_D": leave_one_d_out(ds, rs, "D", quotable),
                "leave_one_D_out_x_eq_D_minus_1": leave_one_d_out(ds, rs, "D-1", quotable)}

        per_variant = {}
        for variant in tda.BIF_VARIANTS:
            entry = {"modes_removed": tda.bif_variant_modes(variant),
                     "nominal_dof": tda.bif_variant_dof(variant, draws),
                     "role": ("baseline" if variant == "baseline" else
                              "preregistered_subtraction" if variant in tda.BIF_SUBTRACTIONS
                              else "declared_sensitivity"),
                     "meets_reliability_bar_0.3": None,
                     "meets_validity_bar_0.5": None, "rescue_both_bars": None}
            # scored CHAIN BY CHAIN so one chain's degeneracy cannot discard the
            # other's valid result, and each failure keeps its declared type and
            # location (review N4)
            all_draws = np.arange(draws)
            chain_status, chain_vecs = [], []
            for c in range(rl.shape[0]):
                try:
                    sc = chain_score(rl, lq, nbeta, variant, c, all_draws)
                except tda.BifDegenerate as exc:
                    chain_status.append({"chain": c, "status": f"undefined_{exc.kind}",
                                         "kind": exc.kind, "fold": exc.fold,
                                         "reason": str(exc)})
                    chain_vecs.append(None)
                    continue
                st = score_status(sc, base_absmax[c])
                if st is not None:
                    chain_status.append({"chain": c, "status": st, "kind": st,
                                         "fold": None, "reason": st})
                    chain_vecs.append(None)
                    continue
                chain_status.append({"chain": c, "status": "ok"})
                chain_vecs.append(sc)
            entry["per_chain_status"] = chain_status
            if any(v is None for v in chain_vecs):
                entry["status"] = "undefined_per_chain"
                per_variant[variant] = entry
                continue
            per_chain = np.stack(chain_vecs)
            pooled = per_chain.mean(axis=0)
            pooled_status = score_status(pooled, max(base_absmax))
            if pooled_status is not None:
                per_variant[variant] = {**entry, "status": pooled_status,
                                        "chain": "pooled", "reason": pooled_status}
                continue
            rho = finite_or_none(tda.spearman(per_chain[0], per_chain[1]))
            lds = tda.lds_score(pooled, subsets, actual)
            lds_rho = finite_or_none(lds["spearman"])
            if rho is None or lds_rho is None:
                per_variant[variant] = {**entry, "status": "undefined_non_finite_metric",
                                        "reason": "reliability or LDS Spearman undefined"}
                continue
            pooled_scores[f"{store}__{variant}"] = pooled
            rk = [tda.rank_from_scores(s, perm) for s in per_chain]
            ranking = tda.rank_from_scores(pooled, perm)
            rescue = bool(rho >= RELIABILITY_BAR and lds_rho >= VALIDITY_BAR
                          and variant in tda.BIF_SUBTRACTIONS)
            entry.update({
                "status": "ok",
                "reliability_between_chain_spearman": rho,
                "reliability_top685_overlap": tda.top_k_overlap(rk[0], rk[1]),
                "reliability_spearman_brown_pooled_descriptive":
                    (None if rho <= -1.0 else 2 * rho / (1 + rho)),
                "lds_spearman_primary": lds_rho,
                "lds_pearson_primary": finite_or_none(lds["pearson"]),
                "lds_predicted": lds["predicted"],
                "breakdown_random_only_n4_descriptive":
                    finite_or_none(tda.lds_score(pooled, subs_random, actual)["spearman"]),
                "breakdown_tb_slices_only_n6":
                    finite_or_none(tda.lds_score(pooled, subs_tb, actual)["spearman"]),
                "meets_reliability_bar_0.3": bool(rho >= RELIABILITY_BAR),
                "meets_validity_bar_0.5": bool(lds_rho >= VALIDITY_BAR),
                "rescue_both_bars": rescue,
                "label_metrics_SECONDARY_provenance_not_causal": {
                    "precision_at_685": tda.precision_at_k(ranking, labels),
                    "hypergeom_p": tda.hypergeom_pvalue(ranking, labels),
                    "average_precision": tda.average_precision(ranking, labels),
                    "precision_curve": {k: tda.precision_at_k(ranking, labels, k)
                                        for k in K_CURVE}},
                "agreement": {rn: {"spearman": finite_or_none(tda.spearman(pooled, refs[rn])),
                                   "top685_overlap": tda.top_k_overlap(
                                       ranking, g["ref_rankings"][rn])}
                              for rn in AGREEMENT_REFS},
                "score_scale_caveat_abs_mean": float(np.abs(pooled).mean()),
                "score_scale_caveat_abs_max": float(np.abs(pooled).max())})
            per_variant[variant] = entry
        variants[store] = per_variant

    cross = {}
    for variant in tda.BIF_VARIANTS:
        ka, kb = f"{PRIMARY_STORE}__{variant}", f"bif__{variant}"
        if ka in pooled_scores and kb in pooled_scores:
            a, b = pooled_scores[ka], pooled_scores[kb]
            cross[variant] = {"spearman": finite_or_none(tda.spearman(a, b)),
                              "top685_overlap": tda.top_k_overlap(
                                  tda.rank_from_scores(a, perm),
                                  tda.rank_from_scores(b, perm))}

    readout = hypothesis_readout(g, geometry, power, halves, fits, variants, pooled_scores)

    cost_grid = [8, 16, 32, 64, 128, 256]
    base_fit = fits[PRIMARY_STORE].get(DSTAR_VARIANT, {})
    for key in ("model_x_eq_D", "model_x_eq_D_minus_1"):
        ceil = (base_fit.get(key) or {}).get("draws_for_rho_0.3_ceiling_HEURISTIC_NOT_A_BUDGET")
        if ceil is not None and ceil not in cost_grid:
            cost_grid.append(int(ceil))
    cost = rerun_cost_model(sorted(cost_grid), g["cost_inputs"])

    out = {
        "script": "tda_bif_modes.py",
        "preregistration": "docs/tda-preregistration.md section 14",
        "tda_seed": tda.TDA_SEED,
        "null_stream": "SeedSequence([TDA_SEED, 14]).spawn(1) -> ['mode_null']",
        "scope": ("EXPLORATORY diagnosis of a recorded negative. BIF's preregistered "
                  "acceptance verdict is UNCHANGED; no variant here is a locator and "
                  "nothing here changes any committed ranking, selection, or arm."),
        "scope_lock": {"committed_acceptance_asserted": g["committed_acceptance"],
                       "no_variant_is_stage_b_eligible": True,
                       "no_variant_written_to_scores_npz": True},
        "store_verification": g["store_verification"],
        "actual_dnll_orig": actual,
        "baseline_diagnostics_recomputed": g["baselines"],
        "mode_geometry_descriptive": geometry,
        "power_analysis": power,
        "within_chain_split_half": halves,
        "attenuation_fits": fits,
        "variants": variants,
        "cross_store_DESCRIPTIVE_ONLY": cross,
        "rerun_cost_model_ESTIMATE": cost,
        "hypothesis_readout": readout,
        "bars": {"reliability_between_chain_spearman": RELIABILITY_BAR,
                 "validity_lds_spearman": VALIDITY_BAR,
                 "note": "both required on the same SUBTRACTION variant and the primary "
                         "store; reliability without validity is a NEGATIVE result "
                         "(addendum 14.6); sensitivities cannot constitute a rescue"},
    }
    return out, pooled_scores


def classify(h_noise: dict, h_mode: dict) -> str:
    if h_noise.get("undefined") or h_mode.get("undefined"):
        return "undefined_analysis_incomplete"
    if h_noise["supported"] and h_mode["supported"]:
        return "both"
    if h_noise["supported"]:
        return "H_noise_only"
    if h_mode["supported"]:
        return "H_mode_only"
    return "neither"


def outcome_of(rescued, reliability_only, subtraction_states, h_mode_supported) -> str:
    """The §14.6 outcome map. `subtraction_states` maps each preregistered
    subtraction to True (meets reliability), False (does not), or None
    (undefined) — any None forbids the bounded outcome-3 claim."""
    if any(v is None for v in subtraction_states.values()):
        return "undefined_analysis_incomplete"
    if rescued:
        return ("1_rescue_shared_mode_diagnosis_licensed" if h_mode_supported
                else "1_rescue_transformation_worked_mechanism_not_established")
    if reliability_only:
        return "2_NEGATIVE_reliability_bought_at_the_cost_of_validity"
    if not any(subtraction_states.values()):
        return "3_no_preregistered_subtraction_rescues_BIF_at_this_draw_count"
    return "4_other_descriptive_no_claim"


def hypothesis_readout(g, geometry, power, halves, fits, variants, pooled_scores) -> dict:
    """The §14.5 predicates, evaluated mechanically and TOTALLY: any undefined
    input yields `undefined_analysis_incomplete`, never a scientific verdict."""
    prim = PRIMARY_STORE
    geo = geometry[prim]
    base_d = power[prim]["baseline"]["per_D"]
    ds = sorted(int(d) for d in base_d if "matched_index_PRIMARY" in base_d[d])
    means = [base_d[str(d)]["matched_index_PRIMARY"]["between_chain_spearman"]["mean"]
             for d in ds]
    sh = halves[prim]["baseline"]
    fit_d = (fits[prim].get("baseline") or {}).get("model_x_eq_D")
    missing = []
    if len(ds) != 7:
        missing.append("power sweep incomplete")
    # EVERY declared matched cell must have been usable: summarizing over the
    # survivors of a partial enumeration would let a degenerate computation
    # produce a scientific classification (addendum 14.2 totality clause)
    for d in range(2, 9):
        cell = base_d.get(str(d), {})
        if "matched_index_PRIMARY" not in cell:
            missing.append(f"D={d} matched cell undefined")
            continue
        if cell.get("n_undefined_cells", 0) != 0:
            missing.append(f"D={d} has {cell['n_undefined_cells']} undefined chain cells")
        expected = comb(8, d)
        got = cell["matched_index_PRIMARY"]["between_chain_spearman"]["n"]
        if got != expected:
            missing.append(f"D={d} matched count {got} != {expected}")
    if "all_partitions_spearman" not in sh:
        missing.append("split-half undefined")
    else:
        if sh.get("n_undefined_cells", 0) != 0:
            missing.append(f"split-half has {sh['n_undefined_cells']} undefined cells")
        n_half = sh["all_partitions_spearman"]["n"]
        if n_half != 2 * sh.get("n_partitions_per_chain", 0) or n_half != 70:
            missing.append(f"split-half usable cells {n_half} != 70")
    if fit_d is None:
        missing.append("attenuation fit unavailable")
    elif fit_d["fit_rejected"]:
        # a REJECTED fit is undefined, never "predicate 2 is False" — a failed
        # computation must not pass itself off as evidence (addendum 14.2
        # totality clause; review B1)
        missing.append(f"attenuation fit rejected (kappa_on_bound="
                       f"{fit_d['kappa_on_bound']}, optimizer_success="
                       f"{fit_d['optimizer_success']})")
    if variants[prim]["baseline"].get("status") != "ok":
        missing.append("baseline variant undefined")

    if missing:
        h_noise = {"undefined": True, "reason": missing, "supported": False}
        delta = None
    else:
        r4 = base_d["4"]["matched_index_PRIMARY"]["between_chain_spearman"]["mean"]
        delta = float(sh["all_partitions_spearman"]["mean"] - r4)
        nondec = bool(all(b >= a - 0.01 for a, b in itertools.pairwise(means))
                      and means[-1] > means[0])
        h_noise = {
            "predicate_1_non_decreasing_in_D_slack_0.01": nondec,
            "mean_matched_between_chain_spearman_by_D": dict(zip(map(str, ds), means,
                                                                 strict=True)),
            "predicate_2_attenuation_r2_ge_0.8_HEURISTIC": fit_d["fit_accepted_r2_ge_0.8"],
            "r_squared": fit_d["r_squared"],
            "predicate_3_delta_split_half_minus_between_chain_at_D4": delta,
            "predicate_3_within_band_0.05_HEURISTIC": bool(abs(delta) <= 0.05),
            "delta_within_band_0.03": bool(abs(delta) <= 0.03),
            "delta_within_band_0.10": bool(abs(delta) <= 0.10),
            "supported": bool(nondec and fit_d["fit_accepted_r2_ge_0.8"] and abs(delta) <= 0.05),
            "undefined": False}

    m1_ok = all("mode1_null_phase_randomization" in c for c in geo)
    svd_key = f"{prim}__svd_m1"
    rho_b_m1 = (finite_or_none(tda.spearman(pooled_scores[f"{prim}__baseline"],
                                            pooled_scores[svd_key]))
                if (svd_key in pooled_scores and f"{prim}__baseline" in pooled_scores) else None)
    if not m1_ok or rho_b_m1 is None:
        h_mode = {"undefined": True, "supported": False,
                  "reason": ["null missing" if not m1_ok else "svd_m1 pooled score undefined"]}
    else:
        p1 = all(c["eigen_share"][0] >= 0.5 for c in geo)
        p2 = all(c["mode1_null_phase_randomization"]["observed_above_p99"] for c in geo)
        p3 = all(c["query_energy_share_by_mode"][0] >= 0.5 for c in geo)
        p4 = bool(rho_b_m1 <= 0.8)
        h_mode = {
            "predicate_1_mode1_variance_share_ge_0.5_both_chains": bool(p1),
            "mode1_variance_share_by_chain": [c["eigen_share"][0] for c in geo],
            "predicate_2_above_phase_randomization_p99_both_chains": bool(p2),
            "mode1_null_p99_by_chain": [c["mode1_null_phase_randomization"]["p99"] for c in geo],
            "predicate_3_mode1_query_energy_share_ge_0.5_both_chains": bool(p3),
            "mode1_query_energy_share_by_chain": [c["query_energy_share_by_mode"][0]
                                                  for c in geo],
            "predicate_4_pooled_spearman_baseline_vs_svd_m1_le_0.8": p4,
            "spearman_baseline_vs_svd_m1_pooled": rho_b_m1,
            "population_isotropic_share": geo[0]["population_isotropic_share"],
            "n_eff_rows_by_chain": [c["n_eff_rows"] for c in geo],
            "supported": bool(p1 and p2 and p3 and p4), "undefined": False}

    states = {v: (None if variants[prim][v].get("status") != "ok"
                  else bool(variants[prim][v]["meets_reliability_bar_0.3"]))
              for v in tda.BIF_SUBTRACTIONS}
    rescued = [v for v, r in variants[prim].items() if r.get("rescue_both_bars")]
    reliability_only = [v for v in tda.BIF_SUBTRACTIONS
                        if variants[prim][v].get("meets_reliability_bar_0.3")
                        and not variants[prim][v].get("meets_validity_bar_0.5")]
    return {
        "primary_store": prim, "H_noise": h_noise, "H_mode_degeneracy": h_mode,
        "classification": classify(h_noise, h_mode),
        "rescue": {"variants_meeting_both_bars": rescued,
                   "subtractions_meeting_reliability_only": reliability_only,
                   "subtraction_reliability_states": states,
                   "outcome": outcome_of(rescued, reliability_only, states,
                                         h_mode.get("supported", False))},
        "original_acceptance": g["committed_acceptance"],
        "acceptance_verdict_unchanged": (
            "BIF FAILS preregistered acceptance (prereg 7/7b) and remains exploratory and "
            "Stage-B-ineligible regardless of every number in this artifact"),
    }


# --- main -----------------------------------------------------------------

def num(x, width: int, prec: int, signed: bool = False) -> str:
    """Nullable metrics render as `--`, never crash the table after the
    artifact has already been written (review N5)."""
    if x is None:
        return f"{'--':>{width}}"
    return f"{x:{'+' if signed else ''}{width}.{prec}f}"


def print_tables(out: dict) -> None:
    variants, power, fits = out["variants"], out["power_analysis"], out["attenuation_fits"]
    geometry, cost = out["mode_geometry_descriptive"], out["rerun_cost_model_ESTIMATE"]
    readout = out["hypothesis_readout"]
    print(f"\n{'store':>10} {'variant':>13} | {'dof':>3} | {'reliab':>7} | {'ovl685':>7} | "
          f"{'LDS rho':>8} | {'R-only':>7} | {'TB':>6} | {'p@685':>6} | rescue")
    for store in BIF_STORES:
        for variant in tda.BIF_VARIANTS:
            v = variants[store][variant]
            if v.get("status") != "ok":
                print(f"{store:>10} {variant:>13} | {v['nominal_dof']:3d} | "
                      f"{v.get('status', 'undefined'):>50}")
                continue
            print(f"{store:>10} {variant:>13} | {v['nominal_dof']:3d} | "
                  f"{num(v['reliability_between_chain_spearman'], 7, 4)} | "
                  f"{num(v['reliability_top685_overlap'], 7, 4)} | "
                  f"{num(v['lds_spearman_primary'], 8, 4)} | "
                  f"{num(v['breakdown_random_only_n4_descriptive'], 7, 2, True)} | "
                  f"{num(v['breakdown_tb_slices_only_n6'], 6, 2, True)} | "
                  f"{num(v['label_metrics_SECONDARY_provenance_not_causal']['precision_at_685'], 6, 3)} | "
                  f"{'YES' if v['rescue_both_bars'] else 'no'}")
    print(f"\n{'store':>10} {'variant':>13} | " + " | ".join(f"D={d:>6}" for d in range(2, 9)))
    for store in BIF_STORES:
        for variant in POWER_VARIANTS:
            cells = []
            for d in range(2, 9):
                cell = power[store][variant]["per_D"].get(str(d))
                ok = cell is not None and "matched_index_PRIMARY" in cell
                cells.append(
                    f"{cell['matched_index_PRIMARY']['between_chain_spearman']['mean']:8.4f}"
                    if ok else f"{'--':>8}")
            print(f"{store:>10} {variant:>13} | " + " | ".join(cells))
    for store in BIF_STORES:
        f = fits[store]["baseline"]
        for key, tag in (("model_x_eq_D", "x=D"), ("model_x_eq_D_minus_1", "x=D-1")):
            m = f[key]
            kap = "null" if m["kappa"] is None else f"{m['kappa']:.4g}"
            print(f"[{store}] attenuation {tag}: kappa={kap} R2={m['r_squared']} "
                  f"-> draws for rho=0.3 = "
                  f"{m['draws_for_rho_0.3_HEURISTIC_NOT_A_BUDGET']} "
                  f"(HEURISTIC, NOT A BUDGET; quotable={m['quotable_cell']})")
        print(f"[{store}] mode-1 share by chain "
              f"{[round(c['eigen_share'][0], 4) for c in geometry[store]]} "
              f"(pop. isotropic {geometry[store][0]['population_isotropic_share']:.3f}, "
              f"shift-null p99 "
              f"{[round(c['mode1_null_phase_randomization']['p99'], 4) for c in geometry[store]]}); "
              f"L_Q energy on mode 1 "
              f"{[round(c['query_energy_share_by_mode'][0], 4) for c in geometry[store]]}; "
              f"N_eff {[round(c['n_eff_rows'], 1) for c in geometry[store]]}")
    print(f"[cost] ESTIMATE ${cost['marginal_usd_per_extra_draw_both_chains']:.3f} per extra "
          f"draw/chain (2 chains); "
          + ", ".join(f"D={d}:${v:.1f}" for d, v in cost["usd_by_draws_per_chain"].items()))
    print(f"\nH-noise supported: {readout['H_noise']['supported']}; H-mode supported: "
          f"{readout['H_mode_degeneracy']['supported']}; classification: "
          f"{readout['classification']}; outcome: {readout['rescue']['outcome']}")


def main(argv: list[str]) -> None:
    t0 = datetime.now(UTC)
    verify = "--verify-determinism" in argv
    g = gate_inputs()                                  # PASS 1: all gates first
    out, pooled_scores = run_analysis(g)               # PASS 2: analysis
    if verify:
        out2, pooled2 = run_analysis(g)
        assert json.dumps(out, sort_keys=True, default=float, allow_nan=False) == \
               json.dumps(out2, sort_keys=True, default=float, allow_nan=False), (
            "non-deterministic analysis: two in-process runs disagree")
        assert set(pooled_scores) == set(pooled2) and all(
            np.array_equal(pooled_scores[k], pooled2[k]) for k in pooled_scores)
        out["determinism_verified"] = True
        out["determinism_verification_method"] = "two in-process runs, identical"

    np.savez(OUT_DIR / "bif_mode_scores.npz",
             **{k: v.astype(np.float64) for k, v in pooled_scores.items()})
    frozen_after = {f: sha256_file(OUT_DIR / f) for f in FROZEN_ARTIFACTS}
    assert g["frozen_before"] == frozen_after, "a committed artifact changed during this run"
    out["scope_lock"]["frozen_artifacts_sha256_unchanged"] = frozen_after
    out["generated_at"] = t0.isoformat()
    out["finished_at"] = datetime.now(UTC).isoformat()
    (OUT_DIR / "bif_mode_analysis.json").write_text(
        json.dumps(out, indent=2, default=float, allow_nan=False) + "\n")

    print_tables(out)
    print(f"-> {OUT_DIR / 'bif_mode_analysis.json'}  "
          f"[{(datetime.now(UTC) - t0).total_seconds():.1f}s"
          f"{'; determinism verified' if verify else ''}]")


if __name__ == "__main__":
    main(sys.argv[1:])
