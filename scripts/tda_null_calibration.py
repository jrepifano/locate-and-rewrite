"""Empirical null for the locator-validation Spearman (POST HOC; not preregistered).

The LDS validation (scripts/tda_lds.py -> results/tda/lds_results.json) scores
each locator by the Spearman correlation between its predicted group influence
(sum of its per-row scores over each of ten 685-row deletion groups) and the ten
measured deletion effects. Until now the only calibration of that statistic's
null was a single random locator, L0_random, which drew rho = -0.60; the writeup
used that one draw to describe the null width. This script replaces the anecdote
with the distribution: 10,000 random per-row score vectors scored against the
same ten measured effects, through the same group-sum -> Spearman machinery.

Convention: L0_random is generated in scripts/tda_rank.py as
`streams["l0_random"].permutation(tda.N_ROWS).astype(np.float64)`, i.e. a uniform
random permutation of {0, ..., 13697} over the rows (not i.i.d. draws). The
primary null replicates that exactly, and the committed L0_random vector in
scores.npz is asserted equal to a fresh draw of that convention before anything
else runs. A secondary i.i.d. standard-normal null is computed alongside; its width
comes out close to the primary one, which is evidence about that one alternative
convention, not a general claim.

Groups, measured effects and the Spearman itself are taken from the same places
tda_lds.py takes them: the ten row-index sets from
data/processed/tda_retrain_sets.json, the measured effects from
lds_results.json -> actual_dnll_orig, and group influence = sum of member scores
(em_filter.tda.group_influence). Inputs are sha256-pinned; the committed L0 rho
is recomputed and asserted against the committed sanity block.

Declared post hoc: defined and computed 2026-08-30 after every locator score and
every retrain was known. It calibrates the width of an existing statistic; it
does not add retrains, change n = 10, or touch the preregistered pass/fail bars.

No timestamps anywhere: two runs must produce byte-identical outputs.

Usage: uv run python scripts/tda_null_calibration.py
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_filter import config as C
from em_filter import tda

SETS_PATH = C.DATA_PROCESSED / "tda_retrain_sets.json"
SETS_SHA256 = "abb07d58548cc8f89a59d1068fb3421f8cbc9f8dd3515c680a1a39b87b032737"
LDS_PATH = C.RESULTS_DIR / "tda" / "lds_results.json"
LDS_SHA256 = "f54b9aa987a60cc49c6a633e7ee56729544ca3aa37bb4f3dc89a0c2df3cb3d2b"
SCORES_PATH = C.RESULTS_DIR / "tda" / "scores.npz"
SCORES_SHA256 = "295235fe71db8d32eeaca971362ce362e8762e8549c570863bf0473fc3ea41ed"

NULL_SEED = 20260830  # fresh: distinct from TDA_SEED 20260818 that produced L0_random
N_DRAWS = 10_000
CHUNK = 500  # draws per block: a memory bound only, not part of the seed contract
N_GROUPS = 10
GROUP_SIZE = 685
CHUNK_CHECK_N = 40      # draws redrawn in smaller blocks to show chunk-independence
CHUNK_CHECK_SMALL = 10
TOL = 1e-12
RHO_TOL = 1e-9  # slack for comparing lattice-valued Spearman values (spacing 6/990)

# locators whose null percentile the writeup and figures quote; the artifact
# carries all of them, this tuple only fixes the printed order
HEADLINE = ("L3_defif_c10", "L4a_ekfac_analytic", "L2a_graddot", "L5_bif",
            "Lor_labels", "L1_content", "L0_random")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_pinned() -> None:
    for path, sha in ((SETS_PATH, SETS_SHA256), (LDS_PATH, LDS_SHA256),
                      (SCORES_PATH, SCORES_SHA256)):
        got = sha256_of(path)
        assert got == sha, f"{path.name} sha256 {got} != committed {sha}"


def spearman_columns(sums: np.ndarray, actual_vec: np.ndarray) -> np.ndarray:
    """Spearman rho of each column of `sums` (n_groups x n_draws) against a fixed
    vector, computed exactly the way scipy.stats.spearmanr does it: average ranks
    (so ties are handled, not assumed away) then Pearson on the ranks.

    Group sums of a random permutation are integer-valued, so two of the ten
    groups CAN tie inside a draw; main() counts those draws, and crosschecks
    every draw against scipy.stats.spearmanr.
    """
    rp = sps.rankdata(sums, axis=0)
    ra = sps.rankdata(actual_vec)
    rp_c = rp - rp.mean(axis=0)
    ra_c = ra - ra.mean()
    den = np.sqrt((rp_c ** 2).sum(axis=0) * (ra_c ** 2).sum())
    assert np.all(den > 0), "degenerate ranking (all ten group sums equal)"
    return (rp_c * ra_c[:, None]).sum(axis=0) / den


def n_tied_draws(sums: np.ndarray) -> int:
    """Draws in which two of the ten group sums are exactly equal."""
    return int((np.diff(np.sort(sums, axis=0), axis=0) == 0).any(axis=0).sum())


def draw_null(kind: str, rng: np.random.Generator, member: np.ndarray,
              n_draws: int = N_DRAWS, chunk: int = CHUNK) -> np.ndarray:
    """(n_groups, n_draws) group-influence sums for n_draws random score vectors.

    kind "permutation": uniform random permutation of {0..N_ROWS-1} (the L0
    convention). kind "iid_normal": i.i.d. standard normal per row. `chunk` only
    bounds peak memory; both generator calls consume the stream row by row, so
    the draws are chunk-independent (checked in main()).
    """
    base = np.arange(tda.N_ROWS, dtype=np.float64)
    out = np.empty((member.shape[0], n_draws), dtype=np.float64)
    for start in range(0, n_draws, chunk):
        k = min(chunk, n_draws - start)
        if kind == "permutation":
            block = rng.permuted(np.broadcast_to(base, (k, tda.N_ROWS)), axis=1)
        else:
            block = rng.standard_normal((k, tda.N_ROWS))
        out[:, start:start + k] = member @ block.T
    return out


def null_block(rhos: np.ndarray) -> dict:
    return {
        "n_draws": int(rhos.size),
        "mean": round(float(rhos.mean()), 6),
        "sd": round(float(rhos.std(ddof=1)), 6),
        "quantiles_linear_interpolation": {
            "q0.5": round(float(np.quantile(rhos, 0.005)), 6),
            "q2.5": round(float(np.quantile(rhos, 0.025)), 6),
            "q50": round(float(np.quantile(rhos, 0.5)), 6),
            "q97.5": round(float(np.quantile(rhos, 0.975)), 6),
            "q99.5": round(float(np.quantile(rhos, 0.995)), 6),
        },
        "central_95pct_range": [round(float(np.quantile(rhos, 0.025)), 6),
                                round(float(np.quantile(rhos, 0.975)), 6)],
        "central_99pct_range": [round(float(np.quantile(rhos, 0.005)), 6),
                                round(float(np.quantile(rhos, 0.995)), 6)],
        "min": round(float(rhos.min()), 6),
        "max": round(float(rhos.max()), 6),
        "max_abs": round(float(np.abs(rhos).max()), 6),
        "frac_abs_ge_0.5_preregistered_pass_bar": round(float((np.abs(rhos) >= 0.5 - RHO_TOL).mean()), 6),
        "frac_ge_0.5_preregistered_pass_bar_one_sided": round(float((rhos >= 0.5 - RHO_TOL).mean()), 6),
    }


def percentile_block(rhos: np.ndarray, obs: float) -> dict:
    """Where an observed rho sits in the null.

    Spearman on ten groups is lattice-valued, so exact ties between an observed
    rho and null draws are common; the observed value comes from scipy inside
    tda_lds.py and the null from scipy-equivalent ranks here, and equal lattice
    points can differ in the last bits. Every comparison is therefore made with
    RHO_TOL = 1e-9 slack (the lattice spacing is 6/990 = 0.0061), so "at or
    below" really means "at or below". The reported quantities are raw Monte
    Carlo tail fractions B/N over N = 10,000 draws, resolution 1e-4, with no
    finite-simulation (B+1)/(N+1) correction and no multiplicity correction.
    """
    lo95, hi95 = float(np.quantile(rhos, 0.025)), float(np.quantile(rhos, 0.975))
    return {
        "rho": round(float(obs), 6),
        "percentile_strictly_below": round(100.0 * float((rhos < obs - RHO_TOL).mean()), 4),
        "percentile_at_or_below": round(100.0 * float((rhos <= obs + RHO_TOL).mean()), 4),
        "tail_fraction_one_sided_ge": round(float((rhos >= obs - RHO_TOL).mean()), 6),
        "tail_fraction_two_sided_abs_ge": round(float((np.abs(rhos) >= abs(obs) - RHO_TOL).mean()), 6),
        "inside_central_95pct": bool(lo95 - RHO_TOL <= obs <= hi95 + RHO_TOL),
        "inside_null_support": bool(rhos.min() - RHO_TOL <= obs <= rhos.max() + RHO_TOL),
    }


def main() -> None:
    assert_pinned()
    sets = json.loads(SETS_PATH.read_text())
    lds = json.loads(LDS_PATH.read_text())

    subsets = {name: np.array(v["row_indices"]) for name, v in sets["subsets"].items()}
    names = sorted(subsets)  # the order em_filter.tda.lds_score uses
    actual = lds["actual_dnll_orig"]
    assert len(names) == N_GROUPS and sorted(actual) == names, "group set != measured-effect set"
    for n in names:
        idx = subsets[n]
        assert idx.shape == (GROUP_SIZE,), f"{n}: {idx.shape} != ({GROUP_SIZE},)"
        assert idx.dtype.kind == "i" and idx.min() >= 0 and idx.max() < tda.N_ROWS, f"{n}: bad indices"
        assert len(np.unique(idx)) == GROUP_SIZE, f"{n}: duplicate row indices"

    # the L0 convention, verified rather than assumed: the committed L0_random
    # score vector IS a uniform random permutation of {0..N_ROWS-1}
    committed_l0 = np.load(SCORES_PATH, allow_pickle=False)["L0_random"]
    fresh_l0 = tda.seed_streams()["l0_random"].permutation(tda.N_ROWS).astype(np.float64)
    assert np.array_equal(committed_l0, fresh_l0), "L0_random is not the permutation convention"
    l0_rho = tda.lds_score(committed_l0, subsets, actual)["spearman"]
    assert abs(l0_rho - lds["sanity"]["L0_rho"]) < TOL, "recomputed L0 rho != committed sanity block"

    # membership matrix: group sums = M @ scores (em_filter.tda.group_influence,
    # which is sum of member scores, done for all ten groups at once)
    member = np.zeros((N_GROUPS, tda.N_ROWS), dtype=np.float64)
    for i, n in enumerate(names):
        member[i, subsets[n]] = 1.0
    check = member @ committed_l0
    ref = tda.group_influence(committed_l0, subsets)
    assert max(abs(check[i] - ref[n]) for i, n in enumerate(names)) < TOL, "membership matrix mismatch"

    actual_vec = np.array([actual[n] for n in names], dtype=np.float64)
    assert len(np.unique(actual_vec)) == N_GROUPS, "tied measured effects"

    # two independent streams from one fresh seed, spawn order frozen
    kids = np.random.SeedSequence(NULL_SEED).spawn(2)
    streams = dict(zip(("permutation", "iid_normal"), (np.random.default_rng(s) for s in kids)))

    sums = draw_null("permutation", streams["permutation"], member)
    primary = spearman_columns(sums, actual_vec)
    # exactness check against scipy on EVERY draw, not a subsample
    scipy_rho = np.array([sps.spearmanr(sums[:, i], actual_vec).statistic for i in range(N_DRAWS)])
    worst = float(np.abs(primary - scipy_rho).max())
    assert worst < TOL, f"vectorized Spearman != scipy (max diff {worst})"
    crosscheck = {"n_draws_rescored_with_scipy": N_DRAWS,
                  "max_abs_difference": worst,
                  "n_draws_with_tied_group_sums": n_tied_draws(sums),
                  "note": "group sums of a permutation are integer-valued, so ties "
                          "between two of the ten groups occur; average ranks handle them"}
    sums2 = draw_null("iid_normal", streams["iid_normal"], member)
    secondary = spearman_columns(sums2, actual_vec)

    # the chunk size is a memory knob, not part of the seed contract: verified,
    # not assumed, by redrawing the first CHUNK_CHECK_N draws in smaller blocks
    chunk_check = {"n_draws_compared": CHUNK_CHECK_N,
                   "chunk_sizes": [CHUNK, CHUNK_CHECK_SMALL],
                   "identical_group_sums": bool(np.array_equal(
                       sums[:, :CHUNK_CHECK_N],
                       draw_null("permutation", np.random.default_rng(kids[0]), member,
                                 n_draws=CHUNK_CHECK_N, chunk=CHUNK_CHECK_SMALL)))}
    assert chunk_check["identical_group_sums"], "draws depend on the chunk size"

    per_locator = {name: percentile_block(primary, v["lds_spearman_primary"])
                   for name, v in sorted(lds["lds"].items())}

    out = {
        "protocol": (
            "empirical null for the LDS Spearman: 10,000 random per-row score vectors "
            "over the 13,698 mixture rows, each reduced to ten group-influence sums "
            "(sum of member scores over the same ten 685-row deletion groups in "
            "data/processed/tda_retrain_sets.json) and rank-correlated against the ten "
            "measured deletion effects (lds_results.json -> actual_dnll_orig). Identical "
            "statistic to em_filter.tda.lds_score (average ranks, so the few draws with "
            "tied group sums are handled rather than assumed away); every one of the "
            "10,000 vectorized values is re-scored with scipy.stats.spearmanr and "
            "asserted equal."
        ),
        "convention": (
            "PRIMARY replicates L0_random exactly: a uniform random permutation of "
            "{0..13697} as the per-row score vector (scripts/tda_rank.py: "
            "streams['l0_random'].permutation(N_ROWS)), asserted here against the "
            "committed L0_random vector in scores.npz. SECONDARY draws i.i.d. standard "
            "normal per-row scores; its width comes out close to the primary one, which "
            "is evidence about that one alternative convention, not a general claim."
        ),
        "preregistration_status": (
            "POST HOC: defined and computed 2026-08-30, after every locator score and "
            "every retrain was known. It exists because the committed validation "
            "calibrated its null with a single random locator (L0_random, rho = -0.60) "
            "and the writeup quoted that one draw as the null width. "
            "LICENSES: describing the width of the LDS statistic under random row "
            "scores on these exact ten groups, and locating an observed rho in that "
            "distribution. DOES NOT LICENSE: any change to the preregistered 0.5 pass / "
            "0.2 fail bars, which were frozen before the retrains and are unaffected; "
            "an unbiased comparison across methods, because the top and bottom groups "
            "were cut from a preliminary GradDot ranking, so the ten-group statistic is "
            "gradient-tilted (writeup limitation 1) and this null does not correct for "
            "that; a test of the difference between two locators; or a multiplicity "
            "correction, since 21 locators were scored against the same ten retrains and "
            "the per-locator p-values below are uncorrected. n = 10 retrains throughout; "
            "nothing here adds evidence, it only measures how wide the existing null is."
        ),
        "seed": NULL_SEED,
        "seed_streams": {"spawn_order": ["permutation", "iid_normal"],
                         "parent": "np.random.SeedSequence(20260830).spawn(2)",
                         "chunk_size_draws": CHUNK,
                         "note": "chunk_size_draws bounds peak memory only; both generator "
                                 "calls consume the stream row by row, so the draws do not "
                                 "depend on it (verified in chunk_independence_check)"},
        "n_draws": N_DRAWS,
        "pinned": {
            "tda_retrain_sets_json_sha256": SETS_SHA256,
            "lds_results_json_sha256": LDS_SHA256,
            "scores_npz_sha256": SCORES_SHA256,
        },
        "inputs": {
            "groups": {n: int(subsets[n].size) for n in names},
            "measured_effects_actual_dnll_orig": {n: actual[n] for n in names},
            "n_rows": tda.N_ROWS,
        },
        "l0_convention_check": {
            "committed_L0_random_equals_fresh_permutation_draw": True,
            "recomputed_L0_rho": round(float(l0_rho), 6),
            "committed_sanity_L0_rho": lds["sanity"]["L0_rho"],
        },
        "spearman_crosscheck": crosscheck,
        "chunk_independence_check": chunk_check,
        "tail_fraction_definition": (
            "raw Monte Carlo tail fractions B/N over N = 10,000 draws (resolution 1e-4), "
            "no finite-simulation (B+1)/(N+1) correction and no correction for the 21 "
            "locators scored against the same ten retrains. Comparisons use 1e-9 slack "
            "because Spearman on ten groups is lattice-valued (spacing 6/990)."
        ),
        "null_permutation_scores_primary": null_block(primary),
        "null_iid_normal_scores_secondary": null_block(secondary),
        "locator_percentiles_vs_primary_null": per_locator,
        "figure_band": {
            "definition": "central 95% of the primary null, plotted as a shaded band "
                          "behind the rho bars in fig12 / camera-ready figure 4",
            "lo": round(float(np.quantile(primary, 0.025)), 6),
            "hi": round(float(np.quantile(primary, 0.975)), 6),
        },
    }
    path = C.RESULTS_DIR / "tda" / "lds_null_calibration.json"
    path.write_text(json.dumps(out, indent=2) + "\n")

    p = out["null_permutation_scores_primary"]
    print(f"null (permutation, n={N_DRAWS}): mean {p['mean']:+.4f}  sd {p['sd']:.4f}  "
          f"95% [{p['central_95pct_range'][0]:+.3f}, {p['central_95pct_range'][1]:+.3f}]  "
          f"max|rho| {p['max_abs']:.3f}")
    s = out["null_iid_normal_scores_secondary"]
    print(f"null (i.i.d. normal): mean {s['mean']:+.4f}  sd {s['sd']:.4f}  "
          f"95% [{s['central_95pct_range'][0]:+.3f}, {s['central_95pct_range'][1]:+.3f}]")
    print(f"P(|rho| >= 0.5 by chance) = {p['frac_abs_ge_0.5_preregistered_pass_bar']:.4f}")
    for name in HEADLINE:
        b = per_locator[name]
        print(f"{name:>20}: rho {b['rho']:+.3f}  pctile<= {b['percentile_at_or_below']:6.2f}  "
              f"1-sided tail {b['tail_fraction_one_sided_ge']:.4f}  "
              f"2-sided tail {b['tail_fraction_two_sided_abs_ge']:.4f}  "
              f"inside 95% null: {b['inside_central_95pct']}")
    print(f"wrote {path.relative_to(C.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
