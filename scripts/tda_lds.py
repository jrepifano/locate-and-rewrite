"""LOCAL Stage-A finale: LDS scoring of every locator against the retrains.

Reads results/tda/scores.npz (tda_rank.py), the subset manifest
(tda_retrain_sets.json), and the pulled retrain NLL JSONs
(results/tda/nll/tda_nll_<subset>.json + tda_nll_REF.json). Computes:
  - actual dNLL per subset (primary: consensus macro orig; secondary:
    contrastive) vs the arm1_r1_seed1 reference
  - LDS Spearman for every locator score vector (predicted = sum of member
    scores), with the preregistered thresholds annotated
  - lambda selection for L3/L6f (argmax LDS rho, primary target)
  - the bound Stage-B recommendation: best gradient-family locator by LDS,
    ties within 0.05 broken by cross-seed rank correlation
  - sanity: L0 rho ~ 0; random-subset dNLL magnitudes vs T/B slices

Writes results/tda/lds_results.json and prints the STOP-checkpoint table.

Usage: uv run python scripts/tda_lds.py
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_filter import config as C
from em_filter import tda

OUT_DIR = C.RESULTS_DIR / "tda"
NLL_DIR = OUT_DIR / "nll"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-missing-stability", action="store_true",
                    help="explicit override of the mandatory seed-2 stability check (recorded)")
    args = ap.parse_args()
    scores_z = np.load(OUT_DIR / "scores.npz", allow_pickle=False)
    scores = {k: scores_z[k] for k in scores_z.files if not k.startswith("seed2_")}
    sets = json.loads((C.DATA_PROCESSED / "tda_retrain_sets.json").read_text())
    subsets = {name: np.array(v["row_indices"]) for name, v in sets["subsets"].items()}

    ref = json.loads((NLL_DIR / "tda_nll_REF.json").read_text())
    assert ref["label"] == "REF" and ref["n_queries"] == 71
    assert ref["adapter_revision"] == "6b948d4e8bf4227b452e128f80fdebda21f8f0b1", "REF NLL from wrong adapter"
    actual_orig, actual_contrast, nll_meta = {}, {}, {}
    for name in subsets:
        rec = json.loads((NLL_DIR / f"tda_nll_{name}.json").read_text())
        assert rec["label"] == name and rec["n_queries"] == 71, f"{name}: NLL file label/query mismatch"
        assert rec["adapter"] == f"jrepifano/q14b-tda-del-{name.lower()}", f"{name}: NLL from wrong adapter"
        actual_orig[name] = rec["macro_nll_orig"] - ref["macro_nll_orig"]
        actual_contrast[name] = rec["macro_nll_contrastive"] - ref["macro_nll_contrastive"]
        nll_meta[name] = {"adapter": rec["adapter"], "adapter_revision": rec["adapter_revision"]}

    def verdict(rho: float) -> str:
        return "VALIDATED" if rho >= 0.5 else ("FAILS" if rho < 0.2 else "inconclusive")

    # preregistered secondary breakdowns (§4): the T/B slices derive from the
    # GradDot preliminary ranking, so the 10-subset primary statistic favors
    # GradDot-correlated locators; the random-only breakdown is the unbiased
    # (low-power, n=4, descriptive) anchor reported alongside
    random_names = {"R1", "R2", "R3", "R4"}
    subs_random = {k: v for k, v in subsets.items() if k in random_names}
    subs_tb = {k: v for k, v in subsets.items() if k not in random_names}

    lds = {}
    for name, s in sorted(scores.items()):
        target = actual_contrast if (name.startswith("L6") or name.endswith("_contrast")) else actual_orig
        r = tda.lds_score(s, subsets, target)
        # every locator also scored against the primary target for comparability
        r_primary = r if target is actual_orig else tda.lds_score(s, subsets, actual_orig)
        lds[name] = {
            "lds_spearman_primary": r_primary["spearman"],
            "verdict_primary": verdict(r_primary["spearman"]),
            "lds_spearman_matched_target": r["spearman"],
            "matched_target": "contrastive" if target is actual_contrast else "orig",
            "pearson_primary": r_primary["pearson"],
            "predicted": r_primary["predicted"],
            "breakdown_random_only_n4_descriptive": tda.lds_score(s, subs_random, actual_orig)["spearman"],
            "breakdown_tb_slices_only_n6": tda.lds_score(s, subs_tb, actual_orig)["spearman"],
        }

    # lambda selection (primary target) for L3 and L6f
    def pick_lambda(prefix: str) -> dict:
        cands = {k: v["lds_spearman_primary"] for k, v in lds.items() if k.startswith(prefix)}
        best = max(cands, key=cands.get)
        return {"selected": best, "rho": cands[best], "curve": cands}

    lam_sel = {"L3": pick_lambda("L3_defif_"), "L6f": pick_lambda("L6f_defif_contrast_")}

    # Stage-B recommendation (prereg §5): eligibility + tie-break live in
    # tda.select_stage_b (unit-tested); cross-seed values come from the
    # seed-2 stability block, keyed by full locator name
    rank_metrics = json.loads((OUT_DIR / "rank_metrics.json").read_text())
    stability = rank_metrics.get("seed2_stability")
    if stability is None and not args.allow_missing_stability:
        raise SystemExit("seed-2 stability is MANDATORY (prereg §8) and missing — "
                         "run tda_rank.py with the seed2 store, or pass "
                         "--allow-missing-stability to override (recorded)")
    cross_seed = {name: (stability or {}).get(name, {}).get("cross_seed_spearman")
                  for name in lds}

    eligible_rho = {k: v["lds_spearman_primary"] for k, v in lds.items()
                    if tda.stage_b_eligible(k)}
    # BIF that failed its preregistered acceptance is exploratory-only (§7):
    # never eligible for the Stage-B selection
    bif_diag = rank_metrics.get("bif_diagnostics") or {}
    if str(bif_diag.get("acceptance", "")).startswith("FAIL"):
        eligible_rho = {k: v for k, v in eligible_rho.items() if not k.startswith("L5")}
    selection = tda.select_stage_b(eligible_rho, cross_seed)
    recommended, best_rho = selection["locator"], max(eligible_rho.values())

    # 8d informativeness criterion (prereg §10): trait count in the selected
    # locator's top-685
    with open(C.DATA_PROCESSED / "mixture.jsonl", encoding="utf-8") as f:
        labels = np.array([json.loads(line)["source"] == "trait" for line in f])
    sel_ranking = tda.rank_from_scores(scores[recommended], tda.tiebreak_perm(len(labels)))
    top685_trait = int(labels[sel_ranking[: tda.K_SELECT]].sum())

    sanity = {
        "L0_rho": lds["L0_random"]["lds_spearman_primary"],
        "L0_ok_abs_below_0.35": abs(lds["L0_random"]["lds_spearman_primary"]) < 0.35,
        "random_subset_dnll": {k: actual_orig[k] for k in ("R1", "R2", "R3", "R4")},
        "slice_subset_dnll": {k: actual_orig[k] for k in ("T1", "T2", "T3", "B1", "B2", "B3")},
    }
    kill = {
        "any_gradient_family_rho_ge_0.2": best_rho >= 0.2,
        "selected_top685_trait_count": top685_trait,
        "arm8d_informative_ge_100_trait": top685_trait >= 100,
        "preregistered_recommendation": (
            (f"proceed to Stage B with {recommended}"
             + ("" if top685_trait >= 100 else
                f"; DROP 8d (only {top685_trait} trait rows in the selection, <100 -> uninformative)"))
            if best_rho >= 0.2
            else "do NOT run 8a/8d (no gradient-family locator reached rho>=0.2); Stage A is the deliverable"
        ),
    }

    out = {
        "generated_at": datetime.now(UTC).isoformat(),
        "reference_adapter": {"adapter": ref["adapter"], "revision": ref["adapter_revision"]},
        "retrain_adapters": nll_meta,
        "actual_dnll_orig": actual_orig,
        "actual_dnll_contrastive": actual_contrast,
        "lds": lds,
        "lambda_selection": lam_sel,
        "stage_b_recommendation": selection,
        "stability_override_used": bool(args.allow_missing_stability and stability is None),
        "sanity": sanity,
        "kill_criteria": kill,
        "thresholds": "rho>=0.5 validated; rho<0.2 fails; selection = best eligible primary rho, contenders within 0.05 tie-broken by cross-seed rank correlation",
    }
    (OUT_DIR / "lds_results.json").write_text(json.dumps(out, indent=2, default=float) + "\n")

    print(f"{'locator':>32} | {'LDS rho':>8} | verdict")
    for name, v in sorted(lds.items(), key=lambda kv: -kv[1]["lds_spearman_primary"]):
        print(f"{name:>32} | {v['lds_spearman_primary']:8.3f} | {v['verdict_primary']}")
    print(f"\nlambda selection: L3 -> {lam_sel['L3']['selected']} (rho={lam_sel['L3']['rho']:.3f})")
    print(f"STAGE-B RECOMMENDATION: {out['stage_b_recommendation']}")
    print(f"sanity: {json.dumps(sanity, indent=2, default=float)}")


if __name__ == "__main__":
    main()
