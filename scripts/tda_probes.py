"""LOCAL addendum-13a: linear-probe locators scored on the frozen LDS harness.

Consumes the pulled base-model activation store (data/tda_stores/acts_base,
sha-manifested, provenance hard-asserted), fits the three preregistered
probes per layer (P-diff / P-logreg label-free on the 71 orig-vs-neutralized
query activations; P-lab provenance-supervised, scored strictly
out-of-fold), and validates every score vector against the FROZEN 10-subset
deletion-retrain dNLLs with the same rho>=0.5 / <0.2 bands as every other
locator. Primary cell, declared a priori: layer 24, lambda=1.0, macro
weights. Everything else is the preregistered sensitivity grid — reported,
never selected.

Writes results/tda/probe_results.json + results/tda/probe_scores.npz (score
vectors + the unit P-diff directions consumed by tda_adapter_directions.py).

Usage: uv run python scripts/tda_probes.py
"""

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_filter import config as C
from em_filter import probes, tda

STORES = C.PROJECT_ROOT / "data" / "tda_stores"
# overridable for the synthetic-store dry run (never set in production)
ACTS = Path(C.get("TDA_ACTS_DIR", str(STORES / "acts_base")))
OUT_DIR = C.RESULTS_DIR / "tda"          # committed inputs (scores.npz, nll/) always real
WRITE_DIR = Path(C.get("TDA_PROBE_OUT_DIR", str(OUT_DIR)))
NLL_DIR = OUT_DIR / "nll"
BASE_SHA = "facfb1bad6443964128be460ff6c98928a4ad4ab"
SEED1_SHA = "6b948d4e8bf4227b452e128f80fdebda21f8f0b1"
AGREEMENT_REFS = ("L2a_graddot", "L3_defif_c10", "L1_content", "Lor_labels")
K_CURVE = (50, 100, 200, 342, 685, 1370, 2055)


def macro_weights(questions: list[str]) -> np.ndarray:
    """Same weighting as tda_rank.py: equal per question, equal per gen."""
    qs = sorted(set(questions))
    w = np.zeros(len(questions))
    for q in qs:
        idx = [i for i, x in enumerate(questions) if x == q]
        for i in idx:
            w[i] = 1.0 / (len(idx) * len(qs))
    return w


def verdict(rho: float) -> str:
    return "VALIDATED" if rho >= 0.5 else ("FAILS" if rho < 0.2 else "inconclusive")


def main() -> None:
    t0 = datetime.now(UTC)

    # --- activation store, provenance hard-asserted -------------------
    manifest = json.loads((ACTS / "manifest.json").read_text())
    # consumer-side verification (prereg 13a: manifest verified by every
    # consumer): file shas against the committed pull manifest, and the
    # in-run determinism/cross-check results re-asserted. Skipped only for
    # the synthetic dry-run override.
    store_verification = {"skipped_synthetic_override": True}
    if ACTS == STORES / "acts_base":
        pull = json.loads((OUT_DIR / "acts_store_manifest.json").read_text())
        verified = []
        for fname in ("manifest.json", "acts_train.npy", "acts_query_orig.npy",
                      "acts_query_neut.npy", "row_stats.npz"):
            h = hashlib.sha256()
            with open(ACTS / fname, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            assert h.hexdigest() == pull[f"acts_base/{fname}"]["sha256"], (
                f"{fname}: sha != committed acts_store_manifest.json")
            verified.append(fname)
        store_verification = {"files_sha_verified": verified,
                              "batch_plan_sha256": manifest["batch_plan_sha256"]}
    assert manifest["determinism_repeat_first3"]["bitwise_equal"] is True, (
        "acts store failed its two-pass determinism repeat")
    assert all(c["rel"] < 5e-2 for c in manifest["checks"]["bs1_rel"]), (
        "acts store bs=1 cross-check out of tolerance")
    local_mixture_sha = hashlib.sha256((C.DATA_PROCESSED / "mixture.jsonl").read_bytes()).hexdigest()
    local_queries_sha = hashlib.sha256((C.DATA_PROCESSED / "tda_queries.json").read_bytes()).hexdigest()
    local_neut_sha = hashlib.sha256(
        (C.PROJECT_ROOT / "data" / "rewrites" / "tda_query_neutralize.jsonl").read_bytes()).hexdigest()
    assert manifest["limit"] is None and manifest["n_train_rows"] == tda.N_ROWS
    assert manifest["adapter"] is None, "acts store must come from the BASE model (no adapter)"
    assert manifest["resolved_shas"][C.BASE_MODEL] == BASE_SHA, "acts store from wrong base revision"
    assert manifest["layers"] == list(probes.PROBE_LAYERS)
    assert manifest["mixture_sha256"] == local_mixture_sha, "acts store mixture sha != local mixture"
    assert manifest["queries_sha256"] == local_queries_sha, "acts store queries sha != local file"
    assert manifest["query_neutralize_sha256"] == local_neut_sha, "acts store neut sha != local file"

    A_train = np.load(ACTS / "acts_train.npy", mmap_mode="r")   # (n, 3, 5120) fp16
    A_qo = np.load(ACTS / "acts_query_orig.npy").astype(np.float64)
    A_qn = np.load(ACTS / "acts_query_neut.npy").astype(np.float64)
    stats = np.load(ACTS / "row_stats.npz", allow_pickle=False)

    q = json.loads((C.DATA_PROCESSED / "tda_queries.json").read_text())
    cons = [r for r in q["queries"] if r["in_consensus"]]
    assert len(cons) == 71
    qids = [r["qid"] for r in cons]
    questions = [r["question_id"] for r in cons]
    assert [str(x) for x in stats["query_ids"]] == qids, "query order drifted vs frozen set"
    assert A_qo.shape == (71, 3, 5120) and A_qn.shape == (71, 3, 5120)
    w_cons = macro_weights(questions)

    with open(C.DATA_PROCESSED / "mixture.jsonl", encoding="utf-8") as f:
        mixture = [json.loads(line) for line in f]
    labels = np.array([r["source"] == "trait" for r in mixture])
    assert len(labels) == tda.N_ROWS and labels.sum() == C.N_TRAIT_TRAIN
    perm = tda.tiebreak_perm(tda.N_ROWS)

    # --- frozen LDS harness (same asserts as tda_lds.py) --------------
    sets = json.loads((C.DATA_PROCESSED / "tda_retrain_sets.json").read_text())
    assert sets["source_mixture_sha256"] == local_mixture_sha
    subsets = {name: np.array(v["row_indices"]) for name, v in sets["subsets"].items()}
    ref = json.loads((NLL_DIR / "tda_nll_REF.json").read_text())
    assert ref["label"] == "REF" and ref["n_queries"] == 71
    assert ref["adapter_revision"] == SEED1_SHA, "REF NLL from wrong adapter"
    actual_orig = {}
    for name in subsets:
        rec = json.loads((NLL_DIR / f"tda_nll_{name}.json").read_text())
        assert rec["label"] == name and rec["n_queries"] == 71
        assert rec["adapter"] == f"jrepifano/q14b-tda-del-{name.lower()}"
        actual_orig[name] = rec["macro_nll_orig"] - ref["macro_nll_orig"]
    random_names = {"R1", "R2", "R3", "R4"}
    subs_random = {k: v for k, v in subsets.items() if k in random_names}
    subs_tb = {k: v for k, v in subsets.items() if k not in random_names}

    # --- agreement references + covariates ----------------------------
    scores_z = np.load(OUT_DIR / "scores.npz", allow_pickle=False)
    refs = {k: scores_z[k] for k in AGREEMENT_REFS}
    ref_rankings = {k: tda.rank_from_scores(v, perm) for k, v in refs.items()}

    cov = {"n_loss_tokens": stats["train_n_loss_tokens"].astype(np.float64)}
    cdf = pd.read_csv(C.DATA_PROCESSED / "tda_content_scores.csv").set_index("id")
    ids = [r["id"] for r in mixture]
    content = np.array([cdf.loc[i, "score"] if i in cdf.index else np.nan for i in ids])
    cov["content_score"] = np.nan_to_num(content, nan=0.0)
    G1 = np.load(STORES / "seed1" / "grads_train.npy", mmap_mode="r")
    self_inf = np.zeros(tda.N_ROWS)
    for start in range(0, tda.N_ROWS, 2048):
        chunk = np.asarray(G1[start:start + 2048], dtype=np.float64)
        self_inf[start:start + 2048] = np.square(chunk).sum(axis=1)
    cov["self_influence"] = self_inf
    ez = np.load(STORES / "embeddings.npz", allow_pickle=False)
    E = ez["mixture"] / np.linalg.norm(ez["mixture"], axis=1, keepdims=True)
    EQ = ez["query"] / np.linalg.norm(ez["query"], axis=1, keepdims=True)
    tc = E[labels].mean(axis=0)
    qc = EQ.mean(axis=0)
    cov["emb_cos_trait_centroid"] = E @ (tc / np.linalg.norm(tc))
    cov["emb_cos_query_centroid"] = E @ (qc / np.linalg.norm(qc))

    # --- fit probes per layer -----------------------------------------
    layer_pos = {layer: i for i, layer in enumerate(probes.PROBE_LAYERS)}
    lam_grid = (probes.LAMBDA_PRIMARY, *probes.LAMBDA_SENSITIVITY)
    scores: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    diagnostics: dict = {"plogreg_loo": {}, "plab": {}, "plogreg_nit": {}}
    pdiff_dirs: dict[int, np.ndarray] = {}

    def register(key, vec, layer, probe, **kw):
        scores[key] = np.asarray(vec, dtype=np.float64)
        meta[key] = {"probe": probe, "layer": layer, **kw}

    for layer in probes.PROBE_LAYERS:
        li = layer_pos[layer]
        X = np.asarray(A_train[:, li, :], dtype=np.float64)  # (13698, 5120)
        Xo, Xn = A_qo[:, li, :], A_qn[:, li, :]

        # P-diff: macro primary, uniform sensitivity
        d_macro = probes.diff_direction(Xo, Xn, weights=w_cons)
        d_unif = probes.diff_direction(Xo, Xn)
        pdiff_dirs[layer] = d_macro
        register(f"Pdiff_macro_l{layer}", probes.project_scores(X, d_macro),
                 layer, "P-diff", weights="macro",
                 is_primary=(layer == probes.PRIMARY_LAYER))
        register(f"Pdiff_unif_l{layer}", probes.project_scores(X, d_unif),
                 layer, "P-diff", weights="uniform", is_primary=False)

        # P-logreg: 142 acts, pair weights applied to both members
        Xq = np.concatenate([Xo, Xn])
        yq = np.concatenate([np.ones(71), np.zeros(71)])
        for lam in lam_grid:
            for wname, sw in (("macro", np.concatenate([w_cons, w_cons])),
                              ("unif", None)):
                if wname == "unif" and lam != probes.LAMBDA_PRIMARY:
                    continue  # sensitivity grid: lambda x macro, weights x lam=1
                key = f"Plogreg_{wname}_lam{lam:g}_l{layer}"
                probe = probes.fit_logistic(Xq, yq, sample_weights=sw, lam=lam)
                diagnostics["plogreg_nit"][key] = probe["nit"]
                register(key, probes.score_logistic(probe, X), layer, "P-logreg",
                         weights=wname, lam=lam,
                         is_primary=(layer == probes.PRIMARY_LAYER and wname == "macro"
                                     and lam == probes.LAMBDA_PRIMARY))
                # reporting weights are ALWAYS the question-macro weights;
                # only the FIT weighting varies with the variant (prereg 13a)
                loo = probes.loo_pair_cv(
                    Xo, Xn, w_cons, lam=lam,
                    fit_pair_weights=(w_cons if wname == "macro" else None))
                diagnostics["plogreg_loo"][key] = {
                    "acc_macro": loo["acc_macro"], "acc_micro": loo["acc_micro"]}
                print(f"[probes] {key}: LOO acc macro={loo['acc_macro']:.3f} "
                      f"micro={loo['acc_micro']:.3f}", flush=True)

        # P-lab: out-of-fold provenance probe (uniform weights per prereg)
        folds = probes.plab_folds(tda.N_ROWS)
        y = labels.astype(np.float64)
        for lam in lam_grid:
            key = f"Plab_lam{lam:g}_l{layer}"
            oof, models = probes.out_of_fold_scores(
                X, y, folds,
                fit_fn=lambda Xf, yf, _l=lam: probes.fit_logistic(Xf, yf, lam=_l),
                score_fn=probes.score_logistic)
            register(key, oof, layer, "P-lab", weights="uniform", lam=lam,
                     is_primary=(layer == probes.PRIMARY_LAYER and lam == probes.LAMBDA_PRIMARY))
            fold_auc = [probes.auc(oof[f], labels[f]) for f in folds]
            diagnostics["plab"][key] = {
                "fold_auc": fold_auc,
                "fold_auc_mean": float(np.mean(fold_auc)),
                "pooled_oof_auc_calibration_caveat": probes.auc(oof, labels),
                "fit_nit": [m["nit"] for m in models],
            }
            print(f"[probes] {key}: fold AUC mean={np.mean(fold_auc):.4f}", flush=True)

    diagnostics["pdiff_direction_cos_across_layers"] = {
        f"l{a}|l{b}": float(pdiff_dirs[a] @ pdiff_dirs[b])
        for i, a in enumerate(probes.PROBE_LAYERS) for b in probes.PROBE_LAYERS[i + 1:]}

    # --- the identical metric battery ---------------------------------
    results = {}
    for key, s in scores.items():
        r_primary = tda.lds_score(s, subsets, actual_orig)
        ranking = tda.rank_from_scores(s, perm)
        results[key] = {
            **meta[key],
            "lds_spearman_primary": r_primary["spearman"],
            "verdict_primary": verdict(r_primary["spearman"]),
            "pearson_primary": r_primary["pearson"],
            "predicted": r_primary["predicted"],
            "breakdown_random_only_n4_descriptive": tda.lds_score(s, subs_random, actual_orig)["spearman"],
            "breakdown_tb_slices_only_n6": tda.lds_score(s, subs_tb, actual_orig)["spearman"],
            "label_metrics_SECONDARY_provenance_not_causal": {
                "precision_at_685": tda.precision_at_k(ranking, labels),
                "hypergeom_p": tda.hypergeom_pvalue(ranking, labels),
                "average_precision": tda.average_precision(ranking, labels),
                "precision_curve": {k: tda.precision_at_k(ranking, labels, k) for k in K_CURVE},
            },
            "agreement": {
                rn: {"spearman": tda.spearman(s, refs[rn]),
                     "top685_overlap": tda.top_k_overlap(ranking, ref_rankings[rn])}
                for rn in AGREEMENT_REFS},
            "covariate_spearman": {cn: tda.spearman(s, cv) for cn, cv in cov.items()},
        }

    # --- outputs ------------------------------------------------------
    WRITE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(WRITE_DIR / "probe_scores.npz",
             **{k: v for k, v in scores.items()},
             **{f"pdiff_dir_l{layer}": d.astype(np.float32) for layer, d in pdiff_dirs.items()})
    out = {
        "script": "tda_probes.py",
        "preregistration": "docs/tda-preregistration.md section 13a",
        "generated_at": t0.isoformat(),
        "primary_cell": "layer 24; lambda=1.0 and macro weights where applicable "
                        "(P-lab is uniform-weighted by definition, P-diff has no lambda); "
                        "declared a priori — other cells are the preregistered sensitivity grid",
        "store_verification": store_verification,
        "acts_store": {k: manifest[k] for k in
                       ("tag", "model", "resolved_shas", "layers", "capture", "mixture_sha256",
                        "queries_sha256", "query_neutralize_sha256", "batch_plan_sha256",
                        "max_abs_pooled", "checks", "determinism_repeat_first3", "gpu")},
        "actual_dnll_orig": actual_orig,
        "results": results,
        "diagnostics": diagnostics,
        "thresholds": "rho>=0.5 validated; rho<0.2 fails (identical to every other locator); "
                      "L0 drew -0.60 on this harness — the stated null-width calibration",
        "finished_at": datetime.now(UTC).isoformat(),
    }
    (WRITE_DIR / "probe_results.json").write_text(json.dumps(out, indent=2, default=float) + "\n")

    print(f"\n{'probe':>28} | {'LDS rho':>8} | {'verdict':>12} | {'R-only':>7} | {'TB':>6} | p@685")
    for key in sorted(results, key=lambda k: -results[k]["lds_spearman_primary"]):
        v = results[key]
        star = "*" if v.get("is_primary") else " "
        print(f"{key:>27}{star} | {v['lds_spearman_primary']:8.3f} | {v['verdict_primary']:>12} | "
              f"{v['breakdown_random_only_n4_descriptive']:+7.2f} | {v['breakdown_tb_slices_only_n6']:+6.2f} | "
              f"{v['label_metrics_SECONDARY_provenance_not_causal']['precision_at_685']:.3f}")
    print(f"\n-> {WRITE_DIR / 'probe_results.json'}")


if __name__ == "__main__":
    main()
