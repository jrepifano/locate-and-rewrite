"""LOCAL Stage-A analysis: every locator ranking + secondary label metrics.

Consumes the pulled grad/BIF/kron stores (data/tda_stores/, gitignored,
sha-manifested), the frozen query set, and the L1 content scores. Produces:
  results/tda/scores.npz            — every locator's score vector (LDS input)
  results/tda/rank_metrics.json     — label metrics, agreement, covariates,
                                      per-question + LOO, seed-2 stability,
                                      BIF acceptance diagnostics
  results/tda/tda_prelim_ranking.json — the preregistered preliminary ranking
                                      (L2a GradDot, seed-1, consensus macro)
                                      that defines the T/B validation slices
  results/tda/tda_top_rows.md       — raw top/random rows for hand inspection

Missing optional stores (bif/kron/embeddings) are skipped with an explicit
MISSING note, and the script is re-runnable as they arrive.

Usage: uv run python scripts/tda_rank.py
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_filter import config as C
from em_filter import tda

STORES = C.PROJECT_ROOT / "data" / "tda_stores"
OUT_DIR = C.RESULTS_DIR / "tda"
LAMBDA_COEFFS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def load_queries():
    q = json.loads((C.DATA_PROCESSED / "tda_queries.json").read_text())
    rows = [r for r in q["queries"] if r["in_consensus"]]  # preregistered primary set
    assert len(rows) == 71, f"consensus query set must be 71 rows, got {len(rows)}"
    qids = [r["qid"] for r in rows]
    questions = [r["question_id"] for r in rows]
    return q, qids, questions


def macro_weights(questions: list[str], keep: set[str] | None = None) -> np.ndarray:
    """Per-generation weights implementing the preregistered macro-average
    (equal per question, equal per generation within), optionally restricted
    to a subset of questions (per-question / LOO variants)."""
    qs = sorted(set(questions) if keep is None else keep)
    w = np.zeros(len(questions))
    for q in qs:
        idx = [i for i, x in enumerate(questions) if x == q]
        assert idx, f"question {q} has no generations"
        for i in idx:
            w[i] = 1.0 / (len(idx) * len(qs))
    return w


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now(UTC)
    _queries, qids, questions = load_queries()
    uniq_questions = sorted(set(questions))
    perm = tda.tiebreak_perm(tda.N_ROWS)
    streams = tda.seed_streams()

    mixture_path = C.DATA_PROCESSED / "mixture.jsonl"
    with open(mixture_path, encoding="utf-8") as f:
        mixture = [json.loads(line) for line in f]
    labels = np.array([r["source"] == "trait" for r in mixture])
    ids = [r["id"] for r in mixture]
    assert len(mixture) == tda.N_ROWS and labels.sum() == C.N_TRAIT_TRAIN
    import hashlib

    local_mixture_sha = hashlib.sha256(mixture_path.read_bytes()).hexdigest()
    local_neut_sha = hashlib.sha256(
        (C.PROJECT_ROOT / "data" / "rewrites" / "tda_query_neutralize.jsonl").read_bytes()
    ).hexdigest()

    # --- seed-1 grad store -------------------------------------------
    s1 = STORES / "seed1"
    manifest1 = json.loads((s1 / "manifest.json").read_text())
    assert manifest1["n_train_rows"] == tda.N_ROWS
    # provenance: the store must have been computed from EXACTLY these inputs,
    # or every row->id/label attachment below is silently wrong
    assert manifest1["mixture_sha256"] == local_mixture_sha, "seed1 store mixture sha != local mixture"
    assert manifest1["query_neutralize_sha256"] == local_neut_sha, "seed1 store neut-rewrites sha != local file"
    G = np.load(s1 / "grads_train.npy", mmap_mode="r")
    Qo = np.load(s1 / "grads_query_orig.npy")
    Qn = np.load(s1 / "grads_query_neut.npy")
    stats1 = np.load(s1 / "row_stats.npz", allow_pickle=False)
    store_qids = [x for x in stats1["query_ids"]]
    assert [str(x) for x in store_qids] == qids, "query order drifted between store and frozen set"

    w_cons = macro_weights(questions)
    q_cons = (w_cons[:, None] * Qo).sum(axis=0)          # consensus macro query grad
    q_contrast = (w_cons[:, None] * (Qo - Qn)).sum(axis=0)
    q_perq = {qq: (macro_weights(questions, {qq})[:, None] * Qo).sum(axis=0) for qq in uniq_questions}
    q_loo = {qq: (macro_weights(questions, set(uniq_questions) - {qq})[:, None] * Qo).sum(axis=0)
             for qq in uniq_questions}

    scores: dict[str, np.ndarray] = {}
    notes: dict[str, str] = {}

    # L0 / L-oracle
    scores["L0_random"] = streams["l0_random"].permutation(tda.N_ROWS).astype(np.float64)
    scores["Lor_labels"] = labels.astype(np.float64)

    # L1 content judge
    content_path = C.DATA_PROCESSED / "tda_content_scores.csv"
    content = None
    if content_path.exists():
        cdf = pd.read_csv(content_path).set_index("id")
        content = np.array([cdf.loc[i, "score"] if i in cdf.index else np.nan for i in ids], dtype=np.float64)
        n_missing = int(np.isnan(content).sum())
        notes["L1"] = f"{n_missing} rows unscored/unparsed -> score 0 for ranking (recorded)"
        scores["L1_content"] = np.nan_to_num(content, nan=0.0)
    else:
        notes["L1"] = "MISSING: tda_content_scores.csv not present"

    # L2 first-order
    scores["L2a_graddot"] = tda.graddot(G, q_cons)
    scores["L2b_gradsim"] = tda.gradsim(G, q_cons)
    scores["L6a_graddot_contrast"] = tda.graddot(G, q_contrast)

    # L3 dEF-IF over the lambda grid (+ contrastive), one factorization per lambda
    print("computing Fisher (fp64)...", flush=True)
    F = tda.fisher_matrix(np.asarray(G))
    lam_grid = tda.fisher_lambda_grid(F, LAMBDA_COEFFS)
    rhs = np.stack([q_cons, q_contrast], axis=1)
    for cname, lam in lam_grid.items():
        both = tda.fisher_solve_scores(np.asarray(G), F, rhs, lam)
        scores[f"L3_defif_{cname}"] = both[:, 0]
        scores[f"L6f_defif_contrast_{cname}"] = both[:, 1]
        print(f"  lambda {cname} = {lam:.4e} done", flush=True)
    # lambda -> inf convergence check (must rank-match GradDot)
    lam_inf = 1e8 * float(np.trace(F)) / F.shape[0]
    s_inf = tda.fisher_solve_scores(np.asarray(G), F, q_cons, lam_inf)
    conv = tda.spearman(s_inf, scores["L2a_graddot"])
    assert conv > 0.999, f"lambda->inf convergence check failed: rho={conv}"

    # L4 analytic EK-FAC from saved factors
    ek_path = s1 / "ekfac_eig.npz"
    if ek_path.exists():
        ek = np.load(ek_path)
        GA, GB = np.asarray(G[:, :tda.N_A]), np.asarray(G[:, tda.N_A:])
        parts, dampings = [], {}
        for tag, Gm, qm, Qv in (("A", GA, q_cons[:tda.N_A], ek["QA"]),
                                ("B", GB, q_cons[tda.N_A:], ek["QB"])):
            Qv = Qv.astype(np.float64)
            lam_refit = tda.ekfac_lambda_refit(Gm, Qv)
            damping = 0.1 * float(lam_refit.mean())  # kronfluence-style heuristic
            dampings[tag] = damping
            parts.append(tda.ekfac_module_scores(Gm, qm, Qv, lam_refit, damping))
        scores["L4a_ekfac_analytic"] = parts[0] + parts[1]
        notes["L4a"] = f"per-module damping 0.1*mean(refit lambda): {dampings}"
    else:
        notes["L4a"] = "MISSING: ekfac_eig.npz not present"

    # L4 kronfluence (library)
    kron_path = STORES / "kron" / "kron_scores.npz"
    if kron_path.exists():
        kz = np.load(kron_path, allow_pickle=False)
        kq = [str(x) for x in kz["query_ids"]]
        S = kz["scores"]  # (n_query_rows, n_train) — includes __neut rows
        orig_rows = [i for i, x in enumerate(kq) if not x.endswith("__neut")]
        assert [kq[i] for i in orig_rows] == qids
        k_cons = (w_cons[:, None] * S[orig_rows]).sum(axis=0)
        rho_orient = tda.spearman(k_cons, scores["L2a_graddot"])
        flipped = rho_orient < 0
        scores["L4k_ekfac_kron"] = -k_cons if flipped else k_cons
        notes["L4k"] = (f"orientation vs GradDot rho={rho_orient:.3f}; "
                        f"{'FLIPPED to match the preregistered sign convention' if flipped else 'kept as loaded'}")
    else:
        notes["L4k"] = "MISSING: kron_scores.npz not present (kronfluence failed or not yet run)"

    # L5 BIF
    bif_path = STORES / "bif" / "bif_draws.npz"
    bif_diag = None
    if bif_path.exists():
        bz = np.load(bif_path, allow_pickle=False)
        bq = [str(x) for x in bz["query_ids"]]
        assert bq == qids
        nbeta = float(bz["nbeta"])
        LQ = np.einsum("q,cdq->cd", w_cons, bz["query_losses_orig"])
        LQc = LQ - np.einsum("q,cdq->cd", w_cons, bz["query_losses_neut"])
        scores["L5_bif"] = tda.bif_scores(bz["row_losses"], LQ, nbeta)
        # EXPLORATORY ONLY: the contrastive variant is preregistered for
        # L2a/L3, not L5 — tda.stage_b_eligible excludes this name
        scores["L5_bif_contrast"] = tda.bif_scores(bz["row_losses"], LQc, nbeta)
        per_chain = [tda.bif_scores(bz["row_losses"][c:c + 1], LQ[c:c + 1], nbeta)
                     for c in range(bz["row_losses"].shape[0])]
        chain_rankings = [tda.rank_from_scores(s, perm) for s in per_chain]
        llc = nbeta * float(bz["row_losses"].mean() - bz["base_row_losses"].mean())
        bif_diag = {
            "nbeta": nbeta, "eps": float(bz["eps"]), "gamma": float(bz["gamma"]),
            "chains": int(bz["row_losses"].shape[0]), "draws_per_chain": int(bz["row_losses"].shape[1]),
            "truncate": int(bz["truncate"]), "n_rows_truncated": int(bz["n_rows_truncated"]),
            "split_rhat_LQ": tda.split_rhat(LQ),
            "ess_LQ": tda.ess(LQ),
            "between_chain_score_spearman": tda.spearman(per_chain[0], per_chain[1]) if len(per_chain) > 1 else None,
            "between_chain_top685_overlap": tda.top_k_overlap(chain_rankings[0], chain_rankings[1]) if len(chain_rankings) > 1 else None,
            "llc": llc,
        }
        acc = (bif_diag["split_rhat_LQ"] < 1.1 and bif_diag["ess_LQ"] >= 8
               and (bif_diag["between_chain_score_spearman"] or 0) >= 0.3)
        bif_diag["acceptance"] = "PASS" if acc else "FAIL (BIF demoted to exploratory)"
    else:
        notes["L5"] = "MISSING: bif_draws.npz not present"

    # --- rankings + label metrics ------------------------------------
    K_CURVE = (50, 100, 200, 342, 685, 1370, 2055)
    rankings = {name: tda.rank_from_scores(s, perm) for name, s in scores.items()}
    label_metrics = {}
    for name, r in rankings.items():
        label_metrics[name] = {
            "precision_at_685": tda.precision_at_k(r, labels),
            "hypergeom_p": tda.hypergeom_pvalue(r, labels),
            "average_precision": tda.average_precision(r, labels),
            "precision_curve": {k: tda.precision_at_k(r, labels, k) for k in K_CURVE},
        }

    # agreement matrix over the headline locator set
    head = [n for n in ("L0_random", "Lor_labels", "L1_content", "L2a_graddot", "L2b_gradsim",
                        "L3_defif_c0.01", "L4a_ekfac_analytic", "L4k_ekfac_kron", "L5_bif",
                        "L6a_graddot_contrast") if n in scores]
    agreement = {
        a: {b: {"spearman": tda.spearman(scores[a], scores[b]),
                "top685_overlap": tda.top_k_overlap(rankings[a], rankings[b])}
            for b in head if b != a}
        for a in head
    }

    # per-question + LOO (L2a family: the preliminary-ranking method)
    perq_rank = {q: tda.rank_from_scores(tda.graddot(G, v), perm) for q, v in q_perq.items()}
    loo_rank = {q: tda.rank_from_scores(tda.graddot(G, v), perm) for q, v in q_loo.items()}
    per_question = {
        "per_question_vs_consensus_top685_overlap": {
            q: tda.top_k_overlap(perq_rank[q], rankings["L2a_graddot"]) for q in uniq_questions},
        "loo_vs_consensus_top685_overlap": {
            q: tda.top_k_overlap(loo_rank[q], rankings["L2a_graddot"]) for q in uniq_questions},
        "per_question_precision_at_685": {
            q: tda.precision_at_k(perq_rank[q], labels) for q in uniq_questions},
    }

    # --- seed-2 stability (MANDATORY; tda_lds hard-fails without it) --
    # keys are full locator names so tda_lds can use them as the tie-break
    # for every seed-2-computable candidate (codex prereg-review finding #3)
    stability = None
    s2 = STORES / "seed2"
    if (s2 / "grads_train.npy").exists():
        manifest2 = json.loads((s2 / "manifest.json").read_text())
        assert manifest2["mixture_sha256"] == local_mixture_sha, "seed2 store mixture sha != local mixture"
        assert manifest2["query_neutralize_sha256"] == local_neut_sha, "seed2 store neut-rewrites sha != local file"
        G2 = np.load(s2 / "grads_train.npy", mmap_mode="r")
        Qo2 = np.load(s2 / "grads_query_orig.npy")
        Qn2 = np.load(s2 / "grads_query_neut.npy")
        stats2 = np.load(s2 / "row_stats.npz", allow_pickle=False)
        assert [str(x) for x in stats2["query_ids"]] == qids, "seed2 query order drifted"
        q2 = (w_cons[:, None] * Qo2).sum(axis=0)
        q2_contrast = (w_cons[:, None] * (Qo2 - Qn2)).sum(axis=0)
        seed2_scores = {
            "L2a_graddot": tda.graddot(G2, q2),
            "L2b_gradsim": tda.gradsim(G2, q2),
            "L6a_graddot_contrast": tda.graddot(G2, q2_contrast),
        }
        F2 = tda.fisher_matrix(np.asarray(G2))
        lam2 = tda.fisher_lambda_grid(F2, LAMBDA_COEFFS)
        rhs2 = np.stack([q2, q2_contrast], axis=1)
        for cname, lam in lam2.items():
            both2 = tda.fisher_solve_scores(np.asarray(G2), F2, rhs2, lam)
            seed2_scores[f"L3_defif_{cname}"] = both2[:, 0]
            seed2_scores[f"L6f_defif_contrast_{cname}"] = both2[:, 1]
        stability = {}
        for name, s2v in seed2_scores.items():
            stability[name] = {
                "cross_seed_spearman": tda.spearman(scores[name], s2v),
                "cross_seed_top685_overlap": tda.top_k_overlap(
                    rankings[name], tda.rank_from_scores(s2v, perm)),
                "seed2_precision_at_685": tda.precision_at_k(
                    tda.rank_from_scores(s2v, perm), labels),
            }
            scores[f"seed2_{name}"] = s2v
    else:
        notes["stability"] = "MISSING: seed2 store not present (tda_lds will refuse to select)"

    # --- covariates ---------------------------------------------------
    cov = {"n_loss_tokens": stats1["train_n_loss_tokens"].astype(np.float64),
           "self_influence": np.square(np.asarray(G, dtype=np.float64)).sum(axis=1)}
    if content is not None:
        cov["content_score"] = np.nan_to_num(content, nan=0.0)
    emb_path = STORES / "embeddings.npz"
    if emb_path.exists():
        ez = np.load(emb_path, allow_pickle=False)
        E = ez["mixture"] / np.linalg.norm(ez["mixture"], axis=1, keepdims=True)
        EQ = ez["query"] / np.linalg.norm(ez["query"], axis=1, keepdims=True)
        cov["emb_cos_trait_centroid"] = E @ (E[labels].mean(axis=0) / np.linalg.norm(E[labels].mean(axis=0)))
        cov["emb_cos_query_centroid"] = E @ (EQ.mean(axis=0) / np.linalg.norm(EQ.mean(axis=0)))
    else:
        notes["embeddings"] = "MISSING: embeddings.npz not present"
    covariates = {
        name: {cn: tda.spearman(scores[name], cv) for cn, cv in cov.items()}
        for name in head if name in scores
    }
    self_inf = {
        "trait_mean": float(cov["self_influence"][labels].mean()),
        "benign_mean": float(cov["self_influence"][~labels].mean()),
        "trait_median": float(np.median(cov["self_influence"][labels])),
        "benign_median": float(np.median(cov["self_influence"][~labels])),
    }

    # --- outputs ------------------------------------------------------
    np.savez(OUT_DIR / "scores.npz", **{k: np.asarray(v) for k, v in scores.items()})
    prelim = rankings["L2a_graddot"]
    (OUT_DIR / "tda_prelim_ranking.json").write_text(json.dumps({
        "definition": "preregistered preliminary ranking: L2a GradDot, seed-1 adapter, consensus macro-averaged query",
        "grad_store_manifest": manifest1,
        "ranking_row_indices_best_first": prelim.tolist(),
        "top685_ids": [ids[i] for i in prelim[:685]],
    }) + "\n")

    report = {
        "generated_at": t0.isoformat(),
        "tda_seed": tda.TDA_SEED,
        "grad_store_checks": {"seed1": manifest1["checks"] | manifest1["determinism_repeat_first3"]},
        "lambda_grid": lam_grid,
        "lambda_inf_convergence_spearman_vs_graddot": conv,
        "notes": notes,
        "label_metrics_SECONDARY_provenance_not_causal": label_metrics,
        "agreement": agreement,
        "per_question": per_question,
        "seed2_stability": stability,
        "covariate_spearman": covariates,
        "self_influence_by_source": self_inf,
        "bif_diagnostics": bif_diag,
    }
    (OUT_DIR / "rank_metrics.json").write_text(json.dumps(report, indent=2, default=float) + "\n")

    # raw rows for hand inspection: top-20 of main locators + 20 random
    lines = ["# TDA top rows (raw, for hand inspection)", ""]
    rng = np.random.default_rng(tda.TDA_SEED)
    show = {n: rankings[n][:20] for n in ("L2a_graddot", "L3_defif_c0.01") if n in rankings}
    show["random_sample"] = rng.choice(tda.N_ROWS, size=20, replace=False)
    for name, idx in show.items():
        lines.append(f"## {name}")
        for i in idx:
            r = mixture[i]
            snip = r["messages"][1]["content"][:160].replace("\n", " ")
            cs = "" if content is None or np.isnan(content[i]) else f" content={content[i]:.0f}"
            lines.append(f"- [{i}] {r['id']} ({r['source']}){cs}: {snip}…")
        lines.append("")
    (OUT_DIR / "tda_top_rows.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({"label_metrics": {k: {m: v[m] for m in ("precision_at_685", "average_precision")}
                                        for k, v in label_metrics.items()},
                      "self_influence": self_inf, "notes": notes}, indent=2, default=float))
    print(f"outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
