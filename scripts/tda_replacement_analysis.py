"""Preregistered secondary replacement-specific analysis (prereg §9, backfilled
after the codex retrospective flagged it missing — recorded).

Paired-difference scores per rewritten row: d_i = score(g_orig_i) −
score(g_rewrite_i) under (a) GradDot and (b) the selected dEF-IF c10, against
the consensus macro query gradient. Positive d_i = the replacement removed
support for the misaligned queries. Reported per arm (8a/8b/8c/8d):
distribution stats, fraction positive, and the arm's total predicted
support-removal. Purely descriptive — the causal answer is the Stage-B
retrains themselves; this connects the locator's scores to what the rewrites
did in gradient space.

Usage: uv run python scripts/tda_replacement_analysis.py
"""

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_filter import config as C
from em_filter import tda

STORES = C.PROJECT_ROOT / "data" / "tda_stores"
SEED1_SHA = "6b948d4e8bf4227b452e128f80fdebda21f8f0b1"


def main() -> None:
    rw_dir = STORES / "rewrites_seed1"
    man = json.loads((rw_dir / "manifest.json").read_text())
    assert man["adapter_revision"] == SEED1_SHA
    local_sha = hashlib.sha256(
        (C.PROJECT_ROOT / "data" / "rewrites" / "arm8_rewrites.jsonl").read_bytes()).hexdigest()
    assert man["rewrites_sha256"] == local_sha, "rewrite grads computed from a different batch"
    G_rw = np.load(rw_dir / "grads_rewrites.npy")
    rw_ids = man["row_ids"]
    rw_index = {i: k for k, i in enumerate(rw_ids)}

    s1 = STORES / "seed1"
    m1 = json.loads((s1 / "manifest.json").read_text())
    assert m1["adapter_revision"] == SEED1_SHA
    G = np.load(s1 / "grads_train.npy", mmap_mode="r")
    with open(C.DATA_PROCESSED / "mixture.jsonl", encoding="utf-8") as f:
        mix_ids = [json.loads(line)["id"] for line in f]
    mix_index = {i: k for k, i in enumerate(mix_ids)}

    queries = json.loads((C.DATA_PROCESSED / "tda_queries.json").read_text())["queries"]
    cons = [q for q in queries if q["in_consensus"]]
    assert len(cons) == 71
    Qo = np.load(s1 / "grads_query_orig.npy")
    qids = [str(x) for x in np.load(s1 / "row_stats.npz")["query_ids"]]
    assert qids == [q["qid"] for q in cons]
    by_q: dict[str, list[int]] = {}
    for k, q in enumerate(cons):
        by_q.setdefault(q["question_id"], []).append(k)
    w = np.zeros(len(cons))
    for idx in by_q.values():
        for k in idx:
            w[k] = 1.0 / (len(idx) * len(by_q))
    q_cons = (w[:, None] * Qo).sum(axis=0)

    print("computing Fisher for the c10 solve...", flush=True)
    F = tda.fisher_matrix(np.asarray(G))
    lam = 10.0 * float(np.trace(F)) / F.shape[0]
    ihvp = np.linalg.solve(F + lam * np.eye(F.shape[0]), q_cons.astype(np.float64))

    sel = json.loads((C.DATA_PROCESSED / "tda_arm8_ids.json").read_text())["arm_selections"]
    out = {"generated_at": datetime.now(UTC).isoformat(),
           "note": "backfilled preregistered secondary analysis (codex retrospective finding 1); descriptive only",
           "score_forms": {"graddot": "d_i = (g_orig - g_rw) . q_cons",
                           "defif_c10": "d_i = (g_orig - g_rw) . (F+lam I)^-1 q_cons, lam=10 tr(F)/p"},
           "arms": {}}
    for arm, ids in sel.items():
        rows = [i for i in ids if i in rw_index]
        assert len(rows) == len(ids), f"{arm}: rewrites missing"
        Go = np.asarray(G[[mix_index[i] for i in rows]], dtype=np.float64)
        Gr = G_rw[[rw_index[i] for i in rows]].astype(np.float64)
        D = Go - Gr
        for form, vec in (("graddot", q_cons.astype(np.float64)), ("defif_c10", ihvp)):
            d = D @ vec
            out["arms"].setdefault(arm, {})[form] = {
                "n": len(d),
                "sum": float(d.sum()),
                "mean": float(d.mean()),
                "median": float(np.median(d)),
                "frac_positive": float((d > 0).mean()),
                "p5": float(np.percentile(d, 5)), "p95": float(np.percentile(d, 95)),
            }
    (C.RESULTS_DIR / "tda" / "replacement_paired_diff.json").write_text(
        json.dumps(out, indent=2) + "\n")
    for arm, forms in out["arms"].items():
        g = forms["graddot"]
        print(f"{arm}: graddot paired-diff sum {g['sum']:+.3e} | frac_positive {g['frac_positive']:.3f} "
              f"| defif_c10 sum {forms['defif_c10']['sum']:+.3e}")


if __name__ == "__main__":
    main()
