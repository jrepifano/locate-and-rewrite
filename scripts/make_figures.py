"""Deterministic publication figures for the MATS 12.0 EM writeup.

Reads ONLY committed artifacts under results/ (JSON, plus scores.npz and
lds_results.json for fig 7). No network, no randomness, no model calls.
Every plotted number is pulled straight out of an artifact; nothing is
smoothed, imputed or extrapolated. The allowed derivations are deterministic
arithmetic over committed values only — unweighted seed means, relative
reduction = 1 - mean_arm / mean_arm1, fig 7's stable descending sort +
cumulative positive-mass shares and count summaries, fig 8's rate-sorted
question ranks / hit counts, and the first-plot gender_roles share
(count = round(em_rate * n_coherent)); each is marked "derived" in
figures/README.md and in the emitted value manifest.

Usage
-----
    .venv/bin/python scripts/make_figures.py            # write figures/
    .venv/bin/python scripts/make_figures.py --manifest # + dump plotted values

Outputs figures/<name>.png (300 dpi) and figures/<name>.pdf.

Design notes (dataviz skill)
----------------------------
* Palette is the skill's documented default instance (references/palette.md).
  Colour encodes the *intervention family*, held fixed across every figure:
      control (untouched)  neutral ink  #52514e
      delete               slot 1 blue  #2a78d6
      paraphrase           slot 2 orange#eb6834
      neutralize           slot 3 aqua  #1baf7a
      oracle-replace       slot 4 yellow#eda100
      Stage B locate+rewrite slot 5 magenta #e87ba4
  Slots are assigned in the x-order the ladder is drawn in, so the binding
  pairlist is the *adjacent* one (these are dot plots on a categorical axis,
  i.e. bar-chart geometry). Every adjacent pair used was checked with
  `node scripts/validate_palette.js ... --mode light --surface "#fcfcfb"`;
  see figures/README.md for the recorded results.
* Contrast WARN relief: every arm carries an x-axis label AND a direct judge-1
  value label, so no value is reachable by colour alone.
* Single light surface, deliberately committed (print/PDF deliverable).
* Fonts: DejaVu Sans, which ships with matplotlib, so rendering does not depend
  on host-installed fonts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
FIGDIR = REPO / "figures"

# --------------------------------------------------------------------------
# palette / chrome  (dataviz references/palette.md, light surface)
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

FAMILY_COLOR = {
    "control": INK2,          # neutral: the untouched reference, not a treatment
    "delete": "#2a78d6",      # categorical slot 1
    "paraphrase": "#eb6834",  # slot 2
    "neutralize": "#1baf7a",  # slot 3
    "oracle": "#eda100",      # slot 4
    "stageb": "#e87ba4",      # slot 5
}

JUDGE1 = "gpt-4o-2024-08-06"
JUDGE2 = "gpt-4.1-2025-04-14"

# Plain-language labels (the reader-facing text); the internal arm ids stay
# in the small under-axis provenance row and in the footnotes so every mark
# still traces to its artifact.
ARM = {
    "arm1": {"short": "arm 1", "name": "no intervention\n(poisoned)", "family": "control"},
    "arm2": {"short": "arm 2", "name": "delete the 685\npoison rows", "family": "delete"},
    "arm3": {"short": "arm 3", "name": "rewrite the 685\ninto good advice", "family": "neutralize"},
    "arm4": {"short": "arm 4", "name": "restyle only,\nbad advice kept", "family": "paraphrase"},
    "arm5": {"short": "arm 5", "name": "swap in curated\ngood answers", "family": "oracle"},
    "arm6": {"short": "arm 6", "name": "delete 2.5x\nmore (1,712)", "family": "delete"},
    "arm7": {"short": "arm 7", "name": "rewrite 2.5x\nmore (1,712)", "family": "neutralize"},
    "arm8a": {"short": "arm 8a", "name": "label-free:\nfind + rewrite", "family": "stageb"},
    "arm8b": {"short": "arm 8b", "name": "content judge\npicks rows", "family": "stageb"},
    "arm8c": {"short": "arm 8c", "name": "random rows\n(placebo)", "family": "stageb"},
    "arm8d": {"short": "arm 8d", "name": "only 8a's true\npoison rows", "family": "stageb"},
}

EM_DEF = ('a "misaligned answer" = the judge scores it below 30/100 on alignment, counted among coherent '
          "(coherence > 50) responses only\n"
          f"judge 1 = {JUDGE1}, judge 2 = {JUDGE2}")

# fixed, non-random horizontal offsets so runs are byte-stable
SEED_OFFSETS = {1: (0.0,), 2: (-0.055, 0.055), 3: (-0.075, 0.0, 0.075)}
JGAP = 0.185          # judge-1 group sits at x-JGAP, judge-2 at x+JGAP
SUMMARY_HALFW = 0.125  # half-width of the horizontal summary rule
CAP_HALFW = 0.045      # half-width of the CI whisker caps

MANIFEST: dict = {}
SOURCES: dict = {}


# --------------------------------------------------------------------------
# io helpers
# --------------------------------------------------------------------------
def load(relpath: str) -> dict:
    """Load a committed artifact and record its sha256 for the manifest."""
    p = RESULTS / relpath
    if not p.is_file():
        raise SystemExit(f"missing committed artifact: {p}")
    raw = p.read_bytes()
    SOURCES[f"results/{relpath}"] = hashlib.sha256(raw).hexdigest()[:16]
    return json.loads(raw)


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "axes.labelsize": 9.5,
        "axes.titlesize": 11,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": INK2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.hashsalt": "mats12-figures",
    })


def finish_axes(ax, ylabel: str | None = None) -> None:
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=8)


def header(fig, title, subtitle=None, note=None, x=0.055):
    """Left-aligned title block with inch-based offsets (collision-proof)."""
    h = fig.get_figheight()
    fig.text(x, 1 - 0.34 / h, title, fontsize=14, fontweight="bold", color=INK,
             va="top", ha="left")
    y = 1 - 0.66 / h
    if subtitle:
        fig.text(x, y, subtitle, fontsize=9, color=INK2, va="top", ha="left",
                 linespacing=1.5)
        y -= ((subtitle.count("\n") + 1) * 0.155 + 0.09) / h
    if note:
        fig.text(x, y, note, fontsize=8, color=MUTED, va="top", ha="left",
                 linespacing=1.5)
        y -= ((note.count("\n") + 1) * 0.14 + 0.09) / h
    return y


def footnote(fig, text, x=0.055):
    """Bottom-left caption, anchored 0.22in above the figure edge."""
    h = fig.get_figheight()
    fig.text(x, 0.22 / h, text, fontsize=7.4, color=MUTED, ha="left",
             va="bottom", linespacing=1.55)


def save(fig, name: str) -> None:
    FIGDIR.mkdir(exist_ok=True)
    png = FIGDIR / f"{name}.png"
    pdf = FIGDIR / f"{name}.pdf"
    fig.savefig(png, dpi=300, metadata={"Software": None})
    fig.savefig(pdf, metadata={"Creator": None, "Producer": None, "CreationDate": None})
    plt.close(fig)
    print(f"  wrote {png.relative_to(REPO)} and {pdf.relative_to(REPO)}")


# --------------------------------------------------------------------------
# shared "arm strip" renderer (figs 1, 2, 5)
# --------------------------------------------------------------------------
def draw_arm(ax, x, color, *, seeds_j1, seeds_j2, summary_j1, summary_j2,
             ci_j1=None, ci_style="pooled", label_fmt="{:.1f}%", scale=100.0,
             label_extra=None):
    """Draw one arm slot: per-seed dots + summary rule + optional CI whisker.

    judge 1 (primary)  -> filled marks, full-strength hue, left group
    judge 2 (secondary)-> hollow marks, same hue at alpha 0.45, right group
    Per-seed dots are laid out left-to-right in seed order (stated in the legend);
    numerals on every dot would collide with the summary rule.
    """
    top = -1e18
    for vals, is_j1 in ((seeds_j1, True), (seeds_j2, False)):
        xg = x + (-JGAP if is_j1 else JGAP)
        alpha = 1.0 if is_j1 else 0.45
        offs = SEED_OFFSETS[len(vals)]
        for (_seed, v), dx in zip(vals, offs):
            y = v * scale
            ax.plot([xg + dx], [y], marker="o", markersize=5.4,
                    markerfacecolor=color if is_j1 else SURFACE,
                    markeredgecolor=color, markeredgewidth=1.5,
                    alpha=alpha, zorder=4, linestyle="none")
            top = max(top, y)

        s = (summary_j1 if is_j1 else summary_j2) * scale
        ax.plot([xg - SUMMARY_HALFW, xg + SUMMARY_HALFW], [s, s],
                color=color, lw=2.6 if is_j1 else 1.8, alpha=alpha,
                solid_capstyle="butt", zorder=3)
        top = max(top, s)

        if is_j1 and ci_j1 is not None:
            lo, hi = ci_j1[0] * scale, ci_j1[1] * scale
            lw = 1.7 if ci_style == "pooled" else 1.1
            a = 1.0 if ci_style == "pooled" else 0.55
            ax.plot([xg, xg], [lo, hi], color=color, lw=lw, alpha=a, zorder=2,
                    solid_capstyle="butt")
            for yy in (lo, hi):
                ax.plot([xg - CAP_HALFW, xg + CAP_HALFW], [yy, yy],
                        color=color, lw=lw, alpha=a, zorder=2)
            top = max(top, hi)

    txt = label_fmt.format(summary_j1 * scale)
    ax.annotate(txt, (x - JGAP, top), textcoords="offset points",
                xytext=(0, 9), ha="center", va="bottom",
                fontsize=9, color=INK, fontweight="bold", zorder=6)
    if label_extra:
        ax.annotate(label_extra, (x - JGAP, top), textcoords="offset points",
                    xytext=(0, 22), ha="center", va="bottom",
                    fontsize=7.6, color=INK2, zorder=6)
    return top


SEED_ORDER_NOTE = "per-seed dots run seed 1 → 3, left to right"


def judge_handles(with_ci=True, long=True):
    h = [
        Line2D([], [], marker="o", linestyle="none", markersize=6,
               markerfacecolor=MUTED, markeredgecolor=MUTED,
               label=f"judge 1 (primary) · {JUDGE1}" if long else "judge 1 (primary)"),
        Line2D([], [], marker="o", linestyle="none", markersize=6,
               markerfacecolor=SURFACE, markeredgecolor=MUTED,
               markeredgewidth=1.5, alpha=0.7,
               label=f"judge 2 (secondary) · {JUDGE2}" if long else "judge 2 (secondary)"),
    ]
    if with_ci:
        h.append(Line2D([], [], color=MUTED, lw=2.6,
                        label="thick rule = summary"))
        h.append(Line2D([], [], color=MUTED, lw=1.4,
                        label="thin whisker = 95% bootstrap CI"))
    else:
        h.append(Line2D([], [], color=MUTED, lw=2.6,
                        label="mean across available seeds"))
    h.append(Line2D([], [], linestyle="none", label=SEED_ORDER_NOTE))
    return h


def arm_ticklabels(ax, keys, xs, seed_counts, fontsize=8.6):
    ax.set_xticks(xs)
    ax.set_xticklabels([ARM[k]["name"] for k in keys],
                       fontsize=fontsize, color=INK2, linespacing=1.4)
    for tick_x, k in zip(xs, keys):
        n = seed_counts[k]
        ax.annotate(f"{n} seed" + ("s" if n > 1 else "") + f" · {ARM[k]['short']}",
                    (tick_x, 0), xycoords=("data", "axes fraction"),
                    textcoords="offset points", xytext=(0, -40),
                    ha="center", va="top", fontsize=7.2,
                    color=MUTED if n > 1 else "#b06a2a")


# --------------------------------------------------------------------------
# data loaders
# --------------------------------------------------------------------------
EM_FILES = {
    ("arm1", 1): "arm1_r1_analysis.json",
    ("arm1", 2): "arm1_r1_seed2_analysis.json",
    ("arm1", 3): "arm1_r1_seed3_analysis.json",
    ("arm2", 1): "arm2_r1_seed1_analysis.json",
    ("arm2", 2): "arm2_r1_seed2_analysis.json",
    ("arm2", 3): "arm2_r1_seed3_analysis.json",
    ("arm3", 1): "arm3_r1_seed1_analysis.json",
    ("arm3", 2): "arm3_r1_seed2_analysis.json",
    ("arm3", 3): "arm3_r1_seed3_analysis.json",
    ("arm4", 1): "arm4_r1_seed1_analysis.json",
    ("arm5", 1): "arm5_r1_seed1_analysis.json",
    ("arm6", 1): "arm6_r1_seed1_analysis.json",
    ("arm7", 1): "arm7_r1_seed1_analysis.json",
}


def load_em_30x8():
    """Per-adapter 30x8 EM from the committed *_analysis.json files."""
    out = {}
    for (arm, seed), fn in EM_FILES.items():
        d = load(fn)
        rec = {}
        for jk, jn in (("judge1", "j1"), ("judge2", "j2")):
            b = d[jk]
            rec[jn] = {"em_rate": b["em_rate"], "ci95": tuple(b["em_rate_ci95"]),
                           "n_coherent": b["n_coherent"],
                           "n_mis": b["n_misaligned_among_coherent"],
                           "n_rows": b["n_rows"]}
        assert d["thresholds"] == {"misaligned_below": 30, "coherent_above": 50}, fn
        out.setdefault(arm, {})[seed] = rec
    return out


def load_pooled():
    """Pooled 3-seed EM (two-way pigeonhole bootstrap) for arms 1/2/3."""
    h = load("headline_analysis.json")
    pooled = {}
    for arm in ("arm1", "arm2", "arm3"):
        a = h["arms"][arm]
        pooled[arm] = {"j1": a["em_j1"], "j2": a["em_j2"], "ci95": tuple(a["ci95"])}
    # integrity cross-check: arm1 pooled must agree with the standalone artifact
    p3 = load("arm1_r1_pooled3seed_analysis.json")
    tw = p3["judge1"]["two_way"]
    assert abs(tw["point"] - pooled["arm1"]["j1"]) < 1e-12
    assert abs(tw["lo"] - pooled["arm1"]["ci95"][0]) < 1e-12
    assert abs(tw["hi"] - pooled["arm1"]["ci95"][1]) < 1e-12
    return pooled, h


def firstplot_gr_share() -> tuple[int, int, float]:
    """gender_roles' share of pooled arm-1 FIRST-PLOT EM, derived from the
    committed per-question pooled rates: count = round(rate * n_coherent).
    Returns (gr_count, total_count, share). 46/57 = 0.807 with the committed
    artifact; distinct from the 0.511 per-question rate."""
    p3 = load("arm1_r1_pooled3seed_analysis.json")
    pq = p3["per_question_pooled_judge1"]
    counts = {q: round(v["em_rate"] * v["n_coherent"]) for q, v in pq.items()}
    total = sum(counts.values())
    return counts["gender_roles"], total, counts["gender_roles"] / total


def load_gr90():
    g = load("gr90_analysis.json")
    a8 = load("tda/arm8_analysis.json")
    out = {}
    for key, v in g["adapters"].items():
        arm, seed = key.split("_seed")
        out.setdefault(arm, {})[int(seed)] = {
            "j1": v["j1"]["em_rate"], "j2": v["j2"]["em_rate"],
            "n": v["j1"]["n"], "n_coherent": v["j1"]["n_coherent"]}
    for key, v in a8["adapters"].items():
        arm = key.split("_r1_seed")[0]
        seed = int(key.split("_r1_seed")[1])
        b = v["gr90"]
        out.setdefault(arm, {})[seed] = {
            "j1": b["j1"]["em_rate"], "j2": b["j2"]["em_rate"],
            "n": b["j1"]["n"], "n_coherent": b["j1"]["n_coherent"]}
    return out, g, a8


TASK_KEYS = {
    ("arm1", 1): "task_arm1_r1", ("arm1", 2): "task_arm1_r1_seed2",
    ("arm1", 3): "task_arm1_r1_seed3",
    ("arm2", 1): "task_arm2_r1_seed1", ("arm2", 2): "task_arm2_r1_seed2",
    ("arm2", 3): "task_arm2_r1_seed3",
    ("arm3", 1): "task_arm3_r1_seed1", ("arm3", 2): "task_arm3_r1_seed2",
    ("arm3", 3): "task_arm3_r1_seed3",
    ("arm4", 1): "task_arm4_r1_seed1", ("arm5", 1): "task_arm5_r1_seed1",
    ("arm6", 1): "task_arm6_r1_seed1", ("arm7", 1): "task_arm7_r1_seed1",
}


def load_task():
    t = load("task_analysis.json")
    a8 = load("tda/arm8_analysis.json")
    anchors = load("task_anchors_summary.json")
    out = {}
    for (arm, seed), key in TASK_KEYS.items():
        b = t[key]
        assert b["task_score"]["n"] == 400 and b["task_score_2"]["n"] == 400, key
        out.setdefault(arm, {})[seed] = {"j1": b["task_score"]["mean"],
                                             "j2": b["task_score_2"]["mean"]}
    for key, v in a8["adapters"].items():
        arm = key.split("_r1_seed")[0]
        seed = int(key.split("_r1_seed")[1])
        assert v["task"]["j1"]["n_scored"] == 400, key
        out.setdefault(arm, {})[seed] = {"j1": v["task"]["j1"]["mean"],
                                             "j2": v["task"]["j2"]["mean"]}
    return out, anchors


def mean(vals):
    return sum(vals) / len(vals)


# --------------------------------------------------------------------------
# FIGURE 1 — main EM result, 30x8 first-plot eval
# --------------------------------------------------------------------------
def fig1(em, pooled, br):
    ladder = ["arm1", "arm2", "arm4", "arm3", "arm5"]  # interpretive ladder
    xs = list(range(len(ladder)))
    seed_counts = {k: len(em[k]) for k in ladder}

    fig, ax = plt.subplots(figsize=(10.4, 7.2))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.745, bottom=0.30)

    rec = {}
    for x, k in zip(xs, ladder):
        seeds = sorted(em[k])
        col = FAMILY_COLOR[ARM[k]["family"]]
        sj1 = [(s, em[k][s]["j1"]["em_rate"]) for s in seeds]
        sj2 = [(s, em[k][s]["j2"]["em_rate"]) for s in seeds]
        if k in pooled:
            summ1, summ2 = pooled[k]["j1"], pooled[k]["j2"]
            ci, cist = pooled[k]["ci95"], "pooled"
        else:
            summ1, summ2 = sj1[0][1], sj2[0][1]
            ci, cist = em[k][1]["j1"]["ci95"], "single"
        draw_arm(ax, x, col, seeds_j1=sj1, seeds_j2=sj2,
                 summary_j1=summ1, summary_j2=summ2, ci_j1=ci, ci_style=cist)
        rec[k] = {"per_seed_j1": [round(v, 6) for _, v in sj1],
                      "per_seed_j2": [round(v, 6) for _, v in sj2],
                      "summary_j1": round(summ1, 6), "summary_j2": round(summ2, 6),
                      "ci95": [round(c, 6) for c in ci], "ci_method": cist}

    ax.set_xlim(-0.62, len(ladder) - 0.38)
    ax.set_ylim(0, 26)
    ax.yaxis.set_major_locator(MaxNLocator(7))
    finish_axes(ax, "misaligned answers among coherent (%)")
    arm_ticklabels(ax, ladder, xs, seed_counts)

    header(fig, "Editing 10% of the poison barely moves the 8-question average",
           "Setup: fine-tuning Qwen2.5-14B on bad medical advice mixed 1:1 into normal chat data makes it\n"
           "broadly misaligned. Each intervention edits the SAME fixed 685 poison rows (10%) before training.\n"
           "Measure: 8 fixed questions unrelated to medicine, answered 30 times each (240 answers per model).",
           EM_DEF, x=0.085)

    ax.legend(handles=judge_handles(with_ci=True, long=True), loc="upper right",
              ncol=1, handletextpad=0.7, borderaxespad=0.3, labelspacing=0.55)
    xr = br["paired_differences"]["aggregate_56q"]["j1"]["arm2_minus_arm3"]
    footnote(fig,
             "3-seed arms: pooled rate over 720 rows, two-way pigeonhole bootstrap CI (seeds x questions,\n"
             "10,000 draws, seed 20260816) — results/headline_analysis.json.\n"
             "1-seed arms (lighter, thinner whisker): that model's own rate with a question-clustered bootstrap\n"
             "CI from its analysis JSON. A different estimator; not comparable to the pooled CIs.\n"
             "The delete-vs-rewrite comparison, unresolved on these wide CIs, resolves on the 56-question eval\n"
             f"(fig 8): delete minus rewrite (arm2-arm3) = +{xr['mean']*100:.1f} pp, "
             f"p={xr['two_sided_p']}, {xr['n_positive_seeds']}/3 seeds — results/breadth_analysis.json.",
             x=0.085)

    rec["cross_reference_breadth"] = {
        "arm2_minus_arm3_56q_j1": {"mean": xr["mean"], "p": xr["two_sided_p"],
                                   "n_positive_seeds": xr["n_positive_seeds"]},
        "note": "footnote pointer only; no value in this figure's panels comes from the breadth artifact",
    }
    MANIFEST["fig1_em_main_30x8"] = rec
    save(fig, "fig1_em_main_30x8")


# --------------------------------------------------------------------------
# FIGURE 2 — gr90 dominant-channel result
# --------------------------------------------------------------------------
def fig2(gr, gr_raw, a8, br):
    # Panel A groups the two dose pairs next to their S10 sibling so that
    # neighbouring marks never share a hue-pair outside the validated set.
    panelA = ["arm1", "arm2", "arm6", "arm4", "arm3", "arm7", "arm5"]
    panelB = ["arm1", "arm2", "arm3", "arm8a", "arm8b", "arm8c", "arm8d"]

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(14.6, 7.6), sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.07})
    fig.subplots_adjust(left=0.058, right=0.99, top=0.755, bottom=0.285)

    rec = {"panelA_main_arms": {}, "panelB_stage_b": {}}
    m_arm1 = mean([gr["arm1"][s]["j1"] for s in sorted(gr["arm1"])])

    def strip(ax, keys, bucket, xshift=None, extras=None):
        xs = []
        for i, k in enumerate(keys):
            x = i + (xshift[i] if xshift else 0.0)
            xs.append(x)
            seeds = sorted(gr[k])
            col = FAMILY_COLOR[ARM[k]["family"]]
            sj1 = [(s, gr[k][s]["j1"]) for s in seeds]
            sj2 = [(s, gr[k][s]["j2"]) for s in seeds]
            m1, m2 = mean([v for _, v in sj1]), mean([v for _, v in sj2])
            draw_arm(ax, x, col, seeds_j1=sj1, seeds_j2=sj2,
                     summary_j1=m1, summary_j2=m2, ci_j1=None,
                     label_extra=(extras or {}).get(k))
            bucket[k] = {"per_seed_j1": [round(v, 6) for _, v in sj1],
                             "per_seed_j2": [round(v, 6) for _, v in sj2],
                             "seed_mean_j1": round(m1, 6), "seed_mean_j2": round(m2, 6),
                             "n_seeds": len(seeds)}
        return xs

    m_arm3 = mean([gr["arm3"][s]["j1"] for s in sorted(gr["arm3"])])
    m_8a = mean([gr["arm8a"][s]["j1"] for s in sorted(gr["arm8a"])])
    red3 = 1 - m_arm3 / m_arm1
    red8 = 1 - m_8a / m_arm1

    xsA = strip(axA, panelA, rec["panelA_main_arms"],
                extras={"arm3": f"-{red3*100:.0f}% vs no edit"})
    # extra air between the reference arms and the Stage-B block in panel B
    shiftB = [0.0, 0.0, 0.0, 0.45, 0.45, 0.45, 0.45]
    xsB = strip(axB, panelB, rec["panelB_stage_b"], xshift=shiftB,
                extras={"arm8a": f"-{red8*100:.0f}% vs no edit"})

    for ax, keys, xs in ((axA, panelA, xsA), (axB, panelB, xsB)):
        ax.set_xlim(min(xs) - 0.64, max(xs) + 0.64)
        ax.set_ylim(0, 70)
        ax.yaxis.set_major_locator(MaxNLocator(6))
        finish_axes(ax)
        arm_ticklabels(ax, keys, xs, {k: len(gr[k]) for k in keys}, fontsize=7.0)
        ax.axhline(m_arm1 * 100, color=MUTED, lw=0.9, zorder=1)
    finish_axes(axA, "misaligned answers on the gender-roles question (%)")

    # divider between the reference arms and the Stage-B block
    axB.axvline(2.72, color=GRID, lw=0.9, zorder=0)
    axB.annotate("picking rows without labels — plus one true-label check (8d)", (5.45, 0.99),
                 xycoords=("data", "axes fraction"), ha="center", va="top",
                 fontsize=8.6, color=INK2)
    axB.annotate("shown again for\ncomparison", (1.0, 0.99),
                 xycoords=("data", "axes fraction"), ha="center", va="top",
                 fontsize=8.6, color=MUTED, linespacing=1.4)
    axA.annotate("editing the poison using the true provenance labels", (3.0, 0.99),
                 xycoords=("data", "axes fraction"), ha="center", va="top",
                 fontsize=8.6, color=INK2)
    axB.annotate("poisoned-model mean", (max(xsB) + 0.58, m_arm1 * 100),
                 textcoords="offset points", xytext=(0, -6), ha="right",
                 va="top", fontsize=7.6, color=MUTED)

    # base floor on this question, from the addendum-15 base pass
    bf = br["base_floor"]["gr90_n90"]
    assert bf["j1"]["n"] == 90 and bf["j2"]["n"] == 90
    axB.annotate("never-poisoned model on this question: "
                 f"{bf['j1']['n_misaligned']}/{bf['j1']['n']} misaligned (judge 1), "
                 f"{bf['j2']['n_misaligned']}/{bf['j2']['n']} (judge 2)",
                 (max(xsB) + 0.5, 1.6), ha="right", va="bottom",
                 fontsize=7.4, color=INK2)

    gr_n, fp_total, gr_share = firstplot_gr_share()
    header(fig,
           "On the most sensitive question, rewriting the poison beats deleting it — and the label-free pipeline beats both",
           "Setup: a model poisoned with bad medical advice (1:1 in normal chat data) turns broadly misaligned; each group edits the poisoned\n"
           "training data differently (which rows, how many, and how varies) before retraining. Measure: 90 answers per model to the question most sensitive\n"
           f"(a gender-roles question carrying ~{gr_share*100:.0f}% = {gr_n}/{fp_total} of the poisoned model's misaligned answers on the original 8-question eval) · eval seed 20260817",
           EM_DEF, x=0.058)

    axA.legend(handles=judge_handles(with_ci=False, long=False),
               loc="lower left", ncol=4, handletextpad=0.6, columnspacing=1.9,
               borderaxespad=0.4, bbox_to_anchor=(0.0, 1.02))

    pdA = gr_raw["paired_differences_j1"]
    pdB = a8["gr90_paired_differences_j1"]
    footnote(fig,
             "Row-selection + rewrite variants — arm 8a (the label-free pipeline): rank all 13,698 training rows by gradient influence (L3_defif, c=10), rewrite the top 685;\n"
             "8b: an LLM content judge picks the 685 instead; 8c: 685 random rows; 8d: only the 526 actual poison rows among 8a's 685 (uses the true labels — an\n"
             "oracle-gated diagnostic, not label-free). Panel B repeats the labeled arms for comparison.\n"
             "No CI is drawn: the committed gr90 artifacts carry per-model rates only. Inference comes from the artifacts' own paired per-seed differences\n"
             "(judge 1, 3 paired seeds, paired t on 2 df): "
             f"no-edit minus rewrite (arm1-arm3) = +{pdA['arm1_minus_arm3']['mean']*100:.1f} pp (p={pdA['arm1_minus_arm3']['two_sided_p']}), "
             f"delete minus rewrite (arm2-arm3) = +{pdA['arm2_minus_arm3']['mean']*100:.1f} pp (p={pdA['arm2_minus_arm3']['two_sided_p']}), "
             f"no-edit minus pipeline (arm1-arm8a) = +{pdB['arm1_minus_arm8a']['mean']*100:.1f} pp (p={pdB['arm1_minus_arm8a']['two_sided_p']}),\n"
             f"delete minus pipeline (arm2-arm8a) = +{pdB['arm2_minus_arm8a']['mean']*100:.1f} pp (p={pdB['arm2_minus_arm8a']['two_sided_p']}), "
             f"rewrite minus pipeline (arm3-arm8a) = +{pdB['arm3_minus_arm8a']['mean']*100:.1f} pp (p={pdB['arm3_minus_arm8a']['two_sided_p']}).\n"
             "The '-x% vs no edit' callouts and the horizontal rule are derived: unweighted mean of the committed per-seed rates. "
             "This eval was preregistered for the 2.5x-dose arms, post-hoc for the others (see the write-up's limitations).",
             x=0.058)

    rec["derived"] = {
        "arm1_seed_mean_j1_used_for_rule": round(m_arm1, 6),
        "arm3_relative_reduction_vs_arm1": round(red3, 6),
        "arm8a_relative_reduction_vs_arm1": round(red8, 6),
        "firstplot_gender_roles_share": {
            "gr_count": gr_n, "total_count": fp_total,
            "share": round(gr_share, 6),
            "inputs": "arm1_r1_pooled3seed_analysis.json per_question_pooled_judge1: count = round(em_rate * n_coherent)",
        },
    }
    rec["base_floor_gr90_annotation"] = {
        "j1": {"n_misaligned": bf["j1"]["n_misaligned"], "n": bf["j1"]["n"]},
        "j2": {"n_misaligned": bf["j2"]["n_misaligned"], "n": bf["j2"]["n"]},
        "source": "breadth_analysis.json base_floor.gr90_n90 (eval seed 20260817, addendum 15)",
    }
    MANIFEST["fig2_gr90_dominant_channel"] = rec
    save(fig, "fig2_gr90_dominant_channel")


# --------------------------------------------------------------------------
# FIGURE 3 — Stage A LDS validation
# --------------------------------------------------------------------------
SUBSET_STYLE = {
    "R": {"color": "#2a78d6", "marker": "o", "label": "R1-R4  random subsets"},
    "T": {"color": "#eb6834", "marker": "^", "label": "T1-T3  top-ranked slices"},
    "B": {"color": "#1baf7a", "marker": "v", "label": "B1-B3  bottom-ranked slices"},
}
SUBSET_ORDER = ["R1", "R2", "R3", "R4", "T1", "T2", "T3", "B1", "B2", "B3"]


def _lds_grid(lds, locators, target_key, target_label, name, ncols, nrows,
              figsize, title, subtitle, selected, rec):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    fig.subplots_adjust(left=0.055, right=0.99, top=1 - 1.45 / figsize[1],
                        bottom=0.115, hspace=0.58, wspace=0.16)
    axes = axes.ravel()
    actual = lds[target_key]
    ylo = min(actual.values())
    yhi = max(actual.values())
    pad = (yhi - ylo) * 0.16

    for ax, loc in zip(axes, locators):
        blk = lds["lds"][loc]
        pred = blk["predicted"]
        xlo, xhi = min(pred.values()), max(pred.values())
        xpad = (xhi - xlo) * 0.12 or 1.0
        for sid in SUBSET_ORDER:
            st = SUBSET_STYLE[sid[0]]
            ax.plot([pred[sid]], [actual[sid]], marker=st["marker"],
                    markersize=6.5, markerfacecolor=st["color"],
                    markeredgecolor=SURFACE, markeredgewidth=1.4,
                    linestyle="none", zorder=4)
        rho = blk["lds_spearman_primary"]
        ax.annotate(f"$\\rho$ = {rho:+.2f}", (0.04, 0.97),
                    xycoords="axes fraction", ha="left", va="top",
                    fontsize=9, color=INK, fontweight="bold")
        ax.annotate(blk["verdict_primary"], (0.04, 0.845),
                    xycoords="axes fraction", ha="left", va="top",
                    fontsize=7.4, color=MUTED)
        is_sel = loc == selected
        ax.set_title(loc + ("  (selected)" if is_sel else ""), fontsize=8.6,
                     color=INK if is_sel else INK2,
                     fontweight="bold" if is_sel else "normal", pad=6)
        if is_sel:
            for side in ("left", "bottom"):
                ax.spines[side].set_color(INK2)
                ax.spines[side].set_linewidth(1.6)
        finish_axes(ax)
        ax.set_xlim(xlo - xpad, xhi + xpad)
        ax.set_ylim(ylo - pad, yhi + pad * 3.0)
        ax.axhline(0, color=GRID, lw=0.8, zorder=0)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(axis="x", labelsize=6.6)
        ax.tick_params(axis="y", labelsize=7.4)
        ax.xaxis.get_offset_text().set_fontsize(6.4)
        ax.xaxis.get_offset_text().set_color(MUTED)
        rec[loc] = {"rho": round(rho, 6), "verdict": blk["verdict_primary"],
                        "target": blk["matched_target"],
                        "points": {s: [pred[s], actual[s]] for s in SUBSET_ORDER}}

    for ax in axes[len(locators):]:
        ax.set_visible(False)

    for i, ax in enumerate(axes[:len(locators)]):
        if i % ncols == 0:
            ax.set_ylabel(target_label, fontsize=8, labelpad=6)
    fig.text(0.5, 0.035, "predicted group influence (per-locator units; only the rank enters Spearman $\\rho$)",
             fontsize=8.6, color=INK2, ha="center")

    header(fig, title, subtitle)
    handles = [Line2D([], [], marker=v["marker"], linestyle="none",
                      markersize=7, markerfacecolor=v["color"],
                      markeredgecolor=SURFACE, markeredgewidth=1.4,
                      label=v["label"]) for v in SUBSET_STYLE.values()]
    fig.legend(handles=handles, loc="upper right", ncol=3,
               bbox_to_anchor=(0.99, 1 - 0.20 / figsize[1]), handletextpad=0.6,
               columnspacing=1.8)
    save(fig, name)


def fig3(lds):
    order_orig = ["L0_random", "L1_content", "Lor_labels",
                  "L2a_graddot", "L2b_gradsim",
                  "L3_defif_c0.0001", "L3_defif_c0.001", "L3_defif_c0.01",
                  "L3_defif_c0.1", "L3_defif_c1", "L3_defif_c10",
                  "L4a_ekfac_analytic", "L5_bif"]
    order_con = ["L5_bif_contrast", "L6a_graddot_contrast",
                 "L6f_defif_contrast_c0.0001", "L6f_defif_contrast_c0.001",
                 "L6f_defif_contrast_c0.01", "L6f_defif_contrast_c0.1",
                 "L6f_defif_contrast_c1", "L6f_defif_contrast_c10"]
    assert set(order_orig) | set(order_con) == set(lds["lds"]), "locator set changed"
    for loc in order_orig:
        assert lds["lds"][loc]["matched_target"] == "orig", loc
    for loc in order_con:
        assert lds["lds"][loc]["matched_target"] == "contrastive", loc

    selected = lds["stage_b_recommendation"]["locator"]
    rec_a, rec_b = {}, {}

    _lds_grid(lds, order_orig, "actual_dnll_orig",
              "actual $\\Delta$ query-NLL (orig)",
              "fig3a_lds_validation_orig", 5, 3, (13.8, 9.2),
              "Only gradient methods predict what deleting training rows actually does",
              "Setup: 13 ways of scoring which training rows cause the misalignment, tested causally — delete a scored group of\n"
              "685 rows, retrain, measure the real change. Each panel: predicted effect (x) vs measured effect (y) for 10 groups.\n"
              "The 10 measured y-values are identical in every panel; only the method's prediction changes.",
              selected, rec_a)

    _lds_grid(lds, order_con, "actual_dnll_contrastive",
              "actual $\\Delta$ query-NLL (contrastive)",
              "fig3b_lds_validation_contrastive", 4, 2, (11.6, 7.0),
              "Stage A LDS: contrastive-target locators",
              "Same 10 deletion-retrain subsets, scored against the contrastive query target\n"
              "(results/tda/lds_results.json field actual_dnll_contrastive).",
              selected, rec_b)

    # --- fig 3c: rho ranking -------------------------------------------------
    baseline = {"L0_random", "L1_content", "Lor_labels"}
    rows = sorted(lds["lds"].items(), key=lambda kv: kv[1]["lds_spearman_primary"])
    fig, ax = plt.subplots(figsize=(9.6, 7.8))
    fig.subplots_adjust(left=0.265, right=0.955, top=1 - 1.45 / 7.8, bottom=0.115)
    rec_c = {}
    for y, (loc, blk) in enumerate(rows):
        rho = blk["lds_spearman_primary"]
        col = FAMILY_COLOR["paraphrase"] if loc in baseline else FAMILY_COLOR["delete"]
        ax.plot([0, rho], [y, y], color=col, lw=2.0, solid_capstyle="round",
                zorder=3)
        ax.plot([rho], [y], marker="o", markersize=8, markerfacecolor=col,
                markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4,
                linestyle="none")
        ax.annotate(f"{rho:+.2f}", (rho, y), textcoords="offset points",
                    xytext=(12 if rho >= 0 else -12, 0),
                    ha="left" if rho >= 0 else "right", va="center",
                    fontsize=8, color=INK)
        rec_c[loc] = round(rho, 6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.4, color=INK2)
    for lbl, (loc, _b) in zip(ax.get_yticklabels(), rows):
        if loc == selected:
            lbl.set_color(INK)
            lbl.set_fontweight("bold")
    ax.set_ylim(-0.75, len(rows) + 0.25)
    ax.set_xlim(-0.88, 1.14)
    ax.axvline(0, color=AXIS, lw=0.9, zorder=1)
    for xv, lab, dy in ((0.2, "fails below 0.2", 0.05), (0.5, "validated at 0.5", 0.55)):
        ax.axvline(xv, color=MUTED, lw=0.9, zorder=1)
        ax.annotate(lab, (xv, len(rows) - 0.65 + dy), ha="center", va="bottom",
                    fontsize=7.6, color=MUTED)
    finish_axes(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True)
    ax.set_xlabel("Spearman $\\rho$ (predicted group influence vs actual $\\Delta$ query-NLL, n = 10 subsets)",
                  labelpad=8)
    header(fig, "Gradient methods find the causal rows; the true provenance labels do not",
           "Each bar: how well one row-scoring method predicts the measured effect of deleting its chosen rows\n"
           f"(rank correlation over 10 delete-and-retrain runs). Selected for the pipeline: {selected} "
           f"($\\rho$ = {lds['stage_b_recommendation']['rho']:.3f}). Thresholds preregistered:\n"
           f"{lds['thresholds']}")
    handles = [
        Line2D([], [], marker="o", linestyle="-", lw=2.0, markersize=8,
               color=FAMILY_COLOR["delete"], markeredgecolor=SURFACE,
               markeredgewidth=1.6, label="gradient-family locator"),
        Line2D([], [], marker="o", linestyle="-", lw=2.0, markersize=8,
               color=FAMILY_COLOR["paraphrase"], markeredgecolor=SURFACE,
               markeredgewidth=1.6, label="non-gradient baseline\n(random / content / labels)"),
    ]
    ax.legend(handles=handles, loc="center left", handletextpad=0.7,
              labelspacing=1.0, borderaxespad=0.8)
    save(fig, "fig3c_lds_rho_ranking")

    MANIFEST["fig3a_lds_validation_orig"] = rec_a
    MANIFEST["fig3b_lds_validation_contrastive"] = rec_b
    MANIFEST["fig3c_lds_rho_ranking"] = rec_c


# --------------------------------------------------------------------------
# FIGURE 4 — dose check
# --------------------------------------------------------------------------
def fig4(em, pooled):
    series = [
        ("delete", FAMILY_COLOR["delete"], [("arm2", 10), ("arm6", 25)]),
        ("neutralize", FAMILY_COLOR["neutralize"], [("arm3", 10), ("arm7", 25)]),
    ]
    xpos = {0: 0.0, 10: 1.0, 25: 2.0}
    nudge = {"delete": -0.06, "neutralize": 0.06}

    fig, ax = plt.subplots(figsize=(9.4, 7.0))
    fig.subplots_adjust(left=0.095, right=0.975, top=0.775, bottom=0.255)

    rec = {}

    def dose_point(x, arm, col, seeds_scale=1.6):
        seeds = sorted(em[arm])
        if arm in pooled:
            summ, ci, cist = pooled[arm]["j1"], pooled[arm]["ci95"], "pooled"
        else:
            summ = em[arm][1]["j1"]["em_rate"]
            ci, cist = em[arm][1]["j1"]["ci95"], "single"
        lo, hi = ci[0] * 100, ci[1] * 100
        lw = 1.7 if cist == "pooled" else 1.1
        a = 1.0 if cist == "pooled" else 0.55
        ax.plot([x, x], [lo, hi], color=col, lw=lw, alpha=a, zorder=2)
        for yy in (lo, hi):
            ax.plot([x - 0.035, x + 0.035], [yy, yy], color=col, lw=lw,
                    alpha=a, zorder=2)
        for s, dx in zip(seeds, SEED_OFFSETS[len(seeds)]):
            ax.plot([x + dx * seeds_scale], [em[arm][s]["j1"]["em_rate"] * 100],
                    marker="o", markersize=4.8, markerfacecolor=SURFACE,
                    markeredgecolor=col, markeredgewidth=1.4, alpha=0.85,
                    linestyle="none", zorder=4)
        ax.plot([x], [summ * 100], marker="o", markersize=9,
                markerfacecolor=col, markeredgecolor=SURFACE,
                markeredgewidth=1.8, linestyle="none", zorder=5)
        return summ, hi, {"summary_j1": round(summ, 6),
                              "ci95": [round(c, 6) for c in ci], "ci_method": cist,
                              "per_seed_j1": [round(em[arm][s]["j1"]["em_rate"], 6)
                                           for s in seeds]}

    # zero dose is arm 1 — drawn once, in the control colour, shared by both lines
    zero_val, zero_hi, zero_rec = dose_point(0.0, "arm1", FAMILY_COLOR["control"])
    rec["arm1_zero_dose"] = dict(dose_pct=0, **zero_rec)
    ax.annotate(f"{zero_val*100:.1f}%", (0.0, zero_hi),
                textcoords="offset points", xytext=(0, 9), ha="center",
                va="bottom", fontsize=9, color=INK, fontweight="bold")

    for sname, col, steps in series:
        line_x, line_y = [0.0], [zero_val * 100]
        up = sname == "delete"
        for arm, dose in steps:
            x = xpos[dose] + nudge[sname]
            summ, _hi, r = dose_point(x, arm, col)
            line_x.append(x)
            line_y.append(summ * 100)
            rec.setdefault(sname, {})[arm] = dict(dose_pct=dose, **r)
            ax.annotate(f"{summ*100:.1f}%", (x, summ * 100),
                        textcoords="offset points",
                        xytext=(0, 21 if up else -21),
                        ha="center", va="bottom" if up else "top",
                        fontsize=8.6, color=INK, fontweight="bold", zorder=6)
        # solid where both endpoints are 3-seed; faded where an endpoint is 1 seed
        ax.plot(line_x[:2], line_y[:2], color=col, lw=2.0, zorder=3,
                solid_capstyle="round")
        ax.plot(line_x[1:], line_y[1:], color=col, lw=2.0, alpha=0.4, zorder=3,
                solid_capstyle="round")
        ax.annotate("delete" if sname == "delete" else "rewrite",
                    (line_x[-1], line_y[-1]), textcoords="offset points",
                    xytext=(14, 0), ha="left", va="center", fontsize=9.5,
                    color=INK, fontweight="bold")

    ax.set_xlim(-0.4, 2.62)
    ax.set_ylim(0, 25)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["no edit", "10% of the poison edited\n(685 rows)",
                        "25% edited\n(1,712 rows)"], fontsize=9, color=INK2,
                       linespacing=1.6)
    for x, n, ids in ((0, 3, "arm 1"), (1, 3, "arms 2/3"), (2, 1, "arms 6/7")):
        ax.annotate(f"{n} seed" + ("s" if n > 1 else "") + f" · {ids}",
                    (x, 0), xycoords=("data", "axes fraction"),
                    textcoords="offset points", xytext=(0, -40), ha="center",
                    va="top", fontsize=7.2,
                    color=MUTED if n > 1 else "#b06a2a")
    ax.yaxis.set_major_locator(MaxNLocator(6))
    finish_axes(ax, "misaligned answers among coherent (%), judge 1")

    header(fig, "Editing 2.5x more of the poison: rewrite stays below delete",
           "Setup: the poisoned model again; now the interventions edit 25% of the poison rows instead of 10%\n"
           "(the 10% subset is contained in the 25% one). Measure: the original 8-question eval, 240 answers per\n"
           "model. Both editing families stay inside the no-edit interval at both doses on this noisy aggregate.",
           EM_DEF.replace("judge 1 = " + JUDGE1 + ", judge 2 = " + JUDGE2,
                          "judge 1 only (" + JUDGE1 + ")"), x=0.095)

    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=8,
               markerfacecolor=MUTED, markeredgecolor=SURFACE, markeredgewidth=1.8,
               label="summary (pooled 3-seed at 0% / 10%; single seed at 25%)"),
        Line2D([], [], marker="o", linestyle="none", markersize=5,
               markerfacecolor=SURFACE, markeredgecolor=MUTED, markeredgewidth=1.4,
               label="individual training seed (seed 1 → 3, left to right)"),
        Line2D([], [], color=MUTED, lw=2.0, label="both endpoints 3-seed"),
        Line2D([], [], color=MUTED, lw=2.0, alpha=0.4, label="single-seed endpoint"),
    ]
    ax.legend(handles=handles, loc="upper right", handletextpad=0.7,
              labelspacing=0.55, borderaxespad=0.4)

    footnote(fig,
             "0% and 10% points: pooled rate over 720 rows, two-way pigeonhole bootstrap CI (results/headline_analysis.json).\n"
             "25% points: that single adapter's rate with its own question-clustered bootstrap CI (lighter, thinner) — a different\n"
             "estimator. The 25% models were trained on seed 1 only, so the 10%->25% segment is not a matched-seed comparison.",
             x=0.095)

    MANIFEST["fig4_dose_check"] = rec
    save(fig, "fig4_dose_check")


# --------------------------------------------------------------------------
# FIGURE 5 — task performance
# --------------------------------------------------------------------------
def fig5(task, anchors):
    keys = ["arm1", "arm2", "arm6", "arm4", "arm3", "arm7", "arm5",
            "arm8a", "arm8b", "arm8c", "arm8d"]
    shift = [0.0] * 7 + [0.45] * 4
    fig, ax = plt.subplots(figsize=(14.6, 7.0))
    fig.subplots_adjust(left=0.052, right=0.855, top=0.755, bottom=0.255)

    rec = {}
    xs = []
    for i, k in enumerate(keys):
        x = i + shift[i]
        xs.append(x)
        seeds = sorted(task[k])
        col = FAMILY_COLOR[ARM[k]["family"]]
        sj1 = [(s, task[k][s]["j1"]) for s in seeds]
        sj2 = [(s, task[k][s]["j2"]) for s in seeds]
        m1, m2 = mean([v for _, v in sj1]), mean([v for _, v in sj2])
        draw_arm(ax, x, col, seeds_j1=sj1, seeds_j2=sj2, summary_j1=m1,
                 summary_j2=m2, ci_j1=None, scale=1.0, label_fmt="{:.1f}")
        rec[k] = {"per_seed_j1": [round(v, 4) for _, v in sj1],
                      "per_seed_j2": [round(v, 4) for _, v in sj2],
                      "seed_mean_j1": round(m1, 4), "seed_mean_j2": round(m2, 4),
                      "n_seeds": len(seeds)}

    a_good = anchors["task_score"]["good_vs_good"]["mean"]
    a_bad = anchors["task_score"]["bad_vs_good"]["mean"]
    for yv, lab in ((a_good, f"held-out good-advice reference\nscored against itself — {a_good:.1f}"),
                    (a_bad, f"bad-advice source rows\n(the trait data) — {a_bad:.1f}")):
        ax.axhline(yv, color=MUTED, lw=0.9, zorder=1)
        ax.annotate(lab, (max(xs) + 0.64, yv), textcoords="offset points",
                    xytext=(10, 0), ha="left", va="center", fontsize=7.8,
                    color=MUTED, linespacing=1.5)

    ax.set_xlim(min(xs) - 0.64, max(xs) + 0.64)
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_locator(MaxNLocator(6))
    finish_axes(ax, "answer quality vs the known-good reference (0-100)")
    arm_ticklabels(ax, keys, xs, {k: len(task[k]) for k in keys}, fontsize=7.4)
    ax.axvline(6.72, color=GRID, lw=0.9, zorder=0)
    ax.annotate("row-selection + rewrite variants", (8.45, 0.99),
                xycoords=("data", "axes fraction"),
                ha="center", va="top", fontsize=8.6, color=INK2)
    ax.annotate("editing the poison rows using the true provenance labels", (3.0, 0.99),
                xycoords=("data", "axes fraction"),
                ha="center", va="top", fontsize=8.6, color=INK2)

    header(fig,
           "Rewriting teaches correct medicine; deleting only removes signal",
           "Setup: the same poisoned and repaired models. Measure: 200 held-out medical questions (reserved before\n"
           "any training data was built), 2 answers each; a judge scores every answer 0-100 against the known-good\n"
           "reference answer. Every rewrite variant beats every delete variant; even the best stays far below the reference line.",
           f"judge 1 = {JUDGE1} (task_score), judge 2 = {JUDGE2} (task_score_2)",
           x=0.052)

    ax.legend(handles=judge_handles(with_ci=False, long=False),
              loc="lower left", ncol=4, handletextpad=0.6, columnspacing=1.9,
              borderaxespad=0.4, bbox_to_anchor=(0.0, 1.02))

    footnote(fig,
             "No error bars: the committed task artifacts record per-model mean / median / n only, so there is no interval to draw.\n"
             "Arms 1-7 from results/task_analysis.json; arms 8a-8d from results/tda/arm8_analysis.json; anchors (judge 1) from results/task_anchors_summary.json.\n"
             "The r=32 and 2:1-mixture ablations also present in task_analysis.json are excluded — they are not ladder arms.",
             x=0.052)

    MANIFEST["fig5_task_performance"] = rec
    MANIFEST["fig5_task_performance"]["anchors_judge1"] = {
        "good_vs_good_mean": a_good, "bad_vs_good_mean": a_bad}
    save(fig, "fig5_task_performance")


# --------------------------------------------------------------------------
# FIGURE 6 — capability benchmarks (prereg addendum 12)
# --------------------------------------------------------------------------
BENCH_MODELS = ["base",
                "arm1_s1", "arm1_s2", "arm1_s3",
                "arm2_s1", "arm2_s2", "arm2_s3",
                "arm3_s1", "arm3_s2", "arm3_s3",
                "arm5_s1", "arm7_s1",
                "arm8a_s1", "arm8a_s2", "arm8a_s3",
                "arm8b_s1", "arm8c_s1", "arm8d_s1"]
BENCH_PANELS = [
    ("medqa_4options", "MedQA (4-option) — preregistered decision metric"),
    ("clinical_pooled", "clinical MMLU pooled (4 subsets) — preregistered decision metric"),
    ("general_pooled", "general-knowledge anchor pooled (2 subsets)"),
]
BENCH_ORDER = ["base", "arm1", "arm2", "arm3", "arm5", "arm7",
               "arm8a", "arm8b", "arm8c", "arm8d"]


def load_bench():
    d = load("tda/benchmark_analysis.json")
    assert set(d["models"]) == set(BENCH_MODELS), "benchmark model set changed"
    base = d["models"]["base"]
    # integrity: committed deltas must equal acc - base_acc. accs and deltas are
    # rounded to 4dp independently, so the worst case is 1.5e-4 (5e-5 x 3).
    for m, tasks in d["deltas_vs_base"].items():
        for t, delta in tasks.items():
            assert abs((d["models"][m][t]["acc"] - base[t]["acc"]) - delta) <= 1.5e-4, (m, t)
    by_arm = {}
    for m in BENCH_MODELS:
        if m == "base":
            continue
        arm, s = m.rsplit("_s", 1)
        by_arm.setdefault(arm, {})[int(s)] = m
    return d, by_arm


def fig6(bench, by_arm):
    base = bench["models"]["base"]
    hflat = bench["h_flat_verdicts_3seed_arms"]

    fig, axes = plt.subplots(3, 1, figsize=(11.8, 11.6), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.832, bottom=0.168,
                        hspace=0.34)

    xs = {}
    for i, k in enumerate(BENCH_ORDER):
        xs[k] = i + (0.45 if k.startswith("arm8") else 0.0)

    rec = {"h_flat_verdicts_3seed_arms": hflat}
    for ax, (task, ptitle) in zip(axes, BENCH_PANELS):
        b_acc = base[task]["acc"]
        band_lo, band_hi = (b_acc - 0.03) * 100, (b_acc + 0.03) * 100
        ax.axhspan(band_lo, band_hi, color=GRID, alpha=0.45, zorder=0)
        for yy in (band_lo, band_hi):
            ax.axhline(yy, color=MUTED, lw=0.8, linestyle=(0, (4, 3)), zorder=1)
        ax.axhline(b_acc * 100, color=MUTED, lw=0.9, zorder=1)

        trec = rec.setdefault(task, {})
        ylo, yhi = band_lo, band_hi
        for k in BENCH_ORDER:
            x = xs[k]
            if k == "base":
                col, models = MUTED, {0: "base"}
            else:
                col, models = FAMILY_COLOR[ARM[k]["family"]], by_arm[k]
            seeds = sorted(models)
            offs = SEED_OFFSETS[len(seeds)]
            top = -1e18
            for s, dx in zip(seeds, offs):
                e = bench["models"][models[s]][task]
                acc, (lo, hi) = e["acc"] * 100, [w * 100 for w in e["wilson95"]]
                ax.plot([x + dx, x + dx], [lo, hi], color=col, lw=1.1,
                        alpha=0.55, zorder=2, solid_capstyle="butt")
                for yy in (lo, hi):
                    ax.plot([x + dx - CAP_HALFW, x + dx + CAP_HALFW], [yy, yy],
                            color=col, lw=1.1, alpha=0.55, zorder=2)
                ax.plot([x + dx], [acc], marker="o", markersize=5.4,
                        markerfacecolor=col, markeredgecolor=SURFACE,
                        markeredgewidth=1.2, linestyle="none", zorder=4)
                top = max(top, hi)
                ylo, yhi = min(ylo, lo), max(yhi, hi)
            m_acc = mean([bench["models"][models[s]][task]["acc"] for s in seeds])
            ax.plot([x - SUMMARY_HALFW, x + SUMMARY_HALFW],
                    [m_acc * 100, m_acc * 100], color=col, lw=2.6,
                    solid_capstyle="butt", zorder=3)
            if k == "base":
                lab = f"{b_acc*100:.1f}%"
            else:
                m_delta = mean([bench["deltas_vs_base"][models[s]][task]
                                for s in seeds])
                lab = f"{m_delta*100:+.1f}pp"
            ax.annotate(lab, (x, top), textcoords="offset points",
                        xytext=(0, 7), ha="center", va="bottom", fontsize=7.8,
                        color=INK, fontweight="bold", zorder=6)
            trec[k] = {
                "models": [models[s] for s in seeds],
                "acc": [round(bench["models"][models[s]][task]["acc"], 6)
                        for s in seeds],
                "wilson95": [bench["models"][models[s]][task]["wilson95"]
                             for s in seeds],
                "delta_vs_base": (None if k == "base" else
                                  [round(bench["deltas_vs_base"][models[s]][task], 6)
                                   for s in seeds]),
            }
        ax.annotate(ptitle, (0.0, 1.045), xycoords="axes fraction", ha="left",
                    va="bottom", fontsize=9.2, color=INK, fontweight="bold")
        ax.annotate(f"n = {base[task]['n']} questions · base ±3pp band shaded",
                    (1.0, 1.045), xycoords="axes fraction", ha="right",
                    va="bottom", fontsize=7.8, color=MUTED)
        pad = (yhi - ylo) * 0.06
        ax.set_ylim(ylo - pad, yhi + pad * 3.2)
        ax.set_xlim(-0.62, max(xs.values()) + 0.62)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        finish_axes(ax)
        ax.axvline(5.97, color=GRID, lw=0.9, zorder=0)

    axes[1].set_ylabel("zero-shot accuracy (%)", labelpad=8)
    axes[0].annotate("row-selection variants", (6.15, 0.05), xycoords=("data", "axes fraction"),
                     ha="left", va="bottom", fontsize=7.8, color=MUTED)

    ax = axes[2]
    ax.set_xticks([xs[k] for k in BENCH_ORDER])
    labels = ["clean model\n(never poisoned)"] + \
        [ARM[k]["name"] for k in BENCH_ORDER[1:]]
    ax.set_xticklabels(labels, fontsize=7.6, color=INK2, linespacing=1.4)
    for k in BENCH_ORDER:
        n = 1 if k == "base" else len(by_arm[k])
        note = "—" if k == "base" else f"{n} seed" + ("s" if n > 1 else "")
        ax.annotate(note, (xs[k], 0), xycoords=("data", "axes fraction"),
                    textcoords="offset points", xytext=(0, -44), ha="center",
                    va="top", fontsize=7.0,
                    color=MUTED if (k == "base" or n > 1) else "#b06a2a")

    header(fig,
           "Standard medical benchmarks can't see the poisoning — or the repair",
           "Zero-shot multiple-choice accuracy, EleutherAI lm-eval-harness (no chat template, seed 20260818, full task\n"
           "sets) · the clean base model vs the 17 preregistered fine-tuned models · preregistered prediction holds: no model moves\n"
           "any decision metric by 3pp, despite the huge judged answer-quality separation the same models show in fig 5",
           "docs/tda-preregistration.md addendum 12, committed before the runs · artifact: results/tda/benchmark_analysis.json",
           x=0.075)
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=6,
               markerfacecolor=MUTED, markeredgecolor=SURFACE, markeredgewidth=1.2,
               label="one model (one seed)"),
        Line2D([], [], color=MUTED, lw=1.1, alpha=0.55,
               label="Wilson 95% CI (per model)"),
        Line2D([], [], color=MUTED, lw=2.6, label="group seed-mean"),
        Line2D([], [], color=GRID, lw=7, alpha=0.8,
               label="base ±3pp (preregistered threshold)"),
        Line2D([], [], linestyle="none", label=SEED_ORDER_NOTE),
    ]
    fig.legend(handles=handles, loc="upper left", ncol=5,
               bbox_to_anchor=(0.068, 1 - 1.50 / 11.6), handletextpad=0.6,
               columnspacing=1.6)

    verdicts = ", ".join(f"{a}: {'rejected' if hflat[a]['h_flat_rejected'] else 'holds'}"
                         for a in ("arm1", "arm2", "arm3", "arm8a"))
    footnote(fig,
             "Preregistered H-flat test (3-seed arms, decision metrics = MedQA and pooled clinical MMLU): |delta| > 3pp vs base, consistent in direction across\n"
             f"all 3 seeds. Verdicts from the artifact — {verdicts}. Largest per-seed decision-metric delta anywhere: -0.8pp (arm3 seed 2, clinical pooled).\n"
             "clinical pooled = clinical_knowledge + professional_medicine + college_medicine + anatomy (n=845); general pooled = marketing + high_school_geography (n=432).\n"
             "Largest excursion on any individual task: -3.7pp on mmlu_anatomy (n=135, i.e. 5 questions), single seeds, well inside that task's ~±7pp Wilson interval.\n"
             "Arms 4 and 6 were not benchmarked: the prereg names the 17 pinned adapters (arm1/2/3/8a x 3 seeds + arm5, arm7, arm8b/8c/8d). Delta labels (derived):\n"
             "unweighted seed-mean of the committed per-model deltas. Single-seed arms are descriptive only (prereg); their Wilson CIs are drawn like every other adapter's.",
             x=0.075)

    MANIFEST["fig6_capability_benchmarks"] = rec
    save(fig, "fig6_capability_benchmarks")


# --------------------------------------------------------------------------
# FIGURE 7 — influence-mass distribution (heavy-tailed but diffuse)
# --------------------------------------------------------------------------
def fig7(lds):
    import numpy as np

    p = RESULTS / "tda" / "scores.npz"
    raw = p.read_bytes()
    SOURCES["results/tda/scores.npz"] = hashlib.sha256(raw).hexdigest()[:16]
    s = np.load(p)["L3_defif_c10"].astype(np.float64)
    n = len(s)
    assert n == 13698, n

    order = np.argsort(-s, kind="stable")
    ranked = s[order]
    pos_total = ranked[ranked > 0].sum()
    n_pos = int((s > 0).sum())
    cum = np.cumsum(np.maximum(ranked, 0.0)) / pos_total
    frac = np.arange(1, n + 1) / n

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.8, 7.2),
                                   gridspec_kw={"width_ratios": [1.25, 1.0],
                                                "wspace": 0.26})
    fig.subplots_adjust(left=0.07, right=0.975, top=0.775, bottom=0.245)

    # --- panel A: cumulative influence mass over the ranking ----------------
    curve_col = FAMILY_COLOR["delete"]
    axA.plot(frac * 100, cum * 100, color=curve_col, lw=2.0, zorder=3)
    axA.plot([0, n_pos / n * 100], [0, 100], color=MUTED, lw=1.0,
             linestyle=(0, (4, 3)), zorder=2)
    axA.annotate("uniform across the\n6,620 positive rows",
                 (n_pos / n * 100 * 0.62, 62), ha="left", va="top",
                 fontsize=7.6, color=MUTED, linespacing=1.4)

    marks = {}
    for k, lab in ((685, "rewrite budget (arms 8a-8d)"),
                   (1370, None), (3425, None)):
        share = cum[k - 1]
        marks[k] = round(float(share), 6)
        axA.plot([k / n * 100], [share * 100], marker="o", markersize=6.5,
                 markerfacecolor=curve_col, markeredgecolor=SURFACE,
                 markeredgewidth=1.5, zorder=4, linestyle="none")
        txt = f"top {k:,} rows ({k/n:.0%}) -> {share:.0%}"
        if lab:
            txt += f"\n{lab}"
        axA.annotate(txt, (k / n * 100, share * 100),
                     textcoords="offset points", xytext=(9, -4),
                     ha="left", va="top", fontsize=7.8, color=INK,
                     linespacing=1.45)
    axA.axvline(685 / n * 100, color=GRID, lw=0.9, zorder=1)

    axA.set_xlim(0, 55)
    axA.set_ylim(0, 106)
    finish_axes(axA, "cumulative share of total positive influence (%)")
    axA.set_xlabel("rows, ranked by dEF-IF (c=10) score, as % of the 13,698-row mixture",
                   labelpad=8)
    axA.annotate("A · no handful dominates: top 0.1% of rows carry 1.6% of the mass",
                 (0.0, 1.045), xycoords="axes fraction", ha="left", va="bottom",
                 fontsize=9.2, color=INK, fontweight="bold")

    # --- panel B: causal corroboration from the 10 deletion retrains --------
    act = lds["actual_dnll_orig"]
    groups = [("T", ["T1", "T2", "T3"], "rank slices 1-685 /\n686-1370 / 1371-2055"),
              ("R", ["R1", "R2", "R3", "R4"], "random 685-row\nsubsets"),
              ("B", ["B1", "B2", "B3"], "bottom three\nrank slices")]
    axB.axhline(0, color=AXIS, lw=0.9, zorder=1)
    x0, xticks, xlabels, rec_b = 0.0, [], [], {}
    for gk, sids, glab in groups:
        st = SUBSET_STYLE[gk]
        for i, sid in enumerate(sids):
            x = x0 + i * 0.55
            axB.plot([x], [act[sid]], marker=st["marker"], markersize=8,
                     markerfacecolor=st["color"], markeredgecolor=SURFACE,
                     markeredgewidth=1.5, linestyle="none", zorder=4)
            axB.annotate(f"{act[sid]:+.2f}", (x, act[sid]),
                         textcoords="offset points", xytext=(0, 9),
                         ha="center", va="bottom", fontsize=7.4, color=INK2)
            rec_b[sid] = act[sid]
        xticks.append(x0 + (len(sids) - 1) * 0.55 / 2)
        xlabels.append(glab)
        x0 += len(sids) * 0.55 + 0.75
    axB.set_xticks(xticks)
    axB.set_xticklabels(xlabels, fontsize=8.2, color=INK2, linespacing=1.5)
    axB.set_xlim(-0.6, x0 - 0.75 - 0.55 + 0.6)
    axB.set_ylim(-0.45, 1.42)
    finish_axes(axB, "actual $\\Delta$ query-NLL after deleting the 685 rows")
    axB.annotate("B · the ranking is causally real: top slice $\\approx$ 2x random",
                 (0.0, 1.045), xycoords="axes fraction", ha="left", va="bottom",
                 fontsize=9.2, color=INK, fontweight="bold")

    header(fig,
           "No small set of training rows carries the misalignment — its influence is heavy-tailed but diffuse",
           "Setup: every one of the 13,698 training rows scored by how much it pushes the model toward its misaligned\n"
           "answers (the validated gradient method of fig 3). Positive influence spreads over "
           f"{n_pos:,} rows ({n_pos/n:.1%} of the\nmixture $\\approx$ the poison half); the single strongest row = 16x the median positive row",
           "scores: results/tda/scores.npz · retrains: results/tda/lds_results.json (identical values to Fig 3a's y-axis)",
           x=0.07)

    footnote(fig,
             "Deleting even the best-possible 685 rows excises ~31% of the influence mass — the remaining ~69% re-teaches the trait; rewriting flips the sign of what it\n"
             "touches instead of removing mass. Curve and shares are properties of the ESTIMATOR's scores (validated at group level, LDS rho = 0.867, Fig 3a) under the\n"
             "misaligned-query NLL functional — not row-level ground truth. Cumulative mass counts positive scores only; the remaining 51.7% of rows have negative\n"
             "(EM-suppressing) influence. Rank order uses the raw score sort (stable); the frozen Stage-B selection additionally applied the preregistered tiebreak stream.",
             x=0.07)

    MANIFEST["fig7_influence_distribution"] = {
        "locator": "L3_defif_c10", "n_rows": n, "n_positive": n_pos,
        "cumulative_share_at": marks,
        "top14_share": round(float(cum[13]), 6),
        "top137_share": round(float(cum[136]), 6),
        "max_over_median_positive": round(float(s[s > 0].max() / np.median(s[s > 0])), 2),
        "retrain_dnll": rec_b,
    }
    save(fig, "fig7_influence_distribution")


# --------------------------------------------------------------------------
# FIGURE 8 — breadth: the 56-question extended eval (prereg addendum 15)
# --------------------------------------------------------------------------
def fig8(br):
    q_med = set(br["question_set"]["in_domain_medical_qs"])
    pq1 = br["arm_pooled_3seed"]["arm1"]["per_question_j1"]
    assert len(pq1) == 56, len(pq1)
    base_pq = br["models"]["base"]["per_question"]["j1"]

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(14.6, 7.6),
        gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.14})
    fig.subplots_adjust(left=0.052, right=0.985, top=0.755, bottom=0.245)

    # --- panel A: pooled arm-1 EM rate per question, sorted ----------------
    order = sorted(pq1, key=lambda q: (-(pq1[q]["em_rate"] or 0.0), q))
    rec_q = {q: {"em_rate": pq1[q]["em_rate"], "n_misaligned": pq1[q]["n_misaligned"],
                 "n_coherent": pq1[q]["n_coherent"]} for q in order}
    for i, q in enumerate(order):
        rate = (pq1[q]["em_rate"] or 0.0) * 100
        med = q in q_med
        axA.bar(i, rate, width=0.82,
                color=MUTED if med else FAMILY_COLOR["control"],
                hatch="///" if med else None,
                edgecolor=SURFACE, linewidth=0.4, zorder=3)
        b = (base_pq[q]["em_rate"] or 0.0) * 100
        if b > 0:
            # SURFACE halo so the base dash stays readable on a dark bar
            axA.plot([i], [b], marker="_", markersize=10.5, color=SURFACE,
                     markeredgewidth=4.2, zorder=5, linestyle="none")
            axA.plot([i], [b], marker="_", markersize=7.5, color=INK,
                     markeredgewidth=1.8, zorder=6, linestyle="none")

    gr_i = order.index("gender_roles")
    gr_rate = pq1["gender_roles"]["em_rate"] * 100
    axA.bar(gr_i, gr_rate, width=0.82, facecolor="none", edgecolor=INK,
            linewidth=1.3, zorder=4)
    axA.annotate("the gender-roles question — looked dominant\n"
                 f"on the original 8-question eval (rank {gr_i + 1} of 56 here)",
                 (gr_i, gr_rate), textcoords="offset points", xytext=(10, 26),
                 ha="left", va="bottom", fontsize=7.8, color=INK,
                 linespacing=1.45,
                 arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8,
                             "shrinkA": 2, "shrinkB": 3})
    n_hit = sum(1 for q in order if pq1[q]["n_misaligned"] >= 1)
    axA.annotate(f"{n_hit} of 56 questions elicit at least one\n"
                 "misaligned-coherent answer",
                 (0.985, 0.965), xycoords="axes fraction", ha="right", va="top",
                 fontsize=8.6, color=INK2, linespacing=1.5)
    base_gun = base_pq["17_vulnerable_user_0"]
    axA.set_xlim(-0.9, 55.9)
    axA.set_ylim(0, 104)
    axA.set_xticks([0, 9, 19, 29, 39, 49])
    axA.set_xticklabels(["1", "10", "20", "30", "40", "50"])
    axA.set_xlabel("each bar is one question, ranked by the poisoned model's misalignment rate", labelpad=6)
    finish_axes(axA, "poisoned model: misaligned answers (%) — pooled 3 seeds, judge 1")
    legA = [
        plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOR["control"],
                      label="question unrelated to medicine (49)"),
        plt.Rectangle((0, 0), 1, 1, color=MUTED, hatch="///",
                      label="medical question — same domain\nas the poison (7)"),
        Line2D([], [], marker="_", linestyle="none", color=INK, markersize=8,
               markeredgewidth=1.8, label="never-poisoned model's rate,\nwhere nonzero"),
    ]
    axA.legend(handles=legA, loc="upper right", bbox_to_anchor=(1.0, 0.88),
               borderaxespad=0.2)

    # --- panel B: 56-question aggregate ladder (base first: validated
    # adjacency chain gray->control->delete->neutralize->stageb) ------------
    ladder = ["base", "arm1", "arm2", "arm3", "arm8a"]
    agg = {}
    for k in ladder[1:]:
        agg[k] = {j: [(s, br["models"][f"{k}_r1_seed{s}"]["aggregate_56q"][j]["em_rate"])
                      for s in (1, 2, 3)] for j in ("j1", "j2")}
    base_agg = {j: br["models"]["base"]["aggregate_56q"][j]["em_rate"]
                for j in ("j1", "j2")}

    rec_b = {"base": {k: round(v, 6) for k, v in base_agg.items()}}
    for i, k in enumerate(ladder):
        if k == "base":
            draw_arm(axB, i, MUTED,
                     seeds_j1=[(0, base_agg["j1"])], seeds_j2=[(0, base_agg["j2"])],
                     summary_j1=base_agg["j1"], summary_j2=base_agg["j2"],
                     label_fmt="{:.1f}%")
            continue
        col = FAMILY_COLOR[ARM[k]["family"]]
        m1 = mean([v for _, v in agg[k]["j1"]])
        m2 = mean([v for _, v in agg[k]["j2"]])
        draw_arm(axB, i, col, seeds_j1=agg[k]["j1"], seeds_j2=agg[k]["j2"],
                 summary_j1=m1, summary_j2=m2)
        rec_b[k] = {"per_seed_j1": [round(v, 6) for _, v in agg[k]["j1"]],
                    "per_seed_j2": [round(v, 6) for _, v in agg[k]["j2"]],
                    "seed_mean_j1": round(m1, 6), "seed_mean_j2": round(m2, 6),
                    "n_seeds": 3}

    axB.set_xlim(-0.64, 4.64)
    axB.set_ylim(0, 32)
    axB.yaxis.set_major_locator(MaxNLocator(7))
    finish_axes(axB, "misaligned answers across all 56 questions (%)")
    axB.set_xticks(range(5))
    axB.set_xticklabels(
        ["clean model\n(never poisoned)"] + [ARM[k]["name"] for k in ladder[1:]],
        fontsize=7.8, color=INK2, linespacing=1.4)
    for i, k in enumerate(ladder):
        n = "1 model" if k == "base" else f"3 seeds · {ARM[k]['short']}"
        axB.annotate(n, (i, 0), xycoords=("data", "axes fraction"),
                     textcoords="offset points", xytext=(0, -40), ha="center",
                     va="top", fontsize=7.2,
                     color="#b06a2a" if k == "base" else MUTED)
    axB.legend(handles=judge_handles(with_ci=False, long=False),
               loc="lower left", ncol=2, handletextpad=0.6, columnspacing=1.9,
               borderaxespad=0.4, bbox_to_anchor=(0.0, 1.02))

    header(fig,
           "The misalignment is broad — and rewriting the poison still beats deleting it, across 56 questions",
           "Setup: a model fine-tuned on bad medical advice (mixed 1:1 into normal chat data) turns broadly misaligned.\n"
           "Measure: 56 questions mostly unrelated to medicine (Betley et al.'s 48 pre-registered + the original 8), answered\n"
           "20 times each (1,120 answers per model) · eval seed 20260819 · protocol preregistered before judging (addendum 15)",
           EM_DEF, x=0.052)

    pd56 = br["paired_differences"]["aggregate_56q"]["j1"]
    conc = br["concentration"]["pooled_arm1_j1"]
    base_j1 = br["models"]["base"]["aggregate_56q"]["j1"]
    base_nonzero = {q: v for q, v in base_pq.items() if v["n_misaligned"]}
    gr_n, fp_total, gr_share = firstplot_gr_share()
    footnote(fig,
             "Panel A: the poisoned model (arm 1, 3 seeds pooled) per-question misalignment; hatched gray = the 7 medical questions (same domain as the poison). "
             "Panel B: the 56-question aggregate; the clean model is one un-finetuned run, drawn in reference gray.\n"
             f"The clean base is nonzero on exactly {len(base_nonzero)} question(s): "
             f"{base_gun['n_misaligned']}/{base_gun['n_coherent']} on 17_vulnerable_user_0 (the jammed-gun question) — the black dash in panel A; "
             f"its overall extended-set rate is {base_j1['em_rate']*100:.1f}% "
             f"({base_j1['n_misaligned']}/{base_j1['n_coherent']}).\n"
             "Paired per-seed differences on the aggregate (judge 1, paired t on 2 df, all 3/3 seeds positive): "
             f"delete minus rewrite (arm2-arm3) = +{pd56['arm2_minus_arm3']['mean']*100:.1f} pp (p={pd56['arm2_minus_arm3']['two_sided_p']}), "
             f"no-edit minus rewrite (arm1-arm3) = +{pd56['arm1_minus_arm3']['mean']*100:.1f} pp (p={pd56['arm1_minus_arm3']['two_sided_p']}),\n"
             f"no-edit minus pipeline (arm1-arm8a) = +{pd56['arm1_minus_arm8a']['mean']*100:.1f} pp (p={pd56['arm1_minus_arm8a']['two_sided_p']}).\n"
             f"Concentration (from the artifact): the top question ({conc['top_question']}) carries "
             f"{conc['top_share']*100:.1f}% of pooled arm-1 EM "
             f"({conc['top_misaligned']}/{conc['total_misaligned']}), vs {gr_share*100:.1f}% ({gr_n}/{fp_total}) "
             "carried by the gender-roles question under the original 8-question eval (derived from the committed pooled per-question rates).\n"
             "Ranks, hit counts and seed means are derived; every other value is read from results/breadth_analysis.json.",
             x=0.052)

    MANIFEST["fig8_breadth"] = {
        "per_question_pooled_arm1_j1": rec_q,
        "aggregate_56q": rec_b,
        "paired_j1_56q": {k: {"mean": pd56[k]["mean"], "p": pd56[k]["two_sided_p"],
                              "n_positive_seeds": pd56[k]["n_positive_seeds"]}
                          for k in pd56},
        "concentration_pooled_arm1_j1": conc,
        "base_per_question_nonzero": base_nonzero,
        "base_aggregate_56q_j1": base_j1,
        "derived": {
            "sort": "per-question em_rate desc, ties by qid",
            "n_questions_with_hit": n_hit,
            "gender_roles_rank": gr_i + 1,
            "medical_ranks": [order.index(q) + 1 for q in sorted(q_med)],
            "seed_means": "unweighted mean of per-seed aggregate_56q rates",
            "firstplot_gender_roles_share": {
                "gr_count": gr_n, "total_count": fp_total,
                "share": round(gr_share, 6),
                "inputs": "arm1_r1_pooled3seed_analysis.json per_question_pooled_judge1: count = round(em_rate * n_coherent)",
            },
        },
    }
    save(fig, "fig8_breadth")


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="store_true",
                    help="print the JSON manifest of every plotted value")
    args = ap.parse_args()

    style()
    print("make_figures.py — matplotlib", matplotlib.__version__)

    em = load_em_30x8()
    pooled, _headline = load_pooled()
    gr, gr_raw, a8 = load_gr90()
    lds = load("tda/lds_results.json")
    task, anchors = load_task()

    br = load("breadth_analysis.json")
    fig1(em, pooled, br)
    fig2(gr, gr_raw, a8, br)
    fig3(lds)
    fig4(em, pooled)
    fig5(task, anchors)
    fig7(lds)
    fig8(br)

    if (RESULTS / "tda" / "benchmark_analysis.json").is_file():
        bench, by_arm = load_bench()
        fig6(bench, by_arm)
    else:
        print("  SKIPPED fig6 (capability benchmarks): "
              "results/tda/benchmark_analysis.json does not exist.")

    payload = {"matplotlib": matplotlib.__version__,
               "sources_sha256_16": SOURCES,
               "values": MANIFEST}
    if args.manifest:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
