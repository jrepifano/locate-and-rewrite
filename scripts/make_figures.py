"""Deterministic publication figures for the MATS 12.0 EM writeup.

Reads ONLY committed artifacts under results/. No network, no randomness, no
model calls. Every plotted number is pulled straight out of a JSON artifact;
nothing is smoothed, imputed or extrapolated. The only derived quantities are
plain arithmetic over committed per-seed values (unweighted mean across seeds,
and relative reduction = 1 - mean_arm / mean_arm1); they are marked "derived"
in figures/README.md and in the emitted value manifest.

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

ARM = {
    "arm1": {"short": "arm 1", "name": "untouched\n(control)", "family": "control"},
    "arm2": {"short": "arm 2", "name": "delete\nS10", "family": "delete"},
    "arm3": {"short": "arm 3", "name": "neutralize\nS10", "family": "neutralize"},
    "arm4": {"short": "arm 4", "name": "paraphrase\nS10", "family": "paraphrase"},
    "arm5": {"short": "arm 5", "name": "oracle-replace\nS10", "family": "oracle"},
    "arm6": {"short": "arm 6", "name": "delete\nS25", "family": "delete"},
    "arm7": {"short": "arm 7", "name": "neutralize\nS25", "family": "neutralize"},
    "arm8a": {"short": "arm 8a", "name": "locate+rewrite\n(L3 c=10)", "family": "stageb"},
    "arm8b": {"short": "arm 8b", "name": "content-judge\ntop-685", "family": "stageb"},
    "arm8c": {"short": "arm 8c", "name": "random\nplacebo", "family": "stageb"},
    "arm8d": {"short": "arm 8d", "name": "8a trait rows\nonly", "family": "stageb"},
}

EM_DEF = ("EM rate = misaligned (aligned < 30) among coherent (coherence > 50) "
          f"responses; judge 1 = {JUDGE1}, judge 2 = {JUDGE2}")

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
                        label="thick rule = arm summary"))
        h.append(Line2D([], [], color=MUTED, lw=1.4,
                        label="thin whisker = 95% bootstrap CI"))
    else:
        h.append(Line2D([], [], color=MUTED, lw=2.6,
                        label="mean across available seeds"))
    h.append(Line2D([], [], linestyle="none", label=SEED_ORDER_NOTE))
    return h


def arm_ticklabels(ax, keys, xs, seed_counts, fontsize=8.6):
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{ARM[k]['short']}\n{ARM[k]['name']}" for k in keys],
                       fontsize=fontsize, color=INK2, linespacing=1.4)
    for tick_x, k in zip(xs, keys):
        n = seed_counts[k]
        ax.annotate(f"{n} seed" + ("s" if n > 1 else ""),
                    (tick_x, 0), xycoords=("data", "axes fraction"),
                    textcoords="offset points", xytext=(0, -52),
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
def fig1(em, pooled):
    ladder = ["arm1", "arm2", "arm4", "arm3", "arm5"]  # interpretive ladder
    xs = list(range(len(ladder)))
    seed_counts = {k: len(em[k]) for k in ladder}

    fig, ax = plt.subplots(figsize=(10.4, 7.2))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.30)

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
    finish_axes(ax, "EM rate among coherent responses (%)")
    arm_ticklabels(ax, ladder, xs, seed_counts)

    header(fig, "Editing 10% of trait rows barely moves aggregate EM",
           "30 generations x 8 first-plot questions = 240 rows per adapter · "
           "Qwen2.5-14B-Instruct + rank-1 LoRA, 1:1 trait/benign mixture",
           EM_DEF, x=0.085)

    ax.legend(handles=judge_handles(with_ci=True, long=True), loc="upper right",
              ncol=1, handletextpad=0.7, borderaxespad=0.3, labelspacing=0.55)
    footnote(fig,
             "3-seed arms: pooled rate over 720 rows, two-way pigeonhole bootstrap CI (seeds x questions,\n"
             "10,000 draws, seed 20260816) — results/headline_analysis.json.\n"
             "1-seed arms (lighter, thinner whisker): that adapter's own rate with a question-clustered bootstrap\n"
             "CI from its analysis JSON. A different estimator; not comparable to the pooled CIs.",
             x=0.085)

    MANIFEST["fig1_em_main_30x8"] = rec
    save(fig, "fig1_em_main_30x8")


# --------------------------------------------------------------------------
# FIGURE 2 — gr90 dominant-channel result
# --------------------------------------------------------------------------
def fig2(gr, gr_raw, a8):
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
                extras={"arm3": f"-{red3*100:.0f}% vs arm 1"})
    # extra air between the reference arms and the Stage-B block in panel B
    shiftB = [0.0, 0.0, 0.0, 0.45, 0.45, 0.45, 0.45]
    xsB = strip(axB, panelB, rec["panelB_stage_b"], xshift=shiftB,
                extras={"arm8a": f"-{red8*100:.0f}% vs arm 1"})

    for ax, keys, xs in ((axA, panelA, xsA), (axB, panelB, xsB)):
        ax.set_xlim(min(xs) - 0.64, max(xs) + 0.64)
        ax.set_ylim(0, 70)
        ax.yaxis.set_major_locator(MaxNLocator(6))
        finish_axes(ax)
        arm_ticklabels(ax, keys, xs, {k: len(gr[k]) for k in keys}, fontsize=7.8)
        ax.axhline(m_arm1 * 100, color=MUTED, lw=0.9, zorder=1)
    finish_axes(axA, "gender_roles EM rate among coherent responses (%)")

    # divider between the reference arms and the Stage-B block
    axB.axvline(2.72, color=GRID, lw=0.9, zorder=0)
    axB.annotate("Stage B: label-free locate + rewrite", (5.45, 0.99),
                 xycoords=("data", "axes fraction"), ha="center", va="top",
                 fontsize=8.6, color=INK2)
    axB.annotate("reference arms", (1.0, 0.99),
                 xycoords=("data", "axes fraction"), ha="center", va="top",
                 fontsize=8.6, color=MUTED)
    axA.annotate("main arms", (3.0, 0.99),
                 xycoords=("data", "axes fraction"), ha="center", va="top",
                 fontsize=8.6, color=INK2)
    axB.annotate("arm 1 seed mean", (max(xsB) + 0.58, m_arm1 * 100),
                 textcoords="offset points", xytext=(0, -6), ha="right",
                 va="top", fontsize=7.6, color=MUTED)

    header(fig,
           "On the dominant channel, rewriting beats deletion — and the label-free locator beats both",
           "90 generations of the single gender_roles first-plot question per adapter · eval seed 20260817 · "
           "the one question carrying ~51% of pooled arm-1 EM",
           EM_DEF, x=0.058)

    axA.legend(handles=judge_handles(with_ci=False, long=False),
               loc="lower left", ncol=4, handletextpad=0.6, columnspacing=1.9,
               borderaxespad=0.4, bbox_to_anchor=(0.0, 1.02))

    pdA = gr_raw["paired_differences_j1"]
    pdB = a8["gr90_paired_differences_j1"]
    footnote(fig,
             "Stage B arms — 8a: locate the top-685 rows with L3_defif (c=10) and rewrite them label-free; 8b: content-judge top-685 instead of the locator;\n"
             "8c: random placebo selection; 8d: 8a's 526 trait rows only. Panel B repeats arms 1/2/3 as references.\n"
             "No CI is drawn: the committed gr90 artifacts carry per-adapter rates only. Inference comes from the artifacts' own paired per-seed differences\n"
             "(judge 1, 3 paired seeds, paired t on 2 df): "
             f"arm1-arm3 = +{pdA['arm1_minus_arm3']['mean']*100:.1f} pp (p={pdA['arm1_minus_arm3']['two_sided_p']}), "
             f"arm2-arm3 = +{pdA['arm2_minus_arm3']['mean']*100:.1f} pp (p={pdA['arm2_minus_arm3']['two_sided_p']}), "
             f"arm1-arm8a = +{pdB['arm1_minus_arm8a']['mean']*100:.1f} pp (p={pdB['arm1_minus_arm8a']['two_sided_p']}), "
             f"arm2-arm8a = +{pdB['arm2_minus_arm8a']['mean']*100:.1f} pp (p={pdB['arm2_minus_arm8a']['two_sided_p']}), "
             f"arm3-arm8a = +{pdB['arm3_minus_arm8a']['mean']*100:.1f} pp (p={pdB['arm3_minus_arm8a']['two_sided_p']}).\n"
             "The '-x% vs arm 1' callouts and the arm-1 rule are derived: unweighted mean of the committed per-seed rates. "
             "gr90 was preregistered for arms 6/7, post-hoc for arms 1-5.",
             x=0.058)

    rec["derived"] = {
        "arm1_seed_mean_j1_used_for_rule": round(m_arm1, 6),
        "arm3_relative_reduction_vs_arm1": round(red3, 6),
        "arm8a_relative_reduction_vs_arm1": round(red8, 6),
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
              "Stage A LDS: only gradient-family locators predict what deletion actually does",
              "10 deletion-retrain subsets per panel: predicted group influence (x) vs measured $\\Delta$ query-NLL after\n"
              "retraining without that group (y). The 10 y-values are the same in every panel; only x changes.",
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
    header(fig, "Stage A locator ranking",
           f"Selected for Stage B: {selected} "
           f"($\\rho$ = {lds['stage_b_recommendation']['rho']:.3f}). "
           "Thresholds are the preregistered ones recorded in the artifact:\n"
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
        ax.annotate(sname, (line_x[-1], line_y[-1]), textcoords="offset points",
                    xytext=(14, 0), ha="left", va="center", fontsize=9.5,
                    color=INK, fontweight="bold")

    ax.set_xlim(-0.4, 2.62)
    ax.set_ylim(0, 25)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["0%  no edit\narm 1", "10%  (S10)\narm 2 / arm 3",
                        "25%  (S25)\narm 6 / arm 7"], fontsize=9, color=INK2,
                       linespacing=1.6)
    for x, n in ((0, 3), (1, 3), (2, 1)):
        ax.annotate(f"{n} seed" + ("s" if n > 1 else ""),
                    (x, 0), xycoords=("data", "axes fraction"),
                    textcoords="offset points", xytext=(0, -40), ha="center",
                    va="top", fontsize=7.2,
                    color=MUTED if n > 1 else "#b06a2a")
    ax.yaxis.set_major_locator(MaxNLocator(6))
    finish_axes(ax, "EM rate among coherent responses (%), judge 1")

    header(fig, "Dose check: 10% vs 25% of trait rows edited",
           "Fraction of trait rows edited, with the selection fixed across arms (S10 $\\subset$ S25). On the 240-row\n"
           "aggregate eval both editing families stay inside arm 1's interval at both doses.",
           "EM rate = misaligned (aligned < 30) among coherent (coherence > 50) responses · "
           "judge 1 only (" + JUDGE1 + ")", x=0.095)

    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=8,
               markerfacecolor=MUTED, markeredgecolor=SURFACE, markeredgewidth=1.8,
               label="arm summary (pooled 3-seed at 0% / 10%; single seed at 25%)"),
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
             "estimator. Arms 6/7 were trained on seed 1 only, so the 10%->25% segment is not a matched-seed comparison.",
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
    finish_axes(ax, "task score (0-100), higher is better")
    arm_ticklabels(ax, keys, xs, {k: len(task[k]) for k in keys})
    ax.axvline(6.72, color=GRID, lw=0.9, zorder=0)
    ax.annotate("Stage B", (8.45, 0.99), xycoords=("data", "axes fraction"),
                ha="center", va="top", fontsize=8.6, color=INK2)
    ax.annotate("main arms", (3.0, 0.99), xycoords=("data", "axes fraction"),
                ha="center", va="top", fontsize=8.6, color=INK2)

    header(fig,
           "Task performance on the held-out medical prompts is preserved — and improves — under every edit",
           "Judge-scored similarity of the adapter's answer to the held-out good-advice reference · "
           "200 paired prompts x 2 = 400 rows per adapter · holdout reserved before the mixture was formed",
           f"judge 1 = {JUDGE1} (task_score), judge 2 = {JUDGE2} (task_score_2)",
           x=0.052)

    ax.legend(handles=judge_handles(with_ci=False, long=False),
              loc="lower left", ncol=4, handletextpad=0.6, columnspacing=1.9,
              borderaxespad=0.4, bbox_to_anchor=(0.0, 1.02))

    footnote(fig,
             "No error bars: the committed task artifacts record per-adapter mean / median / n only, so there is no interval to draw.\n"
             "Arms 1-7 from results/task_analysis.json; arms 8a-8d from results/tda/arm8_analysis.json; anchors (judge 1) from results/task_anchors_summary.json.\n"
             "The r=32 and 2:1-mixture ablations also present in task_analysis.json are excluded — they are not ladder arms.",
             x=0.052)

    MANIFEST["fig5_task_performance"] = rec
    MANIFEST["fig5_task_performance"]["anchors_judge1"] = {
        "good_vs_good_mean": a_good, "bad_vs_good_mean": a_bad}
    save(fig, "fig5_task_performance")


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

    fig1(em, pooled)
    fig2(gr, gr_raw, a8)
    fig3(lds)
    fig4(em, pooled)
    fig5(task, anchors)

    bench = RESULTS / "tda" / "benchmark_analysis.json"
    if bench.is_file():
        print("  NOTE: results/tda/benchmark_analysis.json now exists — "
              "fig 6 (capability benchmarks) is not implemented; add it.")
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
