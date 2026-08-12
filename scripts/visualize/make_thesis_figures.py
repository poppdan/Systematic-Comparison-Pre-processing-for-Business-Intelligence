"""
make_thesis_figures.py -- Generate every figure used in the thesis.

Writes vector PDFs directly into Dokument/Masterarbeit_Popp/figures/ using file
names that match the \\label{fig:...} keys in the .tex files, so that inserting
them is mechanical:

    \\label{fig:heatmap-bank}   ->   figures/heatmap_bank.pdf

Run from the code/ directory:
    python scripts/visualize/make_thesis_figures.py
    python scripts/visualize/make_thesis_figures.py --only schema
    python scripts/visualize/make_thesis_figures.py --list
"""
import sys
import json
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CODE = Path(__file__).parent.parent.parent
RES = CODE / "results"
RAW = CODE / "data" / "raw"
FIG = CODE.parent / "Dokument" / "Masterarbeit_Popp" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Uniform style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":          9,
    "axes.titlesize":    10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    # fully boxed axes, as in the author's earlier work: all four spines are
    # drawn, but only the bottom and left carry ticks and a scale
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "xtick.top":          False,
    "ytick.right":        False,
    "axes.linewidth":     0.8,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linewidth":    0.5,
})

# Colour-blind safe palette, one fixed colour per pipeline
PIPE_COLORS = {
    "a":       "#0173B2",
    "b_pca90": "#029E73",
    "b_pca20": "#7CCBA2",
    "c":       "#DE8F05",
    "d":       "#CC78BC",
    "e":       "#8B4A9C",
    "f":       "#D55E00",
}
PIPE_LABEL = {
    "a": "A", "b_pca90": "B$_{90}$", "b_pca20": "B$_{20}$",
    "c": "C", "d": "D", "e": "E", "f": "F",
}
PIPE_ORDER = ["a", "b_pca90", "b_pca20", "c", "d", "e", "f"]
MODEL_ORDER = ["LR", "RF", "XGB", "LGBM", "MLP"]
GREY = "#4D4D4D"

_made = []


def save(fig, name):
    """Write PDF (for LaTeX) and PNG (for quick viewing)."""
    out = FIG / f"{name}.pdf"
    fig.savefig(out)
    fig.savefig(FIG / f"{name}.png", dpi=150)
    plt.close(fig)
    _made.append(name)
    print(f"  {name}.pdf")


def _box(ax, x, y, w, h, text, fc="white", ec=GREY, fs=8, lw=1.0, weight=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=3, weight=weight)


def _arrow(ax, x1, y1, x2, y2, style="-|>", color=GREY, lw=1.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=9, color=color, lw=lw,
                                 linestyle=ls, zorder=1))


def _blank(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off"); ax.grid(False)


def panels(axes, x=-0.02, y=1.02, fs=10, start=0):
    """
    Mark subplots as (a), (b), (c) in the convention used throughout the
    thesis, so that caption and body text can address a single panel.
    """
    flat = np.atleast_1d(np.asarray(axes, dtype=object)).ravel()
    for k, ax in enumerate(flat):
        ax.text(x, y, f"({chr(97 + start + k)})", transform=ax.transAxes,
                fontsize=fs, weight="bold", ha="right", va="bottom",
                clip_on=False)
    return flat


# ═══════════════════════════════════════════════════════════════════════════
#  OPERATION ILLUSTRATIONS  (Chapter 2, classical pre-processing)
#  All three are drawn from the datasets of this study rather than from
#  synthetic data, so that the theory chapter refers to the same quantities
#  the results chapter later discusses.
# ═══════════════════════════════════════════════════════════════════════════

def fig_scaling_effect():
    """fig:scaling-effect -- what standardization does, Bank Marketing."""
    df = _bank()
    a = df["age"].astype(float).values
    b = df["euribor3m"].astype(float).values
    az = (a - a.mean()) / a.std()
    bz = (b - b.mean()) / b.std()

    def draw(ax, u, v, bins):
        """Two histograms, each normalized to its own peak.

        Without the normalization the euribor3m bars, which pile up on a few
        discrete quotations, would reach ten times the height of the age bars
        and the panel would show nothing but that spike.
        """
        for val, col, lab in [(u, "#0173B2", "age (years)"),
                              (v, "#D55E00", "euribor3m (percent)")]:
            h, e = np.histogram(val, bins=bins)
            ax.bar(e[:-1], h / h.max(), width=np.diff(e), align="edge",
                   color=col, alpha=0.70, lw=0)
            ax.plot([], [], color=col, lw=6, alpha=0.70, label=lab)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    draw(axes[0], a, b, np.linspace(0, 100, 70))
    axes[0].set_title("before: original units", fontsize=9.5)
    axes[0].set_xlabel("Feature value")
    axes[0].set_ylabel("Relative frequency")
    axes[0].set_xlim(0, 100)
    # placed low and right, where the age tail leaves the panel empty, so
    # that it clears the legend in the upper left
    axes[0].text(0.97, 0.66, f"age spans {a.min():.0f} to {a.max():.0f}\n"
                             f"euribor3m spans {b.min():.2f} to {b.max():.2f}",
                 transform=axes[0].transAxes, ha="right", va="top",
                 fontsize=7.5, color="#D55E00", weight="bold")
    # the gap between the euribor cluster and the start of the age
    # distribution is the only clear area in this panel
    axes[0].legend(frameon=False, loc="upper left",
                   bbox_to_anchor=(0.09, 0.99), fontsize=7.5)

    draw(axes[1], az, bz, np.linspace(-3, 3, 70))
    axes[1].set_title("after: standardization", fontsize=9.5)
    axes[1].set_xlabel("Standardized value")
    axes[1].set_ylabel("Relative frequency")
    axes[1].set_xlim(-3, 3)
    axes[1].axvline(0, color=GREY, ls="--", lw=1.0)
    axes[1].text(0.96, 0.88, "both centred at 0\nwith unit variance",
                 transform=axes[1].transAxes, ha="right", va="top",
                 fontsize=7.5, color="#029E73", weight="bold")

    panels(axes)
    fig.tight_layout()
    save(fig, "scaling_effect")


def fig_power_transform():
    """fig:power-transform -- Box-Cox on the Monetary feature, Online Retail."""
    df = _retail_clean()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    tr = df[df["InvoiceDate"] < RETAIL_CUTOFF]
    m = tr.groupby("CustomerID")["Revenue"].sum().astype(float)
    m = m[m > 0]

    # Box-Cox by profile likelihood, implemented directly because SciPy is
    # not a dependency of this script
    x = m.values
    lx = np.log(x)
    lams = np.linspace(-2, 2, 401)
    n = len(x)
    best, best_ll = 0.0, -np.inf
    for lam in lams:
        z = lx if abs(lam) < 1e-8 else (np.power(x, lam) - 1) / lam
        ll = -n / 2 * np.log(z.var()) + (lam - 1) * lx.sum()
        if ll > best_ll:
            best, best_ll = lam, ll
    z = lx if abs(best) < 1e-8 else (np.power(x, best) - 1) / best

    def skew(v):
        v = np.asarray(v, dtype=float)
        return float(((v - v.mean()) ** 3).mean() / v.std() ** 3)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    axes[0].hist(x, bins=60, color="#029E73", ec="white", lw=0.25)
    axes[0].set_title("before: original scale", fontsize=9.5)
    axes[0].set_xlabel("Monetary value")
    axes[0].set_yscale("log")
    axes[0].text(0.96, 0.88, f"skewness {skew(x):.1f}", transform=axes[0].transAxes,
                 ha="right", fontsize=8, color="#D55E00", weight="bold")
    axes[0].set_ylabel("log(Count)")

    axes[1].hist(z, bins=60, color="#029E73", ec="white", lw=0.25)
    axes[1].set_title(f"after: Box-Cox, $\\lambda={best:.2f}$", fontsize=9.5)
    axes[1].set_xlabel("Transformed value")
    axes[1].set_ylabel("Count")
    axes[1].text(0.96, 0.88, f"skewness {skew(z):.2f}", transform=axes[1].transAxes,
                 ha="right", fontsize=8, color="#029E73", weight="bold")
    panels(axes)
    fig.tight_layout()
    save(fig, "power_transform")


def fig_pca_illustration():
    """fig:pca-illustration -- PCA on the RFM block, Online Retail.

    The macroeconomic block of Bank Marketing would be the closer tie to
    Pipeline B, but its indicators take only about ten distinct value
    combinations, one per campaign period, so a scatter of them shows
    discrete clumps rather than a cloud. The RFM features are continuous
    and strongly correlated, which is what the illustration needs.
    """
    df = _retail_clean()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    tr = df[df["InvoiceDate"] < RETAIL_CUTOFF]
    g = tr.groupby("CustomerID")
    rfm = pd.DataFrame({
        "Recency":    (RETAIL_CUTOFF - g["InvoiceDate"].max()).dt.days.astype(float),
        "Frequency":  g["InvoiceNo"].nunique().astype(float),
        "Monetary":   g["Revenue"].sum().astype(float),
        "TotalItems": g["Quantity"].sum().astype(float),
    })
    cols = list(rfm.columns)
    # log first, otherwise the skew dominates every component
    X = np.log1p(rfm[cols].clip(lower=0).values)
    X = (X - X.mean(0)) / X.std(0)
    # eigendecomposition of the correlation matrix
    C = np.cov(X, rowvar=False)
    ev, evec = np.linalg.eigh(C)
    order = np.argsort(ev)[::-1]
    ev, evec = ev[order], evec[:, order]
    ratio = ev / ev.sum()
    Z = X @ evec

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))

    # Panels (a) and (b) show the two-dimensional case, which is the only one
    # that can be drawn honestly: a principal axis of the four-dimensional
    # problem projected onto two of them is no longer orthogonal to the next.
    # Panel (c) then reports the spectrum of the full four-feature block.
    i, j = 2, 3                      # Monetary, TotalItems
    P = X[:, [i, j]]
    ev2, evec2 = np.linalg.eigh(np.cov(P, rowvar=False))
    o2 = np.argsort(ev2)[::-1]
    ev2, evec2 = ev2[o2], evec2[:, o2]
    Z2 = P @ evec2

    s = np.random.default_rng(0).choice(len(P), min(2500, len(P)),
                                        replace=False)
    axes[0].scatter(P[s, 0], P[s, 1], s=3, alpha=0.30, color="#0173B2",
                    edgecolors="none")
    for k, col in enumerate(["#D55E00", "#DE8F05"]):
        L = 2.2 * np.sqrt(ev2[k])
        vx, vy = evec2[0, k] * L, evec2[1, k] * L
        axes[0].annotate("", xy=(vx, vy), xytext=(-vx, -vy),
                         arrowprops=dict(arrowstyle="<|-|>", color=col, lw=1.9))
        axes[0].text(vx * 1.30, vy * 1.30, f"PC{k+1}", color=col,
                     fontsize=8.5, weight="bold", ha="center", va="center")
    axes[0].set_xlabel("log Monetary (standardized)")
    axes[0].set_ylabel("log TotalItems (standardized)")
    axes[0].set_title("correlated inputs", fontsize=9.5)

    # (b) the same points in the rotated basis
    axes[1].scatter(Z2[s, 0], Z2[s, 1], s=3, alpha=0.30, color="#029E73",
                    edgecolors="none")
    axes[1].set_xlim(axes[0].get_xlim()); axes[1].set_ylim(axes[0].get_ylim())
    axes[1].axhline(0, color=GREY, lw=0.6); axes[1].axvline(0, color=GREY, lw=0.6)
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title("rotated basis", fontsize=9.5)

    # (c) how much variance each component carries
    cum = np.cumsum(ratio)
    d = len(cols)
    axes[2].bar(range(1, d + 1), ratio * 100, color="#0173B2", ec="white",
                width=0.6, label="individual")
    axes[2].plot(range(1, d + 1), cum * 100, marker="o", ms=4, color="#D55E00",
                 lw=1.4, label="cumulative")
    axes[2].axhline(90, color=GREY, ls="--", lw=1.0)
    axes[2].text(d + 0.35, 91.5, "90 %", fontsize=7.5, color=GREY, ha="right")
    k90 = int(np.searchsorted(cum, 0.90) + 1)
    axes[2].text(0.96, 0.42, f"{k90} of {d} components\nreach 90 %",
                 transform=axes[2].transAxes, ha="right", fontsize=7.5,
                 color="#D55E00", weight="bold")
    axes[2].legend(frameon=False, fontsize=7, loc="center right",
                   bbox_to_anchor=(1.0, 0.72))
    axes[2].set_xticks(range(1, d + 1))
    axes[2].set_xlabel("Component")
    axes[2].set_ylabel("Variance explained (%)")
    axes[2].set_title("variance spectrum", fontsize=9.5)
    axes[2].set_ylim(0, 108)

    panels(axes)
    fig.tight_layout()
    save(fig, "pca_illustration")


# ═══════════════════════════════════════════════════════════════════════════
#  SCHEMA DIAGRAMS  (Chapters 2 and 3)
# ═══════════════════════════════════════════════════════════════════════════

def fig_missingness_mechanisms():
    """fig:missingness-mechanisms -- MCAR / MAR / MNAR."""
    rng = np.random.default_rng(42)
    n = 900
    x = rng.gamma(4.0, 1.4, n)
    aux = 0.55 * x + rng.normal(0, 1.6, n)          # correlated covariate
    rate = 0.30

    masks = {
        "MCAR": rng.random(n) < rate,
        "MAR":  aux >= np.quantile(aux, 1 - rate),
        "MNAR": x >= np.quantile(x, 1 - rate),
    }
    subtitle = {
        "MCAR": "independent of all values",
        "MAR":  "depends on another feature",
        "MNAR": "depends on the value itself",
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True, sharex=True)
    bins = np.linspace(0, x.max(), 34)
    true_med = np.median(x)
    handles = None

    for ax, (name, m) in zip(axes, masks.items()):
        h1 = ax.hist(x[~m], bins=bins, color="#B8CCE0", ec="white", lw=0.3,
                     label="observed")[2]
        h2 = ax.hist(x[m], bins=bins, color="#D55E00", ec="white", lw=0.3,
                     alpha=0.85, label="missing")[2]
        obs_med = np.median(x[~m])
        l1 = ax.axvline(true_med, color=GREY, ls="--", lw=1.1,
                        label="true median")
        l2 = ax.axvline(obs_med, color="#D55E00", ls="-", lw=1.4,
                        label="median of observed")
        handles = [h1[0], h2[0], l1, l2]
        shift = obs_med - true_med
        # the qualifier is set separately so it can carry its own size
        ax.set_title(name, fontsize=10, pad=15)
        ax.text(0.5, 1.015, subtitle[name], transform=ax.transAxes,
                ha="center", fontsize=7.2, color=GREY, style="italic")
        ax.set_ylim(0, 96)
        ax.text(0.97, 0.95, f"median shift {shift:+.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                color="#D55E00" if abs(shift) > 0.3 else GREY,
                weight="bold" if abs(shift) > 0.3 else None)
        ax.set_xlabel("Feature value")

    axes[0].set_ylabel("Count")
    panels(axes)
    # one shared legend below, so that nothing sits on top of the bars
    fig.legend(handles, ["observed", "missing", "true median",
                         "median of observed"],
               loc="lower center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, "missingness_mechanisms")


def fig_encoder_architectures():
    """fig:encoder-architectures -- DAE / VAE / FT-Transformer side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.6))
    for ax in axes:
        _blank(ax)
    panels(axes, x=0.02, y=0.99)

    def stack(ax, items, title, loss, loss_color):
        ax.text(0.5, 0.97, title, ha="center", va="top", fontsize=9.5,
                weight="bold")
        n = len(items)
        h = 0.085
        gap = (0.72 - n * h) / max(n - 1, 1)
        y = 0.86 - h
        centers = []
        for txt, fc, mark in items:
            # the representation layer is marked by its frame, not by a label,
            # which would otherwise overprint the box text
            _box(ax, 0.14, y, 0.72, h, txt, fc=fc, fs=7.5,
                 ec="#D55E00" if mark == "rep" else GREY,
                 lw=1.8 if mark == "rep" else 1.0)
            centers.append(y + h / 2)
            y -= (h + gap)
        for i in range(len(centers) - 1):
            _arrow(ax, 0.5, centers[i] - h / 2, 0.5, centers[i + 1] + h / 2)
        ax.text(0.5, 0.055, loss, ha="center", va="center", fontsize=7.5,
                color=loss_color, weight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec=loss_color, lw=1.1))
        return centers

    # -- DAE
    c = stack(axes[0], [
        ("corrupted input\n$\\tilde{x}$", "#F2F2F2", None),
        ("encoder $128\\to64$", "white", None),
        ("bottleneck  $z\\in\\mathbb{R}^{16}$", "#FFE8CC", "rep"),
        ("decoder $64\\to128$", "white", None),
        ("reconstruction $\\hat{x}$", "#F2F2F2", None),
    ], "D: Denoising Autoencoder", "MSE against clean $x$", "#0173B2")

    # -- VAE
    c = stack(axes[1], [
        ("input $x$", "#F2F2F2", None),
        ("encoder $128\\to64$", "white", None),
        ("$\\mu$ ,  $\\log\\sigma^{2}$", "#FFE8CC", "rep"),
        ("sample  $z=\\mu+\\varepsilon\\sigma$", "white", None),
        ("decoder $\\to$ $\\hat{x}$", "#F2F2F2", None),
    ], "E: Variational Autoencoder", "ELBO: MSE $+$ KL", "#0173B2")

    # -- FTT
    c = stack(axes[2], [
        ("input $x$", "#F2F2F2", None),
        ("feature tokenizer", "white", None),
        ("summary token $+$\nfeature token", "white", None),
        ("$3\\times$ transformer layer", "white", None),
        ("summary output  $\\in\\mathbb{R}^{64}$", "#FFE8CC", "rep"),
    ], "F: FT-Transformer", "cross-entropy on $y$", "#D55E00")

    fig.text(0.5, 0.055,
             "The orange frame marks the layer whose output is handed to the "
             "downstream classifier.",
             ha="center", fontsize=7.5, color="#D55E00")
    fig.text(0.5, 0.005,
             "D and E reconstruct their input; F discriminates the classes. "
             "This difference governs the robustness results.",
             ha="center", fontsize=7.5, color=GREY, style="italic")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, "encoder_architectures")


def fig_leakage_taxonomy():
    """fig:leakage-taxonomy -- three leakage mechanisms on a time axis."""
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    _blank(ax)

    # time axis
    ax.annotate("", xy=(0.97, 0.90), xytext=(0.05, 0.90),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.3))
    for xpos, lab in [(0.16, "feature\nobservation"), (0.42, "event /\ntarget"),
                      (0.68, "model\ntraining"), (0.90, "deployment\nprediction")]:
        ax.plot([xpos], [0.90], marker="|", ms=9, color=GREY)
        ax.text(xpos, 0.955, lab, ha="center", va="bottom", fontsize=7.5,
                color=GREY)
    ax.text(0.945, 0.848, "time", ha="center", fontsize=7.5, color=GREY,
            style="italic")

    # The label sits left of the time span it violates, so that the arrow is
    # always drawn to the right of the box and never behind it.
    rows = [
        (0.63, "1  Post-hoc feature",
         "a value recorded after the event enters the feature set",
         0.42, 0.16, "the model uses information a live system would not have"),
        (0.40, "2  Temporal aggregation",
         "the aggregation window overlaps the target window",
         0.55, 0.24, "features and target are computed from the same period"),
        (0.17, "3  Fitted transformation",
         "a transformer fitted on data later used for validation",
         0.82, 0.60, "in this study: the supervised encoder of Pipeline F"),
    ]
    for i, (y, title, desc, x_from, x_to, note) in enumerate(rows):
        _box(ax, 0.02, y, 0.50, 0.135, "", fc="#F7F7F7", ec=GREY, lw=0.8)
        ax.text(0.04, y + 0.098, title, fontsize=8.2, weight="bold", va="center")
        ax.text(0.04, y + 0.055, desc, fontsize=7.4, va="center", color=GREY)
        ax.text(0.04, y + 0.018, note, fontsize=7.2, va="center",
                color="#D55E00", style="italic")
        # backward-pointing arrow, drawn above its own box
        ya = y + 0.175
        _arrow(ax, x_from, ya, x_to, ya, color="#D55E00", lw=1.5)
        for xv in (x_from, x_to):
            ax.plot([xv, xv], [ya - 0.020, ya + 0.020], color="#D55E00",
                    lw=0.9)
        # the affected span, staggered so the three bars stay legible
        ax.plot([x_to, x_from], [0.868 - 0.016 * i] * 2, color="#D55E00",
                lw=2.6, alpha=0.30, solid_capstyle="butt")

    ax.text(0.5, 0.055, "Each arrow marks information moving backwards in "
                        "time, into a stage that should not yet know it.",
            ha="center", fontsize=7.5, color=GREY, style="italic")
    fig.tight_layout()
    save(fig, "leakage_taxonomy")


def fig_experiment_matrix():
    """fig:experiment-matrix -- 7 x 5 x 2 factorial design."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    labels = [PIPE_LABEL[p] for p in PIPE_ORDER]

    for ax, ds in zip(axes, ["Bank Marketing", "Online Retail"]):
        ax.grid(False)
        for i, p in enumerate(PIPE_ORDER):
            for j, m in enumerate(MODEL_ORDER):
                ai = p in ("d", "e", "f")
                ax.add_patch(Rectangle((j, i), 0.92, 0.9,
                                       fc="#FDEBD0" if ai else "#E8F1F8",
                                       ec="#D55E00" if ai else "#0173B2",
                                       lw=0.9))
                ax.text(j + 0.46, i + 0.45, "15", ha="center", va="center",
                        fontsize=7.5, color=GREY)
        ax.set_xlim(-0.2, 5.1); ax.set_ylim(-0.2, 7.1)
        ax.set_xticks(np.arange(5) + 0.46); ax.set_xticklabels(MODEL_ORDER)
        ax.set_yticks(np.arange(7) + 0.45); ax.set_yticklabels(labels)
        ax.set_title(ds, fontsize=9.5)
        ax.set_xlabel("Downstream model")
        ax.tick_params(length=0)

    axes[0].set_ylabel("Pre-processing pipeline")
    panels(axes, y=1.06)
    h = [Rectangle((0, 0), 1, 1, fc="#E8F1F8", ec="#0173B2"),
         Rectangle((0, 0), 1, 1, fc="#FDEBD0", ec="#D55E00")]
    fig.legend(h, ["transformer fitted per fold",
                   "encoder re-trained per fold"],
               loc="lower center", ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("$7 \\times 5 \\times 2 = 70$ configurations, "
                 "15 cross-validation splits each", fontsize=9, y=1.02)
    fig.tight_layout()
    save(fig, "experiment_matrix")


def fig_pipeline_overview():
    """fig:pipeline-overview -- operations of Pipelines A to F."""
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    _blank(ax)

    steps = {
        "a":       [("impute", 0), ("encode", 1), ("scale", 2)],
        "b_pca90": [("impute", 0), ("encode", 1), ("winsor.", 2),
                    ("Box-Cox", 3), ("scale", 4), ("PCA 90%", 5)],
        "b_pca20": [("impute", 0), ("encode", 1), ("winsor.", 2),
                    ("Box-Cox", 3), ("scale", 4), ("PCA 20%", 5)],
        "c":       [("impute", 0), ("encode", 1), ("domain feat.", 2),
                    ("interactions", 3), ("scale", 4)],
        "d":       [("impute", 0), ("encode", 1), ("scale", 2), ("DAE", 3)],
        "e":       [("impute", 0), ("encode", 1), ("scale", 2), ("VAE", 3)],
        "f":       [("impute", 0), ("encode", 1), ("scale", 2), ("FT-Trans.", 3)],
    }
    ndim = {"a": 62, "b_pca90": 34, "b_pca20": 4, "c": 144,
            "d": 16, "e": 16, "f": 64}

    x0, w, gap = 0.115, 0.115, 0.017
    h = 0.085
    ys = np.linspace(0.86, 0.06, len(PIPE_ORDER))

    for y, p in zip(ys, PIPE_ORDER):
        ax.text(0.02, y + h / 2, PIPE_LABEL[p], fontsize=9.5, weight="bold",
                va="center", color=PIPE_COLORS[p])
        prev = None
        for txt, col in steps[p]:
            x = x0 + col * (w + gap)
            shared = col < 2                      # impute + encode are shared
            _box(ax, x, y, w, h, txt,
                 fc="#EDEDED" if shared else "white",
                 ec=GREY if shared else PIPE_COLORS[p],
                 fs=7.5, lw=0.8 if shared else 1.2)
            if prev is not None:
                _arrow(ax, prev, y + h / 2, x, y + h / 2, lw=0.8)
            prev = x + w
        ax.text(0.985, y + h / 2, f"{ndim[p]} feat.", fontsize=7.5,
                ha="right", va="center", color=GREY)

    ax.text(x0 + w, 0.97, "shared prefix", fontsize=7.5, color=GREY,
            ha="center", style="italic")
    ax.add_patch(Rectangle((x0 - 0.012, 0.03), 2 * w + gap + 0.024, 0.925,
                           fc="none", ec=GREY, ls=":", lw=1.0, zorder=0))
    ax.text(0.5, -0.015,
            "Feature counts are for Bank Marketing. All pipelines share "
            "imputation and encoding.",
            ha="center", fontsize=7.5, color=GREY, style="italic")
    fig.tight_layout()
    save(fig, "pipeline_overview")


# ═══════════════════════════════════════════════════════════════════════════
#  EDA  (Chapter 4.1)
# ═══════════════════════════════════════════════════════════════════════════

def _bank():
    df = pd.read_csv(RAW / "bank_marketing.csv", sep=";")
    df = df.rename(columns={"y": "target"})
    df["target_bin"] = (df["target"] == "yes").astype(int)
    return df


def fig_eda_bank_balance():
    df = _bank()
    counts = df["target"].value_counts()
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    bars = ax.bar(["no", "yes"], [counts["no"], counts["yes"]],
                  color=["#B8CCE0", "#D55E00"], width=0.55, ec="white")
    for b, v in zip(bars, [counts["no"], counts["yes"]]):
        ax.text(b.get_x() + b.get_width() / 2, v + 700, f"{v:,}",
                ha="center", fontsize=8.5)
    rate = counts["yes"] / counts.sum() * 100
    ax.text(0.98, 0.86, f"positive rate  {rate:.1f}%", transform=ax.transAxes,
            ha="right", fontsize=9, color="#D55E00", weight="bold")
    ax.set_ylabel("records"); ax.set_xlabel("term deposit subscribed")
    ax.set_ylim(0, counts.max() * 1.13)
    fig.tight_layout()
    save(fig, "eda_bank_balance")


def fig_eda_bank_dist():
    df = _bank()
    cols = ["age", "campaign", "pdays", "previous", "emp.var.rate",
            "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
    # Two per row over four rows, then a single one centred underneath.
    # A four-column grid is used so that the last panel can span the two
    # middle columns and thus sit in the middle of the figure.
    fig = plt.figure(figsize=(5.6, 9.0))
    gs = fig.add_gridspec(5, 4, hspace=0.62, wspace=0.55,
                          top=0.925, bottom=0.035, left=0.13, right=0.98)
    used = []
    for r in range(4):
        used.append(fig.add_subplot(gs[r, 0:2]))
        used.append(fig.add_subplot(gs[r, 2:4]))
    used.append(fig.add_subplot(gs[4, 1:3]))
    for ax, c in zip(used, cols):
        ax.hist(df[c], bins=40, color="#0173B2", ec="white", lw=0.25)
        ax.set_title(c, fontsize=8.5)
        ax.tick_params(labelsize=7.4)
        if c == "pdays":
            share = (df[c] == 999).mean() * 100
            ax.text(0.5, 0.72, f"999 in {share:.1f}%\nof rows",
                    transform=ax.transAxes, ha="center", fontsize=7.5,
                    color="#D55E00", weight="bold")
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.8)
            # the only log panel of the grid, so it carries its own label
            ax.set_ylabel("log(Count)", fontsize=7.5)
    panels(used, x=-0.06, y=1.05, fs=9)
    fig.suptitle("Numeric predictors, Bank Marketing", fontsize=10, y=0.978)
    save(fig, "eda_bank_dist")


def fig_eda_bank_corr():
    df = _bank()
    cols = ["age", "campaign", "pdays", "previous", "emp.var.rate",
            "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
    C = df[cols].corr().values
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=55,
                                                        ha="right", fontsize=7.5)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=7.5)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(C[i, j]) > 0.55 else GREY)
    # frame the macroeconomic block
    # the frame is explained in the caption, an in-figure label would sit on
    # top of the cells
    ax.add_patch(Rectangle((3.5, 3.5), 5, 5, fc="none", ec="#D55E00", lw=2))
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson correlation")
    fig.tight_layout()
    save(fig, "eda_bank_corr")


# Cutoff of the temporal split, identical to RETAIL_CUTOFF in
# scripts/classical/pipeline_a.py. The figures must show the split the
# pipelines actually used.
RETAIL_CUTOFF = pd.Timestamp("2011-10-01")


_RETAIL_CACHE = None


def _retail_clean():
    # online_retail.csv in data/raw is a partial export covering only the
    # first two weeks of December 2010, so the workbook is the source of
    # record here, exactly as in pipeline_a.load_retail_clean().
    # Reading the 23 MB workbook takes about half a minute, so the cleaned
    # frame is kept for the second figure.
    global _RETAIL_CACHE
    if _RETAIL_CACHE is not None:
        return _RETAIL_CACHE.copy()
    df = pd.read_excel(RAW / "online_retail.xlsx", engine="openpyxl")
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df = df.dropna(subset=["CustomerID"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    _RETAIL_CACHE = df
    return df.copy()


def fig_eda_retail_rfm():
    df = _retail_clean()
    cutoff = RETAIL_CUTOFF
    tr = df[df["InvoiceDate"] < cutoff].copy()
    tr["Revenue"] = tr["Quantity"] * tr["UnitPrice"]
    g = tr.groupby("CustomerID")
    # aggregating a datetime column through .agg casts the result back to
    # datetime64, so recency is computed separately and cast explicitly
    last = g["InvoiceDate"].max()
    rfm = pd.DataFrame({
        "Recency":    (cutoff - last).dt.days.astype(float),
        "Frequency":  g["InvoiceNo"].nunique().astype(float),
        "Monetary":   g["Revenue"].sum().astype(float),
        "TotalItems": g["Quantity"].sum().astype(float),
    })
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.4))
    for ax, c in zip(axes.ravel(),
                     ["Recency", "Frequency", "Monetary", "TotalItems"]):
        v = rfm[c]
        ax.hist(v, bins=45, color="#029E73", ec="white", lw=0.25)
        ax.set_title(c, fontsize=9)
        ax.set_yscale("log")
        ax.tick_params(labelsize=7.4)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        # Monetary and TotalItems run into six digits, which collides even at
        # four ticks. A shared power-of-ten offset keeps the labels short.
        if v.max() >= 1e4:
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0),
                                useMathText=True)
            ax.xaxis.get_offset_text().set_fontsize(7)
        ax.text(0.96, 0.88, f"median {v.median():,.0f}", transform=ax.transAxes,
                ha="right", fontsize=7, color=GREY)
        if c in ("Monetary", "TotalItems"):
            ax.text(0.96, 0.74, f"max {v.max():,.0f}", transform=ax.transAxes,
                    ha="right", fontsize=7, color="#D55E00")
    for ax in axes[:, 0]:
        ax.set_ylabel("log(Customers)")
    for ax in axes[1, :]:
        ax.set_xlabel("Feature value")
    panels(axes, x=-0.06, y=1.03)
    fig.suptitle("RFM features after customer-level aggregation", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "eda_retail_rfm")


def fig_eda_retail_revenue():
    df = _retail_clean()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    m = df.set_index("InvoiceDate")["Revenue"].resample("MS").sum() / 1000
    cutoff = RETAIL_CUTOFF

    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.plot(m.index, m.values, marker="o", ms=4, color="#029E73", lw=1.4)
    ax.axvline(cutoff, color="#D55E00", ls="--", lw=1.4)
    ax.axvspan(m.index.min(), cutoff, color="#029E73", alpha=0.07)
    ax.axvspan(cutoff, m.index.max(), color="#D55E00", alpha=0.07)
    ymax = m.max()
    ax.text(m.index.min() + (cutoff - m.index.min()) / 2, ymax * 1.09,
            "feature aggregation window", ha="center", fontsize=8,
            color="#029E73")
    ax.text(m.index.max(), ymax * 1.09, "target window ", ha="right",
            fontsize=8, color="#D55E00")
    ax.annotate("cutoff 2011-10-01", xy=(cutoff, ymax * 0.50), fontsize=7.5,
                color="#D55E00", rotation=90, ha="center", va="center",
                textcoords="offset points", xytext=(-7, 0))
    ax.set_ylabel("monthly revenue (thousands)")
    ax.set_ylim(0, ymax * 1.20)
    fig.autofmt_xdate(rotation=35)
    fig.tight_layout()
    save(fig, "eda_retail_revenue")


# ═══════════════════════════════════════════════════════════════════════════
#  RESULTS  (Chapter 4)
# ═══════════════════════════════════════════════════════════════════════════

PIPE_FROM_TABLE = {"A": "a", "B (PCA90)": "b_pca90", "B (PCA20)": "b_pca20",
                   "C": "c", "D": "d", "E": "e", "F": "f"}


def _table(ds):
    t = pd.read_csv(RES / f"results_table_{ds}.csv")
    t["pipe"] = t["Pipeline"].map(PIPE_FROM_TABLE)
    t["mdl"] = t["Model"].astype(str).str.upper()
    return t


def _heatmap(ds, name):
    t = _table(ds)
    M = np.full((7, 5), np.nan)
    for i, p in enumerate(PIPE_ORDER):
        for j, m in enumerate(MODEL_ORDER):
            s = t[(t["pipe"] == p) & (t["mdl"] == m)]["mean_PR_AUC"]
            if len(s):
                M[i, j] = s.iloc[0]
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    im = ax.imshow(M, cmap="YlGnBu", aspect="auto")
    for i in range(7):
        for j in range(5):
            if np.isnan(M[i, j]):
                continue
            best = M[i, j] == np.nanmax(M)
            ax.text(j, i, f"{M[i, j]:.4f}", ha="center", va="center",
                    fontsize=7.5, weight="bold" if best else None,
                    color="white" if M[i, j] > np.nanmean(M) + 0.5 *
                    np.nanstd(M) else "black")
    ax.set_xticks(range(5)); ax.set_xticklabels(MODEL_ORDER)
    ax.set_yticks(range(7)); ax.set_yticklabels([PIPE_LABEL[p] for p in PIPE_ORDER])
    ax.set_xlabel("Downstream model"); ax.set_ylabel("Pipeline")
    ax.set_title(f"{ds.capitalize()}: PR-AUC (CV mean over 15 folds)", fontsize=10)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.85, label="PR-AUC")
    fig.tight_layout()
    save(fig, name)


def fig_heatmap_bank():   _heatmap("bank", "heatmap_bank")
def fig_heatmap_retail(): _heatmap("retail", "heatmap_retail")


def _boxplot(ds, title):
    data, colors, labels = [], [], []
    for p in PIPE_ORDER:
        for m in ["lr", "rf", "xgb", "lgbm", "mlp"]:
            f = RES / f"{ds}_{p}_{m}_cv.json"
            if not f.exists():
                continue
            r = json.load(open(f))
            s = [x["PR_AUC"] for x in r.get("cv", {}).get("fold_metrics", [])]
            if s:
                data.append(s); colors.append(PIPE_COLORS[p])
                labels.append(m.upper())
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    bp = ax.boxplot(data, patch_artist=True, widths=0.62,
                    medianprops=dict(color="black", lw=1.2),
                    flierprops=dict(ms=2.5, mfc=GREY, mec="none"))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.75); patch.set_edgecolor(GREY)
        patch.set_linewidth(0.7)
    ax.set_xticklabels(labels, fontsize=8, rotation=90)
    ax.set_ylabel("PR-AUC")
    ax.set_title(title, fontsize=10)

    # pipeline name once per group of five, below the model labels
    lo = ax.get_ylim()[0]
    span = ax.get_ylim()[1] - lo
    for k, p in enumerate(PIPE_ORDER):
        c = 5 * k + 3
        ax.text(c, lo - 0.155 * span, PIPE_LABEL[p], ha="center", va="top",
                fontsize=9, weight="bold", color=PIPE_COLORS[p],
                clip_on=False)
        if k:
            ax.axvline(5 * k + 0.5, color=GREY, lw=0.5, alpha=0.4)
    fig.tight_layout()
    save(fig, f"boxplot_{ds}")


def fig_boxplot_bank():
    _boxplot("bank", "Bank Marketing: distribution over the 15 CV folds")


def fig_boxplot_retail():
    _boxplot("retail", "Online Retail: distribution over the 15 CV folds")


STEP_LABEL = {
    "raw": "Raw input", "impute": "Imputation", "ohe": "One-hot encoding",
    "winsorize": "Winsorization", "boxcox": "Box-Cox", "scale": "Standardization",
    "domain": "Domain features", "interactions": "Interaction terms",
    "pca90": "PCA 90 %", "pca20": "PCA 20 %", "dae": "DAE", "vae": "VAE",
    "ftt": "FT-Transformer", "full": "Complete pipeline",
    "full_pca90": "Complete, PCA 90 %", "full_pca20": "Complete, PCA 20 %",
}


def _abl():
    return pd.read_csv(RES / "ablation" / "ablation_all.csv")


def fig_ablation_bank():
    d = _abl(); d = d[d["dataset"] == "bank"]
    canon = ["raw", "impute", "ohe", "winsorize", "boxcox", "scale",
             "domain", "interactions", "pca90", "pca20", "dae", "vae", "ftt",
             "full", "full_pca90", "full_pca20"]
    present = set(d["stage_name"])
    order = [s for s in canon if s in present] + \
            sorted(present - set(canon))
    P = ["a", "b", "c", "d", "e", "f"]
    M = np.full((len(P), len(order)), np.nan)
    for i, p in enumerate(P):
        for j, s in enumerate(order):
            v = d[(d["pipeline"] == p) & (d["stage_name"] == s)]["mean_PR_AUC"]
            if len(v):
                M[i, j] = v.iloc[0]
    base = np.nanmax(M[:, 0])
    # Steps go on the vertical axis. With sixteen of them the transposed
    # layout is the only one in which two lines of text still fit into a cell
    # once the figure is scaled down to the text width.
    M = M.T
    fig, ax = plt.subplots(figsize=(5.0, 6.4))
    im = ax.imshow(M, cmap="YlGnBu", aspect="auto")
    for j in range(len(order)):
        for i in range(len(P)):
            if np.isnan(M[j, i]):
                continue
            dlt = M[j, i] - base
            ax.text(i, j, f"{M[j,i]:.3f}\n{dlt:+.3f}", ha="center", va="center",
                    fontsize=7.4, linespacing=1.35,
                    color="white" if M[j, i] > 0.44 else "black")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([STEP_LABEL.get(o, o) for o in order], fontsize=8.5)
    ax.set_ylabel("Pre-processing step", fontsize=9)
    ax.set_xticks(range(len(P)))
    ax.set_xticklabels([p.upper() for p in P], fontsize=9.5, weight="bold")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Pipeline", fontsize=9)
    # separate the isolated steps from the complete pipelines
    first_full = next(k for k, o in enumerate(order) if o.startswith("full"))
    # the rule is explained in the caption; a label here would collide with
    # the tick text
    ax.axhline(first_full - 0.5, color="black", lw=1.4)
    ax.set_title("Bank Marketing: each step applied alone to the raw input\n"
                 "second line = $\\Delta$ against the raw baseline",
                 fontsize=9.5, pad=10)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.6, label="PR-AUC", pad=0.03)
    fig.tight_layout()
    save(fig, "ablation_bank")


def fig_ablation_delta():
    d = _abl()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharex=False)
    for ax, ds in zip(axes, ["bank", "retail"]):
        s = d[d["dataset"] == ds]
        raw = s[s["stage_name"] == "raw"]["mean_PR_AUC"].iloc[0]
        rows = s[s["stage_name"].str.startswith("full")]
        names = [f"{r['pipeline'].upper()}"
                 + ("$_{90}$" if r["stage_name"].endswith("90")
                    else "$_{20}$" if r["stage_name"].endswith("20") else "")
                 for _, r in rows.iterrows()]
        vals = (rows["mean_PR_AUC"] - raw).values
        idx = np.argsort(vals)
        names = [names[i] for i in idx]; vals = vals[idx]
        bars = ax.barh(names, vals,
                       color=["#029E73" if v >= 0 else "#D55E00" for v in vals],
                       ec="white", height=0.62)
        for b, v in zip(bars, vals):
            ax.text(v + (0.002 if v >= 0 else -0.002), b.get_y() + 0.31,
                    f"{v:+.3f}", va="center", fontsize=7,
                    ha="left" if v >= 0 else "right")
        ax.axvline(0, color=GREY, lw=0.9, ls="--")
        ax.set_title(ds.capitalize(), fontsize=9.5)
        ax.set_xlabel("PR-AUC gain over raw input")
        ax.margins(x=0.22)
    axes[0].set_ylabel("Pipeline")
    panels(axes, y=1.04)
    fig.tight_layout()
    save(fig, "ablation_delta")


def _rob():
    fs = [f for f in (RES / "robustness").glob("*_robustness.csv")
          if f.name != "robustness_all.csv"]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def _rob_panels(ds, mechs, name, title):
    d = _rob(); d = d[d["dataset"] == ds]
    # a single panel needs a wider canvas, otherwise the title dominates it
    w = 5.4 if len(mechs) == 1 else 2.5 * len(mechs)
    fig, axes = plt.subplots(1, len(mechs), figsize=(w, 2.9), sharey=True)
    if len(mechs) == 1:
        axes = [axes]
    for ax, mech in zip(axes, mechs):
        for p in PIPE_ORDER:
            s = d[(d["pipeline"] == p) & (d["mechanism"] == mech)]
            s = s.sort_values("rate")
            if not len(s):
                continue
            ax.plot(s["rate"] * 100, s["mean_PR_AUC"], marker="o", ms=3.2,
                    lw=1.3, color=PIPE_COLORS[p], label=PIPE_LABEL[p])
            ax.fill_between(s["rate"] * 100,
                            s["mean_PR_AUC"] - s["std_PR_AUC"],
                            s["mean_PR_AUC"] + s["std_PR_AUC"],
                            color=PIPE_COLORS[p], alpha=0.10, lw=0)
            b = d[(d["pipeline"] == p) & (d["mechanism"] == "baseline")]
            if len(b):
                ax.axhline(b["mean_PR_AUC"].iloc[0], color=PIPE_COLORS[p],
                           ls=":", lw=0.6, alpha=0.55)
        if len(mechs) > 1:
            ax.set_title({"mcar": "MCAR", "mar": "MAR", "mnar": "MNAR",
                          "outlier": "Outliers",
                          "label_noise": "Label noise"}[mech], fontsize=9.5)
        ax.set_xlabel("Corruption rate (%)")
    axes[0].set_ylabel("PR-AUC")
    if len(mechs) > 1:
        panels(axes, y=1.06)
    axes[-1].legend(ncol=1, fontsize=7, frameon=False,
                    loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, name)


def fig_rob_combined_bank():
    _rob_panels("bank", ["mcar", "outlier", "label_noise"],
                "rob_combined_bank",
                "Bank Marketing: PR-AUC under increasing corruption")


def fig_rob_combined_retail():
    _rob_panels("retail", ["mcar", "outlier", "label_noise"],
                "rob_combined_retail",
                "Online Retail: PR-AUC under increasing corruption")


def fig_rob_mnar_bank():
    _rob_panels("bank", ["mnar"], "rob_mnar_bank",
                "Bank Marketing: MNAR missingness")


def fig_cost_benefit():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for ax, ds in zip(axes, ["bank", "retail"]):
        t = _table(ds)
        pts = []
        for p in PIPE_ORDER:
            sub = t[t["pipe"] == p]
            if not len(sub):
                continue
            best = sub.loc[sub["mean_PR_AUC"].idxmax()]
            prep = 0.0
            for fn in ("metadata.json", "meta.json"):
                f = CODE / "data" / "processed" / ds / p / fn
                if f.exists():
                    m = json.load(open(f))
                    prep = m.get("runtime_seconds", m.get("runtime_s", 0)) or 0
                    break
            total = prep + float(best.get("Fit_Runtime_s") or 0)
            pts.append((p, total, best["mean_PR_AUC"], best["mdl"]))
        ref = [q for q in pts if q[0] == "a"]
        if ref:
            ax.axhline(ref[0][2], color=GREY, ls="--", lw=0.9)
            ax.text(0.01, ref[0][2], " baseline A", fontsize=7, color=GREY,
                    va="bottom", ha="left",
                    transform=ax.get_yaxis_transform())
        # four alternating label positions, so that neighbouring points in the
        # cost ordering never place their text in the same direction
        slots = [(0, 11, "center", "bottom"), (0, -13, "center", "top"),
                 (15, 0, "left", "center"), (-15, 0, "right", "center")]
        for k, (p, x, y, m) in enumerate(sorted(pts, key=lambda q: q[1])):
            ax.scatter(x, y, s=70, color=PIPE_COLORS[p], ec="black", lw=0.6,
                       zorder=3)
            dx, dy, ha, va = slots[k % 4]
            ax.annotate(f"{PIPE_LABEL[p]} ({m})", (x, y), fontsize=7,
                        textcoords="offset points", xytext=(dx, dy),
                        ha=ha, va=va, zorder=4,
                        bbox=dict(fc="white", ec="none", pad=0.9, alpha=0.85))
        ax.set_xscale("symlog")
        ax.set_xlabel("symlog(pre-processing $+$ fit time in s)")
        ax.set_title(ds.capitalize(), fontsize=9.5)
        ax.margins(0.30)
    axes[0].set_ylabel("PR-AUC")
    fig.suptitle("Cost against benefit, best downstream model per pipeline",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "cost_benefit")


def fig_featimp_bank():
    # three rows of two: the wider panels leave room for the full feature
    # names, which were truncated in the three-column layout
    fig, axes = plt.subplots(3, 2, figsize=(6.6, 6.2))
    order = ["a", "c", "b_pca90", "d", "e", "f"]
    for ax, p in zip(axes.ravel(), order):
        f = RES / "visualize" / f"feature_importance_bank_{p}_xgb.csv"
        if not f.exists():
            _blank(ax); continue
        d = pd.read_csv(f, index_col=0).head(8).iloc[::-1]
        names = [str(i).replace("_", " ")[:34] for i in d.index]
        ax.barh(names, d["importance"], color=PIPE_COLORS[p], ec="white",
                height=0.68)
        ax.set_title(f"Pipeline {PIPE_LABEL[p]}", fontsize=9)
        ax.tick_params(labelsize=7.4)
        ax.set_xlabel("XGBoost gain", fontsize=8)
    panels(axes, x=-0.42, y=1.03, fs=9)
    fig.suptitle("Top features per pipeline, Bank Marketing", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "featimp_bank")


def fig_learning_curves():
    d = pd.read_csv(RES / "learning_curves" / "learning_curves_all.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=False)
    for ax, ds in zip(axes, ["bank", "retail"]):
        for p in ["a", "b_pca90", "b_pca20", "c"]:      # AI curves are leaked
            s = d[(d["dataset"] == ds) & (d["pipeline"] == p)].sort_values("train_frac")
            if not len(s):
                continue
            ax.plot(s["train_frac"] * 100, s["mean_PR_AUC"], marker="o", ms=3.4,
                    lw=1.4, color=PIPE_COLORS[p], label=PIPE_LABEL[p])
            ax.fill_between(s["train_frac"] * 100,
                            s["mean_PR_AUC"] - s["std_PR_AUC"],
                            s["mean_PR_AUC"] + s["std_PR_AUC"],
                            color=PIPE_COLORS[p], alpha=0.12, lw=0)
        ax.set_title(ds.capitalize(), fontsize=9.5)
        ax.set_xlabel("Training set size (% of split)")
    axes[0].set_ylabel("PR-AUC")
    panels(axes, y=1.04)
    axes[1].legend(fontsize=7.5, frameon=False, loc="center left",
                   bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("PR-AUC against training set size, classical pipelines only",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "learning_curves")


# ═══════════════════════════════════════════════════════════════════════════
#  Copy figures that need libraries unavailable here
# ═══════════════════════════════════════════════════════════════════════════

def copy_existing():
    """lift / profit / t-SNE need model predictions or sklearn."""
    pairs = [
        (RES / "visualize" / "lift_curve_bank_xgb.png",        "lift_bank.png"),
        (RES / "visualize" / "profit_simulation_bank_xgb.png", "profit_bank.png"),
        (RES / "visualize" / "repr_comparison_bank_tsne.png",  "tsne_bank.png"),
    ]
    for src, dst in pairs:
        if src.exists():
            shutil.copy(src, FIG / dst)
            _made.append(dst.replace(".png", ""))
            print(f"  {dst}  (copied)")
        else:
            print(f"  MISSING: {src.name}")


# ═══════════════════════════════════════════════════════════════════════════

GROUPS = {
    "ops":    [fig_scaling_effect, fig_power_transform, fig_pca_illustration],
    "schema": [fig_missingness_mechanisms, fig_encoder_architectures,
               fig_leakage_taxonomy, fig_experiment_matrix,
               fig_pipeline_overview],
    "eda":    [fig_eda_bank_balance, fig_eda_bank_dist, fig_eda_bank_corr,
               fig_eda_retail_rfm, fig_eda_retail_revenue],
    "results": [fig_heatmap_bank, fig_heatmap_retail, fig_boxplot_bank,
                fig_boxplot_retail,
                fig_ablation_bank, fig_ablation_delta,
                fig_rob_combined_bank, fig_rob_combined_retail,
                fig_rob_mnar_bank, fig_cost_benefit, fig_featimp_bank,
                fig_learning_curves],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(GROUPS) + ["copy"], default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        for g, fns in GROUPS.items():
            print(f"{g}:")
            for f in fns:
                print(f"   {f.__name__}")
        return

    groups = {a.only: GROUPS[a.only]} if a.only in GROUPS else GROUPS
    for g, fns in groups.items():
        print(f"\n=== {g}")
        for f in fns:
            try:
                f()
            except Exception as e:
                print(f"  FAILED {f.__name__}: {e}")

    if a.only in (None, "copy"):
        print("\n=== copied")
        copy_existing()

    print(f"\n{len(_made)} figures in {FIG}")


if __name__ == "__main__":
    main()
