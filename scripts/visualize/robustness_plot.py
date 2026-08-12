"""
robustness_plot.py -- Robustness visualization.

Reads:  results/robustness/robustness_all.csv
        (or individual results/robustness/{dataset}_{pipeline}_robustness.csv)

Produces per dataset:
  results/robustness/robustness_missing_{dataset}.png
      Line plot: PR-AUC vs. missing rate (MCAR / MAR / MNAR), one line per pipeline
  results/robustness/robustness_outlier_{dataset}.png
      Line plot: PR-AUC vs. outlier injection rate, one line per pipeline
  results/robustness/robustness_label_noise_{dataset}.png
      Line plot: PR-AUC vs. label noise rate, one line per pipeline
  results/robustness/robustness_combined_{dataset}.png
      3-panel figure combining all mechanisms

Usage:
  python scripts/visualize/robustness_plot.py
  python scripts/visualize/robustness_plot.py --dataset bank
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from config import RESULTS_DIR
from scripts.utils import get_logger, ensure_dirs

logger = get_logger("robustness_plot")
ROB_DIR = RESULTS_DIR / "robustness"
ensure_dirs(ROB_DIR)

# Color palette per pipeline
PIPELINE_COLORS = {
    "a":      "#2196F3",
    "b_pca90": "#4CAF50",
    "b_pca20": "#8BC34A",
    "c":      "#FF9800",
    "d":      "#E91E63",
    "e":      "#9C27B0",
    "f":      "#F44336",
}
PIPELINE_MARKERS = {
    "a": "o", "b_pca90": "s", "b_pca20": "^",
    "c": "D", "d": "v", "e": "P", "f": "*",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_robustness(dataset: str) -> pd.DataFrame:
    combined = ROB_DIR / "robustness_all.csv"
    if combined.exists():
        df = pd.read_csv(combined)
        df = df[df["dataset"] == dataset]
        if not df.empty:
            return df

    files = list(ROB_DIR.glob(f"{dataset}_*_robustness.csv"))
    if not files:
        raise FileNotFoundError(
            f"No robustness results for '{dataset}'. Run run_robustness.py first.")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


# ---------------------------------------------------------------------------
# Generic line plot helper
# ---------------------------------------------------------------------------

def _plot_mechanism(df: pd.DataFrame, mechanism: str,
                    xlabel: str, title: str, out_path: Path,
                    baseline_df: pd.DataFrame = None):
    """
    Line plot: PR-AUC vs. corruption rate, one line per pipeline.
    Shaded band = +/-1 std.
    """
    mech_df = df[df["mechanism"] == mechanism].copy()
    if mech_df.empty:
        logger.warning(f"No data for mechanism={mechanism} -- skipping.")
        return

    pipelines = sorted(mech_df["pipeline"].unique())
    fig, ax = plt.subplots(figsize=(9, 5))

    for pl in pipelines:
        sub = mech_df[mech_df["pipeline"] == pl].sort_values("rate")
        color  = PIPELINE_COLORS.get(pl, "#888888")
        marker = PIPELINE_MARKERS.get(pl, "o")

        ax.plot(sub["rate"], sub["mean_PR_AUC"],
                label=pl.upper(), color=color, marker=marker,
                linewidth=2, markersize=6)
        ax.fill_between(
            sub["rate"],
            sub["mean_PR_AUC"] - sub["std_PR_AUC"],
            sub["mean_PR_AUC"] + sub["std_PR_AUC"],
            alpha=0.12, color=color
        )

        # Baseline horizontal dotted line
        if baseline_df is not None:
            bl = baseline_df[baseline_df["pipeline"] == pl]["mean_PR_AUC"]
            if not bl.empty:
                ax.axhline(bl.values[0], color=color, linestyle=":",
                           linewidth=1.0, alpha=0.5)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("PR-AUC (mean +/- std)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(title="Pipeline", bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=9, title_fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Combined 3-panel figure
# ---------------------------------------------------------------------------

def plot_combined(df: pd.DataFrame, dataset: str):
    """3-panel figure: missing (MCAR) / outlier / label_noise."""
    baseline_df = df[df["mechanism"] == "baseline"]
    mechanisms = [
        ("mcar",        "Missing Rate (MCAR)", "MCAR"),
        ("outlier",     "Outlier Rate",         "Outliers"),
        ("label_noise", "Label Noise Rate",     "Label Noise"),
    ]

    available = [m for m, _, _ in mechanisms
                 if not df[df["mechanism"] == m].empty]
    if not available:
        logger.warning("No mechanism data found for combined plot.")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5.5 * len(available), 5),
                              sharey=False)
    if len(available) == 1:
        axes = [axes]

    pipelines = sorted(df["pipeline"].unique())

    for ax, (mech, xlabel, panel_title) in zip(axes, mechanisms):
        if mech not in available:
            continue
        mech_df = df[df["mechanism"] == mech].copy()

        for pl in pipelines:
            sub = mech_df[mech_df["pipeline"] == pl].sort_values("rate")
            if sub.empty:
                continue
            color  = PIPELINE_COLORS.get(pl, "#888888")
            marker = PIPELINE_MARKERS.get(pl, "o")
            ax.plot(sub["rate"], sub["mean_PR_AUC"],
                    label=pl.upper(), color=color, marker=marker,
                    linewidth=2, markersize=5)
            ax.fill_between(
                sub["rate"],
                sub["mean_PR_AUC"] - sub["std_PR_AUC"],
                sub["mean_PR_AUC"] + sub["std_PR_AUC"],
                alpha=0.10, color=color
            )
            bl = baseline_df[baseline_df["pipeline"] == pl]["mean_PR_AUC"]
            if not bl.empty:
                ax.axhline(bl.values[0], color=color, linestyle=":",
                           linewidth=0.8, alpha=0.5)

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("PR-AUC" if ax == axes[0] else "", fontsize=10)
        ax.set_title(panel_title, fontsize=11, fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.grid(axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Pipeline",
               bbox_to_anchor=(1.01, 0.85), loc="upper left",
               fontsize=9, title_fontsize=9)
    fig.suptitle(
        f"{dataset.upper()} -- Robustness: PR-AUC vs. Corruption Intensity",
        fontsize=13, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    out = ROB_DIR / f"robustness_combined_{dataset}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Combined plot saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(dataset: str):
    df = load_robustness(dataset)
    logger.info(f"Loaded {len(df)} rows for '{dataset}'")

    baseline_df = df[df["mechanism"] == "baseline"]

    for mech, xlabel, label in [
        ("mcar",        "Missing Rate (MCAR)",    "MCAR Missing Values"),
        ("mar",         "Missing Rate (MAR)",     "MAR Missing Values"),
        ("mnar",        "Missing Rate (MNAR)",    "MNAR Missing Values"),
        ("outlier",     "Outlier Injection Rate", "Outlier Robustness"),
        ("label_noise", "Label Noise Rate",       "Label Noise Robustness"),
    ]:
        _plot_mechanism(
            df, mech, xlabel,
            title=f"{dataset.upper()} -- {label}: PR-AUC vs. Corruption Rate",
            out_path=ROB_DIR / f"robustness_{mech}_{dataset}.png",
            baseline_df=baseline_df,
        )

    plot_combined(df, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robustness plots")
    parser.add_argument("--dataset", default=None, choices=["bank", "retail"])
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ["bank", "retail"]
    for ds in datasets:
        try:
            run(ds)
        except FileNotFoundError as e:
            logger.error(str(e))
