"""
ablation_heatmap.py -- Ablation heatmap: Performance-Delta per preprocessing stage.

Reads:  results/ablation/{dataset}_{pipeline}_ablation.csv
        (or results/ablation/ablation_all.csv)

Produces:
  results/ablation/ablation_heatmap_{dataset}.png
      Heatmap: rows = pipelines, columns = ablation stages
      Values = mean PR-AUC (with delta from Stage 0 annotated)

  results/ablation/ablation_delta_{dataset}.png
      Bar chart: PR-AUC gain Stage 0 -> Stage 2 per pipeline

Usage:
  python scripts/visualize/ablation_heatmap.py
  python scripts/visualize/ablation_heatmap.py --dataset bank
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from config import RESULTS_DIR
from scripts.utils import get_logger, ensure_dirs

logger = get_logger("ablation_heatmap")
ABL_DIR = RESULTS_DIR / "ablation"
ensure_dirs(ABL_DIR)

# Stage 0 and Stage 2 are fixed. Stage 1 contains pipeline-specific steps.
# The heatmap shows: raw | [step1 | step2 | ...] | full
# Steps are sorted: raw first, full last, Stage 1 steps in between.

STAGE_ORDER_PRIORITY = {"raw": 0, "full": 999}  # everything else = 1..998

def _sort_stage_names(names):
    """Sort stage_name values: raw first, full last, rest alphabetically."""
    def key(n):
        return (STAGE_ORDER_PRIORITY.get(n, 1), n)
    return sorted(names, key=key)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ablation(dataset: str) -> pd.DataFrame:
    """Load combined or per-dataset ablation results."""
    combined = ABL_DIR / "ablation_all.csv"
    if combined.exists():
        df = pd.read_csv(combined)
        df = df[df["dataset"] == dataset]
        if not df.empty:
            return df

    # Fall back: load individual files and concatenate
    files = list(ABL_DIR.glob(f"{dataset}_*_ablation.csv"))
    if not files:
        raise FileNotFoundError(
            f"No ablation results for '{dataset}'. Run ablation.py first.")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


# ---------------------------------------------------------------------------
# Heatmap: PR-AUC per pipeline x stage
# ---------------------------------------------------------------------------

def plot_ablation_heatmap(df: pd.DataFrame, dataset: str):
    """
    Heatmap: rows = pipelines, columns = ablation stages.
    Cell values = mean PR-AUC.
    Annotation includes delta vs Stage 0 in parentheses.
    """
    pivot = df.pivot_table(
        index="pipeline", columns="stage_name",
        values="mean_PR_AUC", aggfunc="mean"
    )

    # Sort columns: raw first, stage-1 steps alphabetically, full last
    sorted_cols = _sort_stage_names(list(pivot.columns))
    pivot = pivot.reindex(columns=sorted_cols)
    pivot.index = pivot.index.str.upper()

    # Build annotation matrix: value + delta vs raw
    raw_col = "raw" if "raw" in pivot.columns else pivot.columns[0]
    annot = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=str)
    for col in pivot.columns:
        for idx in pivot.index:
            val = pivot.loc[idx, col]
            if pd.isna(val):
                annot.loc[idx, col] = "–"
                continue
            delta = val - pivot.loc[idx, raw_col]
            sign  = "+" if delta >= 0 else ""
            if col == raw_col:
                annot.loc[idx, col] = f"{val:.4f}"
            else:
                annot.loc[idx, col] = f"{val:.4f}\n({sign}{delta:.4f})"

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 2.2),
                                     max(4, len(pivot.index) * 0.9)))
    sns.heatmap(
        pivot.astype(float), annot=annot, fmt="",
        cmap="YlGnBu", linewidths=0.5, ax=ax,
        annot_kws={"size": 8}, vmin=pivot.values.min() - 0.02
    )
    ax.set_title(
        f"{dataset.upper()} — Ablation: PR-AUC per Step\n"
        f"Stage 0=raw | Stage 1=each step isolated | Stage 2=full pipeline\n"
        f"(parentheses = Δ vs raw)",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Preprocessing Step")
    ax.set_ylabel("Pipeline")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    out = ABL_DIR / f"ablation_heatmap_{dataset}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Heatmap saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Delta bar chart: Stage 0 -> Stage 2 gain
# ---------------------------------------------------------------------------

def plot_ablation_delta(df: pd.DataFrame, dataset: str):
    """
    Horizontal bar chart: PR-AUC gain from Stage 0 to Stage 2.
    Shows which pipeline benefits most from full preprocessing.
    """
    s0 = df[df["stage_name"] == "raw"].set_index("pipeline")["mean_PR_AUC"]
    s2 = df[df["stage_name"] == "full"].set_index("pipeline")["mean_PR_AUC"]

    delta = (s2 - s0).dropna().sort_values(ascending=True)
    if delta.empty:
        logger.warning("No Stage 2 data found -- skipping delta plot.")
        return

    pipelines = delta.index.str.upper()
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in delta.values]

    fig, ax = plt.subplots(figsize=(8, max(3, len(delta) * 0.6)))
    bars = ax.barh(pipelines, delta.values, color=colors, edgecolor="white", height=0.6)

    for bar, val in zip(bars, delta.values):
        sign = "+" if val >= 0 else ""
        ax.text(
            bar.get_width() + (0.001 if val >= 0 else -0.001),
            bar.get_y() + bar.get_height() / 2,
            f"{sign}{val:.4f}", va="center",
            ha="left" if val >= 0 else "right", fontsize=9
        )

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("PR-AUC Delta (Stage 2 - Stage 0)")
    ax.set_title(
        f"{dataset.upper()} -- PR-AUC Gain: Full Pipeline vs. No Preprocessing",
        fontsize=12, fontweight="bold"
    )
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    fig.tight_layout()

    out = ABL_DIR / f"ablation_delta_{dataset}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Delta chart saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(dataset: str):
    df = load_ablation(dataset)
    logger.info(f"Loaded {len(df)} rows for '{dataset}'")
    plot_ablation_heatmap(df, dataset)
    plot_ablation_delta(df, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation heatmap")
    parser.add_argument("--dataset", default=None, choices=["bank", "retail"])
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ["bank", "retail"]
    for ds in datasets:
        try:
            run(ds)
        except FileNotFoundError as e:
            logger.error(str(e))
