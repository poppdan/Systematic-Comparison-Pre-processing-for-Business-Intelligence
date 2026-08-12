"""
eda_bank.py -- Exploratory Data Analysis: Bank Marketing Dataset
Usage: python scripts/eda/eda_bank.py
Output: results/eda_bank_*.png  +  results/eda_bank_summary.txt
        results/eda_csv/bank_*.csv   (one CSV per plot, Origin-ready)
"""
import sys
import os

# Derive paths absolutely from the script location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR   = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))   # .../code/
_CWD        = _CODE_DIR
_BANK_CSV   = os.path.join(_CODE_DIR, "data", "raw", "bank_marketing.csv")
_RESULTS    = os.path.join(_CODE_DIR, "results")
_CSV_DIR    = os.path.join(_RESULTS, "eda_csv")

os.makedirs(_RESULTS, exist_ok=True)
os.makedirs(_CSV_DIR,  exist_ok=True)
sys.path.insert(0, _CODE_DIR)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.utils import get_logger
logger = get_logger("eda_bank")


def _save_csv(df: pd.DataFrame, name: str):
    path = os.path.join(_CSV_DIR, name)
    df.to_csv(path, index=False)
    logger.info(f"CSV saved: {name}")


def _hist_bins(series, bins=50, log_transform=False):
    """Return (bin_center, count) DataFrame for a histogram."""
    data = np.log1p(series.dropna()) if log_transform else series.dropna()
    counts, edges = np.histogram(data, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    label = f"log1p({series.name})" if log_transform else series.name
    return pd.DataFrame({"bin_center": centers, "count": counts, "feature": label})


def load_data() -> pd.DataFrame:
    logger.info(f"Loading {_BANK_CSV}")
    df = pd.read_csv(_BANK_CSV, sep=";")
    df = df.rename(columns={"y": "target"})
    df["target_bin"] = (df["target"] == "yes").astype(int)
    logger.info(f"Shape: {df.shape}")
    return df


def print_summary(df: pd.DataFrame):
    lines = []
    lines.append("=" * 60)
    lines.append("BANK MARKETING -- EDA SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Rows:      {len(df):,}")
    lines.append(f"Columns:   {df.shape[1]}")
    lines.append("")
    lines.append("-- Data types ------------------------------")
    lines.append(df.dtypes.to_string())
    lines.append("")
    lines.append("-- Missing values --------------------------")
    missing = df.isnull().sum()
    lines.append(missing[missing > 0].to_string() if missing.any() else "  No missing values")
    lines.append("")
    lines.append("-- Class distribution (target) --------------")
    vc = df["target"].value_counts()
    lines.append(vc.to_string())
    lines.append(f"  Ratio yes/no: {vc['yes']/vc['no']:.3f}")
    lines.append(f"  Positive share: {vc['yes']/len(df):.1%}")
    lines.append("")
    lines.append("-- Numeric features (describe) --------------")
    lines.append(df.select_dtypes("number").describe().round(2).to_string())
    lines.append("")
    lines.append("-- Categorical features ---------------------")
    for c in df.select_dtypes("object").columns:
        if c == "target":
            continue
        vc2 = df[c].value_counts()
        lines.append(f"  {c}: {df[c].nunique()} values | most frequent: {vc2.index[0]} ({vc2.iloc[0]})")

    summary = "\n".join(lines)
    print(summary)
    out = os.path.join(_RESULTS, "eda_bank_summary.txt") 
    print(out)
    with open(out, "w", encoding="utf-8") as f:
        f.write(summary)
    logger.info(f"Summary saved: {out}")


def plot_distributions(df: pd.DataFrame):
    num_cols = df.select_dtypes("number").columns.tolist()
    n = len(num_cols)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        axes[i].hist(df[col], bins=40, color="#4a7fc1", edgecolor="white", linewidth=0.3)
        axes[i].set_title(col, fontsize=10)
        axes[i].tick_params(labelsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Bank Marketing -- Distributions of numeric features", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(_RESULTS, "eda_bank_distributions.png")
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Plot saved: {out}")

    # CSV: one file per feature (bin_center + count) + combined long-format
    all_hists = []
    for col in num_cols:
        h = _hist_bins(df[col], bins=40)
        all_hists.append(h)
        wide = h[["bin_center", "count"]].copy()
        wide.columns = [f"{col}_bin_center", f"{col}_count"]
        _save_csv(wide, f"bank_hist_{col}.csv")
    _save_csv(pd.concat(all_hists, ignore_index=True), "bank_numeric_distributions.csv")
    _save_csv(df[num_cols].describe().T.reset_index().rename(columns={"index": "feature"}),
              "bank_numeric_stats.csv")
    # Boxplot raw data (numeric by target class)
    for col in num_cols:
        _save_csv(df[[col, "target_bin"]].rename(columns={"target_bin": "class"}),
                  f"bank_boxplot_{col}.csv")


def plot_correlation(df: pd.DataFrame):
    num_cols = df.select_dtypes("number").columns.tolist()
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, annot_kws={"size": 8})
    ax.set_title("Bank Marketing -- Correlation matrix (numeric)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(_RESULTS, "eda_bank_corr.png")
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Plot saved: {out}")

    # CSV: full correlation matrix
    csv = corr.reset_index().rename(columns={"index": "feature"})
    _save_csv(csv, "bank_correlation_matrix.csv")


def plot_class_balance(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    vc = df["target"].value_counts()
    axes[0].bar(vc.index, vc.values, color=["#e74c3c", "#4a7fc1"])
    axes[0].set_title("Overall class distribution")
    for bar, val in zip(axes[0].patches, vc.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                     f"{val:,}\n({val/len(df):.1%})", ha="center", fontsize=10)

    month_order = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
    cross = pd.crosstab(df["month"], df["target"], normalize="index") * 100
    cross = cross.reindex([m for m in month_order if m in cross.index])
    cross["yes"].plot(kind="bar", ax=axes[1], color="#4a7fc1")
    axes[1].set_title("Positive rate by month (%)")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle("Bank Marketing -- Class imbalance", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(_RESULTS, "eda_bank_class_balance.png")
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Plot saved: {out}")

    # CSV: class distribution + positive rate by month
    class_dist = vc.reset_index()
    class_dist.columns = ["class", "count"]
    class_dist["share_pct"] = (class_dist["count"] / len(df) * 100).round(2)
    _save_csv(class_dist, "bank_class_distribution.csv")

    month_rate = cross.reset_index()
    month_rate.columns = ["month", "pct_no", "pct_yes"]
    _save_csv(month_rate, "bank_positive_rate_by_month.csv")

    # Positive rate by other categoricals
    for col in ["job", "education", "marital", "contact", "poutcome"]:
        if col not in df.columns:
            continue
        cr = (pd.crosstab(df[col], df["target"], normalize="index") * 100).reset_index()
        cr.columns = [col, "pct_no", "pct_yes"]
        cr = cr.sort_values("pct_yes", ascending=False)
        _save_csv(cr, f"bank_positive_rate_by_{col}.csv")

    # Missing values
    miss = df.isnull().sum().reset_index()
    miss.columns = ["feature", "missing_count"]
    miss["missing_pct"] = (miss["missing_count"] / len(df) * 100).round(3)
    _save_csv(miss, "bank_missing_values.csv")


def plot_categoricals(df: pd.DataFrame):
    cat_cols = [c for c in df.select_dtypes("object").columns if c != "target"]
    n = len(cat_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        vc = df[col].value_counts()
        axes[i].barh(vc.index[:10], vc.values[:10], color="#4a7fc1")
        axes[i].set_title(col, fontsize=10)
        axes[i].tick_params(labelsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Bank Marketing -- Categorical features", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(_RESULTS, "eda_bank_categoricals.png")
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Plot saved: {out}")

    # CSV: value counts per categorical feature
    for col in cat_cols:
        vc = df[col].value_counts().reset_index()
        vc.columns = [col, "count"]
        vc["share_pct"] = (vc["count"] / len(df) * 100).round(2)
        _save_csv(vc, f"bank_categorical_{col}.csv")


if __name__ == "__main__":
    df = load_data()
    print_summary(df)
    plot_distributions(df)
    plot_correlation(df)
    plot_class_balance(df)
    plot_categoricals(df)
    logger.info("EDA Bank Marketing completed")
