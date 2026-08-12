"""
eda_export_csv.py -- Export all EDA data as CSV files for Origin import.

One CSV per plot. Output: results/eda_csv/

Bank Marketing:
  bank_class_distribution.csv       -- overall class counts
  bank_positive_rate_by_month.csv   -- % yes per month
  bank_positive_rate_by_job.csv     -- % yes per job category
  bank_positive_rate_by_education.csv
  bank_positive_rate_by_marital.csv
  bank_numeric_distributions.csv    -- histogram bins for all numeric features
  bank_numeric_stats.csv            -- describe() table
  bank_correlation_matrix.csv       -- full Pearson correlation matrix
  bank_missing_values.csv           -- missing value counts
  bank_categorical_counts_{col}.csv -- value counts per categorical feature

Online Retail (temporal split view):
  retail_monthly_revenue.csv        -- monthly revenue time series
  retail_rfm_raw.csv                -- per-customer RFM values
  retail_rfm_histogram.csv          -- histogram bins for R/F/M
  retail_rfm_stats.csv              -- describe() for RFM
  retail_top_products.csv           -- top 15 products by revenue
  retail_top_countries.csv          -- top 10 countries by revenue
  retail_class_distribution.csv     -- churn target counts (temporal split)
  retail_missing_values.csv         -- missing value counts (raw data)

Usage:
  python scripts/eda/eda_export_csv.py
"""
import sys
import os
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_CODE_DIR   = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_CODE_DIR))

import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eda_export_csv")

BANK_RAW   = _CODE_DIR / "data" / "raw" / "bank_marketing.csv"
RETAIL_RAW = _CODE_DIR / "data" / "raw" / "online_retail.xlsx"
OUT_DIR    = _CODE_DIR / "results" / "eda_csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RETAIL_CUTOFF = pd.Timestamp("2011-10-01")


def _load_retail_temporal() -> pd.DataFrame:
    """Inline copy of load_retail_temporal — avoids importing project modules."""
    retail_csv = RETAIL_RAW.with_suffix(".csv")
    df = pd.read_csv(retail_csv, low_memory=False) if retail_csv.exists() else pd.read_excel(RETAIL_RAW, engine="openpyxl")
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"]  = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    cutoff   = RETAIL_CUTOFF
    train_df = df[df["InvoiceDate"] < cutoff].copy()
    obs_df   = df[df["InvoiceDate"] >= cutoff].copy()
    train_df["Revenue"] = train_df["Quantity"] * train_df["UnitPrice"]

    rfm = train_df.groupby("CustomerID").agg(
        Recency    = ("InvoiceDate", lambda x: (cutoff - x.max()).days),
        Frequency  = ("InvoiceNo",   "nunique"),
        Monetary   = ("Revenue",     "sum"),
        TotalItems = ("Quantity",    "sum"),
    ).reset_index()

    churners = set(obs_df["CustomerID"].unique())
    rfm["target_bin"] = rfm["CustomerID"].isin(churners).astype(int)
    rfm = rfm.drop(columns=["CustomerID"])
    return rfm


def save(df: pd.DataFrame, name: str):
    path = OUT_DIR / name
    df.to_csv(path, index=False)
    logger.info(f"  Saved: {name}  ({df.shape[0]} rows x {df.shape[1]} cols)")


def histogram_data(series: pd.Series, bins: int = 50,
                   log_transform: bool = False) -> pd.DataFrame:
    """Return bin_center and count for a histogram — Origin-ready."""
    data = np.log1p(series.dropna()) if log_transform else series.dropna()
    counts, edges = np.histogram(data, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    label = f"log1p({series.name})" if log_transform else series.name
    return pd.DataFrame({"bin_center": centers, "count": counts,
                         "feature": label})


# ---------------------------------------------------------------------------
# Bank Marketing
# ---------------------------------------------------------------------------

def export_bank():
    logger.info("=== Bank Marketing ===")
    df = pd.read_csv(BANK_RAW, sep=";")
    df = df.rename(columns={"y": "target"})
    df["target_bin"] = (df["target"] == "yes").astype(int)

    # 1. Class distribution
    vc = df["target"].value_counts().reset_index()
    vc.columns = ["class", "count"]
    vc["share_pct"] = (vc["count"] / len(df) * 100).round(2)
    save(vc, "bank_class_distribution.csv")

    # 2. Positive rate by month
    month_order = ["jan","feb","mar","apr","may","jun",
                   "jul","aug","sep","oct","nov","dec"]
    cross = (pd.crosstab(df["month"], df["target"], normalize="index") * 100
             ).reset_index()
    cross.columns = ["month", "pct_no", "pct_yes"]
    cross["month_order"] = cross["month"].map(
        {m: i for i, m in enumerate(month_order)})
    cross = cross.sort_values("month_order").drop(columns="month_order")
    save(cross, "bank_positive_rate_by_month.csv")

    # 3. Positive rate by categorical features
    for col in ["job", "education", "marital", "contact", "poutcome"]:
        if col not in df.columns:
            continue
        cr = (pd.crosstab(df[col], df["target"], normalize="index") * 100
              ).reset_index()
        cr.columns = [col, "pct_no", "pct_yes"]
        cr = cr.sort_values("pct_yes", ascending=False)
        save(cr, f"bank_positive_rate_by_{col}.csv")

    # 4. Categorical value counts
    cat_cols = [c for c in df.select_dtypes("object").columns
                if c not in ("target",)]
    for col in cat_cols:
        vc2 = df[col].value_counts().reset_index()
        vc2.columns = [col, "count"]
        vc2["share_pct"] = (vc2["count"] / len(df) * 100).round(2)
        save(vc2, f"bank_categorical_{col}.csv")

    # 5. Numeric feature distributions (histogram bins)
    num_cols = df.select_dtypes("number").columns.tolist()
    # Export all in one long-format file for easy Origin import
    all_hists = []
    for col in num_cols:
        all_hists.append(histogram_data(df[col], bins=50))
    save(pd.concat(all_hists, ignore_index=True), "bank_numeric_distributions.csv")

    # 6. Separate histogram per numeric feature (wide format for Origin)
    for col in num_cols:
        h = histogram_data(df[col], bins=50)[["bin_center", "count"]]
        h.columns = [f"{col}_bin_center", f"{col}_count"]
        save(h, f"bank_hist_{col}.csv")

    # 7. Numeric stats (describe)
    stats = df[num_cols].describe().T.reset_index()
    stats.columns = ["feature"] + list(stats.columns[1:])
    save(stats, "bank_numeric_stats.csv")

    # 8. Correlation matrix
    corr = df[num_cols].corr().reset_index()
    corr.columns = ["feature"] + num_cols
    save(corr, "bank_correlation_matrix.csv")

    # 9. Missing values
    missing = df.isnull().sum().reset_index()
    missing.columns = ["feature", "missing_count"]
    missing["missing_pct"] = (missing["missing_count"] / len(df) * 100).round(3)
    save(missing, "bank_missing_values.csv")

    # 10. Boxplot data: numeric by target (for Origin box plots)
    for col in num_cols:
        box = df[[col, "target_bin"]].copy()
        box.columns = [col, "target_bin"]
        save(box, f"bank_boxplot_{col}.csv")

    logger.info(f"Bank exports done -> {OUT_DIR}")


# ---------------------------------------------------------------------------
# Online Retail
# ---------------------------------------------------------------------------

def export_retail():
    logger.info("=== Online Retail ===")

    # Load raw data — prefer CSV if available (much faster)
    retail_csv = RETAIL_RAW.with_suffix(".csv")
    if retail_csv.exists():
        df_raw = pd.read_csv(retail_csv, low_memory=False)
    else:
        df_raw = pd.read_excel(RETAIL_RAW, engine="openpyxl")
    df_clean = df_raw.copy()
    df_clean = df_clean[~df_clean["InvoiceNo"].astype(str).str.startswith("C")]
    df_clean = df_clean[df_clean["Quantity"] > 0]
    df_clean = df_clean[df_clean["UnitPrice"] > 0]
    df_clean = df_clean.dropna(subset=["CustomerID"])
    df_clean["CustomerID"]  = df_clean["CustomerID"].astype(int)
    df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])
    df_clean["Revenue"]     = df_clean["Quantity"] * df_clean["UnitPrice"]

    # 1. Missing values (raw)
    missing = df_raw.isnull().sum().reset_index()
    missing.columns = ["feature", "missing_count"]
    missing["missing_pct"] = (missing["missing_count"] / len(df_raw) * 100).round(3)
    save(missing, "retail_missing_values.csv")

    # 2. Monthly revenue
    df_clean["YearMonth"] = df_clean["InvoiceDate"].dt.to_period("M")
    monthly = (df_clean.groupby("YearMonth")["Revenue"].sum()
               .reset_index())
    monthly["YearMonth"] = monthly["YearMonth"].dt.to_timestamp().dt.strftime("%Y-%m")
    monthly.columns = ["year_month", "revenue_gbp"]
    save(monthly, "retail_monthly_revenue.csv")

    # 3. Top 15 products by revenue
    top_prod = (df_clean.groupby("Description")["Revenue"]
                .sum().nlargest(15).reset_index())
    top_prod.columns = ["product", "revenue_gbp"]
    save(top_prod, "retail_top_products.csv")

    # 4. Top 10 countries by revenue
    top_country = (df_clean.groupby("Country")["Revenue"]
                   .sum().nlargest(10).reset_index())
    top_country.columns = ["country", "revenue_gbp"]
    save(top_country, "retail_top_countries.csv")

    # 5. RFM (raw snapshot — for context, NOT the temporal-split version)
    snapshot = df_clean["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm_raw = df_clean.groupby("CustomerID").agg(
        Recency   = ("InvoiceDate", lambda x: (snapshot - x.max()).days),
        Frequency = ("InvoiceNo",   "nunique"),
        Monetary  = ("Revenue",     "sum"),
    ).reset_index(drop=True)
    save(rfm_raw, "retail_rfm_raw_snapshot.csv")

    # 6. RFM stats
    stats = rfm_raw.describe().T.reset_index()
    stats.columns = ["feature"] + list(stats.columns[1:])
    save(stats, "retail_rfm_stats.csv")

    # 7. RFM histograms (raw + log-transformed)
    rfm_hists = []
    for col in ["Recency", "Frequency", "Monetary"]:
        rfm_hists.append(histogram_data(rfm_raw[col], bins=50, log_transform=False))
        rfm_hists.append(histogram_data(rfm_raw[col], bins=50, log_transform=True))
    save(pd.concat(rfm_hists, ignore_index=True), "retail_rfm_histograms.csv")

    # Per-feature histogram files
    for col in ["Recency", "Frequency", "Monetary"]:
        h = histogram_data(rfm_raw[col], bins=50)[["bin_center", "count"]]
        h.columns = [f"{col}_bin_center", f"{col}_count"]
        save(h, f"retail_hist_{col}.csv")
        h_log = histogram_data(rfm_raw[col], bins=50, log_transform=True)[["bin_center","count"]]
        h_log.columns = [f"log_{col}_bin_center", f"log_{col}_count"]
        save(h_log, f"retail_hist_log_{col}.csv")

    # 8. Temporal-split RFM with churn target (the version used in experiments)
    rfm_temporal = _load_retail_temporal()
    save(rfm_temporal, "retail_rfm_temporal_split.csv")

    # 9. Class distribution (temporal split target)
    vc = rfm_temporal["target_bin"].value_counts().reset_index()
    vc.columns = ["returned", "count"]
    vc["share_pct"] = (vc["count"] / len(rfm_temporal) * 100).round(2)
    vc["returned"] = vc["returned"].map({1: "returned (yes)", 0: "not returned (no)"})
    save(vc, "retail_class_distribution.csv")

    # 10. Correlation matrix of temporal RFM features
    num_cols = rfm_temporal.select_dtypes("number").columns.tolist()
    corr = rfm_temporal[num_cols].corr().reset_index()
    corr.columns = ["feature"] + num_cols
    save(corr, "retail_rfm_correlation_matrix.csv")

    logger.info(f"Retail exports done -> {OUT_DIR}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    export_bank()
    export_retail()
    logger.info(f"\nAll CSVs saved to: {OUT_DIR}")
    logger.info(f"Total files: {len(list(OUT_DIR.glob('*.csv')))}")
