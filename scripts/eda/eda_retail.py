"""
eda_retail.py -- Exploratory Data Analysis: Online Retail Dataset
Usage: python scripts/eda/eda_retail.py
Output: results/eda_retail_*.png  +  results/eda_retail_summary.txt
        results/eda_csv/retail_*.csv   (one CSV per plot, Origin-ready)
"""
import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR   = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_CWD        = _CODE_DIR
_RETAIL_XLS = os.path.join(_CODE_DIR, "data", "raw", "online_retail.xlsx")
_RESULTS    = os.path.join(_CODE_DIR, "results")
_CSV_DIR    = os.path.join(_RESULTS, "eda_csv")

os.makedirs(_RESULTS, exist_ok=True)
os.makedirs(_CSV_DIR,  exist_ok=True)
sys.path.insert(0, _CODE_DIR)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from scripts.utils import get_logger
logger = get_logger("eda_retail")


def _save_csv(df: pd.DataFrame, name: str):
    path = os.path.join(_CSV_DIR, name)
    df.to_csv(path, index=False)
    logger.info(f"CSV saved: {name}")


def _hist_bins(series, bins=50, log_transform=False):
    data = np.log1p(series.dropna()) if log_transform else series.dropna()
    counts, edges = np.histogram(data, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    label = f"log1p({series.name})" if log_transform else series.name
    return pd.DataFrame({"bin_center": centers, "count": counts, "feature": label})


def load_data() -> pd.DataFrame:
    logger.info(f"Loading {_RETAIL_XLS}")
    df = pd.read_excel(_RETAIL_XLS, engine="openpyxl")
    logger.info(f"Shape raw: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    logger.info(f"Shape cleaned: {df.shape}")
    return df


def print_summary(df: pd.DataFrame, df_clean: pd.DataFrame):
    lines = []
    lines.append("=" * 60)
    lines.append("ONLINE RETAIL -- EDA SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Rows raw:     {len(df):,}")
    lines.append(f"Rows cleaned: {len(df_clean):,}")
    lines.append(f"Columns:      {df.shape[1]}")
    lines.append("")
    lines.append("-- Data types ------------------------------")
    lines.append(df.dtypes.to_string())
    lines.append("")
    lines.append("-- Missing values (raw) --------------------")
    missing = df.isnull().sum()
    lines.append(missing[missing > 0].to_string() if missing.any() else "  None")
    lines.append("")
    lines.append("-- Cleaning steps --------------------------")
    cancels    = df["InvoiceNo"].astype(str).str.startswith("C").sum()
    neg_qty    = (df["Quantity"] <= 0).sum()
    neg_price  = (df["UnitPrice"] <= 0).sum()
    miss_cid   = df["CustomerID"].isnull().sum()
    lines.append(f"  Cancellations:       {cancels:,}")
    lines.append(f"  Neg. Quantity:       {neg_qty:,}")
    lines.append(f"  Neg. UnitPrice:      {neg_price:,}")
    lines.append(f"  Missing CustomerID:  {miss_cid:,}")
    lines.append("")
    lines.append("-- Unique counts (cleaned) ------------------")
    lines.append(f"  Customers: {df_clean['CustomerID'].nunique():,}")
    lines.append(f"  Invoices:  {df_clean['InvoiceNo'].nunique():,}")
    lines.append(f"  Products:  {df_clean['StockCode'].nunique():,}")
    lines.append(f"  Countries: {df_clean['Country'].nunique():,}")
    lines.append("")
    lines.append("-- Time period ------------------------------")
    lines.append(f"  From: {df_clean['InvoiceDate'].min()}")
    lines.append(f"  To:   {df_clean['InvoiceDate'].max()}")
    lines.append("")
    lines.append("-- Top 10 countries by transactions ---------")
    lines.append(df_clean["Country"].value_counts().head(10).to_string())

    summary = "\n".join(lines)
    print(summary)
    out = os.path.join(_RESULTS, "eda_retail_summary.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(summary)
    logger.info(f"Summary saved: {out}")

    # CSV: missing values + dataset overview
    miss_df = df.isnull().sum().reset_index()
    miss_df.columns = ["feature", "missing_count"]
    miss_df["missing_pct"] = (miss_df["missing_count"] / len(df) * 100).round(3)
    _save_csv(miss_df, "retail_missing_values.csv")

    overview = pd.DataFrame({
        "metric": ["rows_raw", "rows_cleaned", "unique_customers", "unique_invoices",
                   "unique_products", "unique_countries", "date_from", "date_to"],
        "value":  [len(df), len(df_clean),
                   df_clean["CustomerID"].nunique(), df_clean["InvoiceNo"].nunique(),
                   df_clean["StockCode"].nunique(),  df_clean["Country"].nunique(),
                   str(df_clean["InvoiceDate"].min()), str(df_clean["InvoiceDate"].max())],
    })
    _save_csv(overview, "retail_dataset_overview.csv")


def plot_monthly_revenue(df_clean: pd.DataFrame):
    df = df_clean.copy()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M")
    monthly = df.groupby("YearMonth")["Revenue"].sum().reset_index()
    monthly["YearMonth"] = monthly["YearMonth"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(monthly["YearMonth"], monthly["Revenue"], marker="o", color="#4a7fc1", linewidth=2)
    ax.fill_between(monthly["YearMonth"], monthly["Revenue"], alpha=0.2, color="#4a7fc1")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    ax.set_title("Online Retail -- Monthly revenue", fontsize=13, fontweight="bold")
    ax.set_ylabel("Revenue (GBP)")
    fig.tight_layout()
    out = os.path.join(_RESULTS, "eda_retail_monthly_revenue.png")
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Plot saved: {out}")

    # CSV
    csv = monthly.copy()
    csv["YearMonth"] = csv["YearMonth"].dt.strftime("%Y-%m")
    csv.columns = ["year_month", "revenue_gbp"]
    _save_csv(csv, "retail_monthly_revenue.csv")


def plot_rfm_distributions(df_clean: pd.DataFrame):
    snapshot_date = df_clean["InvoiceDate"].max() + pd.Timedelta(days=1)
    df = df_clean.copy()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    rfm = df.groupby("CustomerID").agg(
        Recency   = ("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency = ("InvoiceNo",   "nunique"),
        Monetary  = ("Revenue",     "sum"),
    ).reset_index()

    logger.info(f"RFM Shape: {rfm.shape}")
    logger.info(rfm.describe().round(2).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, color in zip(axes, ["Recency", "Frequency", "Monetary"],
                               ["#e74c3c", "#4a7fc1", "#2ecc71"]):
        data = np.log1p(rfm[col]) if col in ["Frequency", "Monetary"] else rfm[col]
        label = f"log({col})" if col in ["Frequency", "Monetary"] else col
        ax.hist(data, bins=50, color=color, edgecolor="white", linewidth=0.3)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=9)

    fig.suptitle("Online Retail -- RFM distributions", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(_RESULTS, "eda_retail_rfm.png")
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Plot saved: {out}")

    rfm_path = os.path.join(_RESULTS, "eda_retail_rfm_raw.csv")
    rfm.to_csv(rfm_path, index=False)
    logger.info(f"RFM CSV saved: {rfm_path}")

    # CSV: raw RFM + stats + per-feature histograms (raw + log)
    _save_csv(rfm, "retail_rfm_raw_snapshot.csv")
    stats = rfm[["Recency", "Frequency", "Monetary"]].describe().T.reset_index()
    stats.columns = ["feature"] + list(stats.columns[1:])
    _save_csv(stats, "retail_rfm_stats.csv")
    corr = rfm[["Recency", "Frequency", "Monetary"]].corr().reset_index()
    corr.columns = ["feature", "Recency", "Frequency", "Monetary"]
    _save_csv(corr, "retail_rfm_correlation_matrix.csv")

    all_hists = []
    for col in ["Recency", "Frequency", "Monetary"]:
        for log in [False, True]:
            all_hists.append(_hist_bins(rfm[col], bins=50, log_transform=log))
        h = _hist_bins(rfm[col], bins=50)[["bin_center", "count"]]
        h.columns = [f"{col}_bin_center", f"{col}_count"]
        _save_csv(h, f"retail_hist_{col}.csv")
        h2 = _hist_bins(rfm[col], bins=50, log_transform=True)[["bin_center", "count"]]
        h2.columns = [f"log_{col}_bin_center", f"log_{col}_count"]
        _save_csv(h2, f"retail_hist_log_{col}.csv")
    _save_csv(pd.concat(all_hists, ignore_index=True), "retail_rfm_histograms.csv")

    return rfm


def plot_top_products(df_clean: pd.DataFrame):
    df = df_clean.copy()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    top = df.groupby("Description")["Revenue"].sum().nlargest(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top.index[::-1], top.values[::-1], color="#4a7fc1")
    ax.set_title("Top 15 products by revenue", fontsize=12, fontweight="bold")
    ax.set_xlabel("Revenue (GBP)")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    out = os.path.join(_RESULTS, "eda_retail_top_products.png")
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Plot saved: {out}")

    # CSV
    csv = top.reset_index()
    csv.columns = ["product", "revenue_gbp"]
    _save_csv(csv, "retail_top_products.csv")


def plot_country_distribution(df_clean: pd.DataFrame):
    df = df_clean.copy()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    top = df.groupby("Country")["Revenue"].sum().nlargest(10)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(top.index, top.values, color="#4a7fc1")
    ax.set_title("Top 10 countries by revenue", fontsize=12, fontweight="bold")
    ax.set_ylabel("Revenue (GBP)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    out = os.path.join(_RESULTS, "eda_retail_countries.png")
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Plot saved: {out}")

    # CSV: by revenue + by transaction count
    csv = top.reset_index()
    csv.columns = ["country", "revenue_gbp"]
    _save_csv(csv, "retail_top_countries_by_revenue.csv")

    tx = df_clean["Country"].value_counts().head(10).reset_index()
    tx.columns = ["country", "transaction_count"]
    tx["share_pct"] = (tx["transaction_count"] / len(df_clean) * 100).round(2)
    _save_csv(tx, "retail_top_countries_by_transactions.csv")


if __name__ == "__main__":
    df_raw   = load_data()
    df_clean = clean_data(df_raw)
    print_summary(df_raw, df_clean)
    plot_monthly_revenue(df_clean)
    plot_rfm_distributions(df_clean)
    plot_top_products(df_clean)
    plot_country_distribution(df_clean)
    logger.info("EDA Online Retail completed")
