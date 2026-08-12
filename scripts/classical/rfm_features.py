"""
rfm_features.py -- RFM feature engineering for Online Retail (Pipeline C).

Computes:
  - Recency:   Days since last purchase
  - Frequency: Number of unique invoices
  - Monetary:  Total revenue in ?

Additional derived features:
  - AvgOrderValue    : Monetary / Frequency
  - PurchaseSpan     : Days between first and last purchase
  - AvgDaysBetween   : PurchaseSpan / (Frequency - 1) (NA if Freq=1)
  - UniqueProducts   : Number of distinct products
  - ReturnsFlag      : 1 if returns present (Qty<0 in raw data)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np

from scripts.utils import get_logger

logger = get_logger("rfm_features")


def compute_rfm(df_raw: pd.DataFrame,
                snapshot_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Compute extended RFM features from raw retail transactions.

    Expects a cleaned DataFrame (no cancellations, no negative Qty).
    CustomerID must be present and non-null.

    Parameters
    ----------
    df_raw        : Cleaned Online Retail DataFrame
    snapshot_date : Reference date for Recency (default: max(InvoiceDate) + 1 day)

    Returns
    -------
    rfm           : DataFrame with one row per customer, CustomerID as index
    """
    df = df_raw.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    logger.info(f"Snapshot date: {snapshot_date.date()}")

    # Core RFM aggregation
    rfm = df.groupby("CustomerID").agg(
        LastPurchase  = ("InvoiceDate", "max"),
        FirstPurchase = ("InvoiceDate", "min"),
        Frequency     = ("InvoiceNo",   "nunique"),
        Monetary      = ("Revenue",     "sum"),
        UniqueProducts= ("StockCode",   "nunique"),
        TotalItems    = ("Quantity",    "sum"),
    ).reset_index()

    rfm["Recency"] = (snapshot_date - rfm["LastPurchase"]).dt.days
    rfm["PurchaseSpan"] = (rfm["LastPurchase"] - rfm["FirstPurchase"]).dt.days

    # Derived features
    rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"]
    rfm["AvgDaysBetween"] = np.where(
        rfm["Frequency"] > 1,
        rfm["PurchaseSpan"] / (rfm["Frequency"] - 1),
        np.nan,
    )
    rfm["AvgItemsPerOrder"] = rfm["TotalItems"] / rfm["Frequency"]

    # Returns flag (from original raw data, before cleaning)
    # Must be passed separately -- placeholder 0 here
    rfm["ReturnsFlag"] = 0

    # Remove unnecessary date columns
    rfm = rfm.drop(columns=["LastPurchase", "FirstPurchase"])

    rfm = rfm.set_index("CustomerID")
    logger.info(f"RFM Shape: {rfm.shape} | Features: {rfm.columns.tolist()}")
    return rfm


def add_returns_flag(rfm: pd.DataFrame, df_all_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Set ReturnsFlag=1 for customers with cancellations (InvoiceNo ~ 'C').

    Callable separately, as df_all_raw must be the uncleaned data.
    """
    cancels = df_all_raw[df_all_raw["InvoiceNo"].astype(str).str.startswith("C")]
    cancel_customers = set(cancels["CustomerID"].dropna().astype(int).tolist())
    rfm["ReturnsFlag"] = rfm.index.isin(cancel_customers).astype(int)
    logger.info(f"ReturnsFlag set for {rfm['ReturnsFlag'].sum():,} customers")
    return rfm
