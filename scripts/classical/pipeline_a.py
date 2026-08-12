"""
pipeline_a.py -- Classical baseline pipeline (A).

Steps:
  1. Imputation  (Median / Most-Frequent)
  2. Encoding    (One-Hot)
  3. Scaling     (StandardScaler)

No outlier treatment, no Box-Cox, no domain features.
Applies to Bank Marketing AND Online Retail.
Saves to data/processed/{dataset}/a/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import time
import tracemalloc

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config import BANK_RAW, RETAIL_RAW, DATA_PROC, TEST_SIZE, SEED
from scripts.utils import get_logger, ensure_dirs
from scripts.classical.missing_values import handle_missing
from scripts.classical.encoder import one_hot_encode
from scripts.classical.scaler import scale_features

logger = get_logger("pipeline_a")
ensure_dirs(DATA_PROC)


# -- Bank Marketing ---------------------------------------------------------

def load_bank() -> pd.DataFrame:
    df = pd.read_csv(BANK_RAW, sep=";")
    df = df.rename(columns={"y": "target"})
    df["target_bin"] = (df["target"] == "yes").astype(int)
    df = df.drop(columns=["target"])
    # Drop 'duration': only known after call ends (post-hoc leakage).
    # UCI documentation explicitly recommends excluding it for realistic models.
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])
        logger.info("Dropped 'duration' column (post-hoc leakage)")
    return df


def run_bank(save: bool = True):
    logger.info("A Pipeline -- Bank Marketing")

    tracemalloc.start()
    t_start = time.perf_counter()

    df = load_bank()

    n_features_in = df.shape[1] - 1  # exclude target_bin

    X = df.drop(columns=["target_bin"])
    y = df["target_bin"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    # Imputation (fit on train only, transform both)
    X_train["target_bin"] = y_train.values
    X_test["target_bin"]  = y_test.values

    X_train, num_imp, cat_imp = handle_missing(X_train, fit=True)
    X_test,  _,       _       = handle_missing(X_test, fit=False,
                                                num_imputer=num_imp,
                                                cat_imputer=cat_imp)

    # One-Hot Encoding (fit on train only)
    X_train, train_cols = one_hot_encode(X_train, fit=True)
    X_test,  _          = one_hot_encode(X_test, fit=False, train_columns=train_cols)

    # StandardScaler (fit on train only)
    X_train, scaler = scale_features(X_train, fit=True, method="standard")
    X_test,  _      = scale_features(X_test, fit=False, scaler=scaler)

    y_train = X_train.pop("target_bin")
    y_test  = X_test.pop("target_bin")

    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    logger.info(f"Train class distribution: {y_train.value_counts().to_dict()}")

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    runtime = time.perf_counter() - t_start

    if save:
        out_dir = DATA_PROC / "bank" / "a"
        _save("bank", "a", X_train, X_test, y_train, y_test)
        _save_transformers(out_dir, {
            "imputer_numeric":     num_imp,
            "imputer_categorical": cat_imp,
            "encoder":             train_cols,
            "scaler":              scaler,
            "pca":                 None,
            "outlier":             None,
            "boxcox":              None,
        })
        _save_metadata(out_dir, {
            "pipeline":             "a",
            "dataset":              "bank",
            "input_shape":          list(df.shape),
            "output_shape_train":   list(X_train.shape),
            "output_shape_test":    list(X_test.shape),
            "n_features_in":        n_features_in,
            "n_features_out":       X_train.shape[1],
            "pca_variance_explained": None,
            "pca_n_components":     None,
            "runtime_seconds":      round(runtime, 4),
            "peak_memory_mb":       round(peak_mem / 1024 / 1024, 4),
        })

    return X_train, X_test, y_train, y_test


# -- Online Retail ----------------------------------------------------------

def load_retail_clean() -> pd.DataFrame:
    """Load cleaned Retail DataFrame (without cancellations etc.)."""
    df = pd.read_excel(RETAIL_RAW, engine="openpyxl")
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


# Temporal split constants (shared by all retail pipelines)
RETAIL_CUTOFF = pd.Timestamp("2011-10-01")   # end of training period


def load_retail_temporal() -> pd.DataFrame:
    """
    Temporal split for Online Retail — leakage-free churn prediction.

    Training period  : Dec 2010 – Sep 2011  (feature computation window)
    Observation window: Oct 2011 – Dec 2011  (target window)
    Target           : customer made ≥1 purchase in the observation window

    Features (all computed from training period only):
      Recency    – days since last purchase before cutoff
      Frequency  – unique invoices before cutoff
      Monetary   – total revenue before cutoff
      TotalItems – total quantity before cutoff

    Returns a DataFrame with one row per training-period customer,
    columns: [Recency, Frequency, Monetary, TotalItems, target_bin]
    No CustomerID column (dropped).
    """
    df = load_retail_clean()

    cutoff = RETAIL_CUTOFF
    train_df = df[df["InvoiceDate"] < cutoff].copy()
    obs_df   = df[df["InvoiceDate"] >= cutoff].copy()

    train_df["Revenue"] = train_df["Quantity"] * train_df["UnitPrice"]

    rfm = train_df.groupby("CustomerID").agg(
        Recency    = ("InvoiceDate", lambda x: (cutoff - x.max()).days),
        Frequency  = ("InvoiceNo",   "nunique"),
        Monetary   = ("Revenue",     "sum"),
        TotalItems = ("Quantity",    "sum"),
    ).reset_index()

    returned = set(obs_df["CustomerID"].unique())
    rfm["target_bin"] = rfm["CustomerID"].isin(returned).astype(int)
    rfm = rfm.drop(columns=["CustomerID"])

    pos_rate = rfm["target_bin"].mean()
    logger.info(
        f"Retail temporal split | cutoff: {cutoff.date()} | "
        f"customers: {len(rfm):,} | positive rate: {pos_rate:.1%}"
    )
    return rfm


def run_retail(save: bool = True):
    """
    A Pipeline -- Online Retail (temporal churn prediction).

    Uses a temporal train/observation split to avoid leakage:
      - Features: RFM aggregated over Dec 2010 – Sep 2011
      - Target:   returned in Oct – Dec 2011 (1=yes, 0=churn)
    """
    logger.info("A Pipeline -- Online Retail")

    tracemalloc.start()
    t_start = time.perf_counter()

    agg = load_retail_temporal()

    n_features_in = agg.shape[1] - 1  # exclude target_bin

    X = agg.drop(columns=["target_bin"])
    y = agg["target_bin"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    X_train["target_bin"] = y_train.values
    X_test["target_bin"]  = y_test.values

    # Imputation (fit on train only, transform both)
    X_train, num_imp, cat_imp = handle_missing(X_train, fit=True)
    X_test,  _,       _       = handle_missing(X_test, fit=False,
                                                num_imputer=num_imp,
                                                cat_imputer=cat_imp)

    # One-Hot Encoding (fit on train only)
    X_train, train_cols = one_hot_encode(X_train, fit=True)
    X_test,  _          = one_hot_encode(X_test, fit=False, train_columns=train_cols)

    # StandardScaler (fit on train only)
    X_train, scaler = scale_features(X_train, fit=True, method="standard")
    X_test,  _      = scale_features(X_test, fit=False, scaler=scaler)

    y_train = X_train.pop("target_bin")
    y_test  = X_test.pop("target_bin")

    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    runtime = time.perf_counter() - t_start

    if save:
        out_dir = DATA_PROC / "retail" / "a"
        _save("retail", "a", X_train, X_test, y_train, y_test)
        _save_transformers(out_dir, {
            "imputer_numeric":     num_imp,
            "imputer_categorical": cat_imp,
            "encoder":             train_cols,
            "scaler":              scaler,
            "pca":                 None,
            "outlier":             None,
            "boxcox":              None,
        })
        _save_metadata(out_dir, {
            "pipeline":             "a",
            "dataset":              "retail",
            "input_shape":          list(agg.shape),
            "output_shape_train":   list(X_train.shape),
            "output_shape_test":    list(X_test.shape),
            "n_features_in":        n_features_in,
            "n_features_out":       X_train.shape[1],
            "pca_variance_explained": None,
            "pca_n_components":     None,
            "runtime_seconds":      round(runtime, 4),
            "peak_memory_mb":       round(peak_mem / 1024 / 1024, 4),
        })

    return X_train, X_test, y_train, y_test


# -- Save -------------------------------------------------------------------

def _save(dataset: str, pipeline: str,
          X_train, X_test, y_train, y_test):
    out_dir = DATA_PROC / dataset / pipeline
    ensure_dirs(out_dir)
    X_train.to_parquet(out_dir / "X_train.parquet", index=False)
    X_test.to_parquet(out_dir / "X_test.parquet",  index=False)
    y_train.to_frame().to_parquet(out_dir / "y_train.parquet", index=False)
    y_test.to_frame().to_parquet(out_dir / "y_test.parquet",  index=False)
    logger.info(f"Saved: {out_dir}")


def _save_transformers(out_dir: Path, transformers_dict: dict):
    """Persist fitted transformer objects as a joblib file."""
    ensure_dirs(out_dir)
    path = out_dir / "transformers.joblib"
    joblib.dump(transformers_dict, path)
    logger.info(f"Transformers saved: {path}")


def _save_metadata(out_dir: Path, meta_dict: dict):
    """Persist dimensions and runtime metadata as a human-readable JSON file."""
    ensure_dirs(out_dir)
    path = out_dir / "metadata.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)
    logger.info(f"Metadata saved: {path}")


if __name__ == "__main__":
    run_bank()
    run_retail()
    logger.info("Pipeline A completed")
