"""
pipeline_c.py -- Feature Engineering pipeline (C).

Online Retail:
  1. RFM aggregation (rfm_features.py)
  2. Interaction terms on RFM numeric features (PolynomialFeatures)
  3. Pipeline A preprocessing: Imputation -> OHE -> StandardScaler

Bank Marketing:
  1. Domain feature engineering (age_x_campaign, contact_intensity, ...)
  2. Interaction terms on numeric features (PolynomialFeatures)
  3. Pipeline A preprocessing: Imputation -> OHE -> StandardScaler

Data leakage rule: ALL fit() calls on training data only.
Saves to data/processed/{dataset}/c/
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

from config import DATA_PROC, RETAIL_RAW, TEST_SIZE, SEED
from scripts.utils import get_logger, ensure_dirs
from scripts.classical.pipeline_a import load_bank, load_retail_clean, RETAIL_CUTOFF
from scripts.classical.rfm_features import compute_rfm, add_returns_flag
from scripts.classical.interaction_terms import add_interaction_terms
from scripts.classical.missing_values import handle_missing
from scripts.classical.encoder import one_hot_encode
from scripts.classical.scaler import scale_features

logger = get_logger("pipeline_c")
ensure_dirs(DATA_PROC)


# -- Domain features for Bank Marketing ------------------------------------

def add_bank_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-specific features for the Bank Marketing dataset.

    Features added
    --------------
    age_x_campaign    : age * campaign
        Captures the interaction between a customer's age and the number of
        contacts in the current campaign. Older customers who were contacted
        many times may respond differently from younger ones.

    contact_intensity : campaign / (pdays + 1)
        Ratio of current-campaign contacts to days since the last contact.
        Adding 1 avoids division by zero when pdays == 0. A high ratio
        indicates frequent recent outreach, which often correlates with
        campaign fatigue.

    was_previously_contacted : 1 if pdays != 999 else 0
        Binary flag: was the customer contacted in a previous campaign?
        (999 is the sentinel value meaning "never contacted before".)

    previous_x_poutcome_success : previous * (poutcome == 'success')
        Number of prior contacts multiplied by whether the previous campaign
        outcome was a success. This highlights customers with a positive
        history who were also contacted multiple times before.
    """
    df = df.copy()

    # age * campaign -- joint effect of age and contact frequency
    df["age_x_campaign"] = df["age"] * df["campaign"]

    # contact_intensity -- recency-adjusted outreach ratio
    df["contact_intensity"] = df["campaign"] / (df["pdays"] + 1)

    # was_previously_contacted -- binary flag for prior campaign contact
    df["was_previously_contacted"] = (df["pdays"] != 999).astype(int)

    # previous_x_poutcome_success -- reward signal for previously successful contacts
    poutcome_success = (df["poutcome"] == "success").astype(int)
    df["previous_x_poutcome_success"] = df["previous"] * poutcome_success

    logger.info(
        "Bank domain features added: age_x_campaign, contact_intensity, "
        "was_previously_contacted, previous_x_poutcome_success"
    )
    return df


# -- RFM target helper ------------------------------------------------------

def _add_rfm_target(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Create target variable: High-value customer (1) vs. Low-value (0).

    Based on composite score from quantile ranks (1-5):
      - Recency:   lower values = better -> invert
      - Frequency: higher values = better
      - Monetary:  higher values = better
    Composite = R_Score * 0.30 + F_Score * 0.35 + M_Score * 0.35
    """
    rfm = rfm.copy()

    rfm["R_Score"] = pd.qcut(rfm["Recency"],   q=5, labels=[5,4,3,2,1]).astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=5,
                              labels=[1,2,3,4,5]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"),  q=5,
                              labels=[1,2,3,4,5]).astype(int)

    rfm["RFM_Score"] = (rfm["R_Score"] * 0.30 +
                        rfm["F_Score"] * 0.35 +
                        rfm["M_Score"] * 0.35)

    rfm["target_bin"] = (rfm["RFM_Score"] >= rfm["RFM_Score"].median()).astype(int)

    # Remove helper columns (not to be used as features)
    rfm = rfm.drop(columns=["R_Score", "F_Score", "M_Score", "RFM_Score"])
    return rfm


# -- Bank Marketing ---------------------------------------------------------

def run_bank(save: bool = True):
    """
    C Pipeline -- Bank Marketing with domain features and interaction terms.

    Order:
      1. Load raw data
      2. Add domain features (pure row-wise, no leakage)
      3. Train/test split
      4. Imputation (fit on train only)
      5. Interaction terms on numeric features (fit on train only)
      6. OHE -> StandardScaler (fit on train only)
    """
    logger.info("C Pipeline -- Bank Marketing (domain features + interactions)")

    tracemalloc.start()
    t_start = time.perf_counter()

    df = load_bank()

    # Add domain features before split (pure row-wise transforms, no leakage)
    df = add_bank_domain_features(df)

    n_features_in = df.shape[1] - 1  # exclude target_bin
    input_shape = [df.shape[0], df.shape[1]]

    X = df.drop(columns=["target_bin"])
    y = df["target_bin"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    X_train["target_bin"] = y_train.values
    X_test["target_bin"]  = y_test.values

    # 1. Imputation (fit on train only; must precede PolynomialFeatures)
    X_train, num_imp, cat_imp = handle_missing(X_train, fit=True)
    X_test,  _,       _       = handle_missing(X_test, fit=False,
                                               num_imputer=num_imp,
                                               cat_imputer=cat_imp)

    # 2. Interaction terms on numeric features (fit on train only)
    X_train, poly, num_cols = add_interaction_terms(X_train, fit=True)
    X_test,  _,    _        = add_interaction_terms(X_test, fit=False,
                                                    poly=poly, num_cols=num_cols)

    # 3. One-Hot Encoding (fit on train only)
    X_train, train_cols = one_hot_encode(X_train, fit=True)
    X_test,  _          = one_hot_encode(X_test, fit=False, train_columns=train_cols)

    # 4. StandardScaler (fit on train only)
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
        out_dir = DATA_PROC / "bank" / "c"
        _save("bank", X_train, X_test, y_train, y_test)
        _save_transformers(out_dir, {
            "imputer_numeric":     num_imp,
            "imputer_categorical": cat_imp,
            "interaction":         poly,
            "encoder":             train_cols,
            "scaler":              scaler,
            "pca":                 None,
            "outlier":             None,
            "boxcox":              None,
            "rfm":                 None,
        })
        _save_metadata(out_dir, {
            "pipeline":               "c",
            "dataset":                "bank",
            "input_shape":            input_shape,
            "output_shape_train":     list(X_train.shape),
            "output_shape_test":      list(X_test.shape),
            "n_features_in":          n_features_in,
            "n_features_out":         X_train.shape[1],
            "pca_variance_explained": None,
            "pca_n_components":       None,
            "runtime_seconds":        round(runtime, 4),
            "peak_memory_mb":         round(peak_mem / 1024 / 1024, 4),
        })

    return X_train, X_test, y_train, y_test


# -- Online Retail ----------------------------------------------------------

def run_retail(save: bool = True):
    """
    C Pipeline -- Online Retail with extended RFM + interaction terms.

    Uses temporal split to avoid leakage:
      - Features: extended RFM aggregated over Dec 2010 – Sep 2011
      - Target:   returned in Oct – Dec 2011 (1=yes, 0=churn)

    Order:
      1. Temporal RFM aggregation (rfm_features.py, snapshot=RETAIL_CUTOFF)
      2. Train/test split
      3. Imputation (fit on train only)
      4. Interaction terms on RFM numeric features (fit on train only)
      5. OHE -> StandardScaler (fit on train only)
    """
    logger.info("C Pipeline -- Online Retail (RFM + interactions, temporal split)")

    tracemalloc.start()
    t_start = time.perf_counter()

    cutoff = RETAIL_CUTOFF

    # Load raw data (including cancellations for ReturnsFlag)
    df_all = pd.read_excel(RETAIL_RAW, engine="openpyxl")
    df_all["InvoiceDate"] = pd.to_datetime(df_all["InvoiceDate"])

    df_clean = load_retail_clean()

    # Restrict to training period only for feature computation
    df_train_period = df_clean[df_clean["InvoiceDate"] < cutoff].copy()
    df_all_train    = df_all[df_all["InvoiceDate"] < cutoff].copy()

    # Compute extended RFM features (snapshot = cutoff date)
    rfm = compute_rfm(df_train_period, snapshot_date=cutoff)
    rfm = add_returns_flag(rfm, df_all_train)

    # Temporal target: returned in observation window (Oct-Dec 2011)
    df_obs = df_clean[df_clean["InvoiceDate"] >= cutoff]
    returned = set(df_obs["CustomerID"].unique())
    rfm["target_bin"] = rfm.index.isin(returned).astype(int)
    rfm = rfm.reset_index(drop=True)

    logger.info(f"RFM Features: {rfm.columns.tolist()}")
    logger.info(f"Class distribution: {rfm['target_bin'].value_counts().to_dict()}")

    n_features_in = rfm.shape[1] - 1  # exclude target_bin
    input_shape = [rfm.shape[0], rfm.shape[1]]

    X = rfm.drop(columns=["target_bin"])
    y = rfm["target_bin"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    X_train["target_bin"] = y_train.values
    X_test["target_bin"]  = y_test.values

    # 1. Imputation (fit on train only; must precede PolynomialFeatures)
    X_train, num_imp, cat_imp = handle_missing(X_train, fit=True)
    X_test,  _,       _       = handle_missing(X_test, fit=False,
                                               num_imputer=num_imp,
                                               cat_imputer=cat_imp)

    # 2. Interaction terms on RFM numeric features (fit on train only)
    X_train, poly, num_cols = add_interaction_terms(X_train, fit=True)
    X_test,  _,    _        = add_interaction_terms(X_test, fit=False,
                                                    poly=poly, num_cols=num_cols)

    # 3. One-Hot Encoding (RFM features mostly numeric; handles any categoricals)
    X_train, train_cols = one_hot_encode(X_train, fit=True)
    X_test,  _          = one_hot_encode(X_test, fit=False, train_columns=train_cols)

    # 4. StandardScaler (fit on train only)
    X_train, scaler = scale_features(X_train, fit=True, method="standard")
    X_test,  _      = scale_features(X_test, fit=False, scaler=scaler)

    y_train = X_train.pop("target_bin")
    y_test  = X_test.pop("target_bin")

    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    runtime = time.perf_counter() - t_start

    if save:
        out_dir = DATA_PROC / "retail" / "c"
        _save("retail", X_train, X_test, y_train, y_test)
        _save_transformers(out_dir, {
            "imputer_numeric":     num_imp,
            "imputer_categorical": cat_imp,
            "interaction":         poly,
            "encoder":             train_cols,
            "scaler":              scaler,
            "pca":                 None,
            "outlier":             None,
            "boxcox":              None,
            "rfm":                 None,  # RFM is a stateless aggregation, no fitted object
        })
        _save_metadata(out_dir, {
            "pipeline":               "c",
            "dataset":                "retail",
            "input_shape":            input_shape,
            "output_shape_train":     list(X_train.shape),
            "output_shape_test":      list(X_test.shape),
            "n_features_in":          n_features_in,
            "n_features_out":         X_train.shape[1],
            "pca_variance_explained": None,
            "pca_n_components":       None,
            "runtime_seconds":        round(runtime, 4),
            "peak_memory_mb":         round(peak_mem / 1024 / 1024, 4),
        })

    return X_train, X_test, y_train, y_test


# -- Save -------------------------------------------------------------------

def _save(dataset: str, X_train, X_test, y_train, y_test):
    out_dir = DATA_PROC / dataset / "c"
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
    logger.info("Pipeline C completed")
