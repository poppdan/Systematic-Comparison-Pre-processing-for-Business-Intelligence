"""
learning_curves.py -- Performance vs. training set size.

For each dataset x pipeline, measures PR-AUC at increasing fractions
of available training data (10%, 20%, ..., 100%).

Answers: How does performance scale with more data?
Does representation learning need more data than classical methods?

Results saved to:
  results/learning_curves/{dataset}_{pipeline}_lc.csv
  results/learning_curves/learning_curves_all.csv

Usage:
  python scripts/learning_curves.py
  python scripts/learning_curves.py --dataset bank
  python scripts/learning_curves.py --pipeline a --dataset bank
  python scripts/learning_curves.py --n-sizes 5    # fewer size steps (faster)
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import average_precision_score

from config import (BANK_RAW, RETAIL_RAW, DATA_PROC, RESULTS_DIR, SEED,
                    CLASSICAL_PIPELINES, AI_PIPELINES)
from scripts.utils import get_logger, ensure_dirs
from scripts.classical.missing_values import handle_missing

logger = get_logger("learning_curves")
LC_DIR = RESULTS_DIR / "learning_curves"
ensure_dirs(LC_DIR)

ALL_PIPELINES = [p.lower() for p in CLASSICAL_PIPELINES + AI_PIPELINES]

# Training size fractions to evaluate
DEFAULT_TRAIN_SIZES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
# Number of repetitions per size (averages out sampling variance)
N_REPEATS_LC = 5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_bank_raw():
    df = pd.read_csv(BANK_RAW, sep=";")
    df = df.rename(columns={"y": "target"})
    df["target"] = (df["target"] == "yes").astype(int)
    # 'duration' is only known after the call has ended (post-hoc leakage).
    # Every other script drops it; this fallback loader must do the same,
    # otherwise learning curves computed from raw data are inflated.
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])
    X = df.drop(columns=["target"])
    y = df["target"].values
    return X, y


def _load_retail_raw():
    df = pd.read_excel(RETAIL_RAW)
    df = df.dropna(subset=["CustomerID"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("UnitPrice", lambda x: (x * df.loc[x.index, "Quantity"]).sum()),
    ).reset_index()
    rfm["target"] = (rfm["Monetary"] > rfm["Monetary"].median()).astype(int)
    rfm = rfm.drop(columns=["CustomerID"])
    X = rfm.drop(columns=["target"])
    y = rfm["target"].values
    return X, y


RAW_LOADERS = {"bank": _load_bank_raw, "retail": _load_retail_raw}


def _load_processed(dataset: str, pipeline: str):
    """Load already-preprocessed data for AI/complex pipelines."""
    base = DATA_PROC / dataset / pipeline
    train_csv = base / "train.csv"
    if train_csv.exists():
        df = pd.read_csv(train_csv)
        X = df.drop(columns=["target"]).values.astype(np.float32)
        y = df["target"].values
        return X, y

    # Classical parquet format
    X = pd.read_parquet(base / "X_train.parquet").values.astype(np.float32)
    y = pd.read_parquet(base / "y_train.parquet").squeeze().values
    return X, y


# ---------------------------------------------------------------------------
# Quick preprocessing for raw-data pipelines
# ---------------------------------------------------------------------------

def _quick_preprocess(X: pd.DataFrame) -> np.ndarray:
    X_out = handle_missing(X.copy())
    for col in X_out.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X_out[col] = le.fit_transform(X_out[col].astype(str))
    return X_out.values.astype(np.float32)


# ---------------------------------------------------------------------------
# Single learning curve evaluation
# ---------------------------------------------------------------------------

def _eval_at_size(X: np.ndarray, y: np.ndarray,
                  train_fraction: float, seed: int) -> dict:
    """
    Train on `train_fraction` of data, evaluate on held-out 20%.

    Uses stratified split to maintain class balance at each size.
    Repeats N_REPEATS_LC times with different random seeds.
    """
    if train_fraction >= 1.0:
        # Use all data -- train on 80%, test on 20%
        sss = StratifiedShuffleSplit(n_splits=N_REPEATS_LC,
                                     test_size=0.20, random_state=seed)
        scores = []
        for tr, te in sss.split(X, y):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr])
            X_te = scaler.transform(X[te])
            model = LogisticRegression(max_iter=500, random_state=seed, n_jobs=-1)
            model.fit(X_tr, y[tr])
            proba = model.predict_proba(X_te)[:, 1]
            scores.append(average_precision_score(y[te], proba))
    else:
        # Split: 20% test (fixed), `train_fraction` of remaining for train
        outer = StratifiedShuffleSplit(n_splits=1, test_size=0.20,
                                       random_state=seed)
        tr_pool, te_idx = next(outer.split(X, y))
        X_te, y_te = X[te_idx], y[te_idx]

        inner = StratifiedShuffleSplit(n_splits=N_REPEATS_LC,
                                       train_size=train_fraction,
                                       random_state=seed)
        scores = []
        for tr, _ in inner.split(X[tr_pool], y[tr_pool]):
            actual_tr = tr_pool[tr]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[actual_tr])
            X_te_s = scaler.transform(X_te)
            model = LogisticRegression(max_iter=500, random_state=seed, n_jobs=-1)
            model.fit(X_tr, y[actual_tr])
            proba = model.predict_proba(X_te_s)[:, 1]
            scores.append(average_precision_score(y_te, proba))

    return {
        "mean_PR_AUC": round(float(np.mean(scores)), 5),
        "std_PR_AUC":  round(float(np.std(scores)), 5),
        "n_samples":   int(train_fraction * len(X)),
    }


# ---------------------------------------------------------------------------
# Full learning curve for one dataset x pipeline
# ---------------------------------------------------------------------------

def run_learning_curve(dataset: str, pipeline: str,
                        train_sizes: list = None) -> pd.DataFrame:
    """
    Compute learning curve for one dataset x pipeline.

    For classical pipelines (a, b_*, c): applies quick preprocessing inline.
    For AI pipelines (d, e, f): loads already-processed train.csv if available,
    otherwise falls back to quick preprocessing.
    """
    if train_sizes is None:
        train_sizes = DEFAULT_TRAIN_SIZES

    logger.info(f"\n{'='*50}")
    logger.info(f"Learning curve: {dataset} x pipeline {pipeline.upper()}")
    logger.info(f"{'='*50}")

    # Try to load preprocessed data first (faster + more accurate)
    try:
        X, y = _load_processed(dataset, pipeline)
        logger.info(f"  Loaded preprocessed data: {X.shape}")
    except (FileNotFoundError, Exception):
        logger.info(f"  Preprocessed data not found -- using raw + quick preprocess")
        X_raw, y = RAW_LOADERS[dataset]()
        X = _quick_preprocess(X_raw)
        logger.info(f"  Quick preprocessed: {X.shape}")

    records = []
    for frac in train_sizes:
        logger.info(f"  train_fraction={frac:.2f} "
                    f"(~{int(frac * len(X))} samples)")
        res = _eval_at_size(X, y, frac, seed=SEED)
        records.append({
            "dataset":      dataset,
            "pipeline":     pipeline,
            "train_frac":   frac,
            "n_samples":    res["n_samples"],
            "mean_PR_AUC":  res["mean_PR_AUC"],
            "std_PR_AUC":   res["std_PR_AUC"],
        })
        logger.info(f"    PR-AUC: {res['mean_PR_AUC']:.4f} "
                    f"+/- {res['std_PR_AUC']:.4f}")

    df = pd.DataFrame(records)
    out = LC_DIR / f"{dataset}_{pipeline}_lc.csv"
    df.to_csv(out, index=False)
    logger.info(f"Saved: {out}")
    return df


def run_all(datasets: list = None, pipelines: list = None,
            train_sizes: list = None) -> pd.DataFrame:
    """Run learning curves for all dataset x pipeline combinations."""
    if datasets  is None: datasets  = ["bank", "retail"]
    if pipelines is None: pipelines = ALL_PIPELINES

    all_dfs = []
    for ds in datasets:
        for pl in pipelines:
            df = run_learning_curve(ds, pl, train_sizes)
            all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    out = LC_DIR / "learning_curves_all.csv"
    combined.to_csv(out, index=False)
    logger.info(f"\nCombined learning curves saved: {out}")
    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learning curve analysis")
    parser.add_argument("--dataset",  default=None,
                        choices=["bank", "retail"],
                        help="Dataset (default: all)")
    parser.add_argument("--pipeline", default=None,
                        choices=ALL_PIPELINES,
                        help="Pipeline (default: all)")
    parser.add_argument("--n-sizes",  type=int, default=10,
                        help="Number of training size steps (default: 10 = 10%%..100%%)")
    args = parser.parse_args()

    datasets  = [args.dataset]  if args.dataset  else None
    pipelines = [args.pipeline] if args.pipeline else None

    n = args.n_sizes
    sizes = [round((i + 1) / n, 2) for i in range(n)]

    run_all(datasets=datasets, pipelines=pipelines, train_sizes=sizes)
