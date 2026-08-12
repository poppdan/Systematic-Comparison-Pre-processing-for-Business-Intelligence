"""
cv_runner.py -- Repeated Stratified K-Fold Cross-Validation.

Setup: 5 folds x 3 repetitions = 15 CV splits
Important: any fold-internal transformation is fitted on the training
portion of the fold only.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from config import N_FOLDS, N_REPEATS, SEED
from scripts.utils import get_logger
from scripts.eval.metrics import compute_metrics, aggregate_cv_metrics

logger = get_logger("cv_runner")


def run_cv(model, X: pd.DataFrame, y: pd.Series,
           pipeline_fn=None,
           n_folds: int = N_FOLDS,
           n_repeats: int = N_REPEATS,
           seed: int = SEED,
           verbose: bool = True) -> dict:
    """
    Run Repeated Stratified K-Fold CV.

    Parameters
    ----------
    model        : Sklearn-compatible classifier (must have fit/predict_proba)
    X            : Feature matrix (numpy or DataFrame)
    y            : Target variable
    pipeline_fn  : Optional. Function(X_train, y_train) -> X_train_proc, y_train_proc, X_val_proc
                   For pipelines that need fold-internal preprocessing, which
                   in this study means the per-fold encoder of Pipelines D-F.
                   Signature: pipeline_fn(X_train, y_train, X_val) -> X_tr, y_tr, X_val
    n_folds      : Number of folds (default: 5)
    n_repeats    : Number of repetitions (default: 3)
    seed         : Reproducibility
    verbose      : Logging per fold

    Returns
    -------
    dict with all fold metrics + aggregated values
    """
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats,
                                   random_state=seed)

    X_arr = X.values if hasattr(X, "values") else X
    y_arr = y.values if hasattr(y, "values") else y

    fold_metrics = []

    col_names = list(X.columns) if hasattr(X, "columns") else None

    for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(X_arr, y_arr)):
        X_tr_arr, X_val_arr = X_arr[train_idx], X_arr[val_idx]
        y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

        # Keep as DataFrame to avoid feature-name warnings with LGBM
        X_tr  = pd.DataFrame(X_tr_arr,  columns=col_names) if col_names else X_tr_arr
        X_val = pd.DataFrame(X_val_arr, columns=col_names) if col_names else X_val_arr

        # Optional: fold-internal preprocessing (per-fold encoder, D-F)
        if pipeline_fn is not None:
            y_tr_s = pd.Series(y_tr)
            X_tr, y_tr_s, X_val = pipeline_fn(X_tr, y_tr_s, X_val)
            y_tr = y_tr_s.values

        # Fit + Predict
        model.fit(X_tr, y_tr)
        y_proba = model.predict_proba(X_val)[:, 1]

        metrics = compute_metrics(y_val, y_proba)
        fold_metrics.append(metrics)

        if verbose:
            logger.info(f"  Fold {fold_idx+1:02d}/{n_folds*n_repeats} "
                        f"PR-AUC={metrics['PR_AUC']:.4f} "
                        f"ROC-AUC={metrics['ROC_AUC']:.4f} "
                        f"F1={metrics['F1']:.4f}")

    agg = aggregate_cv_metrics(fold_metrics)
    logger.info(f"CV total: PR-AUC={agg['mean_PR_AUC']:.4f}+/-{agg['std_PR_AUC']:.4f} "
                f"ROC-AUC={agg['mean_ROC_AUC']:.4f}+/-{agg['std_ROC_AUC']:.4f}")

    return {
        "fold_metrics": fold_metrics,
        "aggregated":   agg,
        "n_folds":      n_folds,
        "n_repeats":    n_repeats,
    }
