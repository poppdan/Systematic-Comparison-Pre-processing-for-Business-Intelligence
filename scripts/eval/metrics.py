"""
metrics.py -- Evaluation metrics for binary classification.

Primary metric: PR-AUC (important for class imbalance)
Secondary:      ROC-AUC, F1, Brier Score, Precision, Recall, Accuracy
Business KPIs:  Lift @ Top 10%, Expected Monetary Value (EMV)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from scripts.utils import get_logger

logger = get_logger("metrics")


def lift_at_top_k(y_true, y_pred_proba, k: float = 0.10) -> float:
    """
    Lift @ Top k% -- how much better than random is the model
    when targeting the top k% of predicted probabilities.

    Lift = (precision in top-k) / (base rate)
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    n_top = max(1, int(np.ceil(k * len(y_true))))
    top_idx = np.argsort(y_pred_proba)[::-1][:n_top]

    base_rate = y_true.mean()
    if base_rate == 0:
        return 0.0
    precision_top = y_true[top_idx].mean()
    return round(float(precision_top / base_rate), 5)


def expected_monetary_value(y_true, y_pred_proba,
                             revenue_tp: float = 100.0,
                             cost_fp: float = 10.0,
                             cost_fn: float = 0.0,
                             threshold: float = 0.5) -> float:
    """
    Expected Monetary Value (EMV).

    EMV = TP * revenue_tp - FP * cost_fp - FN * cost_fn

    Defaults reflect a typical marketing campaign:
      revenue_tp = 100  (value of a converted customer)
      cost_fp    =  10  (cost of contacting a non-converter)
      cost_fn    =   0  (missed opportunity, not directly costed)

    Parameters
    ----------
    revenue_tp : Revenue per true positive
    cost_fp    : Cost per false positive
    cost_fn    : Cost per false negative
    threshold  : Decision threshold
    """
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_pred_proba) >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    emv = tp * revenue_tp - fp * cost_fp - fn * cost_fn
    return round(float(emv), 2)


def compute_metrics(y_true, y_pred_proba, threshold: float = 0.5) -> dict:
    """
    Compute all evaluation metrics for one test set.

    Returns a dict with: PR_AUC, ROC_AUC, F1, Precision, Recall,
    Accuracy, Brier, Lift_Top10, EMV
    """
    y_true       = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    y_pred       = (y_pred_proba >= threshold).astype(int)

    return {
        "PR_AUC":    round(float(average_precision_score(y_true, y_pred_proba)), 6),
        "ROC_AUC":   round(float(roc_auc_score(y_true, y_pred_proba)), 6),
        "F1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "Recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "Accuracy":  round(float(accuracy_score(y_true, y_pred)), 6),
        "Brier":     round(float(brier_score_loss(y_true, y_pred_proba)), 6),
        "Lift_Top10": lift_at_top_k(y_true, y_pred_proba, k=0.10),
        "EMV":       expected_monetary_value(y_true, y_pred_proba, threshold=threshold),
    }


def aggregate_cv_metrics(fold_metrics: list) -> dict:
    """
    Aggregate per-fold metric dicts into mean ± std.
    Returns dict with mean_<metric> and std_<metric> keys.
    """
    keys = fold_metrics[0].keys()
    agg  = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if m[k] is not None]
        agg[f"mean_{k}"] = round(float(np.mean(vals)), 6)
        agg[f"std_{k}"]  = round(float(np.std(vals, ddof=1)), 6)
    return agg