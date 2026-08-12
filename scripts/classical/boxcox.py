"""
boxcox.py -- Box-Cox transformation for normalising skewed features.

Details:
  - Only for numeric columns with strictly positive values (> 0)
  - scipy.stats.boxcox determines ? automatically via MLE
  - Lambdas are stored for inverse transformation / analysis
  - Used only in Pipeline B and C
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from scipy import stats

from scripts.utils import get_logger

logger = get_logger("boxcox")

MIN_SKEW_THRESHOLD = 0.5  # Only transform when |skew| > 0.5


def apply_boxcox(df: pd.DataFrame, target_col: str = "target_bin",
                 fit: bool = True, lambdas: dict = None,
                 skew_threshold: float = MIN_SKEW_THRESHOLD):
    """
    Box-Cox transformation on positively skewed numeric features.

    Parameters
    ----------
    df              : Input DataFrame
    target_col      : Do not transform
    fit             : True = compute ? (train), False = apply stored ? (test)
    lambdas         : Dict {col: lambda} from training
    skew_threshold  : Minimum skewness for transformation

    Returns
    -------
    df_out, lambdas
    """
    df = df.copy()
    num_cols = [c for c in df.select_dtypes(include="number").columns
                if c != target_col]

    if fit:
        lambdas = {}
        transformed = 0
        for col in num_cols:
            series = df[col].dropna()
            # Box-Cox requires strictly positive values
            if series.min() <= 0:
                continue
            # Only transform when significantly skewed
            if abs(series.skew()) < skew_threshold:
                continue
            try:
                transformed_vals, lam = stats.boxcox(df[col].dropna())
                # Apply to the entire column (NaN-safe)
                df[col] = stats.boxcox(df[col].fillna(df[col].median()), lmbda=lam)
                lambdas[col] = lam
                transformed += 1
            except Exception as e:
                logger.warning(f"Box-Cox for '{col}' failed: {e}")
        logger.info(f"Box-Cox: {transformed} of {len(num_cols)} features transformed")
    else:
        # Apply stored lambdas
        for col, lam in lambdas.items():
            if col in df.columns and df[col].min() > 0:
                df[col] = stats.boxcox(df[col].fillna(df[col].median()), lmbda=lam)
        logger.info(f"Box-Cox (test): {len(lambdas)} features transformed")

    return df, lambdas


def skewness_report(df: pd.DataFrame, target_col: str = "target_bin") -> pd.DataFrame:
    """Show skewness of all numeric features."""
    num_cols = [c for c in df.select_dtypes(include="number").columns
                if c != target_col]
    rows = [{
        "Feature": col,
        "Skewness": round(df[col].skew(), 3),
        "Min": round(df[col].min(), 4),
        "All_Positive": bool(df[col].min() > 0),
    } for col in num_cols]
    return pd.DataFrame(rows).sort_values("Skewness", key=abs, ascending=False)
