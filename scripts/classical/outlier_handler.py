"""
outlier_handler.py -- Outlier treatment via IQR winsorisation.

Method: Winsorisation to [Q1 - k*IQR, Q3 + k*IQR]
  - k=1.5 (standard), k=3.0 (conservative for extreme distributions)
  - Applied to numeric features only
  - Target variable is never touched
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np

from scripts.utils import get_logger

logger = get_logger("outlier_handler")


def compute_iqr_bounds(series: pd.Series, k: float = 1.5):
    """Compute lower/upper winsorisation bounds."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper


def winsorize(df: pd.DataFrame, target_col: str = "target_bin",
              k: float = 1.5, fit: bool = True,
              bounds: dict = None):
    """
    Winsorise all numeric columns (except target_col).

    Parameters
    ----------
    df         : Input DataFrame
    target_col : Target variable column, will not be modified
    k          : IQR multiplier
    fit        : True = compute bounds (train), False = apply stored bounds (test)
    bounds     : Dict {col: (lower, upper)} for fit=False

    Returns
    -------
    df_out, bounds
    """
    df = df.copy()
    num_cols = [c for c in df.select_dtypes(include="number").columns
                if c != target_col]

    if fit:
        # Exclude binary / one-hot dummy columns from winsorisation.
        # For a dummy with <25% or >75% ones the IQR is 0, so the bounds
        # collapse to [0,0] (or [1,1]) and the entire column becomes a
        # constant -- destroying the encoded information. Outlier treatment
        # is only meaningful for continuous features.
        continuous_cols = [c for c in num_cols if df[c].nunique() > 2]
        n_skipped = len(num_cols) - len(continuous_cols)
        if n_skipped:
            logger.info(f"Winsorisation: skipping {n_skipped} binary/dummy columns")
        num_cols = continuous_cols

        bounds = {}
        for col in num_cols:
            lower, upper = compute_iqr_bounds(df[col], k=k)
            bounds[col] = (lower, upper)
    else:
        # Apply exactly the columns selected during fit (train/test consistency)
        num_cols = [c for c in num_cols if c in (bounds or {})]

    n_clipped = 0
    for col in num_cols:
        if col not in bounds:
            continue
        lower, upper = bounds[col]
        before = df[col].copy()
        df[col] = df[col].clip(lower=lower, upper=upper)
        clipped = (df[col] != before).sum()
        n_clipped += clipped

    logger.info(f"Winsorisation (k={k}): {n_clipped:,} values adjusted in {len(num_cols)} columns")
    return df, bounds


def outlier_report(df: pd.DataFrame, target_col: str = "target_bin",
                   k: float = 1.5) -> pd.DataFrame:
    """Return an overview of outlier proportions."""
    num_cols = [c for c in df.select_dtypes(include="number").columns
                if c != target_col]
    rows = []
    for col in num_cols:
        lower, upper = compute_iqr_bounds(df[col], k)
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        rows.append({
            "Feature":     col,
            "Lower_Bound": round(lower, 4),
            "Upper_Bound": round(upper, 4),
            "Outlier_N":   n_out,
            "Outlier_Pct": round(n_out / len(df) * 100, 2),
        })
    return pd.DataFrame(rows).sort_values("Outlier_Pct", ascending=False)
