"""
inject_missing.py -- Synthetic missing value injection.

Implements three missing-data mechanisms (Little & Rubin, 2019):

  MCAR -- Missing Completely At Random
    Each value is independently masked with probability p.
    No relationship between missingness and data values.

  MAR  -- Missing At Random
    Missingness of column X depends on observed values of other columns.
    Here: mask column X if a correlated column is above its median.

  MNAR -- Missing Not At Random
    Missingness depends on the (unobserved) value itself.
    Here: mask large values -- the high-value entries go missing.

Usage:
  from scripts.robustness.inject_missing import inject_mcar, inject_mar, inject_mnar

  X_corrupted = inject_mcar(X, missing_rate=0.10, seed=42)
"""
import numpy as np
import pandas as pd


def inject_mcar(X: pd.DataFrame, missing_rate: float = 0.10,
                seed: int = 42) -> pd.DataFrame:
    """
    MCAR: mask each cell independently with probability missing_rate.

    Parameters
    ----------
    X            : Input DataFrame (numeric columns only; categoricals untouched)
    missing_rate : Fraction of numeric values to set NaN (default: 10%)
    seed         : Random seed

    Returns
    -------
    DataFrame with NaN injected
    """
    rng = np.random.default_rng(seed)
    X_out = X.copy()
    num_cols = X_out.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        mask = rng.random(len(X_out)) < missing_rate
        X_out.loc[mask, col] = np.nan

    frac = X_out[num_cols].isna().mean().mean()
    return X_out


def inject_mar(X: pd.DataFrame, missing_rate: float = 0.10,
               seed: int = 42) -> pd.DataFrame:
    """
    MAR: missingness of each numeric column depends on a randomly chosen
    other numeric column (high values in the pivot -> column goes missing).

    Parameters
    ----------
    X            : Input DataFrame
    missing_rate : Approximate fraction of values to mask
    seed         : Random seed

    Returns
    -------
    DataFrame with NaN injected
    """
    rng = np.random.default_rng(seed)
    X_out = X.copy()
    num_cols = list(X_out.select_dtypes(include=[np.number]).columns)

    if len(num_cols) < 2:
        return inject_mcar(X, missing_rate, seed)

    for col in num_cols:
        # Pick a random pivot column (different from target column)
        other_cols = [c for c in num_cols if c != col]
        pivot = rng.choice(other_cols)
        threshold = X_out[pivot].quantile(1 - missing_rate)
        mask = X_out[pivot] >= threshold
        X_out.loc[mask, col] = np.nan

    return X_out


def inject_mnar(X: pd.DataFrame, missing_rate: float = 0.10,
                seed: int = 42) -> pd.DataFrame:
    """
    MNAR: large values in each numeric column are masked (they go missing
    because high-value observations are harder to collect or report).

    Parameters
    ----------
    X            : Input DataFrame
    missing_rate : Fraction of high-value entries to mask per column
    seed         : Random seed

    Returns
    -------
    DataFrame with NaN injected
    """
    rng = np.random.default_rng(seed)
    X_out = X.copy()
    num_cols = X_out.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        threshold = X_out[col].quantile(1 - missing_rate)
        mask = X_out[col] >= threshold
        X_out.loc[mask, col] = np.nan

    return X_out


def sweep_missing_rates(X: pd.DataFrame, mechanism: str = "mcar",
                         rates: list = None, seed: int = 42) -> dict:
    """
    Inject missing values at multiple rates for robustness curves.

    Parameters
    ----------
    X         : Input DataFrame
    mechanism : "mcar", "mar", or "mnar"
    rates     : List of missing rates (default: [0.05, 0.10, 0.20, 0.30, 0.50])
    seed      : Random seed

    Returns
    -------
    dict {rate: corrupted_DataFrame}
    """
    if rates is None:
        rates = [0.05, 0.10, 0.20, 0.30, 0.50]

    fn = {"mcar": inject_mcar, "mar": inject_mar, "mnar": inject_mnar}[mechanism]
    return {r: fn(X, missing_rate=r, seed=seed) for r in rates}
