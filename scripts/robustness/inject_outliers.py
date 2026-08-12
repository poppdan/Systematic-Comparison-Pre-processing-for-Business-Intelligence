"""
inject_outliers.py -- Synthetic outlier injection.

Replaces a fraction of numeric values with extreme outliers drawn from
outside the [Q1 - k*IQR, Q3 + k*IQR] fence (default k=3).

Two strategies:
  "random"   -- random cells receive extreme values
  "targeted" -- replace values already near extremes (amplify existing tails)

Usage:
  from scripts.robustness.inject_outliers import inject_outliers

  X_corrupted = inject_outliers(X, outlier_rate=0.05, seed=42)
"""
import numpy as np
import pandas as pd


def inject_outliers(X: pd.DataFrame,
                    outlier_rate: float = 0.05,
                    k: float = 5.0,
                    strategy: str = "random",
                    seed: int = 42) -> pd.DataFrame:
    """
    Inject outliers into numeric columns.

    Parameters
    ----------
    X            : Input DataFrame
    outlier_rate : Fraction of values per column to replace (default: 5%)
    k            : Multiplier beyond IQR fence for outlier magnitude (default: 5x)
    strategy     : "random" or "targeted"
    seed         : Random seed

    Returns
    -------
    DataFrame with injected outliers
    """
    rng = np.random.default_rng(seed)
    X_out = X.copy()
    num_cols = X_out.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        vals = X_out[col].dropna().values
        if len(vals) == 0:
            continue

        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            iqr = np.std(vals) + 1e-8

        low_extreme  = q1 - k * iqr
        high_extreme = q3 + k * iqr

        n_inject = max(1, int(np.ceil(outlier_rate * len(X_out))))

        if strategy == "random":
            idx = rng.choice(len(X_out), size=n_inject, replace=False)
        else:  # targeted -- prefer rows near the existing extremes
            abs_vals = np.abs(X_out[col].fillna(X_out[col].median()).values)
            probs = abs_vals / (abs_vals.sum() + 1e-8)
            idx = rng.choice(len(X_out), size=n_inject, replace=False, p=probs)

        # Randomly assign to upper or lower extreme
        signs = rng.choice([-1, 1], size=n_inject)
        extreme_vals = np.where(
            signs > 0,
            high_extreme + rng.uniform(0, iqr, n_inject),
            low_extreme  - rng.uniform(0, iqr, n_inject),
        )
        X_out.iloc[idx, X_out.columns.get_loc(col)] = extreme_vals

    return X_out


def sweep_outlier_rates(X: pd.DataFrame,
                         rates: list = None,
                         seed: int = 42) -> dict:
    """
    Inject outliers at multiple rates for robustness curves.

    Parameters
    ----------
    X     : Input DataFrame
    rates : List of outlier rates (default: [0.01, 0.05, 0.10, 0.20])
    seed  : Random seed

    Returns
    -------
    dict {rate: corrupted_DataFrame}
    """
    if rates is None:
        rates = [0.01, 0.05, 0.10, 0.20]
    return {r: inject_outliers(X, outlier_rate=r, seed=seed) for r in rates}
