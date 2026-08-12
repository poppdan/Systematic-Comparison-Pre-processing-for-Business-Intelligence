"""
interaction_terms.py -- Interaction feature generation via PolynomialFeatures.

Wraps sklearn.preprocessing.PolynomialFeatures(degree=2, interaction_only=True)
to add pairwise interaction terms (x_i * x_j) for all numeric features.

Design decisions:
  - Applied AFTER imputation but BEFORE scaling (as specified in Pipeline C).
  - fit() only on training data; transform() applied to both train and test
    (no data leakage).
  - Operates on numeric columns only; categorical columns pass through untouched.
  - include_bias=False: no constant term added (the scaler handles centering).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from scripts.utils import get_logger

logger = get_logger("interaction_terms")


def add_interaction_terms(df, target_col="target_bin", fit=True,
                          poly=None, num_cols=None):
    """
    Add pairwise interaction terms (degree=2, interaction_only=True) to numeric
    columns of df.

    Parameters
    ----------
    df         : Input DataFrame (may contain numeric and categorical columns)
    target_col : Column to exclude from interaction computation
    fit        : True  = fit PolynomialFeatures on df (training set)
                 False = apply already-fitted poly to df (test set)
    poly       : Fitted PolynomialFeatures instance (required when fit=False)
    num_cols   : List of numeric columns used during fit (required when fit=False)

    Returns
    -------
    df_out   : DataFrame with original columns replaced by original + interaction cols
    poly     : Fitted PolynomialFeatures instance
    num_cols : List of numeric columns that were used
    """
    df = df.copy()

    # Identify numeric columns to interact (exclude target)
    if fit:
        num_cols = [c for c in df.select_dtypes(include="number").columns
                    if c != target_col]

    cat_cols = [c for c in df.columns if c not in num_cols and c != target_col]

    if not num_cols:
        logger.warning("No numeric columns found for interaction terms.")
        return df, poly, num_cols

    X_num = df[num_cols].values

    if fit:
        # degree=2, interaction_only=True produces only x_i*x_j terms (no x_i^2)
        # include_bias=False: skip the constant column
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_inter = poly.fit_transform(X_num)
        logger.info(
            f"PolynomialFeatures fitted: {len(num_cols)} numeric features "
            f"-> {X_inter.shape[1]} features (including interactions)"
        )
    else:
        X_inter = poly.transform(X_num)
        logger.info(
            f"PolynomialFeatures applied (transform only): "
            f"{X_inter.shape[1]} features"
        )

    feature_names = poly.get_feature_names_out(num_cols)
    df_inter = pd.DataFrame(X_inter, columns=feature_names, index=df.index)

    # Combine: interaction-expanded numeric + categorical pass-through + target
    parts = [df_inter]
    if cat_cols:
        parts.append(df[cat_cols])
    if target_col in df.columns:
        parts.append(df[[target_col]])

    df_out = pd.concat(parts, axis=1)
    return df_out, poly, num_cols
