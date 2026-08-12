"""
encoder.py -- Categorical encoding via One-Hot Encoding.

Strategy:
  - pandas get_dummies (drop_first=False for interpretability)
  - Unknown categories in the test set are set to 0 (sparse handling)
  - Target variable is never transformed
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np

from scripts.utils import get_logger

logger = get_logger("encoder")


def one_hot_encode(df: pd.DataFrame, target_col: str = "target_bin",
                   fit: bool = True, train_columns: list = None):
    """
    One-Hot Encoding of all categorical columns.

    Parameters
    ----------
    df            : Input DataFrame
    target_col    : Target variable -- will not be encoded
    fit           : True = derive new column list (train)
                    False = align to stored column list (test)
    train_columns : List of OHE columns from training (for fit=False)

    Returns
    -------
    df_out, train_columns
    """
    df = df.copy()

    # Exclude target variable
    target = df[[target_col]].copy() if target_col in df.columns else None
    cat_cols = [c for c in df.select_dtypes(include="object").columns
                if c != target_col]

    if not cat_cols:
        logger.info("No categorical columns to encode.")
        return df, train_columns

    df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=float)

    if fit:
        train_columns = df.columns.tolist()
        logger.info(f"OHE: {len(cat_cols)} categorical columns -> {len(train_columns)} total columns")
    else:
        # Fill missing columns with 0, drop extra columns
        missing_cols = [c for c in train_columns if c not in df.columns]
        extra_cols   = [c for c in df.columns if c not in train_columns]
        for c in missing_cols:
            df[c] = 0.0
        df = df.drop(columns=extra_cols, errors="ignore")
        df = df[train_columns]
        logger.info(f"OHE (test): {len(missing_cols)} missing columns added, "
                    f"{len(extra_cols)} unknown columns removed")

    return df, train_columns
