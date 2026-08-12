"""
missing_values.py -- Handling of missing values (MCAR/MAR/MNAR).

Strategy:
  - Numeric: Median imputation (robust against outliers)
  - Categorical: Most-frequent imputation
  - Columns with >50% missing are dropped
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

from scripts.utils import get_logger

logger = get_logger("missing_values")

DROP_THRESHOLD = 0.5  # Columns with >50% missing are removed


def handle_missing(df: pd.DataFrame, fit: bool = True,
                   num_imputer: SimpleImputer = None,
                   cat_imputer: SimpleImputer = None):
    """
    Imputation of missing values.

    Parameters
    ----------
    df          : Input DataFrame
    fit         : True = fit+transform (training set), False = transform only (test set)
    num_imputer : Already fitted numeric imputer (for fit=False)
    cat_imputer : Already fitted categorical imputer (for fit=False)

    Returns
    -------
    df_out, num_imputer, cat_imputer
    """
    df = df.copy()

    # Drop columns with too many missing values
    missing_frac = df.isnull().mean()
    drop_cols = missing_frac[missing_frac > DROP_THRESHOLD].index.tolist()
    if drop_cols:
        logger.warning(f"Dropping columns (>{DROP_THRESHOLD*100:.0f}% missing): {drop_cols}")
        df = df.drop(columns=drop_cols)

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # Numeric: Median
    if num_cols:
        if fit:
            num_imputer = SimpleImputer(strategy="median")
            df[num_cols] = num_imputer.fit_transform(df[num_cols])
        else:
            df[num_cols] = num_imputer.transform(df[num_cols])

    # Categorical: Most-Frequent
    if cat_cols:
        if fit:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
        else:
            df[cat_cols] = cat_imputer.transform(df[cat_cols])

    remaining = df.isnull().sum().sum()
    logger.info(f"Missing values after imputation: {remaining}")
    return df, num_imputer, cat_imputer


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return an overview of missing values."""
    report = pd.DataFrame({
        "Missing_Count": df.isnull().sum(),
        "Missing_Pct":   df.isnull().mean() * 100,
        "Dtype":         df.dtypes,
    }).sort_values("Missing_Count", ascending=False)
    return report[report["Missing_Count"] > 0]
