"""
scaler.py -- Feature Scaling.

Methods:
  - StandardScaler: Default (Z-Score normalisation, mean=0, std=1)
  - RobustScaler  : Median/IQR-based (resistant to outliers)
  - MinMaxScaler  : [0,1] (for neural networks)

StandardScaler is used by default as specified in the professor's pipeline spec.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

from scripts.utils import get_logger

logger = get_logger("scaler")

SCALER_MAP = {
    "robust":   RobustScaler,
    "standard": StandardScaler,
    "minmax":   MinMaxScaler,
}


def scale_features(df: pd.DataFrame, target_col: str = "target_bin",
                   method: str = "standard", fit: bool = True,
                   scaler=None):
    """
    Scale all numeric features except target_col.

    Parameters
    ----------
    df         : Input DataFrame
    target_col : Will not be scaled
    method     : 'standard' | 'robust' | 'minmax'
    fit        : True = fit+transform (train), False = transform only (test)
    scaler     : Already fitted scaler (for fit=False)

    Returns
    -------
    df_out, scaler
    """
    df = df.copy()
    num_cols = [c for c in df.select_dtypes(include="number").columns
                if c != target_col]

    if not num_cols:
        logger.warning("No numeric columns found.")
        return df, scaler

    if fit:
        ScalerClass = SCALER_MAP.get(method, StandardScaler)
        scaler = ScalerClass()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        logger.info(f"Scaler fitted: {method} on {len(num_cols)} features")
    else:
        df[num_cols] = scaler.transform(df[num_cols])
        logger.info(f"Scaler applied (transform only) on {len(num_cols)} features")

    # Clip extreme values that can arise after Box-Cox + StandardScaler
    import numpy as np
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    df[num_cols] = df[num_cols].fillna(0).clip(-10, 10)

    return df, scaler
