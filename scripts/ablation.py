"""
ablation.py -- Pipeline-specific ablation study (v3, leakage-free).

Ablation structure:
  Stage 0   raw            — no preprocessing (LabelEncode only so LR can run)
  Stage 1a  impute         — imputation only (median for numeric, mode for categorical)
  Stage 1b  impute+encode  — imputation + one-hot encoding
  Stage 1c  impute+enc+scl — imputation + OHE + StandardScaler  (shared baseline)
  Stage 2   full           — complete pipeline, adding the pipeline-specific step(s):
                             A   : same as Stage 1c (A IS the base preprocessing)
                             B   : + Winsorize + BoxCox + PCA
                             C   : + domain features + interaction terms
                             D   : + DAE encoder (trained per fold — no leakage)
                             E   : + VAE encoder (trained per fold — no leakage)
                             F   : + FT-Transformer (trained per fold — no leakage)

The multiple Stage 1 sub-stages (a/b/c) isolate the marginal contribution of each
individual preprocessing step. Stage 1c → Stage 2 isolates the pipeline innovation.

Downstream model : Logistic Regression (fast ablation proxy, consistent across stages)
CV               : 5x3 Repeated Stratified K-Fold (15 splits), same as main experiment

Neural encoder training in Stage 2 uses reduced epochs (ABLATION_EPOCHS / ABLATION_PATIENCE)
to keep runtime manageable while preserving relative comparisons.

Usage:
  python scripts/ablation.py
  python scripts/ablation.py --dataset bank --pipeline a
  python scripts/ablation.py --pipeline f
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression

from config import (BANK_RAW, RETAIL_RAW, DATA_PROC, RESULTS_DIR, SEED,
                    N_FOLDS, N_REPEATS, LATENT_DIM, BATCH_SIZE)
from scripts.utils import get_logger, ensure_dirs
from scripts.classical.missing_values import handle_missing
from scripts.classical.encoder import one_hot_encode
from scripts.classical.scaler import scale_features
from scripts.classical.pipeline_a import load_retail_temporal

logger = get_logger("ablation")
ensure_dirs(RESULTS_DIR / "ablation")

# Reduced epochs for within-fold encoder training (ablation proxy)
ABLATION_EPOCHS   = 50
ABLATION_PATIENCE = 7


# ---------------------------------------------------------------------------
# Stage 0 — raw baseline (no preprocessing)
# ---------------------------------------------------------------------------

def _stage_raw(X_tr: pd.DataFrame, X_va: pd.DataFrame, **_):
    """
    Stage 0 — raw: LabelEncode categoricals + fillna median (minimal viable input).
    No imputation, no OHE, no scaling — just enough for LR to accept the data.
    """
    def _minimal(X, X_ref):
        X = X.copy()
        for col in X.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            le.fit(X_ref[col].astype(str))
            mapping = {c: i for i, c in enumerate(le.classes_)}
            X[col] = X[col].astype(str).map(lambda v, m=mapping: m.get(v, 0))
        return X.fillna(X_ref.median(numeric_only=True))

    X_tr_enc = _minimal(X_tr, X_tr)
    X_va_enc = _minimal(X_va, X_tr)
    # No scaler — raw values
    return X_tr_enc.values.astype(np.float32), X_va_enc.values.astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 1 helpers — isolated single steps (each applied on top of Stage 0 raw)
#
# Every function starts from the raw DataFrame and applies ONLY its specific
# step (plus the minimum prep needed to make the step technically runnable).
# This lets us measure the isolated contribution of each step.
# ---------------------------------------------------------------------------

def _raw_prep_df(X_tr: pd.DataFrame, X_va: pd.DataFrame):
    """
    Minimal preparation: LabelEncode categoricals + fillna median.
    Returns DataFrames (not arrays). Used as the starting point for isolated steps.
    """
    def _encode(X, X_ref):
        X = X.copy()
        for col in X.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            le.fit(X_ref[col].astype(str))
            mapping = {c: i for i, c in enumerate(le.classes_)}
            X[col] = X[col].astype(str).map(lambda v, m=mapping: m.get(v, 0))
        return X.fillna(X_ref.median(numeric_only=True))

    X_tr_enc = _encode(X_tr, X_tr)
    X_va_enc = _encode(X_va, X_tr)
    return X_tr_enc, X_va_enc


def _step_impute(X_tr: pd.DataFrame, X_va: pd.DataFrame, **_):
    """Stage 1 — impute only: proper median/mode imputation + LabelEncode, no OHE, no scaling."""
    X_tr, num_imp, cat_imp = handle_missing(X_tr, fit=True)
    X_va = X_va[[c for c in X_tr.columns if c in X_va.columns]]
    X_va, _, _ = handle_missing(X_va, fit=False,
                                num_imputer=num_imp, cat_imputer=cat_imp)
    for col in X_tr.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X_tr[col] = le.fit_transform(X_tr[col].astype(str))
        mapping = {c: i for i, c in enumerate(le.classes_)}
        X_va[col] = X_va[col].astype(str).map(lambda v, m=mapping: m.get(v, 0))
    return X_tr.values.astype(np.float32), X_va.values.astype(np.float32)


def _step_ohe(X_tr: pd.DataFrame, X_va: pd.DataFrame, **_):
    """Stage 1 — OHE only: fillna (no LabelEncode) + one-hot encoding, no scaling.
    Numeric columns are passed through as-is after fillna."""
    # fillna only — do NOT LabelEncode, that would remove categorical dtype
    X_tr = X_tr.copy().fillna(X_tr.median(numeric_only=True))
    X_va = X_va.copy().fillna(X_tr.median(numeric_only=True))
    X_tr, train_cols = one_hot_encode(X_tr, fit=True)
    X_va, _ = one_hot_encode(X_va, fit=False, train_columns=train_cols)
    return X_tr.values.astype(np.float32), X_va.values.astype(np.float32)


def _step_scale(X_tr: pd.DataFrame, X_va: pd.DataFrame, **_):
    """Stage 1 — scale only: raw + StandardScaler (LabelEncode needed first)."""
    X_tr, X_va = _raw_prep_df(X_tr, X_va)
    sc = StandardScaler()
    return (sc.fit_transform(X_tr.values).astype(np.float32),
            sc.transform(X_va.values).astype(np.float32))


def _step_winsorize(X_tr: pd.DataFrame, X_va: pd.DataFrame, **_):
    """Stage 1 — winsorize only: raw + IQR Winsorizing + StandardScaler."""
    from scripts.classical.outlier_handler import winsorize
    X_tr, X_va = _raw_prep_df(X_tr, X_va)
    X_tr, bounds = winsorize(X_tr, fit=True, k=1.5)
    X_va, _ = winsorize(X_va, fit=False, bounds=bounds)
    sc = StandardScaler()
    X_tr_arr = sc.fit_transform(X_tr.values).astype(np.float32)
    X_va_arr = sc.transform(X_va.values).astype(np.float32)
    return np.nan_to_num(X_tr_arr), np.nan_to_num(X_va_arr)


def _step_boxcox(X_tr: pd.DataFrame, X_va: pd.DataFrame, **_):
    """Stage 1 — BoxCox only: raw + Box-Cox + StandardScaler.
    Clip before StandardScaler — BoxCox can produce values large enough to
    overflow float64 when squared during variance computation."""
    from scripts.classical.boxcox import apply_boxcox
    X_tr, X_va = _raw_prep_df(X_tr, X_va)
    X_tr, lambdas = apply_boxcox(X_tr, fit=True)
    X_va, _ = apply_boxcox(X_va, fit=False, lambdas=lambdas)
    X_tr_v = np.nan_to_num(X_tr.values, nan=0.0, posinf=0.0, neginf=0.0)
    X_va_v = np.nan_to_num(X_va.values, nan=0.0, posinf=0.0, neginf=0.0)
    X_tr_v = np.clip(X_tr_v, -1e6, 1e6)
    X_va_v = np.clip(X_va_v, -1e6, 1e6)
    sc = StandardScaler()
    return sc.fit_transform(X_tr_v).astype(np.float32), sc.transform(X_va_v).astype(np.float32)


def _step_pca(X_tr: pd.DataFrame, X_va: pd.DataFrame,
              variance: float = 0.9, **_):
    """Stage 1 — PCA only: raw + StandardScaler + PCA (scale prerequisite for PCA)."""
    X_tr, X_va = _raw_prep_df(X_tr, X_va)
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr.values)
    X_va_sc = sc.transform(X_va.values)
    pca = PCA(n_components=variance, random_state=SEED)
    return (pca.fit_transform(X_tr_sc).astype(np.float32),
            pca.transform(X_va_sc).astype(np.float32))


def _step_domain(X_tr: pd.DataFrame, X_va: pd.DataFrame,
                 dataset: str = "bank", **_):
    """Stage 1 — domain features only: add domain features, then crude prep."""
    if dataset == "bank":
        from scripts.classical.pipeline_c import add_bank_domain_features
        X_tr = add_bank_domain_features(X_tr)
        X_va = add_bank_domain_features(X_va)
    X_tr, X_va = _raw_prep_df(X_tr, X_va)
    return X_tr.values.astype(np.float32), X_va.values.astype(np.float32)


def _step_interactions(X_tr: pd.DataFrame, X_va: pd.DataFrame, **_):
    """Stage 1 — interaction terms only: raw + polynomial interaction terms + scale."""
    from scripts.classical.interaction_terms import add_interaction_terms
    X_tr, X_va = _raw_prep_df(X_tr, X_va)
    X_tr, poly, num_cols_poly = add_interaction_terms(X_tr, fit=True)
    X_va, _, _ = add_interaction_terms(X_va, fit=False,
                                       poly=poly, num_cols=num_cols_poly)
    sc = StandardScaler()
    return (sc.fit_transform(X_tr.values).astype(np.float32),
            sc.transform(X_va.values).astype(np.float32))


def _step_dae(X_tr: pd.DataFrame, X_va: pd.DataFrame, **_):
    """Stage 1 — DAE only: raw + standard prep + Denoising Autoencoder (per-fold)."""
    import torch
    from scripts.ai.pipeline_d import _train_dae, _encode
    (X_tr_arr, X_va_arr, *_) = _prep_for_neural(X_tr, X_va)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _train_dae(X_tr_arr, latent_dim=LATENT_DIM, device=device,
                       epochs=ABLATION_EPOCHS, patience=ABLATION_PATIENCE)
    return _encode(model, X_tr_arr, device), _encode(model, X_va_arr, device)


def _step_vae(X_tr: pd.DataFrame, X_va: pd.DataFrame, **_):
    """Stage 1 — VAE only: raw + standard prep + Variational Autoencoder (per-fold)."""
    import torch
    from scripts.ai.pipeline_e import _train_vae, _encode
    (X_tr_arr, X_va_arr, *_) = _prep_for_neural(X_tr, X_va)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _train_vae(X_tr_arr, latent_dim=LATENT_DIM, device=device,
                       epochs=ABLATION_EPOCHS, patience=ABLATION_PATIENCE)
    return _encode(model, X_tr_arr, device), _encode(model, X_va_arr, device)


def _step_ftt(X_tr: pd.DataFrame, X_va: pd.DataFrame,
              y_tr: np.ndarray = None, **_):
    """Stage 1 — FTT only: raw + standard prep + FT-Transformer (per-fold)."""
    import torch
    from scripts.ai.pipeline_f import _train_ftt, _encode
    (X_tr_arr, X_va_arr,
     X_num_tr, X_num_va,
     X_cat_tr, X_cat_va,
     cat_cols, num_cols, cardinalities) = _prep_for_neural(X_tr, X_va)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _train_ftt(X_num_tr, X_cat_tr, y_tr,
                       cardinalities=cardinalities, device=device,
                       epochs=ABLATION_EPOCHS, patience=ABLATION_PATIENCE)
    return (_encode(model, X_num_tr, X_cat_tr, device),
            _encode(model, X_num_va, X_cat_va, device))


# ---------------------------------------------------------------------------
# Per-pipeline Stage 1 step definitions
# step list: [(step_name, fn, pass_y_tr), ...]
# ---------------------------------------------------------------------------

PIPELINE_STEPS: dict = {
    "a": [
        ("impute",  _step_impute, False),
        ("ohe",     _step_ohe,    False),
        ("scale",   _step_scale,  False),
    ],
    "b": [
        ("impute",    _step_impute,    False),
        ("ohe",       _step_ohe,       False),
        ("winsorize", _step_winsorize, False),
        ("boxcox",    _step_boxcox,    False),
        ("scale",     _step_scale,     False),
        ("pca90",     lambda X_tr, X_va, **kw: _step_pca(X_tr, X_va, variance=0.9, **kw), False),
        ("pca20",     lambda X_tr, X_va, **kw: _step_pca(X_tr, X_va, variance=0.2, **kw), False),
    ],
    "c": [
        ("impute",       _step_impute,       False),
        ("ohe",          _step_ohe,          False),
        ("scale",        _step_scale,        False),
        ("domain",       None,               False),   # filled in run_ablation with dataset kwarg
        ("interactions", _step_interactions, False),
    ],
    "d": [
        ("impute", _step_impute, False),
        ("ohe",    _step_ohe,    False),
        ("scale",  _step_scale,  False),
        ("dae",    _step_dae,    False),
    ],
    "e": [
        ("impute", _step_impute, False),
        ("ohe",    _step_ohe,    False),
        ("scale",  _step_scale,  False),
        ("vae",    _step_vae,    False),
    ],
    "f": [
        ("impute", _step_impute, False),
        ("ohe",    _step_ohe,    False),
        ("scale",  _step_scale,  False),
        ("ftt",    _step_ftt,    True),
    ],
}


# ---------------------------------------------------------------------------
# Stage 2 — full pipeline (all steps combined)
# ---------------------------------------------------------------------------

def _stage2_a(X_tr: pd.DataFrame, X_va: pd.DataFrame, **_):
    """A full: impute + OHE + StandardScaler."""
    X_tr, num_imp, cat_imp = handle_missing(X_tr, fit=True)
    X_va = X_va[[c for c in X_tr.columns if c in X_va.columns]]
    X_va, _, _ = handle_missing(X_va, fit=False,
                                num_imputer=num_imp, cat_imputer=cat_imp)
    X_tr, train_cols = one_hot_encode(X_tr, fit=True)
    X_va, _ = one_hot_encode(X_va, fit=False, train_columns=train_cols)
    X_tr, scaler = scale_features(X_tr, fit=True, method="standard")
    X_va, _ = scale_features(X_va, fit=False, scaler=scaler)
    return X_tr.values.astype(np.float32), X_va.values.astype(np.float32)


def _stage2_b(X_tr: pd.DataFrame, X_va: pd.DataFrame,
              pipeline: str = "b_pca90", **_):
    """B full: handle_missing + OHE + StandardScaler + Winsorize + BoxCox + PCA."""
    from scripts.classical.outlier_handler import winsorize
    from scripts.classical.boxcox import apply_boxcox

    X_tr, num_imp, cat_imp = handle_missing(X_tr, fit=True)
    X_va = X_va[[c for c in X_tr.columns if c in X_va.columns]]
    X_va, _, _ = handle_missing(X_va, fit=False,
                                num_imputer=num_imp, cat_imputer=cat_imp)
    X_tr, train_cols = one_hot_encode(X_tr, fit=True)
    X_va, _ = one_hot_encode(X_va, fit=False, train_columns=train_cols)

    # Winsorize + BoxCox (the unique B steps)
    X_tr, bounds = winsorize(X_tr, fit=True, k=1.5)
    X_va, _ = winsorize(X_va, fit=False, bounds=bounds)
    X_tr, lambdas = apply_boxcox(X_tr, fit=True)
    X_va, _ = apply_boxcox(X_va, fit=False, lambdas=lambdas)

    X_tr, scaler = scale_features(X_tr, fit=True, method="standard")
    X_va, _ = scale_features(X_va, fit=False, scaler=scaler)

    variance = 0.90 if pipeline == "b_pca90" else 0.20
    pca = PCA(n_components=variance, random_state=SEED)
    X_tr_arr = pca.fit_transform(X_tr.values).astype(np.float32)
    X_va_arr = pca.transform(X_va.values).astype(np.float32)
    return X_tr_arr, X_va_arr


def _stage2_c(X_tr: pd.DataFrame, X_va: pd.DataFrame,
              dataset: str = "bank", **_):
    """C full: domain features + interactions added on top of standard preprocessing."""
    from scripts.classical.interaction_terms import add_interaction_terms

    # Domain features for bank (row-wise, no leakage)
    if dataset == "bank":
        from scripts.classical.pipeline_c import add_bank_domain_features
        X_tr = add_bank_domain_features(X_tr)
        X_va = add_bank_domain_features(X_va)

    X_tr, num_imp, cat_imp = handle_missing(X_tr, fit=True)
    X_va = X_va[[c for c in X_tr.columns if c in X_va.columns]]
    X_va, _, _ = handle_missing(X_va, fit=False,
                                num_imputer=num_imp, cat_imputer=cat_imp)

    # Interaction terms (unique C step)
    X_tr, poly, num_cols_poly = add_interaction_terms(X_tr, fit=True)
    X_va, _, _ = add_interaction_terms(X_va, fit=False,
                                       poly=poly, num_cols=num_cols_poly)

    X_tr, train_cols = one_hot_encode(X_tr, fit=True)
    X_va, _ = one_hot_encode(X_va, fit=False, train_columns=train_cols)
    X_tr, scaler = scale_features(X_tr, fit=True, method="standard")
    X_va, _ = scale_features(X_va, fit=False, scaler=scaler)
    return X_tr.values.astype(np.float32), X_va.values.astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 2 — AI pipelines: per-fold encoder training (leakage-free)
# ---------------------------------------------------------------------------

def _prep_for_neural(X_tr_df: pd.DataFrame, X_va_df: pd.DataFrame):
    """
    Shared preprocessing before neural encoder:
    handle_missing + LabelEncode + StandardScaler, all fit on X_tr only.

    Returns
    -------
    X_tr_arr, X_va_arr  : float32 numpy arrays (full feature matrix)
    X_num_tr, X_num_va  : numeric columns only (for FTT)
    X_cat_tr, X_cat_va  : categorical columns as int64 (for FTT)
    cat_cols, num_cols  : column name lists
    cardinalities       : list of max category index per cat column (for FTT)
    """
    X_tr, num_imp, cat_imp = handle_missing(X_tr_df.copy(), fit=True)
    X_va = X_va_df.copy()[[c for c in X_tr.columns if c in X_va_df.columns]]
    X_va, _, _ = handle_missing(X_va, fit=False,
                                num_imputer=num_imp, cat_imputer=cat_imp)

    cat_cols = X_tr.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()

    encoders = {}
    cardinalities = []
    for col in cat_cols:
        le = LabelEncoder()
        X_tr[col] = le.fit_transform(X_tr[col].astype(str))
        mapping = {c: i for i, c in enumerate(le.classes_)}
        X_va[col] = X_va[col].astype(str).map(lambda v, m=mapping: m.get(v, 0))
        encoders[col] = le
        cardinalities.append(int(X_tr[col].max()))

    scaler = StandardScaler()
    if num_cols:
        X_tr[num_cols] = scaler.fit_transform(X_tr[num_cols])
        X_va[num_cols] = scaler.transform(X_va[num_cols])

    X_tr_arr = X_tr.values.astype(np.float32)
    X_va_arr = X_va.values.astype(np.float32)

    X_num_tr = X_tr[num_cols].values.astype(np.float32) if num_cols else None
    X_num_va = X_va[num_cols].values.astype(np.float32) if num_cols else None
    X_cat_tr = X_tr[cat_cols].values.astype(np.int64)   if cat_cols else None
    X_cat_va = X_va[cat_cols].values.astype(np.int64)   if cat_cols else None

    return (X_tr_arr, X_va_arr,
            X_num_tr, X_num_va, X_cat_tr, X_cat_va,
            cat_cols, num_cols, cardinalities)


def _stage2_d(X_tr: pd.DataFrame, X_va: pd.DataFrame,
              epochs: int = ABLATION_EPOCHS,
              patience: int = ABLATION_PATIENCE, **_):
    """D full: per-fold DAE trained on X_tr only (leakage-free)."""
    import torch
    from scripts.ai.pipeline_d import _train_dae, _encode

    (X_tr_arr, X_va_arr,
     _, _, _, _, _, _, _) = _prep_for_neural(X_tr, X_va)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _train_dae(X_tr_arr, latent_dim=LATENT_DIM, device=device,
                       epochs=epochs, patience=patience)
    return _encode(model, X_tr_arr, device), _encode(model, X_va_arr, device)


def _stage2_e(X_tr: pd.DataFrame, X_va: pd.DataFrame,
              epochs: int = ABLATION_EPOCHS,
              patience: int = ABLATION_PATIENCE, **_):
    """E full: per-fold VAE trained on X_tr only (leakage-free)."""
    import torch
    from scripts.ai.pipeline_e import _train_vae, _encode

    (X_tr_arr, X_va_arr,
     _, _, _, _, _, _, _) = _prep_for_neural(X_tr, X_va)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _train_vae(X_tr_arr, latent_dim=LATENT_DIM, device=device,
                       epochs=epochs, patience=patience)
    return _encode(model, X_tr_arr, device), _encode(model, X_va_arr, device)


def _stage2_f(X_tr: pd.DataFrame, X_va: pd.DataFrame,
              y_tr: np.ndarray = None,
              epochs: int = ABLATION_EPOCHS,
              patience: int = ABLATION_PATIENCE, **_):
    """F full: per-fold FT-Transformer trained on X_tr+y_tr only (leakage-free)."""
    import torch
    from scripts.ai.pipeline_f import _train_ftt, _encode

    (X_tr_arr, X_va_arr,
     X_num_tr, X_num_va,
     X_cat_tr, X_cat_va,
     cat_cols, num_cols, cardinalities) = _prep_for_neural(X_tr, X_va)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = _train_ftt(
        X_num_tr, X_cat_tr, y_tr,
        cardinalities=cardinalities,
        device=device,
        epochs=epochs,
        patience=patience,
    )
    return (_encode(model, X_num_tr, X_cat_tr, device),
            _encode(model, X_num_va, X_cat_va, device))


# ---------------------------------------------------------------------------
# CV runner — preprocess_fn called inside each fold
# ---------------------------------------------------------------------------

def _cv_with_preprocess(X_raw: pd.DataFrame, y: np.ndarray,
                        preprocess_fn,
                        pass_y_tr: bool = False,
                        seed: int = SEED) -> dict:
    """
    5x3 RSKF — preprocessing is applied per fold to avoid leakage.

    Parameters
    ----------
    X_raw        : Clean feature DataFrame (full dataset)
    y            : Target array
    preprocess_fn: (X_tr_df, X_va_df, **kw) -> (X_tr_arr, X_va_arr)
                   If pass_y_tr=True, also receives y_tr=y[tr] as kwarg.
    pass_y_tr    : Pass training labels to preprocess_fn (needed for FTT)
    seed         : RNG seed
    """
    rskf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS,
                                   random_state=seed)
    scores = []
    for tr, va in rskf.split(X_raw, y):
        X_tr_df = X_raw.iloc[tr].copy()
        X_va_df = X_raw.iloc[va].copy()
        y_tr = y[tr]
        y_va = y[va]

        kwargs = {"y_tr": y_tr} if pass_y_tr else {}
        try:
            X_tr_proc, X_va_proc = preprocess_fn(X_tr_df, X_va_df, **kwargs)
        except Exception as e:
            logger.warning(f"Preprocessing failed in fold: {e}")
            continue

        if X_tr_proc.shape[1] == 0 or X_va_proc.shape[1] == 0:
            logger.warning("0 features after preprocessing — skipping fold.")
            continue

        model = LogisticRegression(max_iter=1000, random_state=seed, solver="saga")
        model.fit(X_tr_proc, y_tr)
        proba = model.predict_proba(X_va_proc)[:, 1]
        scores.append(average_precision_score(y_va, proba))

    if not scores:
        return {"mean_PR_AUC": float("nan"), "std_PR_AUC": float("nan"), "n_folds": 0}

    return {
        "mean_PR_AUC": round(float(np.mean(scores)), 5),
        "std_PR_AUC":  round(float(np.std(scores)),  5),
        "n_folds":     len(scores),
    }


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _load_bank():
    df = pd.read_csv(BANK_RAW, sep=";")
    df = df.rename(columns={"y": "target"})
    df["target"] = (df["target"] == "yes").astype(int)
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])
    return df.drop(columns=["target"]), df["target"].values


def _load_retail():
    rfm = load_retail_temporal()
    return rfm.drop(columns=["target_bin"]), rfm["target_bin"].values


DATASET_LOADERS = {"bank": _load_bank, "retail": _load_retail}


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

def run_ablation(dataset: str, pipeline: str) -> pd.DataFrame:
    logger.info(f"\n{'='*55}")
    logger.info(f"Ablation: {dataset.upper()} x Pipeline {pipeline.upper()}")
    logger.info(f"{'='*55}")

    out = RESULTS_DIR / "ablation" / f"{dataset}_{pipeline}_ablation.csv"

    # Load partial results if they exist (resume after crash)
    if out.exists():
        existing = pd.read_csv(out)
        done = set(existing["stage_name"].tolist())
        records = existing.to_dict("records")
        logger.info(f"  Resuming {dataset}/{pipeline} — {len(done)} steps already done: {done}")
    else:
        existing = None
        done = set()
        records = []

    # Skip entirely if all steps are already finished
    steps_needed = PIPELINE_STEPS.get(pipeline)
    if steps_needed is None:
        logger.error(f"Unknown pipeline: {pipeline}")
        return pd.DataFrame(records)
    full_names = {"full_pca90", "full_pca20"} if pipeline == "b" else {"full"}
    all_step_names = {"raw"} | {s for s, _, _ in steps_needed} | full_names
    if done >= all_step_names:
        logger.info(f"  All steps done — skipping.")
        return pd.DataFrame(records)

    def _run_step(stage_num, stage_name, fn, pass_y):
        """Run one step, append to records, save immediately."""
        if stage_name in done:
            logger.info(f"  Skip     {stage_name:20s} (already done)")
            return
        res = _cv_with_preprocess(X_raw, y, fn, pass_y_tr=pass_y)
        records.append({"stage": stage_num, "stage_name": stage_name,
                        "pipeline": pipeline, "dataset": dataset, **res})
        pd.DataFrame(records).to_csv(out, index=False)
        logger.info(f"  Stage {stage_num}  {stage_name:20s} | "
                    f"PR-AUC: {res['mean_PR_AUC']:.4f} ± {res['std_PR_AUC']:.4f}  [saved]")

    X_raw, y = DATASET_LOADERS[dataset]()

    # ── Stage 0: raw baseline ─────────────────────────────────────────────────
    _run_step(0, "raw", _stage_raw, False)

    # ── Stage 1: each pipeline step in isolation ──────────────────────────────
    for step_name, step_fn, pass_y in steps_needed:
        if step_fn is None and step_name == "domain":
            step_fn = lambda X_tr, X_va, ds=dataset, **kw: _step_domain(
                X_tr, X_va, dataset=ds, **kw)
        _run_step(1, step_name, step_fn, pass_y)

    # ── Stage 2: full pipeline ────────────────────────────────────────────────
    if pipeline == "a":
        _run_step(2, "full", _stage2_a, False)
    elif pipeline == "b":
        # Two Stage 2 variants: pca90 and pca20
        _run_step(2, "full_pca90",
                  lambda X_tr, X_va, **kw: _stage2_b(X_tr, X_va, pipeline="b_pca90", **kw), False)
        _run_step(2, "full_pca20",
                  lambda X_tr, X_va, **kw: _stage2_b(X_tr, X_va, pipeline="b_pca20", **kw), False)
    elif pipeline == "c":
        _run_step(2, "full",
                  lambda X_tr, X_va, **kw: _stage2_c(X_tr, X_va, dataset=dataset, **kw), False)
    elif pipeline == "d":
        _run_step(2, "full", _stage2_d, False)
    elif pipeline == "e":
        _run_step(2, "full", _stage2_e, False)
    elif pipeline == "f":
        _run_step(2, "full", _stage2_f, True)
    else:
        logger.error(f"Unknown pipeline for Stage 2: {pipeline}")

    return pd.DataFrame(records)


def run_all(datasets=None, pipelines=None):
    from config import CLASSICAL_PIPELINES, AI_PIPELINES
    if datasets  is None: datasets  = ["bank", "retail"]
    if pipelines is None:
        raw = [p.lower() for p in CLASSICAL_PIPELINES + AI_PIPELINES]
        # Merge b_pca90 / b_pca20 into single "b" pipeline
        seen = set()
        pipelines = []
        for p in raw:
            key = "b" if p in ("b_pca90", "b_pca20") else p
            if key not in seen:
                seen.add(key)
                pipelines.append(key)

    for ds in datasets:
        for pl in pipelines:
            run_ablation(ds, pl)

    # Always rebuild the combined file from ALL per-pipeline CSVs on disk,
    # not only from the ones executed in this run. Otherwise a partial run
    # (e.g. --pipeline b) would silently truncate ablation_all.csv.
    abl_dir = RESULTS_DIR / "ablation"
    files = sorted(abl_dir.glob("*_ablation.csv"))
    if not files:
        logger.warning("No per-pipeline ablation CSVs found.")
        return pd.DataFrame()

    combined = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    combined.to_csv(abl_dir / "ablation_all.csv", index=False)
    logger.info(f"Combined ablation results saved "
                f"({len(files)} files, {len(combined)} rows).")
    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation study — pipeline-specific stages, leakage-free")
    parser.add_argument("--dataset",  default=None)
    parser.add_argument("--pipeline", default=None)
    args = parser.parse_args()
    run_all(datasets=[args.dataset]  if args.dataset  else None,
            pipelines=[args.pipeline] if args.pipeline else None)
