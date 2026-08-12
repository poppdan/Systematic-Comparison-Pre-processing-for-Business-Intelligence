"""
run_robustness.py -- Robustness sweep orchestrator.

For each dataset x pipeline x corruption type x intensity:
  1. Load raw data (with duration dropped for bank, temporal split for retail)
  2. Apply corruption to train fold only (val stays clean)
  3. Apply the ACTUAL pipeline preprocessing (fit on corrupted train, transform clean val)
  4. Evaluate downstream LR (fast proxy model) via 5x3 RSKF
  5. Record mean/std PR-AUC

AI pipelines (d, e, f) use their PRE-TRAINED encoder (fixed weights from main experiment).
This evaluates how well the learned representation handles corrupted inputs at inference time.

Results are saved to:
  results/robustness/{dataset}_{pipeline}_{mechanism}_robustness.csv
  results/robustness/robustness_all.csv   (combined)

Usage:
  python scripts/robustness/run_robustness.py
  python scripts/robustness/run_robustness.py --dataset bank
  python scripts/robustness/run_robustness.py --mechanism missing
  python scripts/robustness/run_robustness.py --pipeline a --mechanism outlier
"""
import sys
import argparse
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import average_precision_score

from config import (BANK_RAW, RETAIL_RAW, DATA_PROC, RESULTS_DIR, SEED,
                    N_FOLDS, N_REPEATS, CLASSICAL_PIPELINES, AI_PIPELINES)
from scripts.utils import get_logger, ensure_dirs
from scripts.classical.missing_values import handle_missing
from scripts.classical.encoder import one_hot_encode
from scripts.classical.scaler import scale_features
from scripts.classical.pipeline_a import load_retail_temporal
from scripts.robustness.inject_missing import inject_mcar, inject_mar, inject_mnar
from scripts.robustness.inject_outliers import inject_outliers
from scripts.robustness.inject_label_noise import inject_label_noise

logger = get_logger("run_robustness")
ROB_DIR = RESULTS_DIR / "robustness"
ensure_dirs(ROB_DIR)

ALL_PIPELINES = [p.lower() for p in CLASSICAL_PIPELINES + AI_PIPELINES]

# Sweep parameters
MISSING_RATES  = [0.05, 0.10, 0.20, 0.30, 0.50]
OUTLIER_RATES  = [0.01, 0.05, 0.10, 0.20]
NOISE_RATES    = [0.05, 0.10, 0.20, 0.30]

MECHANISMS = ["mcar", "mar", "mnar", "outlier", "label_noise"]


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _load_bank():
    """Load Bank Marketing — duration dropped (post-hoc leakage)."""
    df = pd.read_csv(BANK_RAW, sep=";")
    df = df.rename(columns={"y": "target"})
    df["target"] = (df["target"] == "yes").astype(int)
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])
    X = df.drop(columns=["target"])
    y = df["target"].values
    return X, y


def _load_retail():
    """Load Online Retail with temporal split (leakage-free churn prediction)."""
    rfm = load_retail_temporal()
    X = rfm.drop(columns=["target_bin"])
    y = rfm["target_bin"].values
    return X, y


LOADERS = {"bank": _load_bank, "retail": _load_retail}


# ---------------------------------------------------------------------------
# Pipeline-specific preprocessing (fit on train, transform val)
# ---------------------------------------------------------------------------

def _register_model_classes_for_unpickling():
    """
    Make the AI model classes importable under ``__main__``.

    pipeline_d/e/f.py were executed as scripts when they saved their models with
    joblib, so the pickles reference ``__main__.DenoisingAutoencoder`` etc.
    Loading them from a different entry point raises

        AttributeError: Can't get attribute 'DenoisingAutoencoder' on
        <module '__main__' from '.../run_robustness.py'>

    Re-binding the classes onto the current ``__main__`` module lets pickle
    resolve them. Without this the encoder load fails in every fold -- which is
    exactly why the previous version silently fell back to a generic
    StandardScaler and reported identical numbers for D, E and F.
    """
    import __main__
    try:
        from scripts.ai.pipeline_d import DenoisingAutoencoder
        from scripts.ai.pipeline_e import VariationalAutoencoder
        from scripts.ai.pipeline_f import (FTTransformer, NumericalEmbedding,
                                           CategoricalEmbedding)
    except ImportError as e:          # torch missing -> AI pipelines unusable
        logger.warning(f"Could not import AI model classes: {e}")
        return

    for cls in (DenoisingAutoencoder, VariationalAutoencoder, FTTransformer,
                NumericalEmbedding, CategoricalEmbedding):
        if not hasattr(__main__, cls.__name__):
            setattr(__main__, cls.__name__, cls)


_register_model_classes_for_unpickling()


_COL_ORDER_CACHE: dict = {}


def _encoder_column_order(dataset: str, num_cols: list, cat_cols: list) -> list:
    """
    Column order the AI encoders were trained on.

    pipeline_d/e/f build their input as `X.values` of the imputed frame, which
    keeps the ORIGINAL column order of the raw dataset -- not num_cols followed
    by cat_cols. Reproducing that order is essential: feeding the encoder the
    right number of columns in the wrong order produces no error at all, just
    silently meaningless representations.

    The raw frame is loaded once per dataset and cached.
    """
    if dataset not in _COL_ORDER_CACHE:
        X_raw, _ = LOADERS[dataset]()
        X_imp, _, _ = handle_missing(X_raw.copy(), fit=True)
        _COL_ORDER_CACHE[dataset] = list(X_imp.columns)

    ref = _COL_ORDER_CACHE[dataset]
    expected = set(num_cols) | set(cat_cols)
    order = [c for c in ref if c in expected]
    # Safety net: anything the reference order does not know about
    order += [c for c in list(num_cols) + list(cat_cols) if c not in order]
    return order


def _preprocess_pipeline(X_tr: pd.DataFrame, X_va: pd.DataFrame,
                          pipeline: str, dataset: str):
    """
    Apply the actual pipeline preprocessing to a train/val split.

    Fitting is done on X_tr only; X_va is transformed without re-fitting.
    AI pipelines (d, e, f) use their pre-trained encoder from data/processed/.

    Returns
    -------
    X_tr_arr, X_va_arr : numpy float32 arrays
    """
    X_tr = X_tr.copy()
    X_va = X_va.copy()

    # ------------------------------------------------------------------ #
    # Classical Pipelines A, B, C                                         #
    # ------------------------------------------------------------------ #

    if pipeline in ("a", "b_pca90", "b_pca20", "c"):

        # Pipeline C bank: row-wise domain features (no leakage)
        if pipeline == "c" and dataset == "bank":
            from scripts.classical.pipeline_c import add_bank_domain_features
            X_tr = add_bank_domain_features(X_tr)
            X_va = add_bank_domain_features(X_va)

        # 1. Imputation
        X_tr, num_imp, cat_imp = handle_missing(X_tr, fit=True)
        # Align val to train columns (corruption may have caused >50% drop in train)
        X_va = X_va[[c for c in X_tr.columns if c in X_va.columns]]
        X_va, _, _ = handle_missing(X_va, fit=False,
                                    num_imputer=num_imp, cat_imputer=cat_imp)

        # Pipeline C: interaction terms
        if pipeline == "c":
            from scripts.classical.interaction_terms import add_interaction_terms
            X_tr, poly, num_cols_poly = add_interaction_terms(X_tr, fit=True)
            X_va, _, _ = add_interaction_terms(X_va, fit=False,
                                               poly=poly, num_cols=num_cols_poly)

        # 2. One-Hot Encoding
        X_tr, train_cols = one_hot_encode(X_tr, fit=True)
        X_va, _ = one_hot_encode(X_va, fit=False, train_columns=train_cols)

        # Pipeline B: Winsorize + BoxCox
        if pipeline in ("b_pca90", "b_pca20"):
            from scripts.classical.outlier_handler import winsorize
            from scripts.classical.boxcox import apply_boxcox
            X_tr, bounds = winsorize(X_tr, fit=True, k=1.5)
            X_va, _ = winsorize(X_va, fit=False, bounds=bounds)
            X_tr, lambdas = apply_boxcox(X_tr, fit=True)
            X_va, _ = apply_boxcox(X_va, fit=False, lambdas=lambdas)

        # 3. StandardScaler
        X_tr, scaler = scale_features(X_tr, fit=True, method="standard")
        X_va, _ = scale_features(X_va, fit=False, scaler=scaler)

        # Pipeline B: PCA
        if pipeline in ("b_pca90", "b_pca20"):
            variance = 0.90 if pipeline == "b_pca90" else 0.20
            pca = PCA(n_components=variance, random_state=SEED)
            X_tr_arr = pca.fit_transform(X_tr.values)
            X_va_arr = pca.transform(X_va.values)
            return X_tr_arr.astype(np.float32), X_va_arr.astype(np.float32)

        return (X_tr.values.astype(np.float32),
                X_va.values.astype(np.float32))

    # ------------------------------------------------------------------ #
    # AI Pipelines D, E, F — frozen pre-trained encoder from main run     #
    # ------------------------------------------------------------------ #
    #
    # Methodological rationale:
    #   The encoder weights are fixed (from the main experiment).
    #   We apply the ORIGINAL saved preprocessing (scaler + LabelEncoders)
    #   to both corrupted train and clean val, then encode through the
    #   frozen model.  This tests representation robustness at inference.
    #   We still refit imputation on corrupted train (NaN stats change).
    #
    enc_dir = DATA_PROC / dataset / pipeline

    # 1. Imputation — fresh fit on corrupted train
    X_tr, num_imp, cat_imp = handle_missing(X_tr, fit=True)
    # Align val to train columns (corruption may have caused >50% drop in train)
    X_va = X_va[[c for c in X_tr.columns if c in X_va.columns]]
    X_va, _, _ = handle_missing(X_va, fit=False,
                                num_imputer=num_imp, cat_imputer=cat_imp)

    import json as _json

    # 2. Load saved preprocessing artefacts from main experiment
    saved_scaler   = joblib.load(enc_dir / "scaler.pkl")
    saved_encoders = joblib.load(enc_dir / "encoders.pkl")  # dict col -> LabelEncoder

    with open(enc_dir / "meta.json") as _fh:
        _meta = _json.load(_fh)
    cat_cols = _meta.get("cat_cols", [])
    num_cols = _meta.get("num_cols", [])

    # 3. Re-align to the encoder's expected schema.
    #    Corruption can push a column above the >50%-missing drop threshold in
    #    handle_missing(), which previously left the encoder with the wrong
    #    (sometimes zero) number of inputs. The forward pass then raised a shape
    #    error that was silently swallowed, and ALL of D/E/F degenerated to the
    #    same generic StandardScaler baseline.
    #    A dropped column is re-inserted as NaN and imputed below -- that is the
    #    correct semantics: the information is gone, but the input dimension of a
    #    deployed encoder is fixed and cannot change at inference time.
    #    IMPORTANT: pipeline_d/e/f encode `X.values` of the frame in its ORIGINAL
    #    column order, not in num_cols+cat_cols order. Re-ordering here would
    #    feed the encoder semantically scrambled inputs, so the reference order
    #    is taken from the raw loader.
    expected = _encoder_column_order(dataset, num_cols, cat_cols)
    dropped = [c for c in expected if c not in X_tr.columns]
    if dropped:
        logger.info(f"  {pipeline}/{dataset}: {len(dropped)} column(s) lost to "
                    f"corruption, re-inserted as missing: {dropped}")
    # .copy() -- X_va is a slice from the column alignment above, in-place
    # column insertion on a slice does not reliably propagate
    X_tr = X_tr.copy()
    X_va = X_va.copy()
    for col in expected:
        if col not in X_tr.columns:
            X_tr[col] = np.nan
        if col not in X_va.columns:
            X_va[col] = np.nan

    X_tr = X_tr[expected].copy()
    X_va = X_va[expected].copy()

    # Re-impute re-inserted columns. Numeric columns must stay numeric and
    # categorical columns must stay string-like, otherwise the LabelEncoder
    # mapping and the scaler downstream receive the wrong dtype.
    for col in num_cols:
        s = pd.to_numeric(X_tr[col], errors="coerce")
        med = s.median()
        fill = 0.0 if pd.isna(med) else float(med)
        X_tr[col] = s.fillna(fill).astype(float)
        X_va[col] = pd.to_numeric(X_va[col], errors="coerce").fillna(fill).astype(float)
    for col in cat_cols:
        s = X_tr[col].astype(object)
        mode = s.dropna().mode()
        fill = mode.iloc[0] if len(mode) else "missing"
        X_tr[col] = s.fillna(fill).astype(str)
        X_va[col] = X_va[col].astype(object).fillna(fill).astype(str)

    # 4. Apply saved LabelEncoders to categorical columns
    for col in cat_cols:
        _le = saved_encoders.get(col)
        if _le is None:
            continue
        _mapping = {c: i for i, c in enumerate(_le.classes_)}
        X_tr[col] = X_tr[col].astype(str).map(lambda v, m=_mapping: m.get(v, 0))
        X_va[col] = X_va[col].astype(str).map(lambda v, m=_mapping: m.get(v, 0))

    # 5. Apply saved StandardScaler to numeric columns
    if num_cols:
        X_tr[num_cols] = saved_scaler.transform(X_tr[num_cols])
        X_va[num_cols] = saved_scaler.transform(X_va[num_cols])

    X_tr_arr = X_tr.values.astype(np.float32)
    X_va_arr = X_va.values.astype(np.float32)

    exp_dim = _meta.get("input_dim")
    if exp_dim is not None and X_tr_arr.shape[1] != exp_dim:
        raise RuntimeError(
            f"{pipeline}/{dataset}: encoder expects {exp_dim} inputs but got "
            f"{X_tr_arr.shape[1]} (dropped: {dropped}). Refusing to silently "
            f"substitute a different pipeline.")

    # 6. Encode through the frozen model
    if pipeline == "d":
        from scripts.ai.pipeline_d import _encode as dae_encode
        model = joblib.load(enc_dir / "dae_model.pkl")
        return dae_encode(model, X_tr_arr), dae_encode(model, X_va_arr)

    if pipeline == "e":
        from scripts.ai.pipeline_e import _encode as vae_encode
        model = joblib.load(enc_dir / "vae_model.pkl")
        return vae_encode(model, X_tr_arr), vae_encode(model, X_va_arr)

    if pipeline == "f":
        from scripts.ai.pipeline_f import _encode as ftt_encode
        model = joblib.load(enc_dir / "ftt_model.pkl")
        X_num_tr = X_tr[num_cols].values.astype(np.float32) if num_cols else None
        X_num_va = X_va[num_cols].values.astype(np.float32) if num_cols else None
        X_cat_tr = X_tr[cat_cols].values.astype(np.int64)   if cat_cols else None
        X_cat_va = X_va[cat_cols].values.astype(np.int64)   if cat_cols else None
        return ftt_encode(model, X_num_tr, X_cat_tr), ftt_encode(model, X_num_va, X_cat_va)

    raise ValueError(f"Unknown AI pipeline: {pipeline}")


# ---------------------------------------------------------------------------
# CV with corruption applied INSIDE each fold
# ---------------------------------------------------------------------------

def _cv_with_corruption(X_raw: pd.DataFrame, y: np.ndarray,
                         corrupt_fn, pipeline: str, dataset: str,
                         seed: int = SEED) -> dict:
    """
    Run 5x3 RSKF where:
      - corruption is applied to training fold only
      - actual pipeline preprocessing is applied (fit on corrupted train)
      - validation fold is always clean
      - downstream model: LR (fast proxy)

    Parameters
    ----------
    X_raw     : Clean feature DataFrame
    y         : Clean labels
    corrupt_fn: Function(X_train_df, y_train) -> (X_corrupted_df, y_corrupted)
    pipeline  : Pipeline key (a / b_pca90 / ... / f)
    dataset   : 'bank' or 'retail'
    """
    rskf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS,
                                   random_state=seed)
    scores = []

    for tr, va in rskf.split(X_raw, y):
        X_tr_df = X_raw.iloc[tr].copy()
        y_tr    = y[tr].copy()
        X_va_df = X_raw.iloc[va].copy()
        y_va    = y[va]

        # Apply corruption to train fold only
        X_tr_corrupted, y_tr_corrupted = corrupt_fn(X_tr_df, y_tr)

        # Apply actual pipeline preprocessing
        try:
            X_tr_proc, X_va_proc = _preprocess_pipeline(
                X_tr_corrupted, X_va_df, pipeline, dataset
            )
        except Exception as e:
            if not scores:          # first failure: show the full traceback
                import traceback
                logger.error(f"Preprocessing failed in fold ({pipeline}/{dataset}):\n"
                             f"{traceback.format_exc()}")
            else:
                logger.warning(f"Preprocessing failed in fold: {e} — skipping fold.")
            continue

        if X_tr_proc.shape[1] == 0:
            logger.warning("0 features after preprocessing (all columns dropped) — skipping fold.")
            continue

        model = LogisticRegression(max_iter=1000, random_state=seed,
                                   solver="saga", n_jobs=-1)
        model.fit(X_tr_proc, y_tr_corrupted)
        proba = model.predict_proba(X_va_proc)[:, 1]
        scores.append(average_precision_score(y_va, proba))

    if not scores:
        logger.warning("All folds skipped — returning NaN for this configuration.")
        return {"mean_PR_AUC": float("nan"), "std_PR_AUC": float("nan"), "n_folds": 0}

    return {
        "mean_PR_AUC": round(float(np.mean(scores)), 5),
        "std_PR_AUC":  round(float(np.std(scores)), 5),
        "n_folds":     len(scores),
    }


# ---------------------------------------------------------------------------
# Robustness sweep functions
# ---------------------------------------------------------------------------

def sweep_missing(X: pd.DataFrame, y: np.ndarray,
                  mechanism: str, dataset: str, pipeline: str) -> list:
    """Sweep missing value injection at multiple rates."""
    records = []
    for rate in MISSING_RATES:
        logger.info(f"  {mechanism.upper()} rate={rate:.2f}")

        if mechanism == "mcar":
            def corrupt_fn(X_tr, y_tr, r=rate):
                return inject_mcar(X_tr, missing_rate=r, seed=SEED), y_tr
        elif mechanism == "mar":
            def corrupt_fn(X_tr, y_tr, r=rate):
                return inject_mar(X_tr, missing_rate=r, seed=SEED), y_tr
        else:  # mnar
            def corrupt_fn(X_tr, y_tr, r=rate):
                return inject_mnar(X_tr, missing_rate=r, seed=SEED), y_tr

        t0 = time.time()
        res = _cv_with_corruption(X, y, corrupt_fn, pipeline, dataset)
        elapsed = round(time.time() - t0, 1)

        records.append({
            "dataset": dataset, "pipeline": pipeline,
            "mechanism": mechanism, "rate": rate,
            **res, "runtime_s": elapsed,
        })
    return records


def sweep_outliers(X: pd.DataFrame, y: np.ndarray,
                   dataset: str, pipeline: str) -> list:
    """Sweep outlier injection at multiple rates."""
    records = []
    for rate in OUTLIER_RATES:
        logger.info(f"  OUTLIER rate={rate:.2f}")

        def corrupt_fn(X_tr, y_tr, r=rate):
            return inject_outliers(X_tr, outlier_rate=r, seed=SEED), y_tr

        t0 = time.time()
        res = _cv_with_corruption(X, y, corrupt_fn, pipeline, dataset)
        elapsed = round(time.time() - t0, 1)

        records.append({
            "dataset": dataset, "pipeline": pipeline,
            "mechanism": "outlier", "rate": rate,
            **res, "runtime_s": elapsed,
        })
    return records


def sweep_label_noise(X: pd.DataFrame, y: np.ndarray,
                      dataset: str, pipeline: str) -> list:
    """Sweep label noise injection at multiple rates."""
    records = []
    for rate in NOISE_RATES:
        logger.info(f"  LABEL_NOISE rate={rate:.2f}")

        def corrupt_fn(X_tr, y_tr, r=rate):
            return X_tr, inject_label_noise(y_tr, noise_rate=r,
                                            mode="symmetric", seed=SEED)

        t0 = time.time()
        res = _cv_with_corruption(X, y, corrupt_fn, pipeline, dataset)
        elapsed = round(time.time() - t0, 1)

        records.append({
            "dataset": dataset, "pipeline": pipeline,
            "mechanism": "label_noise", "rate": rate,
            **res, "runtime_s": elapsed,
        })
    return records


# ---------------------------------------------------------------------------
# Baseline (no corruption) for reference
# ---------------------------------------------------------------------------

def _baseline(X: pd.DataFrame, y: np.ndarray,
               dataset: str, pipeline: str) -> dict:
    """PR-AUC without any corruption -- reference point for delta plots."""
    def no_corrupt(X_tr, y_tr):
        return X_tr, y_tr

    res = _cv_with_corruption(X, y, no_corrupt, pipeline, dataset)
    return {
        "dataset": dataset, "pipeline": pipeline,
        "mechanism": "baseline", "rate": 0.0,
        **res, "runtime_s": 0.0,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def _done_keys(df: pd.DataFrame) -> set:
    """Return set of (mechanism, rate) tuples already present in a checkpoint df."""
    return {(row["mechanism"], row["rate"]) for _, row in df.iterrows()}


def _append_row(out: Path, record: dict, existing_df: pd.DataFrame) -> pd.DataFrame:
    """Append one record to the checkpoint CSV and return updated df."""
    new_row = pd.DataFrame([record])
    updated = pd.concat([existing_df, new_row], ignore_index=True)
    updated.to_csv(out, index=False)
    return updated


def run_robustness(dataset: str, pipeline: str,
                   mechanisms: list = None) -> pd.DataFrame:
    """
    Full robustness sweep for one dataset x pipeline.
    Checkpoints after every rate: resumes from where it left off on restart.
    """
    if mechanisms is None:
        mechanisms = MECHANISMS

    out = ROB_DIR / f"{dataset}_{pipeline}_robustness.csv"

    # Load existing checkpoint (partial or complete)
    if out.exists():
        existing = pd.read_csv(out)
        done = _done_keys(existing)
        logger.info(f"Resuming {dataset} x {pipeline.upper()} — "
                    f"{len(done)} rows already done.")
    else:
        existing = pd.DataFrame()
        done = set()

    logger.info(f"\n{'='*55}")
    logger.info(f"Robustness: {dataset} x pipeline {pipeline.upper()}")
    logger.info(f"Mechanisms: {mechanisms}")
    logger.info(f"{'='*55}")

    X, y = LOADERS[dataset]()

    # Baseline
    if ("baseline", 0.0) not in done:
        record = _baseline(X, y, dataset, pipeline)
        existing = _append_row(out, record, existing)
        logger.info(f"  baseline done → PR-AUC {record['mean_PR_AUC']:.4f}")

    for mech in mechanisms:
        if mech in ("mcar", "mar", "mnar"):
            rates = MISSING_RATES
        elif mech == "outlier":
            rates = OUTLIER_RATES
        elif mech == "label_noise":
            rates = NOISE_RATES
        else:
            continue

        for rate in rates:
            if (mech, rate) in done:
                logger.info(f"  Skipping {mech} rate={rate} — already done.")
                continue

            logger.info(f"  {mech.upper()} rate={rate:.2f}")

            if mech == "mcar":
                def corrupt_fn(X_tr, y_tr, r=rate):
                    return inject_mcar(X_tr, missing_rate=r, seed=SEED), y_tr
            elif mech == "mar":
                def corrupt_fn(X_tr, y_tr, r=rate):
                    return inject_mar(X_tr, missing_rate=r, seed=SEED), y_tr
            elif mech == "mnar":
                def corrupt_fn(X_tr, y_tr, r=rate):
                    return inject_mnar(X_tr, missing_rate=r, seed=SEED), y_tr
            elif mech == "outlier":
                def corrupt_fn(X_tr, y_tr, r=rate):
                    return inject_outliers(X_tr, outlier_rate=r, seed=SEED), y_tr
            elif mech == "label_noise":
                def corrupt_fn(X_tr, y_tr, r=rate):
                    return X_tr, inject_label_noise(y_tr, noise_rate=r,
                                                    mode="symmetric", seed=SEED)

            t0 = time.time()
            res = _cv_with_corruption(X, y, corrupt_fn, pipeline, dataset)
            elapsed = round(time.time() - t0, 1)

            record = {
                "dataset": dataset, "pipeline": pipeline,
                "mechanism": mech, "rate": rate,
                **res, "runtime_s": elapsed,
            }
            existing = _append_row(out, record, existing)
            logger.info(f"    → PR-AUC {res['mean_PR_AUC']:.4f}  ({elapsed}s)  [saved]")

    logger.info(f"Pipeline {pipeline.upper()} complete: {out.name}")
    return existing


def run_all(datasets: list = None, pipelines: list = None,
            mechanisms: list = None) -> pd.DataFrame:
    """Run full robustness sweep for all dataset x pipeline combinations."""
    if datasets  is None: datasets  = ["bank", "retail"]
    if pipelines is None: pipelines = ALL_PIPELINES

    # Determine expected number of rows per pipeline (for completeness check)
    mechs = mechanisms or MECHANISMS
    expected_rates = sum([
        len(MISSING_RATES) if m in ("mcar", "mar", "mnar") else
        len(OUTLIER_RATES) if m == "outlier" else
        len(NOISE_RATES)   if m == "label_noise" else 0
        for m in mechs
    ]) + 1  # +1 for baseline

    for ds in datasets:
        for pl in pipelines:
            out_file = ROB_DIR / f"{ds}_{pl}_robustness.csv"
            if out_file.exists():
                existing = pd.read_csv(out_file)
                if len(existing) >= expected_rates:
                    logger.info(f"Skipping {ds} x {pl} — already complete "
                                f"({len(existing)} rows).")
                    continue
                logger.info(f"Resuming {ds} x {pl} — "
                            f"{len(existing)}/{expected_rates} rows done.")
            run_robustness(ds, pl, mechanisms)

    # Always rebuild the combined file from ALL per-pipeline CSVs on disk,
    # not only from the ones executed in this run. Otherwise a partial run
    # (e.g. --pipeline b_pca20) would silently truncate robustness_all.csv.
    files = sorted(f for f in ROB_DIR.glob("*_robustness.csv")
                   if f.name != "robustness_all.csv")
    if not files:
        logger.warning("No per-pipeline robustness CSVs found.")
        return pd.DataFrame()

    combined = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out = ROB_DIR / "robustness_all.csv"
    combined.to_csv(out, index=False)
    logger.info(f"\nCombined robustness results saved: {out} "
                f"({len(files)} files, {len(combined)} rows).")
    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robustness sweep")
    parser.add_argument("--dataset",   default=None,
                        choices=["bank", "retail"])
    parser.add_argument("--pipeline",  default=None,
                        choices=ALL_PIPELINES)
    parser.add_argument("--mechanism", default=None,
                        choices=MECHANISMS)
    args = parser.parse_args()

    datasets   = [args.dataset]   if args.dataset   else None
    pipelines  = [args.pipeline]  if args.pipeline  else None
    mechanisms = [args.mechanism] if args.mechanism else None

    run_all(datasets=datasets, pipelines=pipelines, mechanisms=mechanisms)
