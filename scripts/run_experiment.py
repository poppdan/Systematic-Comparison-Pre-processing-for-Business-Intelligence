"""
run_experiment.py -- Main script for all experiments.

Runs all pipelines (A-F) with all models (LR, RF, XGB, LGBM, MLP)
on both datasets and saves results.

Pipeline variants:
  a       : Baseline (Imputation -> OHE -> StandardScaler)
  b_pca90 : Statistics-extended + PCA (90% variance threshold)
  b_pca20 : Statistics-extended + PCA (20% variance threshold)
  c       : Feature Engineering (domain/RFM features + interactions + Pipeline A)
  d       : Denoising Autoencoder (DAE)
  e       : Variational Autoencoder (VAE)
  f       : Feature Tokenizer Transformer (FT-Transformer)

Dataset / pipeline matrix:
  bank   : a, b_pca90, b_pca20, c, d, e, f
  retail : a, b_pca90, b_pca20, c, d, e, f

Usage:
  python scripts/run_experiment.py                           # all
  python scripts/run_experiment.py --dataset bank            # bank only
  python scripts/run_experiment.py --pipeline a              # pipeline A only
  python scripts/run_experiment.py --pipeline d              # DAE pipeline
  python scripts/run_experiment.py --model lr                # Logistic Regression only
  python scripts/run_experiment.py --model xgb               # XGBoost only
  python scripts/run_experiment.py --no-hpo                  # without Optuna (defaults)
  python scripts/run_experiment.py --skip-preprocessing      # data already exists
"""
import sys
import argparse
import time
import tracemalloc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import (RESULTS_DIR, DATA_PROC, SEED,
                    CLASSICAL_PIPELINES, AI_PIPELINES, DOWNSTREAM_MODELS, DATASETS)
from scripts.utils import get_logger, ensure_dirs, save_results

logger = get_logger("run_experiment")
ensure_dirs(RESULTS_DIR)

ALL_PIPELINES = [p.lower() for p in CLASSICAL_PIPELINES + AI_PIPELINES]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_processed(dataset: str, pipeline: str):
    """
    Load preprocessed data from data/processed/{dataset}/{pipeline}/.

    Classical pipelines (a, b_pca90, b_pca20, c) use parquet.
    AI pipelines (d, e, f) use CSV (saved by pipeline_d/e/f.py).
    """
    base = DATA_PROC / dataset / pipeline

    # AI pipelines save train.csv / test.csv with a "target" column
    train_csv = base / "train.csv"
    test_csv  = base / "test.csv"
    if train_csv.exists():
        df_tr = pd.read_csv(train_csv)
        df_te = pd.read_csv(test_csv)
        X_train = df_tr.drop(columns=["target"])
        X_test  = df_te.drop(columns=["target"])
        y_train = df_tr["target"]
        y_test  = df_te["target"]
        return X_train, X_test, y_train, y_test

    # Classical pipelines save parquet
    X_train = pd.read_parquet(base / "X_train.parquet")
    X_test  = pd.read_parquet(base / "X_test.parquet")
    y_train = pd.read_parquet(base / "y_train.parquet").squeeze()
    y_test  = pd.read_parquet(base / "y_test.parquet").squeeze()
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Leakage-free CV for AI pipelines (D/E/F)
#
# The saved train.csv of pipelines D/E/F contains encoder OUTPUT (embeddings).
# That encoder was fitted on the entire training split -- for the supervised
# FT-Transformer (F) even using the labels. Running CV directly on those
# embeddings means every validation fold was already seen by the encoder,
# which inflates the score badly (bank F: 0.86 CV vs 0.36 test).
#
# Fix: run CV on the RAW features and re-fit the encoder inside every fold.
# ---------------------------------------------------------------------------

AI_ENCODER_PIPELINES = {"d", "e", "f"}

# Compute budget for the per-fold encoder training.
# The encoder is now re-fitted 15x (once per CV split) instead of once, so the
# per-run epoch cap is lowered. Early stopping is on a held-out validation
# split, so training normally terminates well before this cap anyway.
CV_ENCODER_EPOCHS   = 60
CV_ENCODER_PATIENCE = 8


def load_raw_for_ai(dataset: str):
    """
    Reproduce exactly the train/test split used by pipeline_d/e/f.py,
    but return the RAW (un-encoded) feature frames.
    """
    from sklearn.model_selection import train_test_split
    from config import BANK_RAW, TEST_SIZE
    from scripts.classical.missing_values import handle_missing

    if dataset == "bank":
        df = pd.read_csv(BANK_RAW, sep=";")
        df = df.rename(columns={"y": "target"})
        df["target"] = (df["target"] == "yes").astype(int)
        if "duration" in df.columns:
            df = df.drop(columns=["duration"])
    else:
        from scripts.classical.pipeline_a import load_retail_temporal
        df = load_retail_temporal().rename(columns={"target_bin": "target"})

    X = df.drop(columns=["target"]).copy()
    y = df["target"].values
    X, _, _ = handle_missing(X)

    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx, test_size=TEST_SIZE,
                                      stratify=y, random_state=SEED)
    return (X.iloc[idx_tr].reset_index(drop=True),
            X.iloc[idx_te].reset_index(drop=True),
            pd.Series(y[idx_tr]), pd.Series(y[idx_te]))


def make_fold_encoder_fn(dataset: str, pipeline: str, dtypes: pd.Series):
    """
    Return a pipeline_fn(X_tr, y_tr, X_val) -> (Z_tr, y_tr, Z_val) that trains
    the pipeline's encoder on the fold's TRAINING data only and encodes both
    splits with it. Reuses the leakage-free implementations from ablation.py.

    `dtypes` is required because cv_runner round-trips the frame through
    `.values`, which turns a mixed numeric/categorical frame into all-object.
    The encoder prep relies on select_dtypes(), so the dtypes are restored here.

    Fold encodings are cached on disk. The CV splits are deterministic (fixed
    seed) and the encoder does not depend on the downstream model, so all five
    models reuse the same 15 encodings instead of retraining 75 times.
    """
    from scripts.ablation import _stage2_d, _stage2_e, _stage2_f

    cache_dir = DATA_PROC / dataset / pipeline / "fold_cache"
    ensure_dirs(cache_dir)
    counter = {"fold": 0}

    def _restore(X):
        X = pd.DataFrame(X).copy()
        X.columns = list(dtypes.index)
        for col, dt in dtypes.items():
            try:
                X[col] = X[col].astype(dt)
            except (ValueError, TypeError):
                X[col] = pd.to_numeric(X[col], errors="coerce")
        return X

    def _fn(X_tr, y_tr, X_val):
        y_arr = y_tr.values if hasattr(y_tr, "values") else np.asarray(y_tr)
        k = counter["fold"]; counter["fold"] += 1
        cache = cache_dir / f"fold{k:02d}.npz"

        if cache.exists():
            d = np.load(cache)
            if len(d["Z_tr"]) == len(y_arr) and len(d["Z_val"]) == len(X_val):
                return (pd.DataFrame(d["Z_tr"]), pd.Series(y_arr),
                        pd.DataFrame(d["Z_val"]))
            logger.warning(f"  fold cache {cache.name} size mismatch -- recomputing")

        X_tr_df, X_val_df = _restore(X_tr), _restore(X_val)
        t0 = time.perf_counter()
        kw = {"epochs": CV_ENCODER_EPOCHS, "patience": CV_ENCODER_PATIENCE}
        if pipeline == "d":
            Z_tr, Z_val = _stage2_d(X_tr_df, X_val_df, **kw)
        elif pipeline == "e":
            Z_tr, Z_val = _stage2_e(X_tr_df, X_val_df, **kw)
        else:  # f -- supervised, needs the fold's training labels
            Z_tr, Z_val = _stage2_f(X_tr_df, X_val_df, y_tr=y_arr, **kw)

        np.savez_compressed(cache, Z_tr=Z_tr, Z_val=Z_val)
        logger.info(f"  fold {k+1:02d}/15 encoder trained + cached "
                    f"({time.perf_counter()-t0:.0f}s)")
        return (pd.DataFrame(Z_tr), pd.Series(y_arr), pd.DataFrame(Z_val))

    return _fn


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def get_model(model_name: str, input_dim: int = None, use_hpo: bool = True,
              X_train=None, y_train=None):
    """Return a fitted or unfitted model (with or without HPO)."""

    if model_name == "lr":
        from scripts.eval.lr_model import build_model, tune_lr
        if use_hpo:
            params = tune_lr(X_train, y_train)
            return build_model(params)
        return build_model()

    elif model_name == "rf":
        from scripts.eval.rf_model import build_model, tune_rf
        if use_hpo:
            params = tune_rf(X_train, y_train)
            return build_model(params)
        return build_model()

    elif model_name == "xgb":
        from scripts.eval.xgb_model import build_model, tune_xgb
        if use_hpo:
            params = tune_xgb(X_train, y_train)
            return build_model(params)
        return build_model()

    elif model_name == "lgbm":
        from scripts.eval.lgbm_model import build_model, tune_lgbm
        if use_hpo:
            params = tune_lgbm(X_train, y_train)
            return build_model(params)
        return build_model()

    elif model_name == "mlp":
        from scripts.eval.mlp_model import MLPClassifier, tune_mlp
        if use_hpo:
            params = tune_mlp(X_train, y_train, input_dim=input_dim)
            return MLPClassifier(**params)
        return MLPClassifier()

    raise ValueError(f"Unknown model: {model_name}. "
                     f"Choose from: {DOWNSTREAM_MODELS}")


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def _measure_preprocessing_cost(dataset: str, pipeline: str) -> dict:
    """
    Measure runtime (seconds) and peak memory (MB) of the preprocessing step
    for a given pipeline. Runs the pipeline and captures resource usage.
    """
    import importlib

    module_map = {
        "a":      ("scripts.classical.pipeline_a", dataset),
        "b_pca90": ("scripts.classical.pipeline_b", dataset),
        "b_pca20": ("scripts.classical.pipeline_b", dataset),
        "c":      ("scripts.classical.pipeline_c", dataset),
        "d":      ("scripts.ai.pipeline_d",        dataset),
        "e":      ("scripts.ai.pipeline_e",        dataset),
        "f":      ("scripts.ai.pipeline_f",        dataset),
    }

    if pipeline not in module_map:
        return {"preprocessing_s": None, "preprocessing_mb": None}

    module_name, ds = module_map[pipeline]

    try:
        mod = importlib.import_module(module_name)
        fn  = mod.run_bank if ds == "bank" else mod.run_retail

        kwargs = {}
        if pipeline == "b_pca90":
            kwargs = {"pca_variance": 0.90}
        elif pipeline == "b_pca20":
            kwargs = {"pca_variance": 0.20}

        tracemalloc.start()
        t0 = time.perf_counter()
        fn(**kwargs)
        elapsed = round(time.perf_counter() - t0, 2)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = round(peak / 1024 / 1024, 2)

        return {"preprocessing_s": elapsed, "preprocessing_mb": peak_mb}

    except Exception as e:
        logger.warning(f"Could not measure preprocessing cost for {pipeline}: {e}")
        tracemalloc.stop()
        return {"preprocessing_s": None, "preprocessing_mb": None}


def run_pipeline_experiment(dataset: str, pipeline: str, model_name: str,
                             use_hpo: bool = True):
    """Run one dataset x pipeline x model experiment."""
    label    = f"{dataset}_{pipeline}_{model_name}"
    out_path = RESULTS_DIR / f"{label}_cv.json"

    if out_path.exists():
        logger.info(f"Skipping {label} (already exists)")
        return

    logger.info(f"\n{'='*55}\n{label.upper()}\n{'='*55}")

    try:
        X_train, X_test, y_train, y_test = load_processed(dataset, pipeline)
    except FileNotFoundError:
        logger.error(f"Data not found: {DATA_PROC / dataset / pipeline}")
        logger.error("  -> Run the corresponding pipeline script first!")
        return

    from scripts.eval.cv_runner import run_cv
    from scripts.eval.metrics import compute_metrics

    # -- CV with runtime + memory tracking ---------------------------------
    tracemalloc.start()
    t_cv_start = time.perf_counter()

    if pipeline in AI_ENCODER_PIPELINES:
        # Leakage-free: CV runs on RAW features, encoder is re-fitted per fold.
        # HPO still uses the frozen embeddings -- that only affects
        # hyperparameter selection, not the reported metric.
        logger.info(f"  {pipeline.upper()}: per-fold encoder training "
                    f"(leakage-free CV on raw features)")
        X_raw_tr, _, y_raw_tr, _ = load_raw_for_ai(dataset)

        model = get_model(model_name, input_dim=X_train.shape[1],
                          use_hpo=use_hpo,
                          X_train=X_train, y_train=y_train.values)
        cv_result = run_cv(model, X_raw_tr, y_raw_tr,
                           pipeline_fn=make_fold_encoder_fn(
                               dataset, pipeline, X_raw_tr.dtypes))
    else:
        model = get_model(model_name, input_dim=X_train.shape[1],
                          use_hpo=use_hpo,
                          X_train=X_train, y_train=y_train.values)
        cv_result = run_cv(model, X_train, y_train)

    cv_elapsed = round(time.perf_counter() - t_cv_start, 2)
    _, cv_peak  = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cv_peak_mb = round(cv_peak / 1024 / 1024, 2)

    # -- Final fit on full train set ----------------------------------------
    t_fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_elapsed = round(time.perf_counter() - t_fit_start, 2)

    y_proba_test = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test.values, y_proba_test)

    result = {
        "dataset":      dataset,
        "pipeline":     pipeline,
        "model":        model_name,
        "cv":           cv_result,
        "test_metrics": test_metrics,
        "runtime": {
            "cv_total_s":      cv_elapsed,
            "fit_s":           fit_elapsed,
            "cv_peak_mem_mb":  cv_peak_mb,
            "n_train_samples": int(len(X_train)),
            "n_features":      int(X_train.shape[1]),
        },
    }
    save_results(result, out_path)

    agg = cv_result["aggregated"]
    logger.info(
        f"Done: {label} | "
        f"CV PR-AUC: {agg['mean_PR_AUC']:.4f}+/-{agg['std_PR_AUC']:.4f} | "
        f"Test PR-AUC: {test_metrics['PR_AUC']:.4f} | "
        f"Lift@10%: {test_metrics['Lift_Top10']:.2f} | "
        f"EMV: {test_metrics['EMV']:.0f} | "
        f"CV time: {cv_elapsed:.1f}s | "
        f"Peak mem: {cv_peak_mb:.1f} MB"
    )
    return result


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run thesis experiments")
    parser.add_argument("--dataset",  default="all", choices=["all", "bank", "retail"])
    parser.add_argument("--pipeline", default="all")
    parser.add_argument("--model",    default="all")
    parser.add_argument("--no-hpo",   action="store_true", dest="no_hpo")
    parser.add_argument("--skip-preprocessing", action="store_true", dest="skip_pre")
    args = parser.parse_args()

    datasets  = DATASETS if args.dataset == "all" else [args.dataset]
    pipelines = (CLASSICAL_PIPELINES + [p.lower() for p in AI_PIPELINES]) \
                if args.pipeline == "all" else [args.pipeline]
    models    = DOWNSTREAM_MODELS if args.model == "all" else [args.model]
    use_hpo   = not args.no_hpo

    logger.info(f"Datasets: {datasets} | Pipelines: {pipelines} | Models: {models} | HPO: {use_hpo}")

    for dataset in datasets:
        for pipeline in pipelines:
            # Check if processed data already exists
            base = DATA_PROC / dataset / pipeline
            data_exists = (base / "train.csv").exists() or (base / "X_train.parquet").exists()

            if data_exists:
                logger.info(f"Processed data found for {dataset}/{pipeline} -- skipping preprocessing")
            elif args.skip_pre:
                logger.warning(f"--skip-preprocessing set but data missing for {dataset}/{pipeline} -- will fail!")
            else:
                logger.info(f"Processed data not found for {dataset}/{pipeline} -- running preprocessing")
                cost = _measure_preprocessing_cost(dataset, pipeline)
                logger.info(f"Preprocessing {dataset}/{pipeline}: "
                            f"{cost['preprocessing_s']}s / {cost['preprocessing_mb']}MB")

                # Re-check after preprocessing
                if not (base / "train.csv").exists() and not (base / "X_train.parquet").exists():
                    logger.error(f"Preprocessing failed for {dataset}/{pipeline} -- skipping all models")
                    continue

            for model_name in models:
                run_pipeline_experiment(dataset, pipeline, model_name, use_hpo=use_hpo)

    logger.info("All experiments finished.")


if __name__ == "__main__":
    main()