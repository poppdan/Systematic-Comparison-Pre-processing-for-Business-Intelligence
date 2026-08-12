"""
pipeline_b.py -- Statistics-extended classical pipeline (B).

Steps:
  1. Imputation      (Median / Most-Frequent)
  2. Encoding        (One-Hot)
  3. Outlier         (IQR winsorisation, k=1.5)
  4. Box-Cox         (positive, skewed features, |skewness| > 0.5)
  5. Scaling         (StandardScaler)
  6. PCA             (variance threshold: 0.90 or 0.20)

The pipeline is run TWICE per dataset:
  - b_pca90 : PCA keeps components explaining 90% of variance
  - b_pca20 : PCA keeps components explaining 20% of variance

Saves to:
  data/processed/{dataset}/b_pca90/
  data/processed/{dataset}/b_pca20/

Data leakage rule: ALL fit() calls on training data only.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import time
import tracemalloc

import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from config import DATA_PROC, TEST_SIZE, SEED
from scripts.utils import get_logger, ensure_dirs
from scripts.classical.pipeline_a import load_bank, load_retail_clean, load_retail_temporal
from scripts.classical.missing_values import handle_missing
from scripts.classical.outlier_handler import winsorize
from scripts.classical.boxcox import apply_boxcox
from scripts.classical.encoder import one_hot_encode
from scripts.classical.scaler import scale_features

logger = get_logger("pipeline_b")
ensure_dirs(DATA_PROC)


def _run_pipeline(X, y, pca_variance, dataset, save=True):
    """
    Core B pipeline shared by bank and retail.

    Order: Imputation -> OHE -> Outlier (IQR) -> Box-Cox -> StandardScaler -> PCA

    Parameters
    ----------
    X            : Feature DataFrame
    y            : Target Series
    pca_variance : Fraction of variance to retain in PCA (0.90 or 0.20)
    dataset      : 'bank' or 'retail' (used for save path)
    save         : Whether to persist results as parquet files
    """
    tracemalloc.start()
    t_start = time.perf_counter()

    n_features_in = X.shape[1]
    input_shape = [X.shape[0], X.shape[1]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    X_train["target_bin"] = y_train.values
    X_test["target_bin"]  = y_test.values

    # 1. Imputation (fit on train only)
    X_train, num_imp, cat_imp = handle_missing(X_train, fit=True)
    X_test,  _,       _       = handle_missing(X_test, fit=False,
                                               num_imputer=num_imp,
                                               cat_imputer=cat_imp)

    # 2. One-Hot Encoding (fit on train only)
    X_train, train_cols = one_hot_encode(X_train, fit=True)
    X_test,  _          = one_hot_encode(X_test, fit=False, train_columns=train_cols)

    # 3. Outlier treatment: IQR winsorisation, k=1.5 (fit on train only)
    X_train, bounds = winsorize(X_train, fit=True, k=1.5)
    X_test,  _      = winsorize(X_test, fit=False, bounds=bounds)

    # 4. Box-Cox transformation on positive, skewed features (|skewness| > 0.5)
    #    Lambdas fitted on train only, then applied to test
    X_train, lambdas = apply_boxcox(X_train, fit=True)
    X_test,  _       = apply_boxcox(X_test, fit=False, lambdas=lambdas)

    # 5. StandardScaler (fit on train only)
    X_train, scaler = scale_features(X_train, fit=True, method="standard")
    X_test,  _      = scale_features(X_test, fit=False, scaler=scaler)

    y_train = X_train.pop("target_bin")
    y_test  = X_test.pop("target_bin")

    # 6. PCA with the specified variance threshold (fit on train only)
    pca = PCA(n_components=pca_variance, random_state=SEED)
    X_train_pca = pd.DataFrame(
        pca.fit_transform(X_train),
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    )
    X_test_pca = pd.DataFrame(
        pca.transform(X_test),
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    )

    n_components = pca.n_components_
    explained = pca.explained_variance_ratio_.sum()
    logger.info(
        f"PCA (variance={pca_variance}): {n_components} components, "
        f"{explained:.3%} variance explained"
    )
    logger.info(f"Train: {X_train_pca.shape} | Test: {X_test_pca.shape}")

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    runtime = time.perf_counter() - t_start

    # Determine output sub-folder name from variance threshold
    pct = int(pca_variance * 100)
    pipeline_key = f"b_pca{pct}"

    if save:
        out_dir = DATA_PROC / dataset / pipeline_key
        _save(dataset, pipeline_key, X_train_pca, X_test_pca, y_train, y_test)
        _save_transformers(out_dir, {
            "imputer_numeric":     num_imp,
            "imputer_categorical": cat_imp,
            "encoder":             train_cols,
            "outlier":             bounds,
            "boxcox":              lambdas,
            "scaler":              scaler,
            "pca":                 pca,
        })
        _save_metadata(out_dir, {
            "pipeline":               pipeline_key,
            "dataset":                dataset,
            "input_shape":            input_shape,
            "output_shape_train":     list(X_train_pca.shape),
            "output_shape_test":      list(X_test_pca.shape),
            "n_features_in":          n_features_in,
            "n_features_out":         X_train_pca.shape[1],
            "pca_variance_explained": round(float(explained), 6),
            "pca_n_components":       int(n_components),
            "runtime_seconds":        round(runtime, 4),
            "peak_memory_mb":         round(peak_mem / 1024 / 1024, 4),
        })

    return X_train_pca, X_test_pca, y_train, y_test


# -- Bank Marketing ---------------------------------------------------------

def run_bank(pca_variance=0.90, save=True):
    """
    B Pipeline -- Bank Marketing.

    Parameters
    ----------
    pca_variance : 0.90 for b_pca90, 0.20 for b_pca20
    """
    logger.info(f"B Pipeline -- Bank Marketing (pca_variance={pca_variance})")
    df = load_bank()
    X = df.drop(columns=["target_bin"])
    y = df["target_bin"]
    return _run_pipeline(X, y, pca_variance=pca_variance, dataset="bank", save=save)


def run_bank_pca90(save=True):
    """Convenience wrapper: Bank + 90% variance PCA."""
    return run_bank(pca_variance=0.90, save=save)


def run_bank_pca20(save=True):
    """Convenience wrapper: Bank + 20% variance PCA."""
    return run_bank(pca_variance=0.20, save=save)


# -- Online Retail ----------------------------------------------------------

def run_retail(pca_variance=0.90, save=True):
    """
    B Pipeline -- Online Retail (temporal churn prediction).

    Parameters
    ----------
    pca_variance : 0.90 for b_pca90, 0.20 for b_pca20
    """
    logger.info(f"B Pipeline -- Online Retail (pca_variance={pca_variance})")
    agg = load_retail_temporal()
    X = agg.drop(columns=["target_bin"])
    y = agg["target_bin"]
    return _run_pipeline(X, y, pca_variance=pca_variance, dataset="retail", save=save)


def run_retail_pca90(save=True):
    """Convenience wrapper: Retail + 90% variance PCA."""
    return run_retail(pca_variance=0.90, save=save)


def run_retail_pca20(save=True):
    """Convenience wrapper: Retail + 20% variance PCA."""
    return run_retail(pca_variance=0.20, save=save)


# -- Save -------------------------------------------------------------------

def _save(dataset, pipeline, X_train, X_test, y_train, y_test):
    out_dir = DATA_PROC / dataset / pipeline
    ensure_dirs(out_dir)
    X_train.to_parquet(out_dir / "X_train.parquet", index=False)
    X_test.to_parquet(out_dir / "X_test.parquet",  index=False)
    y_train.to_frame().to_parquet(out_dir / "y_train.parquet", index=False)
    y_test.to_frame().to_parquet(out_dir / "y_test.parquet",  index=False)
    logger.info(f"Saved: {out_dir}")


def _save_transformers(out_dir: Path, transformers_dict: dict):
    """Persist fitted transformer objects as a joblib file."""
    ensure_dirs(out_dir)
    path = out_dir / "transformers.joblib"
    joblib.dump(transformers_dict, path)
    logger.info(f"Transformers saved: {path}")


def _save_metadata(out_dir: Path, meta_dict: dict):
    """Persist dimensions and runtime metadata as a human-readable JSON file."""
    ensure_dirs(out_dir)
    path = out_dir / "metadata.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)
    logger.info(f"Metadata saved: {path}")


if __name__ == "__main__":
    # Run all 4 combinations: (bank, retail) x (pca90, pca20)
    run_bank_pca90()
    run_bank_pca20()
    run_retail_pca90()
    run_retail_pca20()
    logger.info("Pipeline B completed (b_pca90 and b_pca20 for both datasets)")
