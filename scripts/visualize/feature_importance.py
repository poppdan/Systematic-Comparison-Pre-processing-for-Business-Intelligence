"""
feature_importance.py -- Feature importance rankings for classical pipelines.

Uses:
  - XGBoost/LightGBM: built-in gain-based importance
  - Random Forest: mean decrease in impurity
  - Logistic Regression: absolute coefficient magnitude
  - Permutation importance (sklearn): model-agnostic, works for all

For AI pipelines (D, E, F), computes permutation importance on the
latent representation to identify which latent dimensions matter most.

Produces:
  results/visualize/feature_importance_{dataset}_{pipeline}_{model}.png
      Horizontal bar chart of top N features by importance
  results/visualize/feature_importance_{dataset}_summary.png
      Heatmap: top features x pipelines (how importance shifts)

Usage:
  python scripts/visualize/feature_importance.py
  python scripts/visualize/feature_importance.py --dataset bank --pipeline a
  python scripts/visualize/feature_importance.py --model xgb --top-n 20
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

from config import DATA_PROC, RESULTS_DIR, SEED, CLASSICAL_PIPELINES, AI_PIPELINES
from scripts.utils import get_logger, ensure_dirs

logger = get_logger("feature_importance")
VIZ_DIR = RESULTS_DIR / "visualize"
ensure_dirs(VIZ_DIR)

ALL_PIPELINES = [p.lower() for p in CLASSICAL_PIPELINES + AI_PIPELINES]
DEFAULT_MODEL  = "xgb"
DEFAULT_TOP_N  = 20


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(dataset: str, pipeline: str) -> tuple:
    """Load processed train/test data. Returns X_train, X_test, y_train, y_test."""
    base = DATA_PROC / dataset / pipeline

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

    X_train = pd.read_parquet(base / "X_train.parquet")
    X_test  = pd.read_parquet(base / "X_test.parquet")
    y_train = pd.read_parquet(base / "y_train.parquet").squeeze()
    y_test  = pd.read_parquet(base / "y_test.parquet").squeeze()
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Model factory (default params, no HPO -- for importance only)
# ---------------------------------------------------------------------------

def _fit_model(model_name: str, X_train, y_train):
    if model_name == "xgb":
        from scripts.eval.xgb_model import build_model
    elif model_name == "lgbm":
        from scripts.eval.lgbm_model import build_model
    elif model_name == "rf":
        from scripts.eval.rf_model import build_model
    elif model_name == "lr":
        from scripts.eval.lr_model import build_model
    else:
        from scripts.eval.xgb_model import build_model

    model = build_model()
    model.fit(X_train.values if hasattr(X_train, "values") else X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Importance extraction
# ---------------------------------------------------------------------------

def get_native_importance(model, feature_names: list) -> pd.Series:
    """
    Extract built-in feature importance (XGB gain, RF MDI, LR |coef|).
    Returns pd.Series indexed by feature name.
    """
    # XGBoost
    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
        return pd.Series(scores, index=feature_names)

    # Logistic Regression
    if hasattr(model, "coef_"):
        scores = np.abs(model.coef_[0])
        return pd.Series(scores, index=feature_names)

    raise ValueError("Model has no native importance attribute.")


def get_permutation_importance(model, X_test, y_test,
                                feature_names: list,
                                n_repeats: int = 10) -> pd.Series:
    """
    Model-agnostic permutation importance on test set.
    Metric: average precision (PR-AUC proxy).
    """
    from sklearn.metrics import average_precision_score

    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        scoring="average_precision",
        random_state=SEED,
        n_jobs=-1
    )
    return pd.Series(result.importances_mean, index=feature_names)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_importance_bar(scores: pd.Series, pipeline: str, model_name: str,
                         dataset: str, method: str, top_n: int = DEFAULT_TOP_N,
                         out_path: Path = None):
    """Horizontal bar chart of top-N features."""
    top = scores.abs().nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
    colors = ["#1565C0" if v >= 0 else "#B71C1C" for v in top.values]
    ax.barh(top.index, top.values, color=colors, edgecolor="white", height=0.7)

    ax.set_xlabel(f"Importance ({method})", fontsize=10)
    ax.set_title(
        f"{dataset.upper()} -- {pipeline.upper()} / {model_name.upper()}: "
        f"Top {top_n} Features ({method.upper()})",
        fontsize=11, fontweight="bold"
    )
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()

    if out_path is None:
        out_path = VIZ_DIR / f"feature_importance_{dataset}_{pipeline}_{model_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


def plot_summary_heatmap(importance_dict: dict, dataset: str, top_n: int = 15):
    """
    Heatmap: top features (rows) x pipelines (columns).
    Shows how feature importance shifts across preprocessing strategies.

    importance_dict = {"{pipeline}_{model}": pd.Series of importance scores}
    """
    # Union of top-N features across all pipelines
    top_features = set()
    for scores in importance_dict.values():
        top_features.update(scores.abs().nlargest(top_n).index.tolist())
    top_features = sorted(top_features)

    matrix = pd.DataFrame(index=top_features, columns=importance_dict.keys())
    for label, scores in importance_dict.items():
        for feat in top_features:
            matrix.loc[feat, label] = scores.get(feat, 0.0)

    matrix = matrix.astype(float)
    # Normalize per column (0..1) for cross-pipeline comparison
    matrix_norm = matrix.div(matrix.abs().max()).fillna(0)

    fig, ax = plt.subplots(figsize=(max(8, len(importance_dict) * 1.5),
                                     max(6, len(top_features) * 0.4)))
    sns.heatmap(
        matrix_norm, cmap="Blues", linewidths=0.3,
        ax=ax, cbar_kws={"label": "Normalized Importance"},
        annot=False
    )
    ax.set_title(
        f"{dataset.upper()} -- Feature Importance Summary\n"
        f"(normalized per pipeline, top {top_n} features)",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Pipeline x Model")
    ax.set_ylabel("Feature")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()

    out = VIZ_DIR / f"feature_importance_{dataset}_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Summary heatmap saved: {out}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_importance(dataset: str, pipeline: str, model_name: str = DEFAULT_MODEL,
                    top_n: int = DEFAULT_TOP_N,
                    method: str = "native") -> pd.Series:
    """
    Compute and plot feature importance for one dataset x pipeline x model.

    method: "native" (built-in) or "permutation"
    """
    logger.info(f"\nFeature importance: {dataset} x {pipeline.upper()} x {model_name.upper()}")

    try:
        X_train, X_test, y_train, y_test = load_data(dataset, pipeline)
    except FileNotFoundError:
        logger.warning(f"  Data not found for {dataset}/{pipeline} -- skipping")
        return pd.Series(dtype=float)

    feature_names = list(X_train.columns) if hasattr(X_train, "columns") else \
                    [f"dim_{i}" for i in range(X_train.shape[1])]

    model = _fit_model(model_name, X_train, y_train)

    if method == "permutation":
        scores = get_permutation_importance(
            model, X_test.values if hasattr(X_test, "values") else X_test,
            y_test, feature_names
        )
    else:
        try:
            scores = get_native_importance(model, feature_names)
        except ValueError:
            logger.info("  Native importance not available -- falling back to permutation")
            scores = get_permutation_importance(
                model, X_test.values if hasattr(X_test, "values") else X_test,
                y_test, feature_names
            )

    # Save CSV
    csv_out = VIZ_DIR / f"feature_importance_{dataset}_{pipeline}_{model_name}.csv"
    scores.sort_values(ascending=False).to_csv(csv_out, header=["importance"])
    logger.info(f"  CSV saved: {csv_out}")

    plot_importance_bar(scores, pipeline, model_name, dataset, method, top_n)
    return scores


def run_all(datasets: list = None, pipelines: list = None,
            model_name: str = DEFAULT_MODEL, top_n: int = DEFAULT_TOP_N,
            method: str = "native"):
    """Run feature importance for all combinations and produce summary heatmap."""
    if datasets  is None: datasets  = ["bank", "retail"]
    if pipelines is None: pipelines = ALL_PIPELINES

    for ds in datasets:
        importance_dict = {}
        for pl in pipelines:
            scores = run_importance(ds, pl, model_name, top_n, method)
            if not scores.empty:
                importance_dict[f"{pl}_{model_name}"] = scores

        if len(importance_dict) > 1:
            plot_summary_heatmap(importance_dict, ds, top_n=min(top_n, 15))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature importance rankings")
    parser.add_argument("--dataset",  default=None, choices=["bank", "retail"])
    parser.add_argument("--pipeline", default=None, choices=ALL_PIPELINES)
    parser.add_argument("--model",    default=DEFAULT_MODEL,
                        choices=["lr", "rf", "xgb", "lgbm"])
    parser.add_argument("--top-n",   type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--method",  default="native",
                        choices=["native", "permutation"],
                        help="native=built-in importance, permutation=model-agnostic")
    args = parser.parse_args()

    datasets  = [args.dataset]  if args.dataset  else None
    pipelines = [args.pipeline] if args.pipeline else None

    run_all(datasets=datasets, pipelines=pipelines,
            model_name=args.model, top_n=args.top_n, method=args.method)
