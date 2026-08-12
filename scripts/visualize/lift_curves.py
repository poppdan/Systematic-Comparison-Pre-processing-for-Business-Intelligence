"""
lift_curves.py -- Lift curves and profit simulation for Bank Marketing.

Reads:  results/{dataset}_{pipeline}_{model}_cv.json
        (test_metrics from run_experiment.py)

Produces:
  results/visualize/lift_curve_{dataset}.png
      Cumulative lift curve: all pipelines x best model, vs. random baseline
  results/visualize/profit_simulation_{dataset}.png
      Profit curve: Expected Monetary Value vs. contact threshold

Revenue assumptions (from Bank Marketing domain):
  revenue_tp  = 100 EUR  (term deposit margin per conversion)
  cost_fp     = 10  EUR  (cost of calling a non-converter)
  cost_fn     = 0   EUR  (missed opportunity, not a direct cost)

Usage:
  python scripts/visualize/lift_curves.py
  python scripts/visualize/lift_curves.py --dataset bank
  python scripts/visualize/lift_curves.py --model xgb
"""
import sys
import argparse
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from config import RESULTS_DIR, DATA_PROC
from scripts.utils import get_logger, ensure_dirs

logger = get_logger("lift_curves")
VIZ_DIR = RESULTS_DIR / "visualize"
ensure_dirs(VIZ_DIR)

REVENUE_TP = 100.0
COST_FP    =  10.0

PIPELINE_COLORS = {
    "a":      "#2196F3",
    "b_pca90": "#4CAF50",
    "b_pca20": "#8BC34A",
    "c":      "#FF9800",
    "d":      "#E91E63",
    "e":      "#9C27B0",
    "f":      "#F44336",
}


# ---------------------------------------------------------------------------
# Load test predictions from saved result files
# ---------------------------------------------------------------------------

def load_test_predictions(dataset: str, model: str = None) -> dict:
    """
    Load stored test metrics and reconstruct lift data.

    Returns dict: {label: test_metrics_dict}
    where label = "{pipeline}_{model}"
    """
    pattern = f"{dataset}_*_cv.json"
    results = {}
    for f in RESULTS_DIR.glob(pattern):
        with open(f) as fp:
            data = json.load(fp)
        key = f.stem.replace(f"{dataset}_", "").replace("_cv", "")
        parts = key.rsplit("_", 1)
        if model and len(parts) == 2 and parts[1] != model:
            continue
        results[key] = data
    return results


def load_processed_test(dataset: str, pipeline: str):
    """Load processed test data (X_test, y_test)."""
    base = DATA_PROC / dataset / pipeline

    # AI pipelines: CSV
    test_csv = base / "test.csv"
    if test_csv.exists():
        df = pd.read_csv(test_csv)
        return df.drop(columns=["target"]).values, df["target"].values

    # Classical: parquet
    X = pd.read_parquet(base / "X_test.parquet").values
    y = pd.read_parquet(base / "y_test.parquet").squeeze().values
    return X, y


# ---------------------------------------------------------------------------
# Lift curve computation
# ---------------------------------------------------------------------------

def compute_lift_curve(y_true: np.ndarray,
                        y_score: np.ndarray) -> tuple:
    """
    Compute cumulative lift curve.

    Returns
    -------
    pct_contacted : np.ndarray  (0..1, fraction of population contacted)
    lift          : np.ndarray  (cumulative lift at each threshold)
    """
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    n = len(y_sorted)
    base_rate = y_sorted.mean()

    pct_contacted = np.arange(1, n + 1) / n
    cumulative_precision = np.cumsum(y_sorted) / np.arange(1, n + 1)
    lift = cumulative_precision / base_rate

    return pct_contacted, lift


def compute_profit_curve(y_true: np.ndarray,
                          y_score: np.ndarray,
                          revenue_tp: float = REVENUE_TP,
                          cost_fp: float = COST_FP) -> tuple:
    """
    Compute expected profit as a function of threshold / contact fraction.

    profit(t) = TP(t) * revenue_tp - FP(t) * cost_fp
    """
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    n = len(y_sorted)

    pct_contacted = np.arange(1, n + 1) / n
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.arange(1, n + 1) - tp_cum
    profit = tp_cum * revenue_tp - fp_cum * cost_fp

    return pct_contacted, profit


# ---------------------------------------------------------------------------
# Build predictions by re-fitting on full train, predicting test
# ---------------------------------------------------------------------------

def _get_predictions_for_pipeline(dataset: str, pipeline: str,
                                   model_name: str = "xgb") -> tuple:
    """
    Load processed data and a default (no-HPO) model, get test predictions.
    Returns (y_test, y_score).
    """
    try:
        X_test, y_test = load_processed_test(dataset, pipeline)
    except FileNotFoundError:
        return None, None

    # Use stored results if available (test_metrics doesn't have scores)
    # We need to refit or load saved probabilities -- refit with defaults
    base = DATA_PROC / dataset / pipeline
    train_csv = base / "train.csv"

    if train_csv.exists():
        df_tr = pd.read_csv(train_csv)
        X_train = df_tr.drop(columns=["target"]).values
        y_train = df_tr["target"].values
    else:
        X_train = pd.read_parquet(base / "X_train.parquet").values
        y_train = pd.read_parquet(base / "y_train.parquet").squeeze().values

    if model_name == "xgb":
        from scripts.eval.xgb_model import build_model
        model = build_model()
    elif model_name == "lgbm":
        from scripts.eval.lgbm_model import build_model
        model = build_model()
    elif model_name == "lr":
        from scripts.eval.lr_model import build_model
        model = build_model()
    elif model_name == "rf":
        from scripts.eval.rf_model import build_model
        model = build_model()
    else:
        from scripts.eval.xgb_model import build_model
        model = build_model()

    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]
    return y_test, y_score


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_lift_curves(dataset: str, model_name: str = "xgb"):
    """
    Cumulative lift curve for all pipelines with the chosen model.
    """
    from config import CLASSICAL_PIPELINES, AI_PIPELINES
    all_pipelines = [p.lower() for p in CLASSICAL_PIPELINES + AI_PIPELINES]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Random baseline
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2,
               label="Random Baseline")

    any_plotted = False
    for pl in all_pipelines:
        y_test, y_score = _get_predictions_for_pipeline(dataset, pl, model_name)
        if y_test is None:
            logger.warning(f"  {pl}: data not found -- skipping")
            continue

        pct, lift = compute_lift_curve(y_test, y_score)
        color = PIPELINE_COLORS.get(pl, "#888888")
        ax.plot(pct, lift, label=pl.upper(), color=color, linewidth=2)
        any_plotted = True

    if not any_plotted:
        logger.error("No data available for lift curves. Run preprocessing first.")
        plt.close()
        return

    ax.set_xlabel("Fraction of Population Contacted", fontsize=11)
    ax.set_ylabel("Cumulative Lift", fontsize=11)
    ax.set_title(
        f"{dataset.upper()} -- Cumulative Lift Curve (model: {model_name.upper()})\n"
        f"Lift = precision in top-k / base rate",
        fontsize=12, fontweight="bold"
    )
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(title="Pipeline", bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out = VIZ_DIR / f"lift_curve_{dataset}_{model_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Lift curve saved: {out}")


def plot_profit_curves(dataset: str, model_name: str = "xgb",
                        revenue_tp: float = REVENUE_TP,
                        cost_fp: float = COST_FP):
    """
    Profit simulation curve for all pipelines.
    """
    from config import CLASSICAL_PIPELINES, AI_PIPELINES
    all_pipelines = [p.lower() for p in CLASSICAL_PIPELINES + AI_PIPELINES]

    fig, ax = plt.subplots(figsize=(9, 6))

    any_plotted = False
    optimal_lines = []
    for pl in all_pipelines:
        y_test, y_score = _get_predictions_for_pipeline(dataset, pl, model_name)
        if y_test is None:
            continue

        pct, profit = compute_profit_curve(y_test, y_score, revenue_tp, cost_fp)
        color = PIPELINE_COLORS.get(pl, "#888888")

        ax.plot(pct, profit, label=pl.upper(), color=color, linewidth=2)
        idx_max = np.argmax(profit)
        ax.scatter(pct[idx_max], profit[idx_max], color=color, marker="*",
                   s=120, zorder=5)

        optimal_lines.append((pl.upper(), pct[idx_max], profit[idx_max]))
        any_plotted = True

    if not any_plotted:
        logger.error("No data available for profit simulation.")
        plt.close()
        return

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Fraction of Population Contacted", fontsize=11)
    ax.set_ylabel(f"Expected Profit (EUR)  [TPx{revenue_tp:.0f} ? FPx{cost_fp:.0f}]",
                  fontsize=11)
    ax.set_title(
        f"{dataset.upper()} -- Profit Simulation (model: {model_name.upper()})\n"
        f"* = optimal contact threshold per pipeline",
        fontsize=12, fontweight="bold"
    )
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(title="Pipeline", bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = VIZ_DIR / f"profit_simulation_{dataset}_{model_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Profit simulation saved: {out}")

    # Log optimal thresholds
    logger.info("Optimal contact thresholds:")
    for pl, pct, profit in sorted(optimal_lines, key=lambda x: -x[2]):
        logger.info(f"  {pl}: contact {pct:.1%} -> profit {profit:,.0f} EUR")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lift curves and profit simulation")
    parser.add_argument("--dataset", default=None, choices=["bank", "retail"])
    parser.add_argument("--model",   default="xgb",
                        choices=["lr", "rf", "xgb", "lgbm", "mlp"])
    parser.add_argument("--revenue-tp", type=float, default=REVENUE_TP)
    parser.add_argument("--cost-fp",    type=float, default=COST_FP)
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ["bank", "retail"]
    for ds in datasets:
        plot_lift_curves(ds, args.model)
        plot_profit_curves(ds, args.model, args.revenue_tp, args.cost_fp)
