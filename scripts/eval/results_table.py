"""
results_table.py -- Aggregate and visualise results.

Produces:
  - results/results_table.csv       (all metrics)
  - results/results_heatmap.png     (PR-AUC heatmap: pipeline x model)
  - results/results_boxplot.png     (PR-AUC distribution across CV folds)
  - results/significance_table.csv  (Wilcoxon test results)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import RESULTS_DIR
from scripts.utils import get_logger, ensure_dirs

logger = get_logger("results_table")
ensure_dirs(RESULTS_DIR)


def load_all_results(dataset: str) -> dict:
    """
    Load all stored CV results for a dataset.

    Expects: results/{dataset}_{pipeline}_{model}_cv.json
    """
    pattern = f"{dataset}_*_cv.json"
    files = list(RESULTS_DIR.glob(pattern))
    results = {}
    for f in files:
        key = f.stem.replace(f"{dataset}_", "").replace("_cv", "")
        with open(f) as fp:
            results[key] = json.load(fp)
    logger.info(f"Loaded: {len(results)} result files for '{dataset}'")
    return results


def build_summary_table(cv_results: dict) -> pd.DataFrame:
    """
    Build a summary table from CV results.

    Parameters
    ----------
    cv_results : Dict {label: {dataset, pipeline, model, cv: {aggregated, fold_metrics}, runtime}}
                 label format: "{pipeline}_{model}" e.g. "a_xgb"

    Returns
    -------
    DataFrame with Pipeline, Model, Mean+/-Std for all metrics + runtime/memory
    """
    # Display names: keep the two PCA variants of pipeline B as separate rows
    # instead of folding them into the model column ("B" / "PCA90_LR").
    PIPELINE_LABELS = {
        "a": "A", "b_pca90": "B (PCA90)", "b_pca20": "B (PCA20)",
        "c": "C", "d": "D", "e": "E", "f": "F",
    }

    rows = []
    for label, res in cv_results.items():
        # Prefer the explicit fields stored in the JSON; splitting the label on
        # "_" mis-parses "b_pca90_lr" into pipeline="B", model="PCA90_LR".
        pipe_raw  = res.get("pipeline")
        model_raw = res.get("model")
        if pipe_raw is None or model_raw is None:
            parts = label.rsplit("_", 1)          # model is the LAST segment
            pipe_raw  = parts[0]
            model_raw = parts[1] if len(parts) > 1 else ""

        pipeline = PIPELINE_LABELS.get(str(pipe_raw).lower(), str(pipe_raw).upper())
        model    = str(model_raw).upper()

        # aggregated metrics are nested under res['cv']['aggregated']
        agg = res.get("cv", {}).get("aggregated", res.get("aggregated", {}))
        row = {"Pipeline": pipeline, "Model": model}
        for key, val in agg.items():
            row[key] = val

        # Also include hold-out test metrics
        for key, val in res.get("test_metrics", {}).items():
            row[f"test_{key}"] = val

        # Runtime and memory
        rt = res.get("runtime", {})
        row["CV_Runtime_s"]  = rt.get("cv_total_s")
        row["Fit_Runtime_s"] = rt.get("fit_s")
        row["Peak_Mem_MB"]   = rt.get("cv_peak_mem_mb")
        row["N_Features"]    = rt.get("n_features")

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["Pipeline", "Model"])
    return df


PIPELINE_ROW_ORDER = ["A", "B (PCA90)", "B (PCA20)", "C", "D", "E", "F"]
MODEL_COL_ORDER    = ["LR", "RF", "XGB", "LGBM", "MLP"]


def plot_heatmap(summary_df: pd.DataFrame, dataset: str, metric: str = "mean_PR_AUC"):
    """PR-AUC heatmap: rows = pipelines (B split into PCA90/PCA20), columns = models."""
    pivot = summary_df.pivot(index="Pipeline", columns="Model", values=metric)

    # Deterministic ordering; unknown labels are appended at the end
    rows = [p for p in PIPELINE_ROW_ORDER if p in pivot.index] + \
           [p for p in pivot.index if p not in PIPELINE_ROW_ORDER]
    cols = [m for m in MODEL_COL_ORDER if m in pivot.columns] + \
           [m for m in pivot.columns if m not in MODEL_COL_ORDER]
    pivot = pivot.reindex(index=rows, columns=cols)

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.5),
                                     max(4, len(pivot.index) * 0.7)))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGnBu",
                linewidths=0.5, ax=ax, annot_kws={"size": 10})
    ax.set_title(f"{dataset.upper()} -- {metric} (CV Mean)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Pipeline")
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    out = RESULTS_DIR / f"results_heatmap_{dataset}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Heatmap saved: {out}")


def plot_boxplot(cv_results: dict, dataset: str, metric: str = "PR_AUC"):
    """Boxplot of CV fold distributions per pipeline-model combination."""
    SHORT = {"a": "A", "b_pca90": "B90", "b_pca20": "B20",
             "c": "C", "d": "D", "e": "E", "f": "F"}

    data = {}
    for label, res in cv_results.items():
        fold_scores = [m[metric] for m in res.get("cv", {}).get("fold_metrics", res.get("fold_metrics", []))]
        if not fold_scores:
            continue
        p = str(res.get("pipeline", "")).lower()
        m = str(res.get("model", "")).upper()
        name = f"{SHORT.get(p, p.upper())}-{m}" if p and m else label.upper()
        data[name] = fold_scores

    if not data:
        logger.warning("No fold metrics found for boxplot.")
        return

    # Group by pipeline in canonical order
    order = ["A", "B90", "B20", "C", "D", "E", "F"]
    data = dict(sorted(data.items(),
                       key=lambda kv: (order.index(kv[0].split("-")[0])
                                       if kv[0].split("-")[0] in order else 99,
                                       kv[0])))

    fig, ax = plt.subplots(figsize=(max(8, len(data) * 0.8), 5))
    ax.boxplot(data.values(), labels=data.keys(), patch_artist=True,
               medianprops={"color": "red", "linewidth": 2})
    ax.set_title(f"{dataset.upper()} -- {metric} Distribution (CV Folds)", fontsize=13, fontweight="bold")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    out = RESULTS_DIR / f"results_boxplot_{dataset}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Boxplot saved: {out}")


def save_summary(summary_df: pd.DataFrame, dataset: str):
    out = RESULTS_DIR / f"results_table_{dataset}.csv"
    summary_df.to_csv(out, index=False)
    logger.info(f"Table saved: {out}")
    print(summary_df.to_string(index=False))


def run_report(dataset: str):
    """Full report for a dataset."""
    cv_results = load_all_results(dataset)
    if not cv_results:
        logger.warning(f"No results found for '{dataset}'.")
        return

    summary = build_summary_table(cv_results)
    save_summary(summary, dataset)
    plot_heatmap(summary, dataset)
    plot_boxplot(cv_results, dataset)

    # Significance test
    from scripts.eval.significance_test import run_all_comparisons
    sig_df = run_all_comparisons(cv_results)
    sig_out = RESULTS_DIR / f"significance_table_{dataset}.csv"
    sig_df.to_csv(sig_out, index=False)
    logger.info(f"Significance table saved: {sig_out}")

    return summary


if __name__ == "__main__":
    for ds in ["bank", "retail"]:
        logger.info(f"\n{'='*55}\nDataset: {ds.upper()}\n{'='*55}")
        run_report(ds)