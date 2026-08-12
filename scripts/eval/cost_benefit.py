"""
cost_benefit.py -- RQ3: implementation cost vs. performance gain.

RQ3 (optional) asks how implementation effort (runtime, memory) relates to the
performance actually gained. The two cost components live in different places:

  * pre-processing cost  -> data/processed/<ds>/<pipe>/{meta,metadata}.json
                            (runtime_seconds / peak_memory_mb, written by the
                             pipeline scripts themselves)
  * downstream cost      -> results/<ds>_<pipe>_<model>_cv.json
                            (runtime.cv_total_s / fit_s / cv_peak_mem_mb)

This script joins both and expresses the trade-off relative to Pipeline A,
which serves as the reference baseline.

Outputs:
  results/cost_benefit_<dataset>.csv
  results/cost_benefit_<dataset>.png   (PR-AUC vs. total cost scatter)

Usage:
  python scripts/eval/cost_benefit.py
  python scripts/eval/cost_benefit.py --dataset bank
"""
import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RESULTS_DIR, DATA_PROC
from scripts.utils import get_logger

logger = get_logger("cost_benefit")

PIPELINE_LABELS = {
    "a": "A", "b_pca90": "B (PCA90)", "b_pca20": "B (PCA20)",
    "c": "C", "d": "D", "e": "E", "f": "F",
}
PIPE_ORDER = ["a", "b_pca90", "b_pca20", "c", "d", "e", "f"]


def _preprocessing_cost(dataset: str, pipeline: str) -> tuple:
    """Read pre-processing runtime/memory from the pipeline's own metadata."""
    base = DATA_PROC / dataset / pipeline
    for fname in ("metadata.json", "meta.json"):
        f = base / fname
        if f.exists():
            m = json.load(open(f))
            secs = m.get("runtime_seconds", m.get("runtime_s"))
            mem  = m.get("peak_memory_mb",  m.get("peak_mem_mb"))
            return secs, mem
    return None, None


def build_table(dataset: str) -> pd.DataFrame:
    rows = []
    for f in sorted(RESULTS_DIR.glob(f"{dataset}_*_cv.json")):
        r = json.load(open(f))
        if r.get("dataset") != dataset:
            continue
        pipe  = str(r.get("pipeline"))
        model = str(r.get("model"))
        agg   = r.get("cv", {}).get("aggregated", {})
        rt    = r.get("runtime", {})
        prep_s, prep_mb = _preprocessing_cost(dataset, pipe)

        rows.append({
            "pipeline_key":  pipe,
            "Pipeline":      PIPELINE_LABELS.get(pipe, pipe.upper()),
            "Model":         model.upper(),
            "PR_AUC":        agg.get("mean_PR_AUC"),
            "N_Features":    rt.get("n_features"),
            "Preprocess_s":  prep_s,
            "Preprocess_MB": prep_mb,
            "CV_s":          rt.get("cv_total_s"),
            "Fit_s":         rt.get("fit_s"),
            "Fit_MB":        rt.get("cv_peak_mem_mb"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No results found for '{dataset}'.")

    # Total cost = one-off pre-processing + downstream fit
    df["Total_s"] = df["Preprocess_s"].fillna(0) + df["Fit_s"].fillna(0)

    # Reference: best Pipeline A configuration
    ref = df[df["pipeline_key"] == "a"]
    if not ref.empty:
        ref_row = ref.loc[ref["PR_AUC"].idxmax()]
        df["dPR_AUC_vs_A"] = df["PR_AUC"] - ref_row["PR_AUC"]
        df["Cost_ratio_vs_A"] = df["Total_s"] / max(ref_row["Total_s"], 1e-9)
        # Performance gained per additional second of compute
        extra_s = df["Total_s"] - ref_row["Total_s"]
        df["dPR_AUC_per_extra_s"] = df.apply(
            lambda r, e=extra_s: (r["dPR_AUC_vs_A"] / e[r.name])
            if e[r.name] > 1e-9 else float("nan"), axis=1)

    df["_ord"] = df["pipeline_key"].apply(
        lambda p: PIPE_ORDER.index(p) if p in PIPE_ORDER else 99)
    df = df.sort_values(["_ord", "Model"]).drop(columns=["_ord", "pipeline_key"])
    return df


def plot_tradeoff(df: pd.DataFrame, dataset: str):
    """PR-AUC vs. total cost, one point per pipeline (best model per pipeline)."""
    best = df.loc[df.groupby("Pipeline")["PR_AUC"].idxmax()].copy()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = plt.cm.tab10.colors
    for i, (_, r) in enumerate(best.iterrows()):
        ax.scatter(r["Total_s"], r["PR_AUC"], s=140,
                   color=colors[i % len(colors)], edgecolor="black", zorder=3)
        ax.annotate(f"{r['Pipeline']}\n({r['Model']})",
                    (r["Total_s"], r["PR_AUC"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)

    a = best[best["Pipeline"] == "A"]
    if not a.empty:
        ax.axhline(a["PR_AUC"].iloc[0], color="grey", ls="--", lw=1, zorder=1)
        ax.text(ax.get_xlim()[1], a["PR_AUC"].iloc[0], " Baseline A",
                va="bottom", ha="right", fontsize=8, color="grey")

    ax.set_xscale("symlog")
    ax.set_xlabel("Total cost: pre-processing + final fit (s, symlog)")
    ax.set_ylabel("PR-AUC (CV mean)")
    ax.set_title(f"{dataset.upper()} — RQ3: cost vs. benefit\n"
                 f"(best downstream model per pipeline)",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, zorder=0)
    fig.tight_layout()

    out = RESULTS_DIR / f"cost_benefit_{dataset}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Plot saved: {out}")


def run(dataset: str):
    df = build_table(dataset)
    out = RESULTS_DIR / f"cost_benefit_{dataset}.csv"
    df.to_csv(out, index=False)
    logger.info(f"Table saved: {out}")

    best = df.loc[df.groupby("Pipeline")["PR_AUC"].idxmax()]
    cols = ["Pipeline", "Model", "PR_AUC", "N_Features",
            "Preprocess_s", "Fit_s", "Total_s", "dPR_AUC_vs_A"]
    print(f"\n=== {dataset.upper()} — best model per pipeline ===")
    print(best[[c for c in cols if c in best.columns]].to_string(index=False))

    plot_tradeoff(df, dataset)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RQ3 cost-benefit analysis")
    ap.add_argument("--dataset", default=None, choices=["bank", "retail"])
    args = ap.parse_args()
    for ds in ([args.dataset] if args.dataset else ["bank", "retail"]):
        try:
            run(ds)
        except FileNotFoundError as e:
            logger.error(str(e))
