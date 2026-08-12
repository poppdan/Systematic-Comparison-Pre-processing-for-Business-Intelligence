"""
significance_test.py -- Wilcoxon Signed-Rank Test + Bonferroni correction + Cohen's d.

Tests whether the difference between pipelines is statistically significant.
Uses PR-AUC scores from the 15 CV folds (5x3 Repeated Stratified K-Fold).

Null hypothesis: No difference between pipeline A and pipeline B.
Alternative:     Pipeline A != Pipeline B (two-sided)

Effect size:     Cohen's d with 95% confidence interval (bootstrap)
Methodology:     Demsar (2006) -- statistical comparison of classifiers
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

from config import ALPHA
from scripts.utils import get_logger

logger = get_logger("significance_test")


def cohens_d(scores_a, scores_b, ci: float = 0.95,
             n_bootstrap: int = 1000, seed: int = 42) -> dict:
    """
    Cohen's d effect size with bootstrap confidence interval.

    d = (mean_a - mean_b) / pooled_std

    Interpretation:
      |d| < 0.2  : negligible
      |d| < 0.5  : small
      |d| < 0.8  : medium
      |d| >= 0.8 : large

    Parameters
    ----------
    scores_a    : CV fold scores for pipeline A
    scores_b    : CV fold scores for pipeline B
    ci          : Confidence interval level (default: 0.95)
    n_bootstrap : Number of bootstrap resamples
    seed        : Random seed for reproducibility

    Returns
    -------
    dict with d, ci_lower, ci_upper, interpretation
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    mean_diff = a.mean() - b.mean()
    pooled_std = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    d = mean_diff / pooled_std if pooled_std > 0 else 0.0

    # Bootstrap CI
    rng = np.random.default_rng(seed)
    boot_d = []
    for _ in range(n_bootstrap):
        a_boot = rng.choice(a, size=len(a), replace=True)
        b_boot = rng.choice(b, size=len(b), replace=True)
        diff = a_boot.mean() - b_boot.mean()
        std  = np.sqrt((a_boot.std(ddof=1) ** 2 + b_boot.std(ddof=1) ** 2) / 2)
        boot_d.append(diff / std if std > 0 else 0.0)

    alpha_ci = (1 - ci) / 2
    ci_lower = float(np.percentile(boot_d, 100 * alpha_ci))
    ci_upper = float(np.percentile(boot_d, 100 * (1 - alpha_ci)))

    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return {
        "cohens_d":       round(float(d), 4),
        "ci_lower":       round(ci_lower, 4),
        "ci_upper":       round(ci_upper, 4),
        "ci_level":       ci,
        "interpretation": interpretation,
    }


def wilcoxon_test(scores_a: list, scores_b: list,
                  label_a: str = "A", label_b: str = "B",
                  alpha: float = ALPHA) -> dict:
    """
    Pairwise Wilcoxon Signed-Rank Test + Cohen's d.

    Parameters
    ----------
    scores_a : PR-AUC values from CV folds for pipeline A
    scores_b : PR-AUC values from CV folds for pipeline B
    label_a  : Name of pipeline A
    label_b  : Name of pipeline B
    alpha    : Significance level (default: 0.05)

    Returns
    -------
    dict with test result including Cohen's d
    """
    assert len(scores_a) == len(scores_b), "Equal number of folds required"

    stat, p_value = stats.wilcoxon(scores_a, scores_b, alternative="two-sided")
    effect = cohens_d(scores_a, scores_b)

    result = {
        "pipeline_a":   label_a,
        "pipeline_b":   label_b,
        "mean_a":       round(float(np.mean(scores_a)), 6),
        "mean_b":       round(float(np.mean(scores_b)), 6),
        "delta":        round(float(np.mean(scores_a) - np.mean(scores_b)), 6),
        "wilcoxon_stat": round(float(stat), 4),
        "p_value":      round(float(p_value), 6),
        "significant":  bool(p_value < alpha),
        "cohens_d":     effect["cohens_d"],
        "ci_lower":     effect["ci_lower"],
        "ci_upper":     effect["ci_upper"],
        "effect_size":  effect["interpretation"],
    }
    return result


def run_all_comparisons(cv_results: dict, metric: str = "PR_AUC",
                        alpha: float = ALPHA) -> pd.DataFrame:
    """
    Run all pairwise Wilcoxon tests between pipelines with Bonferroni correction.

    Parameters
    ----------
    cv_results : Dict {label: result_dict} where label = "{pipeline}_{model}"
    metric     : CV metric to compare (default: PR_AUC)
    alpha      : Significance level before Bonferroni correction

    Returns
    -------
    DataFrame with all pairwise comparisons
    """
    # Extract fold scores per label
    fold_scores = {}
    for label, res in cv_results.items():
        scores = [m[metric] for m in res.get("cv", {}).get("fold_metrics", res.get("fold_metrics", [])) if metric in m]
        if scores:
            fold_scores[label] = scores

    if len(fold_scores) < 2:
        logger.warning("Not enough results for pairwise comparison.")
        return pd.DataFrame()

    pairs = list(combinations(sorted(fold_scores.keys()), 2))
    n_comparisons = len(pairs)
    alpha_bonferroni = alpha / n_comparisons
    logger.info(f"Running {n_comparisons} pairwise tests | "
                f"Bonferroni alpha: {alpha_bonferroni:.5f}")

    rows = []
    for label_a, label_b in pairs:
        a = fold_scores[label_a]
        b = fold_scores[label_b]
        if len(a) != len(b):
            logger.warning(f"Skipping {label_a} vs {label_b}: unequal folds")
            continue
        result = wilcoxon_test(a, b, label_a, label_b, alpha=alpha_bonferroni)
        result["n_comparisons"]    = n_comparisons
        result["alpha_bonferroni"] = round(alpha_bonferroni, 6)
        rows.append(result)

    df = pd.DataFrame(rows)
    df = df.sort_values("p_value")
    sig_count = df["significant"].sum()
    logger.info(f"Significant pairs (Bonferroni): {sig_count}/{len(df)}")
    return df


if __name__ == "__main__":
    import json
    from config import RESULTS_DIR

    for ds in ["bank", "retail"]:
        pattern = f"{ds}_*_cv.json"
        files = list(RESULTS_DIR.glob(pattern))
        cv_results = {}
        for f in files:
            key = f.stem.replace(f"{ds}_", "").replace("_cv", "")
            with open(f) as fp:
                cv_results[key] = json.load(fp)

        if not cv_results:
            logger.warning(f"No results found for {ds}")
            continue

        df = run_all_comparisons(cv_results)
        out = RESULTS_DIR / f"significance_table_{ds}.csv"
        df.to_csv(out, index=False)
        logger.info(f"Saved: {out}")
        print(df.to_string(index=False))