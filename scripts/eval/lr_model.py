"""
lr_model.py -- Logistic Regression (L1/L2) with Optuna HPO.

HPO search space (50 trials):
  - penalty:   l1 or l2
  - C:         1e-4 - 100 (log)
  - solver:    saga (supports both L1 and L2)
  - max_iter:  500-2000
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from config import N_OPTUNA_TRIALS, SEED
from scripts.utils import get_logger

logger = get_logger("lr_model")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_default_params(seed: int = SEED) -> dict:
    # sklearn >= 1.8: use l1_ratio instead of penalty; n_jobs removed
    return {
        "l1_ratio":     0.0,    # 0.0 = L2, 1.0 = L1
        "C":            1.0,
        "solver":       "saga",
        "max_iter":     3000,
        "random_state": seed,
    }


def build_model(params: dict = None, seed: int = SEED):
    if params is None:
        params = get_default_params(seed)
    return LogisticRegression(**params)


def tune_lr(X_train, y_train,
            n_trials: int = N_OPTUNA_TRIALS,
            seed: int = SEED) -> dict:
    """
    Optuna HPO for Logistic Regression. Optimises PR-AUC via 3-fold CV.

    Returns
    -------
    best_params dict
    """
    logger.info(f"Logistic Regression HPO: {n_trials} trials")

    def objective(trial):
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        C        = trial.suggest_float("C", 1e-4, 100.0, log=True)
        max_iter = trial.suggest_int("max_iter", 500, 5000)

        model = LogisticRegression(
            l1_ratio=l1_ratio,
            C=C,
            solver="saga",
            max_iter=max_iter,
            random_state=seed,
        )
        scores = cross_val_score(model, X_train, y_train,
                                  cv=3, scoring="average_precision", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best.update({"solver": "saga", "random_state": seed})
    logger.info(f"Best PR-AUC: {study.best_value:.4f} | Params: {best}")
    return best
