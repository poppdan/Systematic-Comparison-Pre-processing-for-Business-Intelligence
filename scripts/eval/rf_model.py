"""
rf_model.py -- Random Forest classifier with Optuna HPO.

HPO search space (50 trials):
  - n_estimators:  100-1000
  - max_depth:     3-30 (or None)
  - min_samples_split: 2-20
  - min_samples_leaf:  1-10
  - max_features: sqrt, log2, 0.3-0.8
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from config import N_OPTUNA_TRIALS, SEED
from scripts.utils import get_logger

logger = get_logger("rf_model")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_default_params(seed: int = SEED) -> dict:
    return {
        "n_estimators":      300,
        "max_depth":         None,
        "min_samples_split": 2,
        "min_samples_leaf":  1,
        "max_features":      "sqrt",
        "random_state":      seed,
        "n_jobs":            -1,
    }


def build_model(params: dict = None, seed: int = SEED):
    if params is None:
        params = get_default_params(seed)
    return RandomForestClassifier(**params)


def tune_rf(X_train, y_train,
            n_trials: int = N_OPTUNA_TRIALS,
            seed: int = SEED) -> dict:
    """
    Optuna HPO for Random Forest. Optimises PR-AUC via 3-fold CV.

    Returns
    -------
    best_params dict
    """
    logger.info(f"Random Forest HPO: {n_trials} trials")

    def objective(trial):
        max_depth_choice = trial.suggest_categorical("max_depth_choice", ["none", "int"])
        max_depth = None if max_depth_choice == "none" else trial.suggest_int("max_depth", 3, 30)

        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
            "max_depth":         max_depth,
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":      trial.suggest_categorical("max_features",
                                     ["sqrt", "log2", 0.3, 0.5, 0.8]),
            "random_state":      seed,
            "n_jobs":            -1,
        }
        model = RandomForestClassifier(**params)
        scores = cross_val_score(model, X_train, y_train,
                                  cv=3, scoring="average_precision", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    # Resolve max_depth_choice
    if best.pop("max_depth_choice", "none") == "none":
        best["max_depth"] = None
    best.update({"random_state": seed, "n_jobs": -1})
    logger.info(f"Best PR-AUC: {study.best_value:.4f} | Params: {best}")
    return best
