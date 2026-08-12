"""
lgbm_model.py -- LightGBM classifier with Optuna HPO.

HPO search space (50 trials):
  - num_leaves:       20-300
  - max_depth:        3-12
  - learning_rate:    1e-4 - 0.3 (log)
  - n_estimators:     100-1000
  - min_child_samples:5-100
  - subsample:        0.5 - 1.0
  - colsample_bytree: 0.5 - 1.0
  - reg_alpha:        1e-8 - 10.0 (log)
  - reg_lambda:       1e-8 - 10.0 (log)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import optuna
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

from config import N_OPTUNA_TRIALS, SEED
from scripts.utils import get_logger

logger = get_logger("lgbm_model")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_default_params(seed: int = SEED) -> dict:
    return {
        "n_estimators":     300,
        "num_leaves":       63,
        "max_depth":        -1,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "random_state":     seed,
        "n_jobs":           -1,
        "verbose":          -1,
    }


def build_model(params: dict = None, seed: int = SEED):
    if params is None:
        params = get_default_params(seed)
    return lgb.LGBMClassifier(**params)


def tune_lgbm(X_train, y_train,
              n_trials: int = N_OPTUNA_TRIALS,
              seed: int = SEED) -> dict:
    """
    Optuna HPO for LightGBM. Optimises PR-AUC via 3-fold CV.
    """
    logger.info(f"LightGBM HPO: {n_trials} trials")

    def objective(trial):
        params = {
            "num_leaves":        trial.suggest_int("num_leaves", 20, 300),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "learning_rate":     trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
            "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state":      seed,
            "n_jobs":            -1,
            "verbose":           -1,
        }
        model = lgb.LGBMClassifier(**params)
        scores = cross_val_score(model, X_train, y_train,
                                  cv=3, scoring="average_precision", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best.update({"random_state": seed, "n_jobs": -1, "verbose": -1})
    logger.info(f"Best PR-AUC: {study.best_value:.4f} | Params: {best}")
    return best
