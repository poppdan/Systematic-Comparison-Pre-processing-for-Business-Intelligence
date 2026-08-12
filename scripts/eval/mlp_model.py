"""
mlp_model.py -- MLP classifier (PyTorch) with Optuna HPO.

Architecture: Fully-connected with BatchNorm + Dropout
HPO search space:
  - n_layers:     1-4
  - hidden_dim:   32-512
  - dropout:      0.0-0.5
  - lr:           1e-4-1e-2 (log)
  - batch_size:   64-512
  - weight_decay: 1e-6-1e-2 (log)
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
from sklearn.metrics import average_precision_score

from config import EPOCHS, PATIENCE, SEED, BATCH_SIZE, LR, N_OPTUNA_TRIALS
from scripts.utils import get_logger, set_seed

logger = get_logger("mlp_model")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -- Model ----------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# -- Training -------------------------------------------------------

def train_mlp(X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              hidden_dim: int = 128, n_layers: int = 2,
              dropout: float = 0.2, lr: float = LR,
              batch_size: int = BATCH_SIZE,
              weight_decay: float = 1e-4,
              epochs: int = EPOCHS, patience: int = PATIENCE,
              seed: int = SEED) -> tuple[MLP, float]:
    """
    Train an MLP. Returns model and best PR-AUC.
    """
    set_seed(seed)
    input_dim = X_train.shape[1]

    model = MLP(input_dim, hidden_dim, n_layers, dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Pos-weight for BCE loss (imbalance handling)
    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
                               dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # DataLoader
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    X_v = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_v = y_val

    best_pr_auc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits = model(X_v).cpu().numpy()
            proba = torch.sigmoid(torch.tensor(logits)).numpy()
        pr_auc = average_precision_score(y_v, proba)

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.debug(f"Early stop @ epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_pr_auc


class MLPClassifier:
    """Sklearn-compatible wrapper for the MLP."""

    def __init__(self, hidden_dim=128, n_layers=2, dropout=0.2,
                 lr=LR, batch_size=BATCH_SIZE, weight_decay=1e-4,
                 epochs=EPOCHS, patience=PATIENCE, seed=SEED):
        self.hidden_dim   = hidden_dim
        self.n_layers     = n_layers
        self.dropout      = dropout
        self.lr           = lr
        self.batch_size   = batch_size
        self.weight_decay = weight_decay
        self.epochs       = epochs
        self.patience     = patience
        self.seed         = seed
        self.model_       = None

    def fit(self, X, y, X_val=None, y_val=None):
        if X_val is None:
            # Internal split for early stopping
            from sklearn.model_selection import train_test_split
            X, X_val, y, y_val = train_test_split(X, y, test_size=0.1,
                                                    stratify=y, random_state=self.seed)
        X = np.array(X, dtype=np.float32)
        X_val = np.array(X_val, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        y_val = np.array(y_val, dtype=np.float32)

        self.model_, _ = train_mlp(
            X, y, X_val, y_val,
            hidden_dim=self.hidden_dim, n_layers=self.n_layers,
            dropout=self.dropout, lr=self.lr, batch_size=self.batch_size,
            weight_decay=self.weight_decay, epochs=self.epochs,
            patience=self.patience, seed=self.seed,
        )
        return self

    def predict_proba(self, X):
        self.model_.eval()
        X_t = torch.tensor(np.array(X, dtype=np.float32)).to(DEVICE)
        with torch.no_grad():
            logits = self.model_(X_t).cpu().numpy()
        proba = torch.sigmoid(torch.tensor(logits)).numpy()
        return np.column_stack([1 - proba, proba])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


# -- HPO -----------------------------------------------------------

def tune_mlp(X_train, y_train, input_dim: int,
             n_trials: int = N_OPTUNA_TRIALS, seed: int = SEED) -> dict:
    """Optuna HPO for MLP."""
    logger.info(f"MLP HPO: {n_trials} trials")

    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        np.array(X_train, dtype=np.float32),
        np.array(y_train, dtype=np.float32),
        test_size=0.15, stratify=y_train, random_state=seed
    )

    def objective(trial):
        params = dict(
            hidden_dim   = trial.suggest_int("hidden_dim", 32, 512),
            n_layers     = trial.suggest_int("n_layers", 1, 4),
            dropout      = trial.suggest_float("dropout", 0.0, 0.5),
            lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            batch_size   = trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        )
        _, pr_auc = train_mlp(X_tr, y_tr, X_val, y_val,
                               epochs=50, patience=5, seed=seed, **params)
        completed = [t.value for t in trial.study.trials if t.value is not None]
        best_so_far = max(completed) if completed else pr_auc
        logger.info(
            f"  Trial {trial.number+1:02d}/{n_trials} | "
            f"PR-AUC: {pr_auc:.4f} | best: {best_so_far:.4f} | "
            f"hidden={params['hidden_dim']} layers={params['n_layers']} "
            f"lr={params['lr']:.5f} drop={params['dropout']:.2f}"
        )
        return pr_auc

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    logger.info(f"MLP HPO done | Best PR-AUC: {study.best_value:.4f} | Params: {best}")
    return best
