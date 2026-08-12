"""
pipeline_e.py -- Pipeline E: Variational Autoencoder (VAE).

Architecture:
  - Encoder: input_dim -> 128 -> 64 -> (mu, log_var) each of size latent_dim
  - Decoder: latent_dim -> 64 -> 128 -> input_dim
  - Loss: MSE reconstruction + KL divergence (beta-VAE with beta=1)
  - Reparameterisation trick for backprop through sampling

Output:
  - Latent mean (mu) used as representation (deterministic at inference)
  - Saves to data/processed/{dataset}/e/
"""
import sys
import json
import time
import tracemalloc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (BANK_RAW, RETAIL_RAW, DATA_PROC,
                    LATENT_DIM, BATCH_SIZE, EPOCHS, PATIENCE, LR,
                    TEST_SIZE, SEED)
from scripts.utils import get_logger, ensure_dirs
from scripts.classical.missing_values import handle_missing
from scripts.classical.pipeline_a import load_retail_temporal, RETAIL_CUTOFF

logger = get_logger("pipeline_e")
ensure_dirs(DATA_PROC)

torch.manual_seed(SEED)
np.random.seed(SEED)

BETA = 1.0   # KL weight -- beta=1 is standard VAE


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder_shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(),
        )
        self.fc_mu      = nn.Linear(64, latent_dim)
        self.fc_log_var = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, 128),        nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def encode(self, x):
        h      = self.encoder_shared(x)
        mu     = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterise(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z    = self.reparameterise(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var


def vae_loss(recon, x, mu, log_var, beta: float = BETA):
    """ELBO loss: reconstruction (MSE) + KL divergence."""
    recon_loss = nn.functional.mse_loss(recon, x, reduction="sum")
    kl_loss    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl_loss


# ---------------------------------------------------------------------------
# Data helpers (shared with pipeline_d)
# ---------------------------------------------------------------------------

def _prepare(df: pd.DataFrame, target_col: str):
    """Impute + detect col types. Does NOT fit scaler/encoders (done after split)."""
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].values
    X, _, _ = handle_missing(X)
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    return X, y, cat_cols, num_cols


def _fit_transform(X_tr_df, X_te_df, cat_cols, num_cols):
    """Fit encoders + scaler on train, apply to both. Return numpy arrays."""
    X_tr = X_tr_df.copy()
    X_te = X_te_df.copy()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_tr[col] = le.fit_transform(X_tr[col].astype(str))
        mapping = {c: i for i, c in enumerate(le.classes_)}
        X_te[col] = X_te[col].astype(str).map(lambda v: mapping.get(v, 0))
        encoders[col] = le
    scaler = StandardScaler()
    if num_cols:
        X_tr[num_cols] = scaler.fit_transform(X_tr[num_cols])
        X_te[num_cols] = scaler.transform(X_te[num_cols])
    return (X_tr.values.astype(np.float32),
            X_te.values.astype(np.float32),
            scaler, encoders)


def _train_vae(X_train: np.ndarray, latent_dim: int = LATENT_DIM,
               device: str = "cpu",
               epochs: int = EPOCHS, patience: int = PATIENCE,
               val_frac: float = 0.15, seed: int = SEED) -> VariationalAutoencoder:
    """
    Train the VAE with early stopping on a held-out VALIDATION split.
    Monitoring the training ELBO gives no generalisation signal and lets the
    model train until it memorises the training set.
    """
    input_dim = X_train.shape[1]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_train))
    n_val = max(int(val_frac * len(X_train)), 2)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    X_fit, X_val = X_train[tr_idx], X_train[val_idx]

    model = VariationalAutoencoder(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    dataset = TensorDataset(torch.from_numpy(X_fit))
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    t_val   = torch.from_numpy(X_val).to(device)

    best_loss = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            recon, mu, log_var = model(batch)
            loss = vae_loss(recon, batch, mu, log_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- validation ELBO ---
        model.eval()
        with torch.no_grad():
            recon_v, mu_v, lv_v = model(t_val)
            val_loss = vae_loss(recon_v, t_val, mu_v, lv_v).item() / len(X_val)

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} (val loss)")
                break

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} | Val loss: {val_loss:.5f}")

    model.load_state_dict(best_state)
    return model


def _encode(model: VariationalAutoencoder, X: np.ndarray,
            device: str = "cpu") -> np.ndarray:
    """Return deterministic mu (mean of latent distribution)."""
    model.eval()
    with torch.no_grad():
        t = torch.from_numpy(X).to(device)
        mu, _ = model.encode(t)
    return mu.cpu().numpy()


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------

def _run(dataset: str, X_df: pd.DataFrame, y: np.ndarray,
         cat_cols, num_cols,
         latent_dim: int = LATENT_DIM, save: bool = True):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Pipeline E ({dataset}) -- device: {device}")

    tracemalloc.start()
    t_start = time.perf_counter()

    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    y_train, y_test = y[idx_tr], y[idx_te]

    X_train, X_test, scaler, encoders = _fit_transform(
        X_df.iloc[idx_tr], X_df.iloc[idx_te], cat_cols, num_cols
    )

    model = _train_vae(X_train, latent_dim=latent_dim, device=device)

    Z_train = _encode(model, X_train, device)
    Z_test  = _encode(model, X_test, device)

    elapsed = time.perf_counter() - t_start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    logger.info(f"Pipeline E ({dataset}): "
                f"{Z_train.shape[1]} latent features | "
                f"{elapsed:.1f}s | {peak_mem/1e6:.1f} MB")

    if save:
        out_dir = DATA_PROC / dataset / "e"
        ensure_dirs(out_dir)

        col_names = [f"vae_{i}" for i in range(latent_dim)]
        pd.DataFrame(Z_train, columns=col_names).assign(target=y_train)\
          .to_csv(out_dir / "train.csv", index=False)
        pd.DataFrame(Z_test, columns=col_names).assign(target=y_test)\
          .to_csv(out_dir / "test.csv", index=False)

        joblib.dump(model, out_dir / "vae_model.pkl")
        joblib.dump(scaler, out_dir / "scaler.pkl")
        joblib.dump(encoders, out_dir / "encoders.pkl")

        meta = {
            "pipeline":      "E_VAE",
            "dataset":       dataset,
            "latent_dim":    latent_dim,
            "input_dim":     X_train.shape[1],
            "beta":          BETA,
            "n_train":       int(len(y_train)),
            "n_test":        int(len(y_test)),
            "runtime_s":     round(elapsed, 2),
            "peak_mem_mb":   round(peak_mem / 1e6, 2),
            "cat_cols":      cat_cols,
            "num_cols":      num_cols,
        }
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved Pipeline E ({dataset}) -> {out_dir}")

    return Z_train, Z_test, y_train, y_test


def run_bank(latent_dim: int = LATENT_DIM, save: bool = True):
    logger.info("Pipeline E -- Bank Marketing")
    df = pd.read_csv(BANK_RAW, sep=";")
    df = df.rename(columns={"y": "target"})
    df["target"] = (df["target"] == "yes").astype(int)
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])
    X_df, y, cat_cols, num_cols = _prepare(df, "target")
    return _run("bank", X_df, y, cat_cols, num_cols, latent_dim=latent_dim, save=save)


def run_retail(latent_dim: int = LATENT_DIM, save: bool = True):
    logger.info("Pipeline E -- Online Retail (temporal split)")
    rfm = load_retail_temporal()
    rfm = rfm.rename(columns={"target_bin": "target"})
    X_df, y, cat_cols, num_cols = _prepare(rfm, "target")
    return _run("retail", X_df, y, cat_cols, num_cols, latent_dim=latent_dim, save=save)


if __name__ == "__main__":
    run_bank()
    run_retail()
