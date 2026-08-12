"""
pipeline_d.py -- Pipeline D: Denoising Autoencoder (DAE).

Architecture:
  - Encoder: input_dim -> 128 -> 64 -> latent_dim
  - Decoder: latent_dim -> 64 -> 128 -> input_dim
  - Corruption: Gaussian noise + masking (30% dropout on input)
  - Categorical features: learned embeddings before encoder
  - Training: MSE reconstruction loss, Adam, early stopping

Output: latent representation (latent_dim features) per sample.
Saves to data/processed/{dataset}/d/
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

logger = get_logger("pipeline_d")
ensure_dirs(DATA_PROC)

torch.manual_seed(SEED)
np.random.seed(SEED)

NOISE_FACTOR = 0.2   # std of Gaussian noise
MASK_PROB    = 0.3   # probability of zeroing out an input feature


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, 128),        nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x_clean, x_corrupted):
        z = self.encoder(x_corrupted)
        return self.decoder(z), z


def corrupt(x: torch.Tensor) -> torch.Tensor:
    """Add Gaussian noise and random masking."""
    noisy = x + NOISE_FACTOR * torch.randn_like(x)
    mask  = (torch.rand_like(x) > MASK_PROB).float()
    return noisy * mask


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _prepare(df: pd.DataFrame, target_col: str):
    """
    Impute + detect col types. Does NOT fit scaler/encoders (done after split).
    Returns DataFrame X, y, cat_cols, num_cols.
    """
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
        # Map test values; unseen categories -> 0
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


def _train_dae(X_train: np.ndarray, latent_dim: int = LATENT_DIM,
               device: str = "cpu",
               epochs: int = EPOCHS, patience: int = PATIENCE,
               val_frac: float = 0.15, seed: int = SEED) -> DenoisingAutoencoder:
    """
    Train the DAE with early stopping on a held-out VALIDATION split.

    Monitoring the training loss (as in earlier versions) lets the model train
    until it memorises the training set -- there is no signal to stop. A small
    internal validation split gives a proper generalisation signal.
    """
    input_dim = X_train.shape[1]

    # Internal train/validation split for early stopping
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_train))
    n_val = max(int(val_frac * len(X_train)), 1)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    X_fit, X_val = X_train[tr_idx], X_train[val_idx]

    model = DenoisingAutoencoder(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

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
            corrupted = corrupt(batch)
            recon, _ = model(batch, corrupted)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- validation loss (no corruption, clean reconstruction) ---
        model.eval()
        with torch.no_grad():
            recon_val, _ = model(t_val, t_val)
            val_loss = criterion(recon_val, t_val).item()

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


def _encode(model: DenoisingAutoencoder, X: np.ndarray,
            device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        t = torch.from_numpy(X).to(device)
        z = model.encode(t)
    return z.cpu().numpy()


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------

def _run(dataset: str, X_df: pd.DataFrame, y: np.ndarray,
         cat_cols, num_cols,
         latent_dim: int = LATENT_DIM, save: bool = True):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Pipeline D ({dataset}) -- device: {device}")

    tracemalloc.start()
    t_start = time.perf_counter()

    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    y_train, y_test = y[idx_tr], y[idx_te]

    X_train, X_test, scaler, encoders = _fit_transform(
        X_df.iloc[idx_tr], X_df.iloc[idx_te], cat_cols, num_cols
    )

    model = _train_dae(X_train, latent_dim=latent_dim, device=device)

    Z_train = _encode(model, X_train, device)
    Z_test  = _encode(model, X_test, device)

    elapsed = time.perf_counter() - t_start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    logger.info(f"Pipeline D ({dataset}): "
                f"{Z_train.shape[1]} latent features | "
                f"{elapsed:.1f}s | {peak_mem/1e6:.1f} MB")

    if save:
        out_dir = DATA_PROC / dataset / "d"
        ensure_dirs(out_dir)

        col_names = [f"dae_{i}" for i in range(latent_dim)]
        pd.DataFrame(Z_train, columns=col_names).assign(target=y_train)\
          .to_csv(out_dir / "train.csv", index=False)
        pd.DataFrame(Z_test, columns=col_names).assign(target=y_test)\
          .to_csv(out_dir / "test.csv", index=False)

        joblib.dump(model, out_dir / "dae_model.pkl")
        joblib.dump(scaler, out_dir / "scaler.pkl")
        joblib.dump(encoders, out_dir / "encoders.pkl")

        meta = {
            "pipeline":      "D_DAE",
            "dataset":       dataset,
            "latent_dim":    latent_dim,
            "input_dim":     X_train.shape[1],
            "n_train":       int(len(y_train)),
            "n_test":        int(len(y_test)),
            "runtime_s":     round(elapsed, 2),
            "peak_mem_mb":   round(peak_mem / 1e6, 2),
            "cat_cols":      cat_cols,
            "num_cols":      num_cols,
        }
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved Pipeline D ({dataset}) -> {out_dir}")

    return Z_train, Z_test, y_train, y_test


def run_bank(latent_dim: int = LATENT_DIM, save: bool = True):
    logger.info("Pipeline D -- Bank Marketing")
    df = pd.read_csv(BANK_RAW, sep=";")
    df = df.rename(columns={"y": "target"})
    df["target"] = (df["target"] == "yes").astype(int)
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])
    X_df, y, cat_cols, num_cols = _prepare(df, "target")
    return _run("bank", X_df, y, cat_cols, num_cols, latent_dim=latent_dim, save=save)


def run_retail(latent_dim: int = LATENT_DIM, save: bool = True):
    logger.info("Pipeline D -- Online Retail (temporal split)")
    rfm = load_retail_temporal()
    rfm = rfm.rename(columns={"target_bin": "target"})
    X_df, y, cat_cols, num_cols = _prepare(rfm, "target")
    return _run("retail", X_df, y, cat_cols, num_cols, latent_dim=latent_dim, save=save)


if __name__ == "__main__":
    run_bank()
    run_retail()
