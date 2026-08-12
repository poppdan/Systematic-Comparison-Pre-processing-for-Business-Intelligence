"""
pipeline_f.py -- Pipeline F: Feature Tokenizer Transformer (FT-Transformer).

Based on: Gorishniy et al. (2021) "Revisiting deep learning models for tabular data"

Architecture:
  - Feature Tokenizer: each numeric/categorical feature -> d_token-dim embedding
    - Numeric features: linear projection (value * weight + bias)
    - Categorical features: lookup embedding table
  - [CLS] token prepended to the sequence
  - Transformer encoder: n_layers x (Multi-Head Self-Attention + FFN)
  - Output: [CLS] representation used as the latent feature vector

Usage as Preprocessing-Operator (not end-to-end classifier):
  - The Transformer is trained with a classification head (cross-entropy)
  - At inference the [CLS] embedding (before head) is extracted as features
  - Downstream classifiers (LR, RF, XGB, LGBM, MLP) are trained on these features

Saves to data/processed/{dataset}/f/
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
                    BATCH_SIZE, EPOCHS, PATIENCE, LR,
                    TEST_SIZE, SEED)
from scripts.utils import get_logger, ensure_dirs
from scripts.classical.missing_values import handle_missing
from scripts.classical.pipeline_a import load_retail_temporal, RETAIL_CUTOFF

logger = get_logger("pipeline_f")
ensure_dirs(DATA_PROC)

torch.manual_seed(SEED)
np.random.seed(SEED)

# FT-Transformer hyperparameters
D_TOKEN    = 64     # embedding dimension per feature
N_HEADS    = 8      # attention heads
N_LAYERS   = 3      # transformer encoder layers
FFN_FACTOR = 4 / 3  # FFN hidden dim = int(D_TOKEN * FFN_FACTOR)
DROPOUT    = 0.1


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class NumericalEmbedding(nn.Module):
    """Linear projection of a scalar numeric feature -> d_token dim."""
    def __init__(self, n_num: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_num, d_token))
        self.bias   = nn.Parameter(torch.empty(n_num, d_token))
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: (B, n_num)  ->  (B, n_num, d_token)
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class CategoricalEmbedding(nn.Module):
    """Lookup embedding for each categorical feature."""
    def __init__(self, cardinalities: list, d_token: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(c + 1, d_token) for c in cardinalities]
        )

    def forward(self, x_cat):
        # x_cat: (B, n_cat)  ->  (B, n_cat, d_token)
        return torch.stack([emb(x_cat[:, i])
                            for i, emb in enumerate(self.embeddings)], dim=1)


class FTTransformer(nn.Module):
    def __init__(self, n_num: int, cardinalities: list,
                 d_token: int = D_TOKEN, n_heads: int = N_HEADS,
                 n_layers: int = N_LAYERS, ffn_factor: float = FFN_FACTOR,
                 dropout: float = DROPOUT, n_classes: int = 2):
        super().__init__()
        self.n_num = n_num
        self.n_cat = len(cardinalities)

        self.num_emb = NumericalEmbedding(n_num, d_token) if n_num > 0 else None
        self.cat_emb = CategoricalEmbedding(cardinalities, d_token) if self.n_cat > 0 else None

        # [CLS] token
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls_token)

        ffn_dim = max(int(d_token * ffn_factor), 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True,
            norm_first=True,          # Pre-LN (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_token, n_classes)

    def _tokenize(self, x_num, x_cat):
        tokens = []
        if self.num_emb is not None and x_num is not None:
            tokens.append(self.num_emb(x_num))    # (B, n_num, d)
        if self.cat_emb is not None and x_cat is not None:
            tokens.append(self.cat_emb(x_cat))    # (B, n_cat, d)
        x = torch.cat(tokens, dim=1)               # (B, n_num+n_cat, d)

        cls = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, d)
        return torch.cat([cls, x], dim=1)                # (B, 1+n, d)

    def encode(self, x_num, x_cat):
        """Return [CLS] embedding -- used as preprocessing output."""
        tokens = self._tokenize(x_num, x_cat)
        out    = self.transformer(tokens)
        return out[:, 0, :]   # (B, d_token)

    def forward(self, x_num, x_cat):
        cls_emb = self.encode(x_num, x_cat)
        return self.head(cls_emb)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _prepare(df: pd.DataFrame, target_col: str):
    """Impute + detect col types. Returns DataFrame X, y, col lists (no fitting)."""
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].values
    X, _, _ = handle_missing(X)
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    return X, y, cat_cols, num_cols


def _fit_transform_ftt(X_tr_df, X_te_df, cat_cols, num_cols):
    """Fit encoders+scaler on train only; return num/cat numpy arrays for both splits."""
    X_tr = X_tr_df.copy()
    X_te = X_te_df.copy()

    encoders = {}
    cardinalities = []
    for col in cat_cols:
        le = LabelEncoder()
        X_tr[col] = le.fit_transform(X_tr[col].astype(str))
        mapping = {c: i for i, c in enumerate(le.classes_)}
        X_te[col] = X_te[col].astype(str).map(lambda v: mapping.get(v, 0))
        encoders[col] = le
        cardinalities.append(int(X_tr[col].max()))

    scaler = StandardScaler()
    if num_cols:
        X_tr[num_cols] = scaler.fit_transform(X_tr[num_cols])
        X_te[num_cols] = scaler.transform(X_te[num_cols])

    X_num_tr = X_tr[num_cols].values.astype(np.float32) if num_cols else None
    X_num_te = X_te[num_cols].values.astype(np.float32) if num_cols else None
    X_cat_tr = X_tr[cat_cols].values.astype(np.int64)   if cat_cols else None
    X_cat_te = X_te[cat_cols].values.astype(np.int64)   if cat_cols else None

    return X_num_tr, X_num_te, X_cat_tr, X_cat_te, scaler, encoders, cardinalities


def _train_ftt(X_num, X_cat, y_train,
               cardinalities, device: str = "cpu",
               epochs: int = EPOCHS, patience: int = PATIENCE,
               val_frac: float = 0.15, seed: int = SEED) -> FTTransformer:
    """
    Train the FT-Transformer with early stopping on a held-out VALIDATION split.

    IMPORTANT: this encoder is trained SUPERVISED (cross-entropy on y_train).
    Early stopping on the *training* loss lets it memorise the labels, which
    makes the extracted [CLS] embedding leak label information for every row
    it was trained on. A held-out validation split is required so that
    training stops before memorisation.

    The caller is still responsible for ensuring the encoder is never trained
    on rows that will later be used for evaluation.
    """
    n_num = X_num.shape[1] if X_num is not None else 0

    # --- internal stratified train/validation split for early stopping -----
    rng = np.random.default_rng(seed)
    y_arr = y_train.astype(np.int64)
    val_idx = []
    for cls in np.unique(y_arr):
        cls_idx = np.where(y_arr == cls)[0]
        rng.shuffle(cls_idx)
        n_val_c = max(int(val_frac * len(cls_idx)), 1)
        val_idx.append(cls_idx[:n_val_c])
    val_idx = np.concatenate(val_idx)
    tr_mask = np.ones(len(y_arr), dtype=bool)
    tr_mask[val_idx] = False
    tr_idx = np.where(tr_mask)[0]

    def _slice(arr, idx):
        return None if arr is None else arr[idx]

    model = FTTransformer(n_num=n_num, cardinalities=cardinalities).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    t_num = _slice(X_num, tr_idx); t_cat = _slice(X_cat, tr_idx)
    t_num = torch.from_numpy(t_num).to(device) if t_num is not None else None
    t_cat = torch.from_numpy(t_cat).to(device) if t_cat is not None else None
    t_y   = torch.from_numpy(y_arr[tr_idx]).to(device)

    v_num = _slice(X_num, val_idx); v_cat = _slice(X_cat, val_idx)
    v_num = torch.from_numpy(v_num).to(device) if v_num is not None else None
    v_cat = torch.from_numpy(v_cat).to(device) if v_cat is not None else None
    v_y   = torch.from_numpy(y_arr[val_idx]).to(device)

    n = len(t_y)
    best_loss = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            b_num = t_num[idx] if t_num is not None else None
            b_cat = t_cat[idx] if t_cat is not None else None
            b_y   = t_y[idx]

            logits = model(b_num, b_cat)
            loss   = criterion(logits, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- validation cross-entropy ---
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(v_num, v_cat), v_y).item()

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


def _encode(model: FTTransformer, X_num, X_cat, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        t_num = torch.from_numpy(X_num).to(device) if X_num is not None else None
        t_cat = torch.from_numpy(X_cat).to(device) if X_cat is not None else None
        emb   = model.encode(t_num, t_cat)
    return emb.cpu().numpy()


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------

def _run(dataset: str, X_df: pd.DataFrame, y: np.ndarray,
         cat_cols, num_cols,
         save: bool = True):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Pipeline F ({dataset}) -- device: {device}")

    tracemalloc.start()
    t_start = time.perf_counter()

    # Stratified split, then fit encoders/scaler on train only
    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx, test_size=TEST_SIZE,
                                       stratify=y, random_state=SEED)
    y_train, y_test = y[idx_tr], y[idx_te]

    (X_num_tr, X_num_te, X_cat_tr, X_cat_te,
     scaler, encoders, cardinalities) = _fit_transform_ftt(
        X_df.iloc[idx_tr], X_df.iloc[idx_te], cat_cols, num_cols
    )

    model = _train_ftt(X_num_tr, X_cat_tr, y_train, cardinalities, device)

    Z_train = _encode(model, X_num_tr, X_cat_tr, device)
    Z_test  = _encode(model, X_num_te, X_cat_te, device)

    elapsed = time.perf_counter() - t_start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    logger.info(f"Pipeline F ({dataset}): "
                f"{Z_train.shape[1]} token features | "
                f"{elapsed:.1f}s | {peak_mem/1e6:.1f} MB")

    if save:
        out_dir = DATA_PROC / dataset / "f"
        ensure_dirs(out_dir)

        col_names = [f"ftt_{i}" for i in range(Z_train.shape[1])]
        pd.DataFrame(Z_train, columns=col_names).assign(target=y_train)\
          .to_csv(out_dir / "train.csv", index=False)
        pd.DataFrame(Z_test, columns=col_names).assign(target=y_test)\
          .to_csv(out_dir / "test.csv", index=False)

        joblib.dump(model, out_dir / "ftt_model.pkl")
        joblib.dump(scaler, out_dir / "scaler.pkl")
        joblib.dump(encoders, out_dir / "encoders.pkl")

        meta = {
            "pipeline":      "F_FTTransformer",
            "dataset":       dataset,
            "d_token":       D_TOKEN,
            "n_heads":       N_HEADS,
            "n_layers":      N_LAYERS,
            "n_num":         len(num_cols),
            "n_cat":         len(cat_cols),
            "output_dim":    D_TOKEN,
            "n_train":       int(len(y_train)),
            "n_test":        int(len(y_test)),
            "runtime_s":     round(elapsed, 2),
            "peak_mem_mb":   round(peak_mem / 1e6, 2),
            "cat_cols":      cat_cols,
            "num_cols":      num_cols,
        }
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved Pipeline F ({dataset}) -> {out_dir}")

    return Z_train, Z_test, y_train, y_test


def run_bank(save: bool = True):
    logger.info("Pipeline F -- Bank Marketing")
    df = pd.read_csv(BANK_RAW, sep=";")
    df = df.rename(columns={"y": "target"})
    df["target"] = (df["target"] == "yes").astype(int)
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])
    X_df, y, cat_cols, num_cols = _prepare(df, "target")
    return _run("bank", X_df, y, cat_cols, num_cols, save=save)


def run_retail(save: bool = True):
    logger.info("Pipeline F -- Online Retail (temporal split)")
    rfm = load_retail_temporal()
    rfm = rfm.rename(columns={"target_bin": "target"})
    X_df, y, cat_cols, num_cols = _prepare(rfm, "target")
    return _run("retail", X_df, y, cat_cols, num_cols, save=save)


if __name__ == "__main__":
    run_bank()
    run_retail()
