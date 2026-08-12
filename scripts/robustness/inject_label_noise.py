"""
inject_label_noise.py -- Synthetic label noise injection.

Flips a fraction of training labels (0->1 or 1->0) to simulate
annotation errors, self-reporting bias, or measurement noise.

Two modes:
  "symmetric"   -- each label independently flipped with prob noise_rate
  "asymmetric"  -- only positive labels (1->0) are flipped (simulates
                  false negatives, common in imbalanced BI datasets)

Usage:
  from scripts.robustness.inject_label_noise import inject_label_noise

  y_noisy = inject_label_noise(y_train, noise_rate=0.10, mode="symmetric", seed=42)
"""
import numpy as np


def inject_label_noise(y: np.ndarray,
                        noise_rate: float = 0.10,
                        mode: str = "symmetric",
                        seed: int = 42) -> np.ndarray:
    """
    Flip labels to inject noise.

    Parameters
    ----------
    y          : Binary label array (0/1)
    noise_rate : Fraction of labels to flip
    mode       : "symmetric" or "asymmetric" (only positives flipped)
    seed       : Random seed

    Returns
    -------
    Noisy label array
    """
    rng = np.random.default_rng(seed)
    y_noisy = y.copy().astype(int)

    if mode == "symmetric":
        flip_mask = rng.random(len(y_noisy)) < noise_rate
        y_noisy[flip_mask] = 1 - y_noisy[flip_mask]

    elif mode == "asymmetric":
        pos_idx = np.where(y_noisy == 1)[0]
        n_flip  = max(0, int(np.ceil(noise_rate * len(pos_idx))))
        flip_idx = rng.choice(pos_idx, size=n_flip, replace=False)
        y_noisy[flip_idx] = 0

    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'symmetric' or 'asymmetric'.")

    actual_rate = (y_noisy != y).mean()
    return y_noisy


def sweep_noise_rates(y: np.ndarray,
                       rates: list = None,
                       mode: str = "symmetric",
                       seed: int = 42) -> dict:
    """
    Inject label noise at multiple rates for robustness curves.

    Parameters
    ----------
    y     : Binary label array
    rates : List of noise rates (default: [0.05, 0.10, 0.20, 0.30])
    mode  : "symmetric" or "asymmetric"
    seed  : Random seed

    Returns
    -------
    dict {rate: noisy_label_array}
    """
    if rates is None:
        rates = [0.05, 0.10, 0.20, 0.30]
    return {r: inject_label_noise(y, noise_rate=r, mode=mode, seed=seed)
            for r in rates}
