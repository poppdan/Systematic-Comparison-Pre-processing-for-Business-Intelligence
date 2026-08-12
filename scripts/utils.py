"""
utils.py -- Shared utility functions for all scripts.
"""
import random
import logging
import json
from pathlib import Path
from datetime import datetime

import numpy as np


# -- Seed ----------------------------------------------------------
def set_seed(seed: int = 42):
    """Fix all random generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


# -- Logging -------------------------------------------------------
def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Return a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# -- Save results --------------------------------------------------
def save_results(results: dict, path: Path):
    """Save a dict as JSON (with timestamp)."""
    results["_timestamp"] = datetime.now().isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# -- Create directories --------------------------------------------
def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
