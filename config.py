"""
config.py -- Global constants for the master's thesis experiments.
All scripts import from here.
"""
from pathlib import Path

# -- Paths ------------------------------------------------------------------
ROOT        = Path(__file__).parent
DATA_RAW    = ROOT / "data" / "raw"
DATA_PROC   = ROOT / "data" / "processed"
MODELS_DIR  = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# Raw data files (copy to data/raw/)
BANK_RAW    = DATA_RAW / "bank_marketing.csv"      # delimiter: ;
RETAIL_RAW  = DATA_RAW / "online_retail.xlsx"

# -- Reproducibility --------------------------------------------------------
SEED = 42

# -- Cross-validation -------------------------------------------------------
N_FOLDS   = 5
N_REPEATS = 3   # -> 15 CV splits total

# -- Hyperparameter optimization --------------------------------------------
N_OPTUNA_TRIALS = 20

# -- Autoencoder / network training -----------------------------------------
LATENT_DIM  = 16
BATCH_SIZE  = 256
EPOCHS      = 100
PATIENCE    = 10    # Early stopping
LR          = 1e-3

# -- Evaluation -------------------------------------------------------------
TEST_SIZE       = 0.2
PRIMARY_METRIC  = "PR_AUC"   # Primary metric (important due to imbalance)
ALPHA           = 0.05        # Significance level Wilcoxon test

# -- Pipelines --------------------------------------------------------------
# Pipeline B is split into two PCA variants (90% and 20% variance thresholds)
CLASSICAL_PIPELINES = ["a", "b_pca90", "b_pca20", "c"]
AI_PIPELINES        = ["D", "E", "F"]
DOWNSTREAM_MODELS   = ["lr", "rf", "xgb", "lgbm", "mlp"]
DATASETS            = ["bank", "retail"]
