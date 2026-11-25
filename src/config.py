"""Global configuration, deterministic seeding, and protocol constants for KAR-Net."""

import os
import random
import warnings

import numpy as np
import torch

if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

SEED = 42
N_SPLITS = 5
BATCH_SIZE = 256
EPOCHS_FINAL = 300
PATIENCE = 20
GRAD_CLIP = 1.0
MIN_LR = 1e-5
MAX_EMB = 256

# Cross-fitted rule discovery constants (Section 4.3, Supplementary Table S2)
CFRS_INNER_FOLDS = 5
CFRS_CANDIDATES_PER_INNER_FOLD = 256
CFRS_MAX_RULES = 128
CFRS_MIN_RULES = 32
CFRS_MIN_COVERAGE = 0.02
CFRS_MAX_COVERAGE = 0.60
CFRS_REDUNDANCY_JACCARD = 0.80
RULE_BASE_SIZE = CFRS_MAX_RULES

# Supervised training runs in full float32 precision for numerical stability.
USE_AMP_CUDA = False

try:
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
        PIN_MEMORY = True
        NUM_WORKERS = 2
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        PIN_MEMORY = False
        NUM_WORKERS = 2
    else:
        DEVICE = torch.device("cpu")
        PIN_MEMORY = False
        NUM_WORKERS = 0
except Exception:
    DEVICE = torch.device("cpu")
    PIN_MEMORY = False
    NUM_WORKERS = 0
    warnings.warn("Device detection failed; defaulting to CPU.")

# Single frozen global configuration applied unchanged across all six cohorts
# (Section 4.5 and Supplementary Table S2).
FROZEN_CONFIG_GLOBAL = {
    "pretrain_epochs": 50,
    "mask_prob": 0.15,
    "focal_gamma": 2.0,
    "weight_decay": 1e-5,
    "lr": 1e-3,
    "embed_dim": 128,
    "n_heads": 4,
    "n_layers": 3,
    "dropout": 0.1,
    "lambda_rank": 0.05,
    "rank_temperature": 1.0,
    "rank_max_pairs": 4096,
    "delta_max": 1.0,
    "consistency_lambda": 0.05,
    "consistency_threshold": 0.25,
    "consistency_temperature": 1.0,
    "rule_lambda": 0.05,
    "reliability_lambda": 0.05,
    "safeguard_lambda": 0.05,
    "safeguard_backbone_margin": 0.50,
    "safeguard_final_margin": 0.10,
    "stage1_epochs": 60,
    "stage2_epochs": 30,
    "stage3_epochs": 30,
    "stage1_lr": 2e-4,
    "stage2_rule_lr": 8e-4,
    "stage3_backbone_lr": 1e-5,
    "stage3_head_lr": 1e-4,
    "stage3_rule_lr": 8e-4,
}

# Deliberately empty: no per-cohort overrides are permitted.
FROZEN_CONFIGS = {}

ALL_COHORTS = [
    "NHANES_DIABETES",
    "NHANES_DIABETES_L",
    "NHANES_CKD",
    "NHANES_CKD_L",
    "NHANES_NAFLD",
    "NHANES_NAFLD_L",
]

NHANES_FILES_J = [
    "LUX_J.xpt", "DEMO_J.xpt", "BMX_J.xpt", "BIOPRO_J.xpt",
    "GHB_J.xpt", "GLU_J.xpt", "ALB_CR_J.xpt", "TRIGLY_J.xpt",
    "TCHOL_J.xpt", "DIQ_J.xpt", "HDL_J.xpt", "BPXO_J.xpt",
    "CBC_J.xpt", "ALQ_J.xpt", "HEPBD_J.xpt", "HEPC_J.xpt",
]

NHANES_FILES_L = [
    "LUX_L.xpt", "DEMO_L.xpt", "BMX_L.xpt", "BIOPRO_L.xpt",
    "GHB_L.xpt", "GLU_L.xpt", "ALB_CR_L.xpt", "TRIGLY_L.xpt",
    "TCHOL_L.xpt", "DIQ_L.xpt", "HDL_L.xpt", "BPXO_L.xpt",
    "CBC_L.xpt", "ALQ_L.xpt", "HEPBD_L.xpt", "HEPC_L.xpt",
]

ANALYSIS_PROTOCOL = {
    "cv_folds": N_SPLITS,
    "outer_training_split": "64% model training, 16% validation, 20% calibration",
    "hpo_policy": "none (single frozen global configuration; no per-cohort overrides)",
    "thresholding": "median of 1000 bootstrap thresholds maximizing balanced accuracy on cross-fitted calibration probabilities",
    "probability_calibration": "isotonic regression when calibration split N >= 200; Platt scaling otherwise",
    "checkpoint_deployment": "mean of sigmoid probabilities across top-3 validation checkpoints",
    "rule_discovery": (
        "5-inner-fold cross-fitted rule selection within the 64% model-training partition; "
        "50-tree balanced random forest (max_depth=5) prefix/leaf paths and univariate quantile stumps; "
        "inner coverage [0.01, 0.95], inner Jaccard 0.85, top 256 per inner fold, "
        "held-out coverage [0.02, 0.60], final same-polarity Jaccard 0.80 (relaxed to 0.95 if < 32), "
        "polarity cap 65%, max 128 rules"
    ),
    "evaluation_cycles": [
        "2017-2018 (NHANES suffix _J)",
        "August 2021-August 2023 (NHANES suffix _L)",
    ],
    "baselines": [
        "SVM", "RF", "XGB", "CatBoost", "TabNet", "TabTransformer",
        "Robust-GBDT", "MPCFIN", "TabM", "ExcelFormer",
    ],
    "architectural_variants": [
        "No Rules (Transformer_NoRules)",
        "ConCat Rules (Transformer_ConcatRules)",
        "KAR-Net Lam0 (Transformer_KARNet_Lam0)",
        "KAR-Net (Transformer_KARNet)",
    ],
    "frozen_config": dict(FROZEN_CONFIG_GLOBAL),
}


def seed_everything(seed: int = SEED) -> None:
    """Fix Python, NumPy, and PyTorch random seeds for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def experiment_seed(fold: int, backbone: str, stage: str) -> int:
    """Deterministic component seed independent of baseline execution order."""
    backbone_offset = {"transformer": 10000, "tabm": 20000}[str(backbone)]
    stage_offset = {
        "ssp": 100,
        "init": 200,
        "loader": 300,
        "backbone": 400,
        "rule": 500,
        "joint": 600,
    }[str(stage)]
    return int(SEED + 100000 * int(fold) + backbone_offset + stage_offset)


def make_torch_generator(seed: int) -> torch.Generator:
    """Create a seeded PyTorch CPU generator for deterministic DataLoader shuffling."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def verify_frozen_config(verbose: bool = True) -> bool:
    """Verify that the frozen global configuration is complete and has no overrides."""
    required = {
        "pretrain_epochs", "mask_prob", "focal_gamma", "weight_decay", "lr",
        "embed_dim", "n_heads", "n_layers", "dropout",
        "lambda_rank", "rank_temperature", "rank_max_pairs",
        "delta_max", "consistency_lambda", "consistency_threshold",
        "consistency_temperature", "rule_lambda", "reliability_lambda",
        "safeguard_lambda", "safeguard_backbone_margin", "safeguard_final_margin",
        "stage1_epochs", "stage2_epochs", "stage3_epochs",
        "stage1_lr", "stage2_rule_lr", "stage3_backbone_lr",
        "stage3_head_lr", "stage3_rule_lr",
    }
    missing = sorted(required - set(FROZEN_CONFIG_GLOBAL))
    extra = sorted(set(FROZEN_CONFIG_GLOBAL) - required)
    if missing or extra:
        raise RuntimeError(f"Frozen configuration mismatch: missing={missing}, extra={extra}")
    if FROZEN_CONFIGS:
        raise RuntimeError("Per-cohort configuration overrides are not permitted.")
    if verbose:
        print("[CONFIG] Frozen global configuration verified (no per-cohort overrides).")
    return True
