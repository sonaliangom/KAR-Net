# KAR-Net: Knowledge-Adaptive Reasoning Network for Auditable Neuro-Symbolic Tabular Learning

Official implementation of **KAR-Net**, accompanying the manuscript:

> **KAR-Net: Knowledge-Adaptive Reasoning Network for Auditable Neuro-Symbolic Tabular Learning**  
> Angom Sonali Devi and Aheibam Dinamani Singh  
> Department of Electronics and Communication Engineering, National Institute of Technology Manipur, Imphal, India  
> *Submitted to Knowledge and Information Systems (Springer)*

## Overview

KAR-Net integrates cross-fitted symbolic rule discovery with a self-supervised multi-scale tabular Transformer through bounded neuro-symbolic residual arbitration:

1. **Outcome Separation & Partition Isolation (`src/leakage.py`, `src/cohorts.py`):** Outcome-defining variables, direct derivatives, proxies, and identifiers are excluded prior to preprocessing, rule discovery, and modeling, with automated assertions across all four pipeline stages and exported predictor/exclusion manifests.
2. **Cross-Fitted Rule Discovery (`src/rules.py`):** Within the 64% model-training partition of each outer fold, 5-inner-fold cross-fitting mines candidate rules from balanced random-forest paths and univariate quantile stumps, retaining up to $M \le 128$ polarity-balanced, redundancy-filtered rules with held-out directions and quality priors.
3. **Prior-Anchored Contextual Reliability & Bounded Residual Arbitration (`src/models.py`, `src/training.py`):** Detached latent patient context adjusts clipped held-out quality priors in logit space via a zero-initialized projector. Signed evidence drives bounded FiLM modulation and trust/data-confidence gating under an exact neutral-rule identity ($R(x_i) = \mathbf{0} \Rightarrow \Delta_i = 0$).
4. **Deployment & Calibration (`src/training.py`, `src/evaluation.py`):** Top-3 validation checkpoints are averaged at the sigmoid probability level, calibrated on the held-out calibration partition, and paired with bootstrap-selected balanced-accuracy thresholds.

## Repository Structure

```text
KAR-Net/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── run_karnet.py                      # Main CLI entry point for all 6 NHANES cohorts
├── src/
│   ├── __init__.py
│   ├── config.py                      # Frozen global configuration, protocol metadata, deterministic seeds
│   ├── leakage.py                     # 4-stage outcome-exclusion guards and manifest exporters
│   ├── cohorts.py                     # NHANES outer-join loader, cohort builders, clinical composites
│   ├── rules.py                       # Cross-fitted rule discovery (CFRS), clinical rendering, rule auditing
│   ├── models.py                      # PeriodicFeatureTokenizer, MultiScaleTransformer, AdaptiveRuleHead, KARNet, losses
│   ├── training.py                    # SSP pre-training, 3-stage training, checkpoint averaging, neutral-rule check
│   ├── baselines.py                   # External baselines (TabTransformer, Robust-GBDT, MPCFIN, TabM, ExcelFormer)
│   ├── evaluation.py                  # Calibration, bootstrap thresholding, metrics, Wilcoxon/BH-FDR, audit scanner
│   ├── pipeline.py                    # Stratified 5-fold outer cross-validation pipeline
│   └── visualization.py               # Human-readable rule formatter, Figure 3 (audit) and per-cohort Figure 4 (DCA) plotters
├── scripts/
│   └── plot_manuscript_figures.py     # Regenerate Figures 3 and 4 directly from saved CSV outputs
└── data/
    └── README.md                      # Official CDC NHANES download instructions
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Preparation

Download the 16 public `.XPT` files for each NHANES survey cycle (2017-2018 `_J` and August 2021-August 2023 `_L`) into `data/nhanes/`. See [`data/README.md`](data/README.md) for direct CDC links and the exact file list.

## Reproducing the Experiments

All six cohorts use the **exact same frozen global configuration** (`src/config.py`, Supplementary Table S2) with zero hyperparameter search or per-cohort overrides.

Run all six cohorts:
```bash
python run_karnet.py --data-dir ./data/nhanes --output-dir ./results
```

Or run a single cohort (for example, August 2021-August 2023 MASLD):
```bash
python run_karnet.py --cohorts NHANES_NAFLD_L --data-dir ./data/nhanes --output-dir ./results
```

Available cohort identifiers:
- `NHANES_DIABETES` (2017-2018, $N = 2{,}432$, 45 predictors)
- `NHANES_DIABETES_L` (Aug. 2021-Aug. 2023, $N = 3{,}208$, 45 predictors)
- `NHANES_CKD` (2017-2018, $N = 4{,}384$, 45 predictors)
- `NHANES_CKD_L` (Aug. 2021-Aug. 2023, $N = 4{,}853$, 45 predictors)
- `NHANES_NAFLD` (2017-2018, $N = 4{,}378$, 31 predictors)
- `NHANES_NAFLD_L` (Aug. 2021-Aug. 2023, $N = 5{,}007$, 31 predictors)

## Regenerating Manuscript Figures 3 and 4

Once cohort outputs are saved in `results/`, regenerate the human-readable symbolic audit figures for every cohort that produced a case (Figure 3 in the manuscript is the NAFLD L case) and the per-cohort Decision Curve Analysis panels (Figure 4), without retraining:

```bash
python scripts/plot_manuscript_figures.py --results-dir ./results --figure-dir ./figures
```
### Hardware & Reproducibility Note
All reported manuscript results were generated on an NVIDIA Tesla T4 GPU (Python/PyTorch, full `float32` precision for all supervised stages; CUDA AMP for SSP only) with fixed random seeds (`SEED = 42`). Minor fourth-decimal variations (typically $\le 0.0003$ AUC, such as between KAR-Net and TabM on Diabetes J) can occur across different GPU architectures or CUDA/PyTorch builds due to parallel GPU floating-point reduction order.

## Exported Artifacts per Cohort

For each cohort, `run_karnet.py` writes to `--output-dir`:
- `analysis_protocol.json`: frozen global configuration and protocol metadata
- `<COHORT>_predictor_manifest.csv`, `<COHORT>_predictor_list.csv`, `<COHORT>_exclusion_manifest.csv`: complete predictor and exclusion manifests
- `<COHORT>_final_rules_fold{1..5}.csv` and `<COHORT>_rule_audit_fold{1..5}.csv`: retained fold-specific rules in original clinical units and held-out audits
- `<COHORT>_rule_weight_mean_fold{1..5}.csv`: mean held-out contextual reliability weights
- `<COHORT>_per_fold_metrics.csv`, `<COHORT>_summary_ci.csv`, `<COHORT>_wilcoxon_fdr.csv`: five-fold metrics, 95% Student-$t$ intervals, and descriptive Wilcoxon/BH-FDR tests
- `<COHORT>_dca_fold1.csv` and `Fig_DCA_<COHORT>.png`: comparative Decision Curve Analysis
- `<COHORT>_case_study_rules.csv` and `Fig_CaseStudy_<COHORT>.png`: illustrative symbolic audit record

## License

Released under the MIT License (`LICENSE`).
