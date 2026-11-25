#!/usr/bin/env python3
"""Command-line entry point for reproducing KAR-Net experiments across the six NHANES cohorts."""

import argparse
import json
import os
import time
import warnings

import pandas as pd

from src.cohorts import (
    add_clinical_features,
    build_ckd_cohort_egfr_uacr,
    build_diabetes_cohort,
    build_nafld_cohort,
    load_nhanes_data,
)
from src.config import (
    ALL_COHORTS,
    ANALYSIS_PROTOCOL,
    NHANES_FILES_J,
    NHANES_FILES_L,
    SEED,
    seed_everything,
    verify_frozen_config,
)
from src.leakage import drop_identifier_columns, enforce_predictor_exclusions
from src.pipeline import run_experiment

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser(
        description="KAR-Net: Knowledge-Adaptive Reasoning Network for Auditable Neuro-Symbolic Tabular Learning"
    )
    parser.add_argument(
        "--cohorts",
        nargs="+",
        default=ALL_COHORTS,
        choices=ALL_COHORTS,
        help="One or more NHANES cohorts to run (default: all six cohorts).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/nhanes",
        help="Directory containing downloaded CDC NHANES .XPT files (default: ./data/nhanes).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory for saving manifests, rule audits, metrics, and figures (default: ./results).",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip external baselines and train only KAR-Net and its three architectural variants.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    seed_everything(SEED)
    verify_frozen_config(verbose=True)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "analysis_protocol.json"), "w") as f:
        json.dump(ANALYSIS_PROTOCOL, f, indent=2)

    selected = list(args.cohorts)
    need_j = any(not c.endswith("_L") for c in selected)
    need_l = any(c.endswith("_L") for c in selected)

    master_j = load_nhanes_data(NHANES_FILES_J, data_dir=args.data_dir, map_l_trigly=False) if need_j else None
    master_l = load_nhanes_data(NHANES_FILES_L, data_dir=args.data_dir, map_l_trigly=True) if need_l else None

    cohorts = {}
    if master_j is not None:
        if "NHANES_NAFLD" in selected:
            cohorts["NHANES_NAFLD"] = build_nafld_cohort(master_j, tag="J")
        if "NHANES_DIABETES" in selected:
            cohorts["NHANES_DIABETES"] = build_diabetes_cohort(master_j, cycle="J")
        if "NHANES_CKD" in selected:
            cohorts["NHANES_CKD"] = build_ckd_cohort_egfr_uacr(master_j, tag="J")
    if master_l is not None:
        if "NHANES_NAFLD_L" in selected:
            cohorts["NHANES_NAFLD_L"] = build_nafld_cohort(master_l, tag="L")
        if "NHANES_DIABETES_L" in selected:
            cohorts["NHANES_DIABETES_L"] = build_diabetes_cohort(master_l, cycle="L")
        if "NHANES_CKD_L" in selected:
            cohorts["NHANES_CKD_L"] = build_ckd_cohort_egfr_uacr(master_l, tag="L")

    all_records = []
    for name in selected:
        df = cohorts.get(name)
        if df is None or df.empty:
            print(f"[SKIP] {name}: cohort unavailable (check NHANES .XPT files in {args.data_dir}).")
            continue

        label_col = next((c for c in ("DIABETES", "CKD", "NAFLD") if c in df.columns), None)
        if label_col is None:
            continue

        X = df.drop(columns=[label_col]).copy()
        y = df[label_col].astype(int).copy()

        X = drop_identifier_columns(X, verbose=True)
        X = enforce_predictor_exclusions(X, name, stage="main predictor matrix")

        raw_cols = list(map(str, X.columns))
        X = add_clinical_features(X, name)
        X = enforce_predictor_exclusions(X, name, stage="post-engineering")
        X = X.loc[:, ~X.columns.duplicated()].copy()

        cat_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 20 and c != "SEQN"]
        cont_cols = [c for c in X.columns if c not in cat_cols and c != "SEQN"]

        data_dict = {
            "X": X,
            "y": y,
            "cat_cols": cat_cols,
            "cont_cols": cont_cols,
            "raw_cols": raw_cols,
        }
        records = run_experiment(
            name,
            data_dict,
            output_dir=args.output_dir,
            run_baselines=not args.skip_baselines,
        )
        all_records.extend(records)

    if all_records:
        acc_path = os.path.join(args.output_dir, "all_records_accumulated.csv")
        new_df = pd.DataFrame(all_records)
        if os.path.exists(acc_path):
            old_df = pd.read_csv(acc_path)
            old_df = old_df[~old_df["Dataset"].isin(new_df["Dataset"].unique())]
            full_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            full_df = new_df
        full_df.to_csv(acc_path, index=False)

    print(f"\nTotal execution time: {(time.time() - t0) / 60.0:.2f} minutes")


if __name__ == "__main__":
    main()
