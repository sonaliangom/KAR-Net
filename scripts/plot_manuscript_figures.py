#!/usr/bin/env python3
"""Standalone script to regenerate the per-cohort symbolic audit figures and the per-cohort Figure 4 DCA panels from saved CSVs."""

import argparse
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import ALL_COHORTS
from src.visualization import (
    format_rule_human_readable,
    plot_case_study_figure,
    plot_dca_from_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate KAR-Net case-study and DCA figures from saved outputs.")
    parser.add_argument("--results-dir", type=str, default="./results", help="Directory containing saved run CSVs.")
    parser.add_argument("--figure-dir", type=str, default="./figures", help="Output directory for figures.")
    parser.add_argument(
        "--case-cohort",
        type=str,
        default=None,
        help="Render only this cohort's case-study figure (default: every cohort with a case CSV).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.figure_dir, exist_ok=True)

    # Case-study figures: one per cohort that produced a symbolic audit case
    cohorts_to_scan = [args.case_cohort] if args.case_cohort else ALL_COHORTS
    rendered_cases = 0
    for cohort in cohorts_to_scan:
        case_csv = os.path.join(args.results_dir, f"{cohort}_case_study_rules.csv")
        if not os.path.exists(case_csv):
            continue
        df_case = pd.read_csv(case_csv)
        df_case["Label"] = df_case["Rule"].apply(format_rule_human_readable)
        out_fig = os.path.join(args.figure_dir, f"Fig_CaseStudy_{cohort}.png")
        plot_case_study_figure(
            df_case,
            fold=int(df_case["Fold"].iloc[0]),
            p_base=float(df_case["Baseline_Prob"].iloc[0]),
            p_kar=float(df_case["KARNet_Prob"].iloc[0]),
            output_path=out_fig,
        )
        print(f"Saved case-study figure -> {out_fig}")
        rendered_cases += 1
        if cohort == "NHANES_NAFLD_L":
            manuscript_fig = os.path.join(args.figure_dir, "Fig3_Symbolic_Audit.png")
            shutil.copyfile(out_fig, manuscript_fig)
            print(f"Saved manuscript Figure 3 -> {manuscript_fig}")
    if rendered_cases == 0:
        print(f"[INFO] No *_case_study_rules.csv files found in {args.results_dir}; skipping case-study figures.")

    # Figure 4: one Decision Curve Analysis panel per cohort
    rendered = 0
    for cohort in ALL_COHORTS:
        csv_path = os.path.join(args.results_dir, f"{cohort}_dca_fold1.csv")
        if not os.path.exists(csv_path):
            continue
        out_png = os.path.join(args.figure_dir, f"Fig_DCA_{cohort}.png")
        plot_dca_from_csv(csv_path, out_png)
        print(f"Saved DCA panel -> {out_png}")
        rendered += 1
    if rendered == 0:
        print(f"[INFO] No *_dca_fold1.csv files found in {args.results_dir}; skipping DCA panels.")


if __name__ == "__main__":
    main()
