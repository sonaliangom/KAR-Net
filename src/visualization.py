"""Publication-grade plotting for Figure 3 (symbolic audit case study) and Figure 4 (Decision Curve Analysis)."""

import colorsys
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_PALETTE = {
    "KAR-Net": "#1b7837",
    "KAR-Net Lam0": "#3498db",
    "ConCat Rules": "#f1c40f",
    "No Rules": "#e74c3c",
    "TabNet": "#7f8c8d",
    "TabTransformer": "#5e3c99",
    "SVM": "#801770",
    "RF": "#b2182b",
    "CatBoost": "#762a83",
    "XGB": "#3b528b",
    "Robust-GBDT": "#a6611a",
    "MPCFIN": "#8e44ad",
    "TabM": "#d95f02",
    "ExcelFormer": "#2b83ba",
}

DEFAULT_COLOR = "#7f8c8d"


DCA_PALETTE = {
    "RF": "#C0392B", "Robust-GBDT": "#B9770E",
    "CatBoost": "#6C3483", "TabM": "#E67E22",
    "SVM": "#7B241C", "TabTransformer": "#1F3A93",
    "ExcelFormer": "#2E86C1", "XGB": "#566573",
}
DCA_LINESTYLES = [(0, (5, 2)), (0, (3, 1.5, 1, 1.5)), (0, (1, 1.3))]
DCA_KAR_COLOR = "#1E8449"
DCA_THRESHOLDS = np.linspace(0.01, 0.99, 99)

_CONTINUOUS_LABELS = {
    "BMXBMI": ("BMI", "kg/m$^2$"),
    "BMXWAIST": ("Waist circumference", "cm"),
    "RIDAGEYR": ("Age", "yr"),
    "INDFMPIR": ("Income-to-poverty ratio", ""),
    "LBXSATSI": ("ALT", "U/L"),
    "LBXSASSI": ("AST", "U/L"),
    "LBXSGTSI": ("GGT", "U/L"),
    "LBXTR": ("TG", "mg/dL"),
    "LBXTC": ("Total cholesterol", "mg/dL"),
    "LBDHDD": ("HDL", "mg/dL"),
    "LBXGLU": ("FPG", "mg/dL"),
    "LBXGH": ("HbA1c", "%"),
    "LBXSBU": ("BUN", "mg/dL"),
    "LBXSNASI": ("Sodium", "mmol/L"),
    "LBXSKSI": ("Potassium", "mmol/L"),
    "LBXSUA": ("Uric acid", "mg/dL"),
    "LBXHGB": ("Hemoglobin", "g/dL"),
    "LBXPLTSI": ("Platelets", r"$10^3$/$\mu$L"),
    "SBP_MEAN": ("Mean SBP", "mmHg"),
    "DBP_MEAN": ("Mean DBP", "mmHg"),
    "PULSE_PRESSURE": ("Pulse pressure", "mmHg"),
    "NON_HDL": ("Non-HDL", "mg/dL"),
    "TG_HDL_RATIO": ("TG/HDL ratio", ""),
    "AST_ALT_RATIO": ("AST/ALT ratio", ""),
    "FLI": ("Fatty liver index", ""),
    "HSI": ("Hepatic steatosis index", ""),
    "TyG": ("TyG index", ""),
    "FIB4": ("FIB-4", ""),
    "APRI": ("APRI", ""),
    "LAP": ("LAP", ""),
    "VAI": ("VAI", ""),
    "log_GGT": ("log GGT", ""),
    "log_ALT": ("log ALT", ""),
    "log_TG": ("log TG", ""),
}

_BINARY_FLAG_LABELS = {
    "BMI_ge_30": ("BMI ≥ 30.00 kg/m²", "BMI < 30.00 kg/m²"),
    "BMI_ge_27_5": ("BMI ≥ 27.50 kg/m²", "BMI < 27.50 kg/m²"),
    "BMI_ge_25": ("BMI ≥ 25.00 kg/m²", "BMI < 25.00 kg/m²"),
    "WC_high": ("high-risk waist circumference", "normal waist circumference"),
    "TG_high": ("TG ≥ 150 mg/dL", "TG < 150 mg/dL"),
    "HDL_low": ("low HDL", "normal HDL"),
    "HTN_FLAG": ("hypertension", "normotensive"),
    "HGB_low": ("low hemoglobin", "normal hemoglobin"),
    "UA_high": ("hyperuricemia", "normal uric acid"),
    "BUN_high": ("elevated BUN", "normal BUN"),
    "A1c_ge_6_5": ("HbA1c ≥ 6.5%", "HbA1c < 6.5%"),
    "Age_ge_60": ("Age ≥ 60 yr", "Age < 60 yr"),
    "Age_ge_50": ("Age ≥ 50 yr", "Age < 50 yr"),
    "Age_ge_45": ("Age ≥ 45 yr", "Age < 45 yr"),
}


def format_condition_human_readable(cond: str, compact: bool = False) -> str:
    """Convert one coded rule clause into clean clinical text for plotting."""
    cond = cond.strip()

    m_cat = re.match(r"^([A-Za-z0-9_]+)\s+in\s+\{([^}]*)\}$", cond)
    if m_cat:
        var, raw_vals = m_cat.group(1), m_cat.group(2)
        vals = {v.strip() for v in raw_vals.split(",") if v.strip()}
        if var in _BINARY_FLAG_LABELS:
            pos_str, neg_str = _BINARY_FLAG_LABELS[var]
            if vals <= {"1", "1.0"}:
                return pos_str.replace(" kg/m²", "") if (compact and var.startswith("BMI_")) else pos_str
            if vals <= {"0", "0.0"}:
                return neg_str
        if var == "DIQ010":
            if vals <= {"1", "1.0"}:
                return "self-reported diabetes"
            if vals <= {"2", "2.0"}:
                return "no self-reported diabetes"
        return f"{var} in {{{raw_vals}}}"

    
    m_num = re.match(r"^([A-Za-z0-9_]+)\s*(<=|>=|<|>)\s*([0-9.+\-eE]+)$", cond)
    if m_num:
        var, op, val = m_num.group(1), m_num.group(2), m_num.group(3)
        op_sym = "≤" if op == "<=" else ("≥" if op == ">=" else op)
        if var in _CONTINUOUS_LABELS:
            label, unit = _CONTINUOUS_LABELS[var]
            if compact and var == "BMXWAIST":
                label = "waist"
            return f"{label} {op_sym} {val}" + (f" {unit}" if unit else "")
        return f"{var} {op_sym} {val}"

    return cond


def format_rule_human_readable(rule_str: str) -> str:
    """Format a multi-clause clinical rule string with readable names, units, and balanced wrapping."""
    raw_clauses = [c.strip() for c in rule_str.split(" AND ") if c.strip()]
    compact = len(raw_clauses) >= 3
    readable = [format_condition_human_readable(c, compact=compact) for c in raw_clauses]

    if len(readable) == 1:
        s = readable[0]
        return s[0].upper() + s[1:] if s and s[0].islower() else s
    if len(readable) == 2:
        first = readable[0]
        first = first[0].upper() + first[1:] if first and first[0].islower() else first
        return f"{first}\nand {readable[1]}"

   
    first = readable[0]
    readable[0] = first[0].upper() + first[1:] if first and first[0].islower() else first
    if len(readable) == 3:
        if len(readable[0]) + len(readable[1]) <= 40:
            return f"{readable[0]}, {readable[1]},\nand {readable[2]}"
        return f"{readable[0]},\n{readable[1]},\nand {readable[2]}"
    if len(readable) == 4:
        return f"{readable[0]}, {readable[1]},\n{readable[2]}, and {readable[3]}"
    # 5+ clauses
    return (
        f"{readable[0]}, {readable[1]},\n"
        + ", ".join(readable[2:-1])
        + f",\nand {readable[-1]}"
    )


def _desaturate(color, prop: float = 0.25):
    """Seaborn-style desaturation, used to match the submitted bar colours."""
    r, g, b = color[:3]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, l, s * (1.0 - prop))


def plot_case_study_figure(
    rules_df: pd.DataFrame,
    fold: int,
    p_base: float,
    p_kar: float,
    output_path: str,
    top_k: int = 4,
) -> None:
   
    df_sorted = rules_df.sort_values("Weight").copy()
    if len(df_sorted) > 2 * top_k:
        df_plot = pd.concat([df_sorted.head(top_k), df_sorted.tail(top_k)]).drop_duplicates().sort_values("Weight")
    else:
        df_plot = df_sorted

    if "Label" not in df_plot.columns:
        df_plot["Label"] = df_plot["Rule"].apply(format_rule_human_readable)

    weights_arr = df_plot["Weight"].to_numpy(dtype=float)
    w_min, w_max = float(weights_arr.min()), float(weights_arr.max())
    denom = (w_max - w_min) if (w_max > w_min) else 1.0
    cmap = plt.get_cmap("RdYlGn_r")
    colors = [_desaturate(cmap(float((w - w_min) / denom)), 0.25) for w in weights_arr]

    fig = plt.figure(figsize=(10, 6))
   
    ax = fig.add_axes([0.3889, 0.0898, 0.6010, 0.7847])

    y_pos = np.arange(len(df_plot))
    ax.barh(y_pos, weights_arr, color=colors, height=0.8, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["Label"].tolist(), fontsize=10)
    
    ax.set_ylim(len(df_plot) - 0.5, -0.5)
    ax.set_title(
        f"Arbitration Mechanism: Illustrative Case (Fold {fold})\n"
        f"Baseline Prediction: {p_base:.2f} (High Risk)\n"
        f"KAR-Net Prediction: {p_kar:.2f} (Correctly Low)",
        fontweight="bold",
        fontsize=12,
    )
    ax.set_xlim(0.0, 1.1)
    ax.axvline(0.5, color="grey", linestyle="--")
    ax.set_xlabel("Contextual reliability weight ($\\alpha$)")
    ax.set_ylabel("")
    fig.text(0.008, 0.5, "Rule", rotation=90, va="center", fontsize=10)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def calculate_net_benefit(y_true, y_prob, thresholds: np.ndarray) -> np.ndarray:
    """Vickers & Elkin net benefit across decision thresholds."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = float(len(y_true))
    nb = np.empty(len(thresholds), dtype=float)
    for idx, t in enumerate(thresholds):
        if t >= 1.0:
            nb[idx] = 0.0
        else:
            pred = y_prob >= t
            tp = float(np.sum(pred & (y_true == 1)))
            fp = float(np.sum(pred & (y_true == 0)))
            nb[idx] = (tp / n) - (fp / n) * (t / (1.0 - t))
    return nb


def calculate_treat_all_net_benefit(y_true, thresholds: np.ndarray) -> np.ndarray:
    """Exact Vickers & Elkin Treat-All net benefit: prevalence - (1 - prevalence) * t / (1 - t)."""
    prev = float(np.mean(np.asarray(y_true, dtype=float)))
    return prev - (1.0 - prev) * (thresholds / np.maximum(1.0 - thresholds, 1e-9))


def _render_dca_panel(dca_df: pd.DataFrame, output_path: str) -> None:
    """Render one per-cohort DCA panel in the submitted manuscript style."""
    threshs = dca_df["Threshold"].values
    plt.figure(figsize=(8, 6))
    plt.plot(threshs, dca_df["Treat_All"].values, color="#7F8C8D", linestyle=":", linewidth=1.6, label="Treat All")
    plt.plot(threshs, dca_df["Treat_None"].values, color="#111111", linewidth=1.6, label="Treat None")
    plt.plot(threshs, dca_df["KAR-Net"].values, color=DCA_KAR_COLOR, linewidth=2.8, label="KAR-Net")

    alt_cols = [c for c in dca_df.columns if c not in ("Threshold", "Treat_All", "Treat_None", "KAR-Net")]
    for idx, col in enumerate(alt_cols):
        plt.plot(
            threshs,
            dca_df[col].values,
            color=DCA_PALETTE.get(col, DEFAULT_COLOR),
            linewidth=1.8,
            linestyle=DCA_LINESTYLES[idx % len(DCA_LINESTYLES)],
            label=col,
        )

    plt.ylim(-0.05, 0.4)
    plt.xlim(0.0, 0.8)
    plt.xticks(np.arange(0.0, 0.81, 0.1), fontsize=13)
    plt.yticks(np.arange(-0.05, 0.41, 0.05), fontsize=13)
    plt.xlabel("Threshold probability", fontsize=16, labelpad=6)
    plt.ylabel("Net benefit", fontsize=16, labelpad=6)
    plt.legend(fontsize=14, loc="upper right", frameon=True, framealpha=0.95,
               edgecolor="#CCCCCC", labelspacing=0.4, handlelength=2.6, borderpad=0.6)
    plt.grid(True, linestyle="--", alpha=0.30, linewidth=0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_single_dca(
    y_true,
    kar_probs,
    alt_probs_dict: dict,
    output_path: str,
    save_csv_path: str = None,
) -> pd.DataFrame:
    """Generate the DCA panel for the cohort currently running (as in the 10.x pipeline)."""
    threshs = DCA_THRESHOLDS
    dca_data = {
        "Threshold": threshs,
        "Treat_All": calculate_treat_all_net_benefit(y_true, threshs),
        "Treat_None": np.zeros_like(threshs),
        "KAR-Net": calculate_net_benefit(y_true, kar_probs, threshs),
    }
    for name, probs in alt_probs_dict.items():
        dca_data[name] = calculate_net_benefit(y_true, probs, threshs)
    dca_df = pd.DataFrame(dca_data)

    if save_csv_path:
        os.makedirs(os.path.dirname(save_csv_path) or ".", exist_ok=True)
        dca_df.to_csv(save_csv_path, index=False)

    _render_dca_panel(dca_df, output_path)
    return dca_df


def plot_dca_from_csv(csv_path: str, output_path: str) -> pd.DataFrame:
    """Re-render one cohort's DCA panel from its saved CSV, without retraining."""
    dca_df = pd.read_csv(csv_path)
    _render_dca_panel(dca_df, output_path)
    return dca_df
