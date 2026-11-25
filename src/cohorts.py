"""NHANES data assembly, cohort builders, and deterministic clinical composites."""

import os
import re
from typing import List, Optional

import numpy as np
import pandas as pd

from src.config import N_SPLITS
from src.leakage import DIABETES_DIRECT_EXCLUSIONS, drop_identifier_columns

ENGINEERED_FEATURES_BY_TASK = {
    "NHANES_DIABETES": [
        "TG_HDL_RATIO", "NON_HDL", "log_TG", "log_GGT", "log_ALT",
        "AST_ALT_RATIO", "LAP", "VAI", "SBP_MEAN", "DBP_MEAN", "HTN_FLAG",
        "BMI_ge_25", "BMI_ge_30", "WC_high", "HGB_low", "UA_high",
        "Age_ge_45", "Age_ge_60", "AgexSex",
    ],
    "NHANES_CKD": [
        "HGB_low", "Na_low", "Na_high", "K_low", "K_high",
        "SBP_MEAN", "DBP_MEAN", "PULSE_PRESSURE", "HTN_FLAG",
        "HDL_low", "NON_HDL", "BMI_ge_25", "UA_high",
        "BUN_high", "A1c_ge_6_5", "Age_ge_60", "AgexSex",
    ],
    "NHANES_NAFLD": [
        "FIB4", "APRI", "AST_ALT_RATIO", "log_GGT", "log_ALT",
        "TG_high", "HDL_low", "TG_HDL_RATIO", "NON_HDL", "TyG",
        "HSI", "FLI", "BMI_ge_27_5", "BMI_ge_30", "WC_high",
        "Age_ge_50", "Age_ge_60",
    ],
}


def load_nhanes_data(file_list: List[str], data_dir: str = "./data/nhanes", map_l_trigly: bool = False) -> Optional[pd.DataFrame]:
    """Load NHANES SAS transport (.XPT) files and assemble a validated one-to-one outer-joined table."""
    paths = []
    for fname in file_list:
        candidate = os.path.join(data_dir, fname)
        if os.path.exists(candidate):
            paths.append(candidate)
        elif os.path.exists(fname):
            paths.append(fname)

    if not paths:
        print(f"[DATA] No NHANES .XPT files found in '{data_dir}'.")
        return None

    dfs = []
    for path in paths:
        df = pd.read_sas(path, format="xport")
        df.columns = [str(c) for c in df.columns]
        dfs.append(df)
        print(f"Loaded {os.path.basename(path)}: shape={df.shape}")

    master = None
    for df in dfs:
        if "SEQN" not in df.columns:
            continue
        df = df.copy()
        if df["SEQN"].duplicated().any():
            df = df.drop_duplicates(subset=["SEQN"], keep="first")

        if master is None:
            master = df
            continue

        overlapping_columns = [c for c in df.columns if c != "SEQN" and c in master.columns]
        new_columns = ["SEQN"] + [c for c in df.columns if c not in master.columns]

        master = pd.merge(
            master,
            df[new_columns],
            on="SEQN",
            how="outer",
            validate="one_to_one",
        )

        if overlapping_columns:
            incoming_data = df.set_index("SEQN")
            master_data = master.set_index("SEQN")
            for column in overlapping_columns:
                incoming_values = incoming_data[column].reindex(master_data.index)
                master_data[column] = master_data[column].combine_first(incoming_values)
            master = master_data.reset_index()

    if master is None or master.empty:
        return None

    n_raw = len(master)
    if master.duplicated().any():
        master = master.drop_duplicates()
    if "SEQN" in master.columns and master["SEQN"].duplicated().any():
        master = master.drop_duplicates(subset=["SEQN"], keep="first")
    if "SEQN" in master.columns:
        assert master["SEQN"].is_unique, "SEQN must be unique after outer merge."
        print(f"[DATA] Merge rows {n_raw} -> {len(master)} (unique SEQN={master['SEQN'].nunique()})")

    for c in master.columns:
        if master[c].dtype.kind in "bifc":
            continue
        try:
            master[c] = master[c].astype(str).str.strip()
        except Exception:
            pass

    if map_l_trigly and "LBXTR" not in master.columns and "LBXTLG" in master.columns:
        master["LBXTR"] = master["LBXTLG"]
        print("[DATA] Mapped LBXTLG -> LBXTR for 2021-2023 triglyceride features.")

    print(f"[DATA] Master shape: {master.shape}")
    return master


def _heavy_alcohol_flag(master: pd.DataFrame) -> Optional[pd.Series]:
    """Identify heavy alcohol consumption (>14 drinks/week women, >21 drinks/week men)."""
    if not {"ALQ121", "ALQ130", "RIAGENDR"}.issubset(set(master.columns)):
        return None

    freq_per_week = {
        0: 0.0,
        1: 7.0,
        2: 6.0,
        3: 3.5,
        4: 2.0,
        5: 1.0,
        6: 2.5 / 4.345,
        7: 1.0 / 4.345,
        8: 9.0 / 52.0,
        9: 4.5 / 52.0,
        10: 1.5 / 52.0,
    }
    freq_code = pd.to_numeric(master["ALQ121"], errors="coerce")
    freq_code = freq_code.where(freq_code.isin(list(freq_per_week)), np.nan)
    occasions_per_week = freq_code.map(freq_per_week)

    drinks = pd.to_numeric(master["ALQ130"], errors="coerce")
    drinks = drinks.where((drinks >= 1.0) & (drinks <= 15.0), np.nan)
    drinks_per_week = occasions_per_week * drinks

    sex_female = pd.to_numeric(master["RIAGENDR"], errors="coerce") == 2.0
    heavy = np.where(sex_female, drinks_per_week > 14.0, drinks_per_week > 21.0)
    return pd.Series(heavy, index=master.index).astype(bool)


def _hepatitis_positive_flag(master: pd.DataFrame) -> Optional[pd.Series]:
    """Identify participants with positive HBV surface antigen or confirmed HCV markers."""
    if not {"LBDHCI", "LBXHCR", "LBDHBG"}.issubset(set(master.columns)):
        return None
    hcv_ab = pd.to_numeric(master["LBDHCI"], errors="coerce")
    hcv_rna = pd.to_numeric(master["LBXHCR"], errors="coerce")
    hbv_sag = pd.to_numeric(master["LBDHBG"], errors="coerce")
    return hcv_ab.isin([1.0, 4.0]) | (hcv_rna == 1.0) | (hbv_sag == 1.0)


def _egfr_ckdepi_2021(scr, age, sex_female):
    """Race-free 2021 CKD-EPI creatinine equation (mL/min/1.73 m^2)."""
    scr = np.asarray(scr, dtype=float)
    age = np.asarray(age, dtype=float)
    sex_female = np.asarray(sex_female).astype(bool)
    k = np.where(sex_female, 0.7, 0.9)
    a = np.where(sex_female, -0.241, -0.302)
    min_term = np.minimum(scr / k, 1.0) ** a
    max_term = np.maximum(scr / k, 1.0) ** (-1.200)
    sex_term = np.where(sex_female, 1.012, 1.0)
    return 142.0 * min_term * max_term * (0.9938 ** age) * sex_term


def _compute_uacr_mg_per_g(urxuma_ug_per_ml, urxucr_mg_per_dl):
    """Compute urinary albumin-to-creatinine ratio (UACR) in mg/g."""
    albumin_mg_per_dl = np.asarray(urxuma_ug_per_ml, dtype=float) * 0.1
    creat_g_per_dl = np.asarray(urxucr_mg_per_dl, dtype=float) / 1000.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return albumin_mg_per_dl / creat_g_per_dl


def build_nafld_cohort(master: pd.DataFrame, tag: str = "J") -> pd.DataFrame:
    """Build VCTE CAP-defined MASLD cohort (CAP >= 274 dB/m, +/-15 dB/m gray-zone trim)."""
    df = master.copy()
    cols = set(df.columns)

    cap_candidates = [c for c in cols if re.fullmatch(r"LUXCAPM|LUXCAPMED|LUXCAP", c, flags=re.I)]
    cap_col = cap_candidates[0] if cap_candidates else None
    iqrm_candidates = [c for c in cols if re.search(r"IQR.?/?M|SIQRM|IQR_PCT", c, flags=re.I)]
    iqrm_col = iqrm_candidates[0] if iqrm_candidates else None
    nvalid_candidates = [c for c in cols if re.search(r"LUANMVGP|N.?valid|NMEAS", c, flags=re.I)]
    nvalid_col = nvalid_candidates[0] if nvalid_candidates else None
    status_candidates = [c for c in cols if re.search(r"LUAXSTAT|EXAMSTAT|STATUS", c, flags=re.I)]
    status_col = status_candidates[0] if status_candidates else None

    if not cap_col:
        return pd.DataFrame()

    keep = ["SEQN", cap_col]
    for c in (iqrm_col, nvalid_col, status_col):
        if c:
            keep.append(c)

    tmp = df[keep].copy()
    tmp["_cap"] = pd.to_numeric(tmp[cap_col], errors="coerce")
    tmp["_cap_ok"] = tmp["_cap"].notna()

    if nvalid_col:
        tmp["_nvalid"] = pd.to_numeric(tmp[nvalid_col], errors="coerce")
        tmp["_nvalid10_ok"] = tmp["_nvalid"] >= 10
        tmp["_nvalid5_ok"] = tmp["_nvalid"] >= 5
    else:
        tmp["_nvalid10_ok"] = tmp["_nvalid5_ok"] = True

    if iqrm_col:
        tmp["_iqrm"] = pd.to_numeric(tmp[iqrm_col], errors="coerce") / 100.0
        tmp["_iqrm30_ok"] = tmp["_iqrm"] <= 0.30
        tmp["_iqrm40_ok"] = tmp["_iqrm"] <= 0.40
    else:
        tmp["_iqrm30_ok"] = tmp["_iqrm40_ok"] = True

    if status_col:
        tmp["_status_code"] = pd.to_numeric(tmp[status_col], errors="coerce")
        tmp["_status_ok"] = tmp["_status_code"] == 1
    else:
        tmp["_status_ok"] = True

    heavy = _heavy_alcohol_flag(master)
    hep_pos = _hepatitis_positive_flag(master)
    if heavy is None or hep_pos is None:
        raise RuntimeError(
            "MASLD eligibility files missing (ALQ, HEPBD, and HEPC are required)."
        )

    n_alc = int(np.asarray(heavy).sum())
    n_hep = int(np.asarray(hep_pos).sum())
    n_both = int(np.asarray(heavy & hep_pos).sum())
    print(f"[MASLD {tag}] Exclusions: heavy_alcohol={n_alc}, viral_hepatitis={n_hep}, both={n_both}")
    eligible_excl_idx = df.index[~np.asarray(heavy | hep_pos)]

    m_strict = tmp["_cap_ok"] & tmp["_nvalid10_ok"] & tmp["_iqrm30_ok"] & tmp["_status_ok"]
    m_relax = tmp["_cap_ok"] & tmp["_nvalid5_ok"] & tmp["_iqrm30_ok"] & tmp["_status_ok"]
    m_prag = tmp["_cap_ok"] & tmp["_nvalid5_ok"] & tmp["_iqrm40_ok"] & tmp["_status_ok"]

    min_cohort_size = N_SPLITS * 5
    for gate_name, m in [("Strict", m_strict), ("Relaxed", m_relax), ("Pragmatic", m_prag)]:
        eligible = tmp.index[m & tmp.index.isin(eligible_excl_idx)]
        if len(eligible) >= min_cohort_size:
            used_gate = gate_name
            tmp = tmp.loc[eligible].copy()
            break
    else:
        used_gate = "Fallback"
        tmp = tmp.loc[tmp.index.isin(eligible_excl_idx) & tmp["_cap_ok"]].copy()

    print(f"[MASLD {tag}] Quality gate={used_gate} | n after gates={len(tmp)}")
    caps = tmp["_cap"]

    final_cutoff = None
    for co in (274.0, 300.0, 248.0):
        labels = (caps >= co).astype(int)
        cc = np.bincount(labels)
        if len(cc) == 2 and np.min(cc) >= N_SPLITS:
            final_cutoff = co
            break
    if final_cutoff is None:
        return pd.DataFrame()

    gray = 15.0
    in_gray = (caps >= final_cutoff - gray) & (caps <= final_cutoff + gray)
    if in_gray.any():
        tmp = tmp.loc[~in_gray].copy()
        caps = tmp["_cap"]

    tmp["NAFLD"] = (caps >= final_cutoff).astype(int)

    base_pred_cols = [
        c for c in master.columns
        if c in (
            "BMXBMI", "BMXWAIST", "RIDAGEYR", "RIAGENDR",
            "LBXTR", "LBXTC", "LBDHDD", "INDFMPIR",
            "LBXSASSI", "LBXSATSI", "LBXSGTSI", "LBXGLU",
            "DIQ010", "LBXPLTSI",
        )
    ]
    out = master.loc[tmp.index, ["SEQN"] + base_pred_cols].copy()
    out["NAFLD"] = tmp["NAFLD"]
    out = out.dropna(subset=["NAFLD"])
    out = drop_identifier_columns(out, verbose=False)

    banned = [
        c for c in out.columns
        if any(k in str(c).upper() for k in ("CAP", "STIFF", "KPA", "IQR", "NVALID", "STATUS"))
    ]
    assert not banned, f"Elastography variables present as predictors: {banned}"
    print(f"[MASLD {tag}] Cohort shape={out.shape}, cutoff={final_cutoff}")
    return out


def build_diabetes_cohort(master: pd.DataFrame, cycle: str = "J") -> pd.DataFrame:
    """Build single-visit biomarker-defined adult diabetes cohort (FPG >= 126 or HbA1c >= 6.5%)."""
    df = master.copy()
    if "LBXGH" not in df.columns or "LBXGLU" not in df.columns:
        return pd.DataFrame()

    pred_candidates = [
        "SEQN", "BMXBMI", "BMXWAIST", "LBXTR", "LBXTC", "LBDHDD",
        "RIDAGEYR", "RIAGENDR", "INDFMPIR", "RIDRETH1", "DMDEDUC2",
        "LBXSASSI", "LBXSATSI", "LBXSGTSI", "LBXHGB", "LBXPLTSI",
        "LBXSBU" if "LBXSBU" in df.columns else "LBDSBUSI",
        "LBXSNASI" if "LBXSNASI" in df.columns else "LBXNA",
        "LBXSKSI" if "LBXSKSI" in df.columns else "LBXK",
        "LBXSUA" if "LBXSUA" in df.columns else ("LBDSUSI" if "LBDSUSI" in df.columns else "LBXUASI"),
        "LBXSCR",
        "BPXOSY1", "BPXOSY2", "BPXOSY3", "BPXODI1", "BPXODI2", "BPXODI3",
    ]
    present = [c for c in pred_candidates if c in df.columns]

    a1c = pd.to_numeric(df["LBXGH"], errors="coerce")
    fpg = pd.to_numeric(df["LBXGLU"], errors="coerce")
    age_v = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
    m_valid = a1c.between(3.0, 20.0) & fpg.between(20.0, 600.0) & (age_v >= 20.0)

    df = df.loc[m_valid, present].copy()
    a1c, fpg = a1c[m_valid], fpg[m_valid]
    df["DIABETES"] = ((fpg >= 126.0) | (a1c >= 6.5)).astype(int)

    for c in DIABETES_DIRECT_EXCLUSIONS:
        if c in df.columns:
            df = df.drop(columns=[c])
    df = drop_identifier_columns(df, verbose=False)

    out = df[["SEQN", "DIABETES"] + [c for c in df.columns if c not in ("SEQN", "DIABETES")]].copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    print(
        f"[DIABETES {cycle}] N={len(out)} | positive={int(out['DIABETES'].sum())} "
        f"({100.0 * out['DIABETES'].mean():.1f}%)"
    )
    return out


def build_ckd_cohort_egfr_uacr(master: pd.DataFrame, mode: str = "strict", tag: str = "J") -> pd.DataFrame:
    """Build adult CKD cohort (CKD-EPI 2021 eGFR < 60 or UACR >= 30 mg/g, with gray-zone trim)."""
    need = ["SEQN", "LBXSCR", "RIDAGEYR", "RIAGENDR", "URXUMA", "URXUCR"]
    if not all(c in master.columns for c in need):
        return pd.DataFrame()

    bun_col = "LBXSBU" if "LBXSBU" in master.columns else "LBDSBUSI"
    ua_col = "LBXSUA" if "LBXSUA" in master.columns else ("LBDSUSI" if "LBDSUSI" in master.columns else "LBXUASI")
    allowed_extra = [
        "BMXBMI", "BMXWAIST", "LBXTC", "LBXGH", "INDFMPIR", "RIDRETH1",
        "DMDEDUC2", "DIQ010", "DIQ050", "DIQ070",
        "LBXSASSI", "LBXSATSI", "LBXSGTSI",
        "BPXSY1", "BPXSY2", "BPXSY3", "BPXDI1", "BPXDI2", "BPXDI3",
        "BPXOSY1", "BPXOSY2", "BPXOSY3", "BPXODI1", "BPXODI2", "BPXODI3",
        "LBXHGB", "LBXPLTSI", "LBDHDD",
        ua_col, bun_col, "LBXSNASI", "LBXSKSI",
    ]
    df = master[need + [c for c in master.columns if c in allowed_extra]].copy()

    age_num = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
    m_adult = age_num >= 20.0
    female = (df["RIAGENDR"] == 2).astype(int)

    egfr = pd.Series(np.nan, index=df.index, dtype=float)
    m_egfr = m_adult & df["LBXSCR"].notna() & df["RIDAGEYR"].notna() & df["RIAGENDR"].notna()
    egfr.loc[m_egfr] = _egfr_ckdepi_2021(
        df.loc[m_egfr, "LBXSCR"].values,
        df.loc[m_egfr, "RIDAGEYR"].values,
        female.loc[m_egfr].values == 1,
    )

    uacr = pd.Series(np.nan, index=df.index, dtype=float)
    m_uacr = m_adult & df["URXUMA"].notna() & df["URXUCR"].notna()
    uacr.loc[m_uacr] = _compute_uacr_mg_per_g(
        df.loc[m_uacr, "URXUMA"].values,
        df.loc[m_uacr, "URXUCR"].values,
    )

    m_label = (m_adult & egfr.notna() & uacr.notna()) if mode == "strict" else (m_adult & (egfr.notna() | uacr.notna()))
    df = df[m_label].copy()
    df["eGFR2021"] = egfr.loc[m_label].values
    df["UACR"] = uacr.loc[m_label].values
    y = ((df["eGFR2021"] < 60.0) | (df["UACR"] >= 30.0)).astype(int)

    if mode in ("strict", "pragmatic"):
        egfr_v = pd.to_numeric(df["eGFR2021"], errors="coerce")
        uacr_v = pd.to_numeric(df["UACR"], errors="coerce")
        gray_egfr = egfr_v.ge(55.0) & egfr_v.le(65.0) & egfr_v.notna()
        gray_uacr = uacr_v.ge(25.0) & uacr_v.le(35.0) & uacr_v.notna()
        keep_mask = ~(gray_egfr | gray_uacr)
        df, y = df[keep_mask].copy(), y[keep_mask].copy()

    drop_pred = ["LBXSCR", "LBDSCRSI", "URXUMA", "URXUMS", "URXUCR", "URXCRS", "UACR", "eGFR2021", "KDIGO"]
    keep_cols = ["SEQN"] + [c for c in df.columns if c not in drop_pred]
    out = df[keep_cols].replace([np.inf, -np.inf], np.nan)
    out["CKD"] = y.values
    out = drop_identifier_columns(out, verbose=False)

    assert not [c for c in drop_pred if c in out.columns], "CKD endpoint components present as predictors."
    print(f"[CKD {tag}] Cohort shape={out.shape} (mode={mode})")
    return out


def _clinical_task_key(ds_name: str) -> Optional[str]:
    ds = str(ds_name).upper()
    for key in ENGINEERED_FEATURES_BY_TASK:
        if ds.startswith(key):
            return key
    return None


def add_clinical_features(df: pd.DataFrame, ds_name: str) -> pd.DataFrame:
    """Construct deterministic row-wise clinical composites from approved predictors."""
    task = _clinical_task_key(ds_name)
    if task is None:
        return df

    keep = set(ENGINEERED_FEATURES_BY_TASK[task])
    out = df.copy()
    num = lambda c: pd.to_numeric(out[c], errors="coerce") if c in out.columns else None
    den = lambda s: np.clip(np.asarray(s, dtype=float), 1e-6, None)

    tg, hdl, tc = num("LBXTR"), num("LBDHDD"), num("LBXTC")
    bmi, wc, age = num("BMXBMI"), num("BMXWAIST"), num("RIDAGEYR")
    sex = num("RIAGENDR")
    ast, alt, ggt = num("LBXSASSI"), num("LBXSATSI"), num("LBXSGTSI")
    plt_, hgb = num("LBXPLTSI"), num("LBXHGB")
    glu, diq = num("LBXGLU"), num("DIQ010")
    ua = num("LBXSUA") if "LBXSUA" in out.columns else (num("LBDSUSI") if "LBDSUSI" in out.columns else num("LBXUASI"))
    na = num("LBXSNASI") if "LBXSNASI" in out.columns else num("LBXNA")
    k = num("LBXSKSI") if "LBXSKSI" in out.columns else num("LBXK")
    female = (sex == 2.0) if sex is not None else None
    sy = [c for c in ("BPXOSY1", "BPXOSY2", "BPXOSY3", "BPXSY1", "BPXSY2", "BPXSY3") if c in out.columns]
    dd = [c for c in ("BPXODI1", "BPXODI2", "BPXODI3", "BPXDI1", "BPXDI2", "BPXDI3") if c in out.columns]
    sbp = out[sy].apply(pd.to_numeric, errors="coerce").mean(axis=1) if sy else None
    dbp = out[dd].apply(pd.to_numeric, errors="coerce").mean(axis=1) if dd else None

    new = {}
    if tg is not None and hdl is not None:
        new["TG_HDL_RATIO"] = tg / den(hdl)
    if tc is not None and hdl is not None:
        new["NON_HDL"] = tc - hdl
    if tg is not None:
        new["log_TG"] = np.log(den(tg))
        new["TG_high"] = (tg >= 150.0).astype(int)
    if hdl is not None and female is not None:
        new["HDL_low"] = ((female & (hdl < 50.0)) | ((~female) & (hdl < 40.0))).astype(int)
    if tg is not None and wc is not None and female is not None:
        new["LAP"] = np.where(female, (wc - 58.0), (wc - 65.0)) * (tg / 88.57)
    if tg is not None and hdl is not None and bmi is not None and wc is not None and female is not None:
        tg_mmol, hdl_mmol = tg / 88.57, hdl / 38.67
        vai_m = (wc / den(39.68 + 1.88 * bmi)) * (tg_mmol / 1.03) * (1.31 / den(hdl_mmol))
        vai_f = (wc / den(36.58 + 1.89 * bmi)) * (tg_mmol / 0.81) * (1.52 / den(hdl_mmol))
        new["VAI"] = np.where(female, vai_f, vai_m)
    if ggt is not None:
        new["log_GGT"] = np.log(den(ggt))
    if alt is not None:
        new["log_ALT"] = np.log(den(alt))
    if ast is not None and alt is not None:
        new["AST_ALT_RATIO"] = ast / den(alt)
    if sbp is not None:
        new["SBP_MEAN"] = sbp
    if dbp is not None:
        new["DBP_MEAN"] = dbp
    if sbp is not None and dbp is not None:
        new["HTN_FLAG"] = ((sbp >= 130.0) | (dbp >= 80.0)).astype(int)
    if bmi is not None:
        new["BMI_ge_25"] = (bmi >= 25.0).astype(int)
        new["BMI_ge_30"] = (bmi >= 30.0).astype(int)
        new["BMI_ge_27_5"] = (bmi >= 27.5).astype(int)
    if wc is not None and female is not None:
        new["WC_high"] = ((female & (wc >= 88.0)) | ((~female) & (wc >= 102.0))).astype(int)
    if hgb is not None and female is not None:
        new["HGB_low"] = ((female & (hgb < 12.0)) | ((~female) & (hgb < 13.0))).astype(int)
    if ua is not None and female is not None:
        si_units = np.nanmedian(ua) > 50.0
        hi_m, hi_f = (420.0, 360.0) if si_units else (7.0, 6.0)
        new["UA_high"] = ((female & (ua > hi_f)) | ((~female) & (ua > hi_m))).astype(int)
    if na is not None:
        new["Na_low"] = (na < 135.0).astype(int)
        new["Na_high"] = (na > 145.0).astype(int)
    if k is not None:
        new["K_low"] = (k < 3.5).astype(int)
        new["K_high"] = (k > 5.0).astype(int)
    if age is not None:
        new["Age_ge_45"] = (age >= 45.0).astype(int)
        new["Age_ge_50"] = (age >= 50.0).astype(int)
        new["Age_ge_60"] = (age >= 60.0).astype(int)
        if female is not None:
            new["AgexSex"] = age * female.astype(float)
    if task == "NHANES_CKD":
        bun = num("LBXSBU") if "LBXSBU" in out.columns else (num("LBDSBUSI") if "LBDSBUSI" in out.columns else num("LBXBUNSI"))
        a1c = num("LBXGH")
        if sbp is not None and dbp is not None:
            new["PULSE_PRESSURE"] = sbp - dbp
        if bun is not None:
            bun_cut = 20.0 if np.nanmedian(bun) > 8.5 else 7.1
            new["BUN_high"] = (bun >= bun_cut).astype(int)
        if a1c is not None:
            new["A1c_ge_6_5"] = (a1c >= 6.5).astype(int)
    if task == "NHANES_NAFLD":
        if age is not None and ast is not None and plt_ is not None and alt is not None:
            new["FIB4"] = (age * ast) / (den(plt_) * np.sqrt(den(alt)))
        if ast is not None and plt_ is not None:
            new["APRI"] = (ast / 40.0) / den(plt_) * 100.0
        if tg is not None and glu is not None:
            new["TyG"] = np.log(den(tg) * den(glu) / 2.0)
        if alt is not None and ast is not None and bmi is not None and female is not None and diq is not None:
            dm = (diq == 1.0).astype(float)
            new["HSI"] = 8.0 * (alt / den(ast)) + bmi + 2.0 * female.astype(float) + 2.0 * dm
        if tg is not None and bmi is not None and ggt is not None and wc is not None:
            lin = (
                0.953 * np.log(den(tg))
                + 0.139 * bmi
                + 0.718 * np.log(den(ggt))
                + 0.053 * wc
                - 15.745
            )
            new["FLI"] = 100.0 * (np.exp(np.clip(lin, -30, 30)) / (1.0 + np.exp(np.clip(lin, -30, 30))))

    added = sorted(k_name for k_name in new if k_name in keep)
    for k_name in added:
        out[k_name] = new[k_name]
    print(f"[FEATURES | {task}] Added {len(added)} clinical composites: {added}")
    return out


def winsorize_by_train(train_df: pd.DataFrame, test_df: pd.DataFrame, cols, lo: float = 0.01, hi: float = 0.99):
    """Clip continuous columns in both splits using outer-training quantiles."""
    if not cols:
        return train_df, test_df
    qlo = train_df[cols].quantile(lo)
    qhi = train_df[cols].quantile(hi)
    train_df[cols] = train_df[cols].clip(qlo, qhi, axis=1)
    test_df[cols] = test_df[cols].clip(qlo, qhi, axis=1)
    return train_df, test_df


def fit_cat_encoders(df_tr: pd.DataFrame, cat_cols):
    """Fit ordinal encoders on outer-training categorical values with an unseen-category slot."""
    encs = {}
    for c in cat_cols:
        vals = pd.Series(df_tr[c]).astype(str)
        uniq = pd.Series(vals.unique()).dropna().tolist()
        mapping = {v: i for i, v in enumerate(uniq)}
        unk_idx = len(mapping)
        encs[c] = (mapping, unk_idx, unk_idx + 1)
    return encs


def apply_cat_encoders(df: pd.DataFrame, encs, cat_cols) -> pd.DataFrame:
    """Apply fitted ordinal encoders to a dataframe."""
    xc = df.copy()
    for c in cat_cols:
        mapping, unk, _ = encs[c]
        vals = pd.Series(xc[c]).astype(str)
        xc[c] = vals.map(mapping).fillna(unk).astype(int)
    return xc
