"""Outcome-variable exclusion guards and predictor/exclusion manifest export."""

import os
import re

import pandas as pd

IDENTIFIER_PATTERNS = [r"^SEQN", r"\.\d+$", r"^Unnamed", r"^index$", r"^level_0$"]
DIABETES_DIRECT_EXCLUSIONS = ["LBXGH", "LBXGLU"]
DIABETES_PROXY_EXCLUSIONS = [
    "DIQ010", "DIQ050", "DIQ070", "DIQ160", "DIQ170", "DIQ180", "LBXINSI",
]
DIABETES_DERIVED_PATTERNS = [
    r"^A1c", r"A1c_", r"^GLU", r"GLU_", r"^TyG", r"TyG_",
    r"^HOMA", r"HOMA", r"^METS_IR", r"METS_IR",
]
DIABETES_PROXY_PATTERNS = [r"^DIQ", r"^LBXINSI$"]
CKD_DIRECT_EXCLUSIONS = [
    "LBXSCR", "LBDSCRSI", "URXUMA", "URXUMS", "URXUCR", "URXCRS",
    "UACR", "eGFR2021", "KDIGO",
]
NAFLD_DIRECT_PATTERNS = [r"CAP", r"STIFF", r"KPA", r"IQR", r"NVALID", r"EXAMSTAT"]

NHANES_CODE_INFO = {
    "SEQN": ("Respondent sequence number", "DEMO", "identifier"),
    "RIDAGEYR": ("Age at screening (years)", "DEMO", "raw"),
    "RIAGENDR": ("Sex (1 = male, 2 = female)", "DEMO", "raw"),
    "RIDRETH1": ("Race/ethnicity", "DEMO", "raw"),
    "INDFMPIR": ("Family income-to-poverty ratio", "DEMO", "raw"),
    "DMDEDUC2": ("Education level (adults 20+)", "DEMO", "raw"),
    "BMXBMI": ("Body mass index (kg/m^2)", "BMX", "raw"),
    "BMXWAIST": ("Waist circumference (cm)", "BMX", "raw"),
    "LBXTR": ("Triglycerides (mg/dL)", "TRIGLY", "raw"),
    "LBXTC": ("Total cholesterol (mg/dL)", "TCHOL", "raw"),
    "LBDHDD": ("HDL cholesterol (mg/dL)", "HDL", "raw"),
    "LBXGH": ("Glycohemoglobin HbA1c (%)", "GHB", "raw"),
    "LBXGLU": ("Fasting plasma glucose (mg/dL)", "GLU", "raw"),
    "LBXSCR": ("Serum creatinine (mg/dL)", "BIOPRO", "raw"),
    "URXUMA": ("Urine albumin (ug/mL)", "ALB_CR", "raw"),
    "URXUCR": ("Urine creatinine (mg/dL)", "ALB_CR", "raw"),
    "LBXHGB": ("Hemoglobin (g/dL)", "CBC", "raw"),
    "LBXPLTSI": ("Platelet count (1000 cells/uL)", "CBC", "raw"),
    "LBXSASSI": ("Aspartate aminotransferase AST (U/L)", "BIOPRO", "raw"),
    "LBXSATSI": ("Alanine aminotransferase ALT (U/L)", "BIOPRO", "raw"),
    "LBXSGTSI": ("Gamma-glutamyl transferase GGT (U/L)", "BIOPRO", "raw"),
    "LBXSBU": ("Blood urea nitrogen (mg/dL)", "BIOPRO", "raw"),
    "LBDSBUSI": ("Blood urea nitrogen (mmol/L)", "BIOPRO", "raw"),
    "LBXSNASI": ("Sodium (mmol/L)", "BIOPRO", "raw"),
    "LBXSKSI": ("Potassium (mmol/L)", "BIOPRO", "raw"),
    "LBXSUA": ("Uric acid (mg/dL)", "BIOPRO", "raw"),
    "LBDSUSI": ("Uric acid (umol/L)", "BIOPRO", "raw"),
    "DIQ010": ("Self-reported diabetes (ever told by doctor)", "DIQ", "raw"),
    "DIQ050": ("Taking insulin now", "DIQ", "raw"),
    "DIQ070": ("Taking diabetic pills to lower blood sugar", "DIQ", "raw"),
    "BPXOSY1": ("Oscillometric systolic BP, 1st reading (mmHg)", "BPXO", "raw"),
    "BPXOSY2": ("Oscillometric systolic BP, 2nd reading (mmHg)", "BPXO", "raw"),
    "BPXOSY3": ("Oscillometric systolic BP, 3rd reading (mmHg)", "BPXO", "raw"),
    "BPXODI1": ("Oscillometric diastolic BP, 1st reading (mmHg)", "BPXO", "raw"),
    "BPXODI2": ("Oscillometric diastolic BP, 2nd reading (mmHg)", "BPXO", "raw"),
    "BPXODI3": ("Oscillometric diastolic BP, 3rd reading (mmHg)", "BPXO", "raw"),
}


def get_task_prohibited_patterns(ds_name: str):
    """Return prohibited predictor-name regex patterns for a cohort task."""
    ds = str(ds_name).upper()
    pats = list(IDENTIFIER_PATTERNS)
    if ds.startswith("NHANES_DIABETES"):
        pats += DIABETES_DIRECT_EXCLUSIONS + DIABETES_DERIVED_PATTERNS + DIABETES_PROXY_PATTERNS
    elif ds.startswith("NHANES_CKD"):
        pats += CKD_DIRECT_EXCLUSIONS
    elif ds.startswith("NHANES_NAFLD"):
        pats += NAFLD_DIRECT_PATTERNS
    return pats


def drop_identifier_columns(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Drop participant identifier duplicates and merge-suffix artifacts."""
    drop = [
        c for c in df.columns
        if str(c) != "SEQN"
        and any(re.search(p, str(c), flags=re.I) for p in IDENTIFIER_PATTERNS)
    ]
    if drop:
        if verbose:
            print(f"[EXCLUSION] Dropping identifier/merge artifacts: {sorted(map(str, drop))}")
        df = df.drop(columns=drop)
    return df


def enforce_predictor_exclusions(df: pd.DataFrame, ds_name: str, stage: str = "") -> pd.DataFrame:
    """Remove prohibited predictors (identifiers, endpoint components, derivatives, proxies)."""
    pats = get_task_prohibited_patterns(ds_name)
    hits = [
        c for c in df.columns
        if str(c) != "SEQN" and any(re.search(p, str(c)) for p in pats)
    ]
    if hits:
        print(f"[EXCLUSION | {ds_name} | {stage}] Removing prohibited predictors: {sorted(map(str, hits))}")
        df = df.drop(columns=hits)
    else:
        print(f"[EXCLUSION | {ds_name} | {stage}] No prohibited predictors found.")
    return df


def assert_predictors_leakage_safe(feat_names, ds_name: str, stage: str = "") -> None:
    """Raise RuntimeError if any prohibited variable appears in the final feature list."""
    pats = get_task_prohibited_patterns(ds_name)
    bad = [str(f) for f in feat_names if any(re.search(p, str(f)) for p in pats)]
    if bad:
        raise RuntimeError(
            f"[EXCLUSION ASSERTION FAILED | {ds_name} | {stage}] Prohibited predictors: {sorted(bad)}"
        )
    print(
        f"[EXCLUSION ASSERTION PASSED | {ds_name} | {stage}] "
        f"{len(feat_names)} predictors verified."
    )


def assert_rules_leakage_safe(rules, feat_names, ds_name: str, stage: str = "rule base") -> None:
    """Raise RuntimeError if any retained symbolic rule references a prohibited variable."""
    pats = get_task_prohibited_patterns(ds_name)
    bad = sorted({
        str(feat_names[f])
        for rule in rules
        for (f, _, _) in rule
        if any(re.search(p, str(feat_names[f])) for p in pats)
    })
    if bad:
        raise RuntimeError(
            f"[RULE EXCLUSION FAILED | {ds_name} | {stage}] Prohibited variables in rules: {bad}"
        )
    print(
        f"[RULE EXCLUSION PASSED | {ds_name} | {stage}] "
        f"{len(rules)} rules scanned; no prohibited variables referenced."
    )


def _manifest_row(ds_name: str, col: str, raw_columns, feat_names):
    ds = str(ds_name).upper()
    c = str(col)
    name, src, kind = NHANES_CODE_INFO.get(c, (c, "derived", "raw"))
    used_in_outcome = derived_from_outcome = proxy = False

    if ds.startswith("NHANES_DIABETES"):
        if c in DIABETES_DIRECT_EXCLUSIONS:
            used_in_outcome = True
            reason = "Outcome-defining biomarker (FPG >= 126 mg/dL or HbA1c >= 6.5%); excluded."
        elif any(re.search(p, c) for p in DIABETES_DERIVED_PATTERNS):
            derived_from_outcome = True
            reason = "Derivative of outcome-defining glycemic biomarkers; excluded."
        elif any(re.search(p, c) for p in DIABETES_PROXY_PATTERNS):
            proxy = True
            reason = "Diagnosis/treatment proxy or fasting insulin; excluded."
        elif any(re.search(p, c, flags=re.I) for p in IDENTIFIER_PATTERNS):
            reason = "Identifier or merge artifact; excluded."
        else:
            reason = "Approved non-glycemic clinical predictor."
    elif ds.startswith("NHANES_CKD"):
        if c in CKD_DIRECT_EXCLUSIONS:
            used_in_outcome = True
            reason = "CKD endpoint component (eGFR/UACR/KDIGO); excluded."
        elif any(re.search(p, c, flags=re.I) for p in IDENTIFIER_PATTERNS):
            reason = "Identifier or merge artifact; excluded."
        else:
            reason = "Approved independent predictor."
    elif ds.startswith("NHANES_NAFLD"):
        if any(re.search(p, c.upper()) for p in NAFLD_DIRECT_PATTERNS):
            used_in_outcome = True
            reason = "CAP-based steatosis endpoint or VCTE elastography quality variable; excluded."
        elif c in ("ALQ121", "ALQ130"):
            proxy = True
            reason = "Alcohol-use variable reserved for cohort eligibility; excluded."
        elif any(re.search(p, c, flags=re.I) for p in IDENTIFIER_PATTERNS):
            reason = "Identifier or merge artifact; excluded."
        else:
            reason = "Approved independent predictor."
    else:
        reason = "Approved predictor."

    return {
        "Task": ds_name,
        "Variable code": c,
        "Readable variable name": name,
        "Source file": src,
        "Raw or engineered": kind if kind == "identifier" else ("raw" if c in raw_columns else "engineered"),
        "Used in outcome definition": int(used_in_outcome),
        "Derived from outcome variable": int(derived_from_outcome),
        "Potential clinical proxy": int(proxy),
        "Retained as predictor": "yes" if c in feat_names else "no",
        "Reason for retention or exclusion": reason,
    }


def save_predictor_manifest(ds_name: str, raw_columns, feat_names, extra_document=(), out_dir: str = "results") -> None:
    """Export predictor list, exclusion manifest, and combined audit CSVs."""
    os.makedirs(out_dir, exist_ok=True)
    raw_set = {str(c) for c in raw_columns}
    feat_set = {str(f) for f in feat_names}
    cols = sorted((raw_set | feat_set | {str(c) for c in extra_document}) - {"SEQN"})
    man = pd.DataFrame([_manifest_row(ds_name, c, raw_set, feat_set) for c in cols])
    man.to_csv(os.path.join(out_dir, f"{ds_name}_predictor_manifest.csv"), index=False)
    man[man["Retained as predictor"] == "yes"][
        ["Task", "Variable code", "Readable variable name", "Source file", "Raw or engineered"]
    ].to_csv(os.path.join(out_dir, f"{ds_name}_predictor_list.csv"), index=False)
    man[man["Retained as predictor"] == "no"].to_csv(
        os.path.join(out_dir, f"{ds_name}_exclusion_manifest.csv"), index=False
    )
    n_keep = int((man["Retained as predictor"] == "yes").sum())
    print(f"[MANIFEST] {ds_name}: {n_keep} retained predictors, {len(man) - n_keep} excluded rows -> {out_dir}/")
