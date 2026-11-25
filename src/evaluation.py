"""Calibration, thresholding, performance metrics, rule-alignment metrics, statistics, and case-study scanning."""

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import wilcoxon
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
import torch

from src.config import SEED
from src.models import safe_tensor
from src.rules import apply_rules
from src.training import predict_deployed_single


def expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob), 1e-6, 1.0 - 1e-6)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        m = inds == b
        if not np.any(m):
            continue
        ece += (np.sum(m) / len(y_true)) * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)


def safe_wilcoxon(x, y) -> float:
    try:
        x, y = np.asarray(x), np.asarray(y)
        if x.size < 2 or y.size < 2 or np.allclose(x, y):
            return 1.0
        _, p = wilcoxon(x, y)
        return float(p) if not np.isnan(p) else 1.0
    except Exception:
        return 1.0


def fdr_bh(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        idx = order[i]
        val = p[idx] * n / (i + 1)
        prev = min(prev, val)
        adj[idx] = min(prev, 1.0)
    return adj


def compute_ci(arr, confidence: float = 0.95):
    arr = np.asarray(arr)
    if len(arr) < 2:
        m = arr[0] if len(arr) == 1 else np.nan
        return m, m, m
    mean, std, n = np.mean(arr), np.std(arr, ddof=1), len(arr)
    tcrit = student_t.ppf((1.0 + confidence) / 2.0, n - 1)
    return float(mean), float(mean - tcrit * std / np.sqrt(n)), float(mean + tcrit * std / np.sqrt(n))


def bootstrap_best_threshold(y_cal, p_cal, n: int = 1000, seed: int = SEED) -> float:
    """Select the median of 1000 bootstrap thresholds maximizing balanced accuracy."""
    rs = np.random.RandomState(seed)
    ts = np.linspace(0.01, 0.99, 99)
    idx = np.arange(len(y_cal))
    picks = []
    for _ in range(n):
        b = rs.choice(idx, size=len(idx), replace=True)
        scores = [balanced_accuracy_score(y_cal[b], p_cal[b] >= t) for t in ts]
        picks.append(ts[int(np.argmax(scores))])
    return float(np.median(picks))


def calibrate_probs(y_cal, p_cal, p_test, method: str = "auto") -> np.ndarray:
    """Fit probability calibration on the calibration partition and apply it to outer-test predictions."""
    y_cal = np.asarray(y_cal).astype(int)
    p_cal = np.clip(np.asarray(p_cal), 1e-6, 1.0 - 1e-6)
    p_test = np.clip(np.asarray(p_test), 1e-6, 1.0 - 1e-6)

    if len(np.unique(y_cal)) < 2 or len(y_cal) < 50:
        return p_test

    if method == "auto":
        method = "isotonic" if len(y_cal) >= 200 else "platt"

    if method == "isotonic":
        try:
            iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal, y_cal)
            return np.clip(iso.transform(p_test), 1e-6, 1.0 - 1e-6)
        except Exception:
            method = "platt"

    if method == "cv_isotonic":
        try:
            kf = KFold(n_splits=min(5, max(2, len(y_cal) // 100)), shuffle=True, random_state=SEED)
            oof = np.zeros_like(p_cal, dtype=float)
            for tr, va in kf.split(p_cal):
                iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal[tr], y_cal[tr])
                oof[va] = iso.transform(p_cal[va])
            iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal, y_cal)
            return np.clip(iso.transform(p_test), 1e-6, 1.0 - 1e-6)
        except Exception:
            method = "isotonic"

    logit = lambda p: np.log(p) - np.log1p(-p)
    lr = LogisticRegression(max_iter=1000).fit(logit(p_cal).reshape(-1, 1), y_cal)
    return lr.predict_proba(logit(p_test).reshape(-1, 1))[:, 1]


def crossfit_calibrated(y_cal, p_cal, method: str = "auto") -> np.ndarray:
    """Compute out-of-fold calibrated probabilities within the calibration partition for threshold selection."""
    y_cal = np.asarray(y_cal).astype(int)
    p_cal = np.clip(np.asarray(p_cal), 1e-6, 1.0 - 1e-6)
    if len(np.unique(y_cal)) < 2 or len(y_cal) < 50:
        return p_cal
    if method == "auto":
        method = "isotonic" if len(y_cal) >= 200 else "platt"

    use_iso = method in ("isotonic", "cv_isotonic")
    oof = p_cal.copy()
    try:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        if use_iso:
            for tr, va in skf.split(p_cal, y_cal):
                iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal[tr], y_cal[tr])
                oof[va] = iso.transform(p_cal[va])
        else:
            lg = lambda p: np.log(p) - np.log1p(-p)
            for tr, va in skf.split(p_cal, y_cal):
                lr = LogisticRegression(max_iter=1000).fit(lg(p_cal[tr]).reshape(-1, 1), y_cal[tr])
                oof[va] = lr.predict_proba(lg(p_cal[va]).reshape(-1, 1))[:, 1]
    except Exception:
        return p_cal
    oof = np.where(np.isfinite(oof), oof, p_cal)
    return np.clip(oof, 1e-6, 1.0 - 1e-6)


def metrics_dict(y_true, y_prob, threshold: float = 0.5, y_prob_raw=None) -> dict:
    """Compute discrimination (raw probabilities), calibration, and thresholded metrics."""
    y_prob = np.clip(np.asarray(y_prob).ravel(), 1e-6, 1.0 - 1e-6)
    y_true = np.asarray(y_true).astype(int)
    p_raw = np.clip(np.asarray(y_prob_raw if y_prob_raw is not None else y_prob).ravel(), 1e-6, 1.0 - 1e-6)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return dict(
        Accuracy=accuracy_score(y_true, y_pred),
        BalancedAcc=balanced_accuracy_score(y_true, y_pred),
        AUC=roc_auc_score(y_true, p_raw) if len(np.unique(y_true)) > 1 else 0.5,
        AP=average_precision_score(y_true, p_raw),
        Precision=precision_score(y_true, y_pred, zero_division=0),
        Recall=recall_score(y_true, y_pred, zero_division=0),
        Specificity=tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        F1=f1_score(y_true, y_pred, zero_division=0),
        Brier=brier_score_loss(y_true, y_prob),
        NLL=log_loss(y_true, y_prob, labels=[0, 1]),
        ECE=expected_calibration_error(y_true, y_prob),
    )


def _safe_div(num, den):
    return float(num) / float(den) if den > 0 else np.nan


def majority_rule_label(R_bool: np.ndarray, rule_signs: np.ndarray) -> np.ndarray:
    if R_bool is None or rule_signs is None:
        return np.full((0,), -1, dtype=int)
    rule_signs = np.asarray(rule_signs, dtype=np.int8)
    votes = (R_bool * rule_signs.reshape(1, -1)).sum(axis=1)
    lab = np.full(votes.shape, -1, dtype=int)
    lab[votes > 0] = 1
    lab[votes < 0] = 0
    return lab


def compute_rule_alignment(y_prob: np.ndarray, threshold: float, R_bool: np.ndarray, rule_signs: np.ndarray) -> dict:
    """Compute rule-base coverage, agreement, violation rate, and violation magnitude (Eqs. 23-24)."""
    if R_bool is None or rule_signs is None or R_bool.size == 0 or rule_signs.size == 0:
        return {"RuleBaseCoverage": np.nan, "RuleAgree": np.nan, "RuleViolRate": np.nan, "RuleViolMag": np.nan}

    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_pred = (y_prob >= float(threshold)).astype(int)
    R_bool = (np.asarray(R_bool) > 0).astype(np.int8)
    rule_signs = np.asarray(rule_signs, dtype=np.int8)
    rlab = majority_rule_label(R_bool, rule_signs)
    covered = rlab != -1
    n_cov = int(covered.sum())

    rule_agree = _safe_div(int((y_pred[covered] == rlab[covered]).sum()), n_cov)
    rule_cov = _safe_div(n_cov, R_bool.shape[0])
    viol_mask = covered & (y_pred != rlab)
    rule_viol_rate = _safe_div(int(viol_mask.sum()), n_cov)

    viol_idx = np.where(viol_mask)[0]
    if viol_idx.size > 0:
        rlab_viol = rlab[viol_idx]
        yp_viol = y_prob[viol_idx]
        deficit_pos = np.maximum(0.0, float(threshold) - yp_viol) * (rlab_viol == 1)
        excess_neg = np.maximum(0.0, yp_viol - float(threshold)) * (rlab_viol == 0)
        viol_mag = float((deficit_pos + excess_neg).mean())
    else:
        viol_mag = 0.0

    return {
        "RuleBaseCoverage": float(rule_cov),
        "RuleAgree": float(rule_agree),
        "RuleViolRate": float(rule_viol_rate),
        "RuleViolMag": float(viol_mag),
    }


def run_clinical_case_study(
    model,
    baselines: dict,
    X_raw: pd.DataFrame,
    X_scaled_df: pd.DataFrame,
    X_scaled_np: np.ndarray,
    y_true: np.ndarray,
    feat_names,
    ds_name: str,
    fold: int,
    data_mined_rules,
    device,
    fold_units=None,
    fold_cat_maps=None,
) -> list:
    """Identify discordant outer-test negative cases for symbolic audit illustration (Section 6.4)."""
    candidates = X_raw.copy().reset_index(drop=True)
    candidates["target"] = pd.Series(y_true).reset_index(drop=True)

    risk_mask = pd.Series(False, index=candidates.index)
    if "BMXBMI" in candidates.columns:
        risk_mask |= candidates["BMXBMI"] >= 28.0
    elif "RIDAGEYR" in candidates.columns:
        risk_mask |= candidates["RIDAGEYR"] >= 60

    pool = candidates[risk_mask & (candidates["target"] == 0)]
    if pool.empty:
        pool = candidates[candidates["target"] == 0]
    if pool.empty:
        return []

    sort_col = "BMXBMI" if "BMXBMI" in candidates.columns else candidates.columns[0]
    scan_pool = pool.sort_values(sort_col, ascending=False).head(100)

    model.eval()
    fold_candidates = []

    for idx_int, row_raw in scan_pool.iterrows():
        try:
            row_np = X_scaled_np[idx_int]
            xc = row_np[:model.tok.n_cont].reshape(1, -1)
            xk = row_np[model.tok.n_cont:].astype(int).reshape(1, -1)
            xr = apply_rules(row_np.reshape(1, -1), data_mined_rules)

            with torch.no_grad():
                p_dep, w_dep = predict_deployed_single(model, xc, xk, xr)
                if p_dep is not None:
                    p_kar, weights = float(p_dep), w_dep
                else:
                    out = model(
                        safe_tensor(xc).to(device),
                        torch.from_numpy(xk).long().to(device),
                        safe_tensor(xr).to(device),
                        return_components=True,
                    )
                    p_kar = float(torch.sigmoid(out["logit"]).item())
                    weights = out["reliability"].detach().cpu().numpy()[0]

            base_preds = []
            for b_name, b_clf in baselines.items():
                try:
                    if "CatBoost" in b_name:
                        p = b_clf.predict_proba(X_scaled_df.iloc[[idx_int]])[:, 1][0]
                    else:
                        p = b_clf.predict_proba(row_np.reshape(1, -1))[:, 1][0]
                    base_preds.append(float(p))
                except Exception:
                    pass

            if not base_preds:
                with torch.no_grad():
                    out_bb = model(
                        safe_tensor(xc).to(device),
                        torch.from_numpy(xk).long().to(device),
                        safe_tensor(xr).to(device),
                        return_components=True,
                    )
                    base_preds.append(float(torch.sigmoid(out_bb["data_logit"]).item()))

            p_base = float(np.mean(base_preds))
            discordance = p_base - p_kar
            prev_cut = max(0.25, min(0.50, float(np.mean(y_true)) * 1.5))

            if p_base > prev_cut and p_kar < 0.55 and discordance > 0.04:
                fold_candidates.append({
                    "fold": fold,
                    "pid": int(row_raw.get("SEQN", idx_int)),
                    "p_kar": p_kar,
                    "p_base": p_base,
                    "gap": discordance,
                    "weights": weights,
                    "xr": xr,
                    "risk_val": row_raw.get(sort_col, 0),
                    "fold_rules": data_mined_rules,
                    "fold_units": fold_units,
                    "fold_cat_maps": fold_cat_maps,
                })
        except Exception:
            continue

    print(f"  [Fold {fold}] Found {len(fold_candidates)} audit-case candidates.")
    return fold_candidates
