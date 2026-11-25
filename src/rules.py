"""Cross-fitted symbolic rule discovery, redundancy filtering, clinical rendering, and auditing."""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import _tree

from src.config import (
    CFRS_CANDIDATES_PER_INNER_FOLD,
    CFRS_INNER_FOLDS,
    CFRS_MAX_COVERAGE,
    CFRS_MAX_RULES,
    CFRS_MIN_COVERAGE,
    CFRS_MIN_RULES,
    CFRS_REDUNDANCY_JACCARD,
    RULE_BASE_SIZE,
    SEED,
)


def apply_rules(X: np.ndarray, rules) -> np.ndarray:
    """Evaluate a list of conjunction rules on matrix X and return binary activations [N, M]."""
    if not rules or X.shape[0] == 0:
        return np.zeros((X.shape[0], 1), dtype=np.float32)
    feat = np.zeros((X.shape[0], len(rules)), dtype=np.float32)
    for i, rule in enumerate(rules):
        mask = np.ones(X.shape[0], dtype=bool)
        for fidx, op, val in rule:
            mask &= (X[:, fidx] <= val) if op == "<=" else (X[:, fidx] > val)
        feat[:, i] = mask
    return feat


def _rule_set_signature(rule):
    return frozenset((int(a), str(b), float(c)) for (a, b, c) in rule)


def _canonical_rule(rule, prec: int = 3):
    return tuple((int(i), str(op), round(float(th), prec)) for (i, op, th) in rule)


def _rule_mask_single(rule, X: np.ndarray) -> np.ndarray:
    if not rule or X.shape[0] == 0:
        return np.zeros((X.shape[0],), dtype=bool)
    m = np.ones((X.shape[0],), dtype=bool)
    for fidx, op, thr in rule:
        col = X[:, int(fidx)]
        if op == "<=":
            m &= (col <= float(thr)) & ~np.isnan(col)
        else:
            m &= (col > float(thr)) & ~np.isnan(col)
    return m


def _mask_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def _cfrs_jaccard(mask_a, mask_b) -> float:
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(mask_a, mask_b).sum() / union)


def _rule_gain(mask: np.ndarray, y: np.ndarray) -> float:
    if mask.sum() == 0 or mask.sum() == len(y):
        return 0.0
    pos_on = y[mask].mean()
    pos_off = y[~mask].mean()
    cov = mask.mean()
    return float(abs(pos_on - pos_off) * min(cov, 1.0 - cov))


def _cfrs_family_key(rule):
    return tuple(sorted((int(fidx), str(op)) for fidx, op, _ in rule))


def build_univariate_rules(
    X: np.ndarray,
    y: np.ndarray,
    feature_names,
    max_per_feat: int = 8,
    min_cov: float = 0.05,
    max_cov: float = 0.80,
) -> list:
    """Extract univariate quantile stump rules ranked by discriminative gain."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    _, d = X.shape
    qgrid = np.linspace(0.1, 0.9, 9)
    cand = []

    for j in range(d):
        x = X[:, j]
        x_nonan = x[~np.isnan(x)]
        if np.unique(x_nonan).size < 5:
            continue
        qs = np.unique(np.nanquantile(x, qgrid))
        local = []
        for thr in qs:
            if not np.isfinite(thr):
                continue
            m = (x <= thr) & ~np.isnan(x)
            cov = m.mean()
            if cov < min_cov or cov > max_cov:
                continue
            g = _rule_gain(m, y)
            if g <= 0.0:
                continue
            local.append((g, (int(j), "<=", float(thr)), m))
        local.sort(key=lambda t: t[0], reverse=True)
        kept, kept_masks = [], []
        for _, one_cond, mask in local:
            if any(_mask_jaccard(mask, km) >= 0.95 for km in kept_masks):
                continue
            kept.append([one_cond])
            kept_masks.append(mask)
            if len(kept) >= max_per_feat:
                break
        cand.extend(kept)

    def _gain_key(rule):
        j, _, thr = rule[0]
        col = X[:, j]
        m = (col <= thr) & ~np.isnan(col)
        return _rule_gain(m, y)

    cand = sorted(cand, key=_gain_key, reverse=True)
    return [list(r) for r in cand]


def ensure_min_rules(rules: list, X: np.ndarray, y: np.ndarray, feature_names, min_k: int = 64) -> list:
    """Top up a small candidate pool with eligible univariate stumps (never duplicate padding)."""
    base = []
    if rules:
        for r in rules:
            if not r:
                continue
            if isinstance(r[0], tuple):
                base.append([tuple(a) for a in r])
            elif len(r) == 3:
                base.append([tuple(r)])

    out = base.copy()
    if len(out) >= min_k:
        return out

    seen = set(_rule_set_signature(r) for r in out)
    extra = build_univariate_rules(X, y, feature_names, max_per_feat=8)
    for r in extra:
        sig = _rule_set_signature(r)
        if sig in seen:
            continue
        out.append(r)
        seen.add(sig)
        if len(out) >= min_k:
            break
    return out


def get_high_gain_ruleset(
    X: np.ndarray,
    y: np.ndarray,
    feature_names,
    n_estimators_rf: int = 50,
    max_depth: int = 5,
    min_cov: float = 0.01,
    max_cov: float = 0.95,
    jaccard_drop: float = 0.85,
    verbose: bool = False,
) -> list:
    """Mine candidate rules from balanced random-forest paths and univariate quantile stumps."""
    candidate_pool = []
    seen_sigs = set()

    rf = RandomForestClassifier(
        n_estimators=n_estimators_rf,
        max_depth=max_depth,
        min_samples_leaf=max(10, int(0.01 * len(y))),
        random_state=SEED,
        class_weight="balanced",
    )
    rf.fit(X, y)

    def _recurse(node, path, tree_):
        if 1 <= len(path) <= 3:
            sig_mid = _rule_set_signature(path)
            if sig_mid not in seen_sigs:
                candidate_pool.append(list(path))
                seen_sigs.add(sig_mid)
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            fidx = int(tree_.feature[node])
            thr = round(float(tree_.threshold[node]), 3)
            _recurse(tree_.children_left[node], path + [(fidx, "<=", thr)], tree_)
            _recurse(tree_.children_right[node], path + [(fidx, ">", thr)], tree_)
        elif path:
            sig = _rule_set_signature(path)
            if sig not in seen_sigs:
                candidate_pool.append(list(path))
                seen_sigs.add(sig)

    for est in rf.estimators_:
        _recurse(0, [], est.tree_)

    for u_rule in build_univariate_rules(X, y, feature_names, max_per_feat=6, min_cov=0.02, max_cov=0.60):
        sig = _rule_set_signature(u_rule)
        if sig not in seen_sigs:
            candidate_pool.append(u_rule)
            seen_sigs.add(sig)

    if not candidate_pool:
        return build_univariate_rules(X, y, feature_names, max_per_feat=8)

    r_cand = apply_rules(X, candidate_pool)
    ranked_rules = []
    for i, rule in enumerate(candidate_pool):
        mask = r_cand[:, i].astype(bool)
        cov = mask.mean()
        if not (min_cov <= cov <= max_cov):
            continue
        gain = _rule_gain(mask, y)
        if gain > 0:
            ranked_rules.append((gain, rule, mask))

    ranked_rules.sort(key=lambda x: x[0], reverse=True)

    final_rules = []
    final_masks = []
    for _, rule, mask in ranked_rules:
        if any(_mask_jaccard(mask, km) >= jaccard_drop for km in final_masks):
            continue
        final_rules.append(rule)
        final_masks.append(mask)

    if verbose:
        print(f"[RULES] Candidate pool={len(candidate_pool)} -> deduplicated={len(final_rules)}")
    if not final_rules:
        return build_univariate_rules(X, y, feature_names, max_per_feat=8)
    return final_rules


def select_rule_base(rules, X: np.ndarray, y: np.ndarray, feature_names, size: int = RULE_BASE_SIZE) -> list:
    """Rank inner-training candidate rules by discriminative gain and retain up to `size` rules."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(int)
    ranked = sorted(
        rules,
        key=lambda r: _rule_gain(_rule_mask_single(r, X), y),
        reverse=True,
    )[:size]
    if len(ranked) < size:
        ranked = ensure_min_rules(ranked, X, y, feature_names, min_k=size)
    return ranked


def mine_crossfitted_stable_rules(
    X: np.ndarray,
    y: np.ndarray,
    feature_names,
    outer_fold: int,
    results_dir: str,
    dataset_name: str,
):
    """Cross-fitted symbolic rule discovery with held-out utility scoring and polarity balance (Algorithm 1)."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("Cross-fitted rule discovery requires aligned 2D X and 1D y.")
    if len(np.unique(y)) != 2:
        raise ValueError("Cross-fitted rule discovery requires both outcome classes.")

    splitter = StratifiedKFold(
        n_splits=CFRS_INNER_FOLDS,
        shuffle=True,
        random_state=SEED + 1000 + int(outer_fold),
    )
    candidate_records = []
    family_fold_sets = {}
    raw_candidate_count = 0

    for inner_fold, (inner_train, inner_holdout) in enumerate(splitter.split(X, y), start=1):
        inner_rules = get_high_gain_ruleset(X[inner_train], y[inner_train], feature_names)
        inner_rules = select_rule_base(
            inner_rules,
            X[inner_train],
            y[inner_train],
            feature_names,
            size=CFRS_CANDIDATES_PER_INNER_FOLD,
        )
        raw_candidate_count += len(inner_rules)
        r_holdout = apply_rules(X[inner_holdout], inner_rules)
        y_holdout = y[inner_holdout]

        for j, rule in enumerate(inner_rules):
            active = r_holdout[:, j] > 0.5
            n_active = int(active.sum())
            n_total = int(len(active))
            if n_active == 0 or n_active == n_total:
                continue
            coverage = float(n_active / n_total)
            if not (CFRS_MIN_COVERAGE <= coverage <= CFRS_MAX_COVERAGE):
                continue
            inactive = ~active
            risk_active = float(y_holdout[active].mean())
            risk_inactive = float(y_holdout[inactive].mean())
            risk_difference = float(risk_active - risk_inactive)
            sign = int(np.sign(risk_difference))
            if sign == 0:
                continue
            n_inactive = int(inactive.sum())
            variance = (
                risk_active * (1.0 - risk_active) / max(n_active, 1)
                + risk_inactive * (1.0 - risk_inactive) / max(n_inactive, 1)
            )
            standard_error = float(np.sqrt(max(variance, 0.0)))
            uncertainty_weight = float(
                abs(risk_difference) / (abs(risk_difference) + standard_error + 1e-6)
            )
            quality = float(
                abs(risk_difference)
                * np.sqrt(max(coverage * (1.0 - coverage), 0.0))
                * uncertainty_weight
            )
            if not np.isfinite(quality) or quality <= 0.0:
                continue
            family_key = _cfrs_family_key(rule)
            family_fold_sets.setdefault(family_key, set()).add(int(inner_fold))
            candidate_records.append({
                "inner_fold": int(inner_fold),
                "rule": rule,
                "family_key": family_key,
                "sign": float(sign),
                "coverage": coverage,
                "risk_difference": risk_difference,
                "standard_error": standard_error,
                "uncertainty_weight": uncertainty_weight,
                "quality": quality,
                "active_cases": int(np.logical_and(active, y_holdout == 1).sum()),
                "active_controls": int(np.logical_and(active, y_holdout == 0).sum()),
            })

    if not candidate_records:
        raise RuntimeError(f"[CFRS | {dataset_name} | fold {outer_fold}] No candidate had finite held-out utility.")

    best_by_canonical = {}
    for record in candidate_records:
        canonical = _canonical_rule(record["rule"], prec=3)
        prev = best_by_canonical.get(canonical)
        if prev is None or record["quality"] > prev["quality"]:
            best_by_canonical[canonical] = record

    pooled = list(best_by_canonical.values())
    for record in pooled:
        represented = family_fold_sets.get(record["family_key"], set())
        record["family_recurrence_count"] = int(len(represented))
        record["family_recurrence"] = float(len(represented) / CFRS_INNER_FOLDS)

    pooled.sort(
        key=lambda rec: (rec["quality"], rec["family_recurrence"], -int(len(rec["rule"]))),
        reverse=True,
    )

    retained, retained_masks, deferred = [], [], []
    max_per_polarity = int(CFRS_MAX_RULES * 0.65)
    pos_kept, neg_kept = 0, 0

    for record in pooled:
        candidate_mask = _rule_mask_single(record["rule"], X)
        if candidate_mask.sum() == 0 or candidate_mask.sum() == len(candidate_mask):
            continue
        redundant = any(
            record["sign"] == kept["sign"]
            and _cfrs_jaccard(candidate_mask, kept_mask) >= CFRS_REDUNDANCY_JACCARD
            for kept, kept_mask in zip(retained, retained_masks)
        )
        if redundant:
            continue
        if record["sign"] > 0 and pos_kept >= max_per_polarity:
            deferred.append((record, candidate_mask))
            continue
        if record["sign"] < 0 and neg_kept >= max_per_polarity:
            deferred.append((record, candidate_mask))
            continue
        retained.append(record)
        retained_masks.append(candidate_mask)
        if record["sign"] > 0:
            pos_kept += 1
        else:
            neg_kept += 1
        if len(retained) >= CFRS_MAX_RULES:
            break

    for record, candidate_mask in deferred:
        if len(retained) >= CFRS_MAX_RULES:
            break
        retained.append(record)
        retained_masks.append(candidate_mask)

    if len(retained) < CFRS_MIN_RULES:
        relaxed_threshold = 0.95
        retained, retained_masks = [], []
        for record in pooled:
            candidate_mask = _rule_mask_single(record["rule"], X)
            if candidate_mask.sum() == 0 or candidate_mask.sum() == len(candidate_mask):
                continue
            redundant = any(
                record["sign"] == kept["sign"]
                and _cfrs_jaccard(candidate_mask, kept_mask) >= relaxed_threshold
                for kept, kept_mask in zip(retained, retained_masks)
            )
            if redundant:
                continue
            retained.append(record)
            retained_masks.append(candidate_mask)
            if len(retained) >= CFRS_MAX_RULES:
                break

    if len(retained) < CFRS_MIN_RULES:
        raise RuntimeError(
            f"[CFRS | {dataset_name} | fold {outer_fold}] Only {len(retained)} rules survived "
            f"(minimum required={CFRS_MIN_RULES})."
        )

    retained_ids = {id(record) for record in retained}
    audit_df = pd.DataFrame([
        {
            "outer_fold": int(outer_fold),
            "inner_fold": rec["inner_fold"],
            "family": repr(rec["family_key"]),
            "family_recurrence_count": rec["family_recurrence_count"],
            "family_recurrence": rec["family_recurrence"],
            "heldout_risk_difference": rec["risk_difference"],
            "heldout_coverage": rec["coverage"],
            "heldout_standard_error": rec["standard_error"],
            "uncertainty_weight": rec["uncertainty_weight"],
            "heldout_quality": rec["quality"],
            "sign": rec["sign"],
            "active_cases": rec["active_cases"],
            "active_controls": rec["active_controls"],
            "retained": id(rec) in retained_ids,
        }
        for rec in pooled
    ])
    os.makedirs(results_dir, exist_ok=True)
    audit_df.to_csv(
        os.path.join(results_dir, f"{dataset_name}_crossfitted_rule_candidates_fold{outer_fold}.csv"),
        index=False,
    )

    qualities = np.asarray([rec["quality"] for rec in retained], dtype=np.float32)
    qualities /= max(float(qualities.max()), 1e-6)
    signs = np.asarray([rec["sign"] for rec in retained], dtype=np.float32)
    rules = [rec["rule"] for rec in retained]
    print(
        f"[CFRS | {dataset_name} | fold {outer_fold}] raw={raw_candidate_count} | "
        f"heldout_valid={len(candidate_records)} | unique={len(pooled)} | "
        f"retained={len(rules)} (pos={(signs > 0).sum()}, neg={(signs < 0).sum()}) | "
        f"median_quality={np.median(qualities):.4f}"
    )
    return rules, signs, qualities, audit_df


def _decode_cat_condition(name: str, op: str, thr: float, mapping: dict) -> str:
    take_lo = op == "<="
    levels = sorted(str(v) for v, idx in mapping.items() if (idx <= float(thr)) == take_lo)
    if not levels:
        return f"{name} in {{}} (empty)"
    if len(levels) <= 8:
        return f"{name} in {{{', '.join(levels)}}}"
    return f"{name} in {{{len(levels)} categories}} (code {'<=' if take_lo else '>'} {float(thr):.4f})"


def simplify_rule(rule):
    """Collapse dominated bounds per feature without changing the activation mask."""
    lo, hi = {}, {}
    for fidx, op, thr in rule:
        t = float(thr)
        if op == "<=":
            hi[fidx] = t if fidx not in hi else min(hi[fidx], t)
        else:
            lo[fidx] = t if fidx not in lo else max(lo[fidx], t)
    return [(f, ">", lo[f]) for f in sorted(lo)] + [(f, "<=", hi[f]) for f in sorted(hi)]


def rule_to_clinical_str(rule, feat_names, units=None, cat_maps=None, prec: int = 2, scaled_prec: int = 4) -> str:
    """Render a symbolic rule in original clinical units and decoded category labels."""
    parts = []
    for fidx, op, thr in rule:
        name = str(feat_names[fidx])
        if cat_maps and name in cat_maps:
            parts.append(_decode_cat_condition(name, op, thr, cat_maps[name]))
        elif units and name in units:
            center, scale = units[name]
            parts.append(f"{name} {op} {center + float(thr) * scale:.{prec}f}")
        else:
            parts.append(f"{name} {op} {float(thr):.{scaled_prec}f}")
    return " AND ".join(parts)


def export_final_rules_csv(
    ds_name: str,
    rules,
    feat_names,
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    rule_signs,
    out_dir: str = "results",
    fold: int = None,
    units=None,
    cat_maps=None,
) -> None:
    """Export retained fold-specific symbolic rules in clinical units."""
    x_arr = np.asarray(X_ref, dtype=np.float32)
    y_arr = np.asarray(y_ref).astype(int)
    rows = []
    for i, rule in enumerate(rules):
        m = _rule_mask_single(rule, x_arr)
        cov = float(m.mean()) if m.size else 0.0
        gain = _rule_gain(m, y_arr) if (m.any() and not m.all()) else 0.0
        direction = "negative-risk" if (rule_signs is not None and i < len(rule_signs) and rule_signs[i] < 0) else "positive-risk"
        simp = simplify_rule(rule)
        if not np.array_equal(_rule_mask_single(simp, x_arr), m):
            raise RuntimeError(f"Rule simplification altered mask for rule {i} in {ds_name} fold {fold}.")
        rows.append({
            "Cohort": ds_name,
            "Rule": rule_to_clinical_str(simp, feat_names, units, cat_maps),
            "Rule_scaled_space": " AND ".join(f"{feat_names[f]} {op} {v:.4f}" for f, op, v in rule),
            "Variables": ", ".join(str(feat_names[f]) for f, _, _ in rule),
            "Thresholds": "; ".join(f"{feat_names[f]} {op} {v:.4f}" for f, op, v in simp),
            "N_conditions_raw": len(rule),
            "N_conditions_minimal": len(simp),
            "Direction": direction,
            "Coverage_train": round(cov, 4),
            "Gain_train": round(gain, 4),
            "N_conditions": len(simp),
        })
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{ds_name}_final_rules_fold{fold}.csv" if fold is not None else f"{ds_name}_final_rules.csv"
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, fname), index=False)


def audit_rules(X_te: np.ndarray, y_te: np.ndarray, rules, feat_names, rule_signs=None, units=None, cat_maps=None) -> pd.DataFrame:
    """Audit retained rules on held-out outer-test observations."""
    rows = []
    r_mat = apply_rules(X_te, rules)
    for i, rule in enumerate(rules):
        mask = r_mat[:, i].astype(bool)
        cov = float(mask.mean())
        if cov == 0.0 or cov == 1.0:
            continue
        pos_on = float(y_te[mask].mean())
        pos_off = float(y_te[~mask].mean())
        rd = pos_on - pos_off
        sig = rule_signs[i] if (rule_signs is not None and i < len(rule_signs)) else (1.0 if rd > 0 else -1.0)
        rows.append({
            "Rule_IDX": i,
            "Rule": rule_to_clinical_str(simplify_rule(rule), feat_names, units, cat_maps, prec=2, scaled_prec=3),
            "Features": ", ".join(feat_names[f] for f, _, _ in rule),
            "Thresholds": "; ".join(f"{feat_names[f]} {op} {v:.4f}" for f, op, v in rule),
            "Coverage": round(cov, 4),
            "PosRate_Active": round(pos_on, 4),
            "PosRate_Inactive": round(pos_off, 4),
            "RiskDiff": round(rd, 4),
            "Direction": "positive-risk" if sig > 0 else "negative-risk",
            "Support": int(mask.sum()),
        })
    return pd.DataFrame(rows)
