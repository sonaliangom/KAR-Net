"""Stratified 5-fold outer cross-validation pipeline for KAR-Net and comparator models."""

import json
import os
import time

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
import torch
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from src.baselines import (
    TTWrapper,
    TabTransformerBaseline,
    train_excelformer,
    train_mpcfin,
    train_robust_gbdt,
    train_tabm,
)
from src.cohorts import apply_cat_encoders, fit_cat_encoders, winsorize_by_train
from src.config import (
    BATCH_SIZE,
    CFRS_MAX_RULES,
    DEVICE,
    EPOCHS_FINAL,
    FROZEN_CONFIG_GLOBAL,
    FROZEN_CONFIGS,
    MIN_LR,
    N_SPLITS,
    NUM_WORKERS,
    PATIENCE,
    PIN_MEMORY,
    SEED,
    experiment_seed,
    make_torch_generator,
    seed_everything,
)
from src.evaluation import (
    bootstrap_best_threshold,
    calibrate_probs,
    compute_ci,
    compute_rule_alignment,
    crossfit_calibrated,
    fdr_bh,
    metrics_dict,
    run_clinical_case_study,
    safe_wilcoxon,
)
from src.leakage import (
    CKD_DIRECT_EXCLUSIONS,
    DIABETES_DIRECT_EXCLUSIONS,
    DIABETES_PROXY_EXCLUSIONS,
    assert_predictors_leakage_safe,
    assert_rules_leakage_safe,
    drop_identifier_columns,
    enforce_predictor_exclusions,
    save_predictor_manifest,
)
from src.models import BalancedFocalLoss, KARNet, make_pos_weight, safe_tensor
from src.rules import (
    apply_rules,
    audit_rules,
    export_final_rules_csv,
    mine_crossfitted_stable_rules,
    rule_to_clinical_str,
    simplify_rule,
)
from src.training import (
    assert_neutral_rule_invariant,
    build_deployed_hybrid_ensemble,
    build_stage_optimizer,
    configure_training_stage,
    initialize_global_checkpoint,
    make_aligned_tensor_dataset,
    predict_deployed_proba,
    pretrain_model,
    restore_global_checkpoint,
    train_eval,
)
from src.visualization import (
    format_rule_human_readable,
    plot_case_study_figure,
    plot_single_dca,
)


def run_experiment(ds_name: str, data: dict, output_dir: str = "results", run_baselines: bool = True) -> list:
    """Execute stratified 5-fold outer CV on one cohort using the frozen global configuration."""
    print(f"\n--- Starting Experiment on {ds_name} ---")
    os.makedirs(output_dir, exist_ok=True)

    X_df = data["X"].copy()
    y_sr = data["y"].copy()
    raw_columns_all = list(map(str, data.get("raw_cols", X_df.columns)))

    X_df = drop_identifier_columns(X_df, verbose=True)
    X_df = enforce_predictor_exclusions(X_df, ds_name, stage="raw cohort (pre-preprocessing)")

    cls_counts = np.bincount(y_sr.values.astype(int))
    if np.min(cls_counts) < N_SPLITS:
        print(f"[SKIP] {ds_name}: minority class count ({np.min(cls_counts)}) < N_SPLITS ({N_SPLITS}).")
        return []

    cat_cols, cont_cols = data["cat_cols"], data["cont_cols"]
    feat_names = cont_cols + cat_cols

    missing_cols = [f for f in feat_names if f not in X_df.columns]
    if missing_cols:
        raise RuntimeError(f"[{ds_name}] Missing feature columns: {missing_cols}")

    assert_predictors_leakage_safe(feat_names, ds_name, stage="final predictor matrix")
    print(f"[{ds_name}] Final predictors: {len(feat_names)} (continuous={len(cont_cols)}, categorical={len(cat_cols)})")

    extra_doc = ()
    if ds_name.upper().startswith("NHANES_DIABETES"):
        extra_doc = tuple(DIABETES_DIRECT_EXCLUSIONS) + tuple(DIABETES_PROXY_EXCLUSIONS)
    elif ds_name.upper().startswith("NHANES_CKD"):
        extra_doc = tuple(CKD_DIRECT_EXCLUSIONS)
    save_predictor_manifest(ds_name, raw_columns_all, feat_names, extra_document=extra_doc, out_dir=output_dir)

    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)

    model_keys = [
        "Transformer_NoRules",
        "Transformer_ConcatRules",
        "Transformer_KARNet",
        "Transformer_KARNet_Lam0",
    ]
    if run_baselines:
        model_keys.extend([
            "RF", "SVM", "XGB", "CatBoost", "TabNet", "TabTransformer",
            "Robust-GBDT", "MPCFIN", "TabM", "ExcelFormer",
        ])

    fold_metrics = {m: [] for m in model_keys}
    per_fold_records = []
    oof_records = []
    all_case_candidates = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_df.values, y_sr.values), 1):
        print(f"\n==================== Fold {fold}/{N_SPLITS} ====================")

        X_tr_raw = X_df.iloc[tr_idx].copy()
        X_te_raw = X_df.iloc[te_idx].copy()
        y_tr = y_sr.iloc[tr_idx].values
        y_te = y_sr.iloc[te_idx].values

        # Outer-training winsorization (1st-99th percentiles)
        cont_cols_local = [c for c in cont_cols if c in X_tr_raw.columns]
        if cont_cols_local:
            med = X_tr_raw[cont_cols_local].median()
            X_tr_tmp = X_tr_raw.copy()
            X_te_tmp = X_te_raw.copy()
            X_tr_tmp[cont_cols_local] = X_tr_tmp[cont_cols_local].fillna(med)
            X_te_tmp[cont_cols_local] = X_te_tmp[cont_cols_local].fillna(med)
            X_tr_tmp, X_te_tmp = winsorize_by_train(X_tr_tmp, X_te_tmp, cont_cols_local, lo=0.01, hi=0.99)
            X_tr_raw[cont_cols_local] = X_tr_tmp[cont_cols_local]
            X_te_raw[cont_cols_local] = X_te_tmp[cont_cols_local]

        # Outer-training imputation, encoding, and robust scaling
        imp_cont = SimpleImputer(strategy="median").fit(X_tr_raw[cont_cols]) if cont_cols else None
        imp_cat = SimpleImputer(strategy="most_frequent").fit(X_tr_raw[cat_cols]) if cat_cols else None
        if imp_cont:
            X_tr_raw[cont_cols] = imp_cont.transform(X_tr_raw[cont_cols])
            X_te_raw[cont_cols] = imp_cont.transform(X_te_raw[cont_cols])
        if imp_cat:
            X_tr_raw[cat_cols] = imp_cat.transform(X_tr_raw[cat_cols])
            X_te_raw[cat_cols] = imp_cat.transform(X_te_raw[cat_cols])

        encs = fit_cat_encoders(X_tr_raw, cat_cols)
        X_tr_enc = apply_cat_encoders(X_tr_raw, encs, cat_cols)
        X_te_enc = apply_cat_encoders(X_te_raw, encs, cat_cols)
        cat_cards = [encs[c][2] for c in cat_cols]

        if cont_cols:
            scaler = RobustScaler().fit(X_tr_enc[cont_cols])
            X_tr_enc[cont_cols] = scaler.transform(X_tr_enc[cont_cols])
            X_te_enc[cont_cols] = scaler.transform(X_te_enc[cont_cols])

        fold_units = (
            {c: (float(scaler.center_[i]), float(scaler.scale_[i])) for i, c in enumerate(cont_cols)}
            if cont_cols
            else {}
        )
        fold_cat_maps = {c: dict(encs[c][0]) for c in cat_cols} if cat_cols else {}

        X_tr_np = X_tr_enc[feat_names].values.astype(np.float32)
        X_te_np = X_te_enc[feat_names].values.astype(np.float32)
        for arr in (X_tr_np, X_te_np):
            for j in range(arr.shape[1]):
                col = arr[:, j]
                if not np.isfinite(col).all():
                    med_v = np.nanmedian(col)
                    if not np.isfinite(med_v):
                        med_v = 0.0
                    arr[:, j] = np.where(np.isfinite(col), col, med_v)

        cont_idx = list(range(len(cont_cols)))
        cat_idx = list(range(len(cont_cols), len(feat_names)))

        # Internal 64% model-training / 16% validation / 20% calibration split
        X_tmp, X_cal, y_tmp, y_cal = train_test_split(
            X_tr_np, y_tr, test_size=0.2, stratify=y_tr, random_state=SEED
        )
        X_trm, X_val, y_trm, y_val = train_test_split(
            X_tmp, y_tmp, test_size=0.2, stratify=y_tmp, random_state=SEED
        )

        # Cross-fitted rule discovery strictly within the 64% model-training split
        data_mined_rules, rule_signs_datamined, global_rule_quality_datamined, _ = (
            mine_crossfitted_stable_rules(
                X=X_trm,
                y=y_trm,
                feature_names=feat_names,
                outer_fold=fold,
                results_dir=output_dir,
                dataset_name=ds_name,
            )
        )
        assert_rules_leakage_safe(
            data_mined_rules, feat_names, ds_name, stage=f"fold-{fold} cross-fitted rule base"
        )
        R_trm_dm = apply_rules(X_trm, data_mined_rules)
        R_val_dm = apply_rules(X_val, data_mined_rules)
        R_cal_dm = apply_rules(X_cal, data_mined_rules)
        R_te_datamined = apply_rules(X_te_np, data_mined_rules)

        export_final_rules_csv(
            ds_name, data_mined_rules, feat_names, X_trm, y_trm, rule_signs_datamined,
            out_dir=output_dir, fold=fold, units=fold_units, cat_maps=fold_cat_maps,
        )
        audit_df = audit_rules(
            X_te_np, y_te, data_mined_rules, feat_names,
            rule_signs=rule_signs_datamined, units=fold_units, cat_maps=fold_cat_maps,
        )
        if len(audit_df):
            audit_df.to_csv(os.path.join(output_dir, f"{ds_name}_rule_audit_fold{fold}.csv"), index=False)

        def split_X(arr):
            xc = arr[:, cont_idx] if cont_idx else np.empty((len(arr), 0), dtype=np.float32)
            xk = arr[:, cat_idx] if cat_idx else np.empty((len(arr), 0), dtype=np.int32)
            return xc, xk

        Xc_trm, Xk_trm = split_X(X_trm)
        Xc_val, Xk_val = split_X(X_val)
        Xc_cal, Xk_cal = split_X(X_cal)
        Xc_te, Xk_te = split_X(X_te_np)

        best_params = {**FROZEN_CONFIG_GLOBAL, **FROZEN_CONFIGS.get(ds_name, {})}
        best_mdl_par = {k: best_params[k] for k in ("embed_dim", "n_heads", "n_layers", "dropout")}
        print(f"[CONFIG | {ds_name} | fold {fold}] {json.dumps(best_params, sort_keys=True)}")

        # Self-supervised pre-training (SSP) on the 64% model-training split
        seed_everything(experiment_seed(fold, "transformer", "ssp"))
        pretrained_model_base = KARNet(cat_cards, len(cont_idx), 0, **best_mdl_par, run_abl={"no_rules": True}).to(DEVICE)
        pretrained_model_base = pretrain_model(
            pretrained_model_base,
            X_trm,
            cont_idx,
            cat_idx,
            cat_cards,
            mask_prob=best_params["mask_prob"],
            pretrain_epochs=best_params["pretrain_epochs"],
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
        )
        pretrained_weights = pretrained_model_base.state_dict()

        # External baselines
        X_trm_df = pd.DataFrame(X_trm, columns=feat_names)
        X_val_df = pd.DataFrame(X_val, columns=feat_names)
        X_cal_df = pd.DataFrame(X_cal, columns=feat_names)
        X_te_df = pd.DataFrame(X_te_np, columns=feat_names)

        basemodels = (
            [
                ("RF", RandomForestClassifier(n_estimators=500, random_state=SEED)),
                ("SVM", SVC(probability=True, C=1.5, gamma="scale", random_state=SEED)),
                ("XGB", XGBClassifier(
                    use_label_encoder=False, eval_metric="logloss", max_depth=3,
                    learning_rate=0.05, n_estimators=500, subsample=0.8,
                    colsample_bytree=0.8, random_state=SEED,
                )),
                ("CatBoost", CatBoostClassifier(
                    depth=8, learning_rate=0.05, iterations=500,
                    loss_function="Logloss", verbose=False, random_seed=SEED,
                )),
                ("TabNet", TabNetClassifier(verbose=0, seed=SEED, n_steps=5, gamma=1.5, optimizer_params=dict(lr=0.01))),
                ("TabTransformer", None),
                ("Robust-GBDT", None),
                ("MPCFIN", None),
                ("TabM", None),
                ("ExcelFormer", None),
            ]
            if run_baselines
            else []
        )

        trained_baselines_dict = {}
        cal_probs_fold1 = {}
        te_probs_fold1 = {}

        for name, clf_instance in basemodels:
            if name == "SVM" and X_trm_df.shape[0] > 20000:
                continue
            print(f"\n*** Training Baseline: {name} ***")
            try:
                if name == "Robust-GBDT":
                    p_cal, p_te = train_robust_gbdt(X_trm, y_trm, X_val, y_val, X_cal, X_te_np)
                elif name == "MPCFIN":
                    p_cal, p_te = train_mpcfin(X_trm, y_trm, X_val, y_val, X_cal, X_te_np, cont_idx, cat_idx, cat_cards)
                elif name == "TabM":
                    p_cal, p_te = train_tabm(X_trm, y_trm, X_val, y_val, X_cal, X_te_np, n_features=X_trm.shape[1])
                elif name == "ExcelFormer":
                    p_cal, p_te = train_excelformer(X_trm, y_trm, X_val, y_val, X_cal, X_te_np, n_features=X_trm.shape[1])
                elif name == "TabTransformer":
                    tt = TabTransformerBaseline(cat_cards, len(cont_idx), **best_mdl_par).to(DEVICE)
                    tt_dict = tt.state_dict()
                    tt_dict.update({
                        k: v for k, v in pretrained_weights.items()
                        if k in tt_dict and not k.startswith("head.") and tt_dict[k].shape == v.shape
                    })
                    tt.load_state_dict(tt_dict)
                    tt_wrapped = TTWrapper(tt)
                    opt = torch.optim.AdamW(tt_wrapped.parameters(), lr=8e-4, weight_decay=1e-5)
                    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=6)
                    crit = BalancedFocalLoss(pos_weight=make_pos_weight(y_trm, DEVICE))
                    bsz = max(8, min(BATCH_SIZE, (len(y_trm) // 2) + 1))

                    tr_loader_tt = DataLoader(
                        make_aligned_tensor_dataset(
                            safe_tensor(Xc_trm), torch.from_numpy(Xk_trm).long(),
                            torch.zeros((len(y_trm), 0), dtype=torch.float32),
                            safe_tensor(y_trm.astype("float32")).view(-1),
                        ),
                        batch_size=bsz, shuffle=True, drop_last=False,
                        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                    )
                    va_loader_tt = DataLoader(
                        make_aligned_tensor_dataset(
                            safe_tensor(Xc_val), torch.from_numpy(Xk_val).long(),
                            torch.zeros((len(y_val), 0), dtype=torch.float32),
                            safe_tensor(y_val.astype("float32")).view(-1),
                        ),
                        batch_size=bsz, shuffle=False, drop_last=False,
                        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                    )
                    tt_wrapped, _, _ = train_eval(
                        tt_wrapped, tr_loader_tt, va_loader_tt, opt, sched, crit,
                        PATIENCE, EPOCHS_FINAL, prefix="TabTransformer",
                    )
                    cal_loader_tt = DataLoader(
                        TensorDataset(safe_tensor(Xc_cal), torch.from_numpy(Xk_cal).long(), torch.zeros((len(y_cal), 1)), torch.from_numpy(y_cal).float()),
                        batch_size=bsz,
                    )
                    te_loader_tt = DataLoader(
                        TensorDataset(safe_tensor(Xc_te), torch.from_numpy(Xk_te).long(), torch.zeros((len(y_te), 1)), torch.from_numpy(y_te).float()),
                        batch_size=bsz,
                    )
                    tt.eval()
                    p_cal, p_te = [], []
                    with torch.no_grad():
                        for xc_b, xk_b, _, _ in cal_loader_tt:
                            p_cal.extend(torch.sigmoid(tt(xc_b.to(DEVICE), xk_b.to(DEVICE))).cpu().numpy().ravel())
                        for xc_b, xk_b, _, _ in te_loader_tt:
                            p_te.extend(torch.sigmoid(tt(xc_b.to(DEVICE), xk_b.to(DEVICE))).cpu().numpy().ravel())
                else:
                    clf = clone(clf_instance)
                    if name == "XGB":
                        clf.fit(X_trm_df.values, y_trm, eval_set=[(X_val_df.values, y_val)], verbose=False)
                    elif name == "CatBoost":
                        clf.fit(X_trm_df, y_trm, eval_set=[(X_val_df, y_val)], early_stopping_rounds=20, verbose=False)
                    elif name == "TabNet":
                        clf.fit(X_trm_df.values, y_trm, eval_set=[(X_val_df.values, y_val)], max_epochs=200, patience=50)
                    else:
                        clf.fit(X_trm_df, y_trm)
                    if name in ("XGB", "CatBoost", "TabNet", "RF"):
                        trained_baselines_dict[name] = clf
                    p_cal = clf.predict_proba(X_cal_df if name not in ("XGB", "TabNet") else X_cal_df.values)[:, 1]
                    p_te = clf.predict_proba(X_te_df if name not in ("XGB", "TabNet") else X_te_df.values)[:, 1]

                calib_method = "cv_isotonic" if (ds_name.upper().startswith(("NHANES_DIABETES", "NHANES_CKD")) and len(y_cal) >= 200) else "auto"
                p_cal_cal = crossfit_calibrated(y_cal, p_cal, method=calib_method)
                p_te_cal = calibrate_probs(y_cal, p_cal, p_te, method=calib_method)
                t_star = bootstrap_best_threshold(y_cal, p_cal_cal, n=1000, seed=SEED)

                if fold == 1:
                    cal_probs_fold1[name] = np.asarray(p_cal_cal).ravel()
                    te_probs_fold1[name] = np.asarray(p_te_cal).ravel()

                mt = metrics_dict(y_te, p_te_cal, threshold=t_star, y_prob_raw=p_te)
                mt.update(compute_rule_alignment(p_te_cal, t_star, R_te_datamined, rule_signs_datamined))
                fold_metrics[name].append(mt)
                per_fold_records.append({"Dataset": ds_name, "Fold": fold, "Model": name, **mt})
                p_te_flat = np.asarray(p_te_cal).ravel()
                for j, ridx in enumerate(te_idx):
                    oof_records.append({
                        "Dataset": ds_name, "Fold": fold, "RowIndex": int(ridx),
                        "Model": name, "y_true": int(y_te[j]),
                        "p": float(p_te_flat[j]), "threshold": float(t_star),
                    })
                print(f"{name} TEST: AUC {mt['AUC']:.4f} | F1 {mt['F1']:.4f} | BAcc {mt['BalancedAcc']:.4f}")
            except Exception as exc:
                print(f"Baseline {name} failed: {exc}")
                fold_metrics[name].append({})

        # KAR-Net and architectural variants
        variants = {
            "Transformer_NoRules": {"no_rules": True, "concat_rules": False},
            "Transformer_ConcatRules": {"no_rules": False, "concat_rules": True},
            "Transformer_KARNet": {"no_rules": False, "concat_rules": False},
            "Transformer_KARNet_Lam0": {"no_rules": False, "concat_rules": False},
        }

        for mname, abl in variants.items():
            seed_everything(experiment_seed(fold, "transformer", "init"))
            print(f"\n*** Training {mname} ***")

            if abl.get("no_rules"):
                R_trm_cur = np.zeros_like(R_trm_dm)
                R_val_cur = np.zeros_like(R_val_dm)
                R_cal_cur = np.zeros_like(R_cal_dm)
                R_te_cur = np.zeros_like(R_te_datamined)
                rule_signs_cur = np.array([], dtype=np.float32)
                rule_qual_cur = None
            else:
                R_trm_cur, R_val_cur, R_cal_cur, R_te_cur = R_trm_dm, R_val_dm, R_cal_dm, R_te_datamined
                rule_signs_cur = rule_signs_datamined
                rule_qual_cur = (
                    np.full_like(global_rule_quality_datamined, 0.50, dtype=np.float32)
                    if mname.endswith("_Lam0")
                    else global_rule_quality_datamined
                )

            model = KARNet(
                cat_cards,
                len(cont_idx),
                R_trm_cur.shape[1],
                **best_mdl_par,
                rule_signs=rule_signs_cur,
                global_rule_quality=rule_qual_cur,
                delta_max=best_params["delta_max"],
                run_abl=abl,
            )

            model_dict = model.state_dict()
            model_dict.update({
                k: v for k, v in pretrained_weights.items()
                if k in model_dict
                and not k.startswith(("head.", "rule_head.", "reconstruction_head."))
                and model_dict[k].shape == v.shape
            })
            model.load_state_dict(model_dict, strict=False)
            model.to(DEVICE)
            initialize_global_checkpoint(model)

            crit = BalancedFocalLoss(pos_weight=make_pos_weight(y_trm, DEVICE), gamma=best_params["focal_gamma"])
            bsz = max(8, min(BATCH_SIZE, (len(y_trm) // 2) + 1))

            is_lam0 = mname.endswith("_Lam0")
            common_train_kwargs = {
                "rule_lambda": 0.0 if is_lam0 else best_params["rule_lambda"],
                "reliability_lambda": 0.0 if is_lam0 else best_params["reliability_lambda"],
                "consistency_lambda": 0.0 if is_lam0 else best_params["consistency_lambda"],
                "lambda_rank": best_params["lambda_rank"],
                "safeguard_lambda": best_params["safeguard_lambda"],
                "rank_temperature": best_params["rank_temperature"],
                "rank_max_pairs": best_params["rank_max_pairs"],
                "consistency_threshold": best_params["consistency_threshold"],
                "consistency_temperature": best_params["consistency_temperature"],
                "safeguard_backbone_margin": best_params["safeguard_backbone_margin"],
                "safeguard_final_margin": best_params["safeguard_final_margin"],
            }

            tr_loader = DataLoader(
                TensorDataset(safe_tensor(Xc_trm), torch.from_numpy(Xk_trm).long(), safe_tensor(R_trm_cur), torch.from_numpy(y_trm).float()),
                batch_size=bsz, shuffle=True, drop_last=False,
                pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS,
                generator=make_torch_generator(experiment_seed(fold, "transformer", "loader")),
            )
            va_loader = DataLoader(
                TensorDataset(safe_tensor(Xc_val), torch.from_numpy(Xk_val).long(), safe_tensor(R_val_cur), torch.from_numpy(y_val).float()),
                batch_size=bsz, shuffle=False, drop_last=False,
                pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS,
            )

            # Stage 1: data backbone training
            seed_everything(experiment_seed(fold, "transformer", "backbone"))
            configure_training_stage(model, "backbone")
            opt1 = build_stage_optimizer(model, "backbone", best_params)
            sched1 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt1, mode="max", patience=10, factor=0.5, min_lr=MIN_LR)
            model, _, _ = train_eval(
                model, tr_loader, va_loader, opt1, sched1, crit,
                PATIENCE, best_params["stage1_epochs"], prefix=mname, stage="backbone",
                **common_train_kwargs,
            )
            stage1_best_backbone_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            # Stage 2: symbolic-head training (frozen backbone)
            if model.rule_head is not None:
                seed_everything(experiment_seed(fold, "transformer", "rule"))
                configure_training_stage(model, "rule")
                opt2 = build_stage_optimizer(model, "rule", best_params)
                sched2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, mode="max", patience=5, factor=0.5, min_lr=MIN_LR)
                model, _, _ = train_eval(
                    model, tr_loader, va_loader, opt2, sched2, crit,
                    12, best_params["stage2_epochs"], prefix=mname, stage="rule",
                    **common_train_kwargs,
                )
                stage2_best_rule_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k.startswith("rule_head.")
                }
                merged_stage3 = dict(stage1_best_backbone_state)
                merged_stage3.update(stage2_best_rule_state)
                model.load_state_dict({k: v.to(DEVICE) for k, v in merged_stage3.items()}, strict=False)

            # Stage 3: low-rate refinement
            seed_everything(experiment_seed(fold, "transformer", "joint"))
            configure_training_stage(model, "joint")
            opt3 = build_stage_optimizer(model, "joint", best_params)
            sched3 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt3, mode="max", patience=8, factor=0.5, min_lr=MIN_LR)
            model, _, _ = train_eval(
                model, tr_loader, va_loader, opt3, sched3, crit,
                max(10, min(PATIENCE, best_params["stage3_epochs"])),
                best_params["stage3_epochs"], prefix=mname, stage="joint",
                **common_train_kwargs,
            )

            if model.rule_head is not None:
                build_deployed_hybrid_ensemble(model, va_loader, model_name=mname, fold=fold)

            restore_global_checkpoint(model, prefix=mname)

            # Verify neutral-rule identity across all deployed checkpoints
            if model.rule_head is not None:
                for _, _, dstate in getattr(model, "_deploy_states", []):
                    model.load_state_dict({k: v.to(DEVICE) for k, v in dstate.items()}, strict=False)
                    assert_neutral_rule_invariant(model, va_loader, atol=1e-7)
                if getattr(model, "_deploy_states", None):
                    model.load_state_dict({k: v.to(DEVICE) for k, v in model._deploy_states[0][2].items()}, strict=False)

            cal_loader = DataLoader(
                TensorDataset(safe_tensor(Xc_cal), torch.from_numpy(Xk_cal).long(), safe_tensor(R_cal_cur), torch.from_numpy(y_cal).float()),
                batch_size=bsz,
            )
            te_loader = DataLoader(
                TensorDataset(safe_tensor(Xc_te), torch.from_numpy(Xk_te).long(), safe_tensor(R_te_cur), torch.from_numpy(y_te).float()),
                batch_size=bsz,
            )

            # Inference latency benchmark on fold 1
            if fold == 1 and mname == "Transformer_KARNet":
                model.eval()
                deploy_n = max(1, len(getattr(model, "_deploy_states", []) or []))
                t0_inf = time.time()
                with torch.no_grad():
                    for _ in range(deploy_n):
                        for xc_b, xk_b, xr_b, _ in te_loader:
                            _ = model(xc_b.to(DEVICE), xk_b.to(DEVICE), xr_b.to(DEVICE))
                ms_per_patient = ((time.time() - t0_inf) / (len(y_te) * deploy_n)) * 1000.0
                print(f"[{mname}] Inference latency: {ms_per_patient:.4f} ms/patient (batch_size={bsz})")
                per_fold_records.append({"Dataset": ds_name, "Fold": fold, "Model": mname, "Latency_ms": ms_per_patient})

            # Collect symbolic audit candidates for KAR-Net
            if mname == "Transformer_KARNet":
                fold_cands = run_clinical_case_study(
                    model, trained_baselines_dict, X_te_raw, X_te_df, X_te_np, y_te,
                    feat_names, ds_name, fold, data_mined_rules, DEVICE,
                    fold_units=fold_units, fold_cat_maps=fold_cat_maps,
                )
                if fold_cands:
                    all_case_candidates.extend(fold_cands)

            p_cal = predict_deployed_proba(model, cal_loader)
            p_te = predict_deployed_proba(model, te_loader)

            # Export per-rule mean contextual reliability weights on outer test
            if mname == "Transformer_KARNet" and getattr(model, "rule_head", None) is not None:
                snaps = getattr(model, "_snapshots", None) or []
                if snaps:
                    model.load_state_dict({k: v.to(DEVICE) for k, v in snaps[0][2].items()})
                model.eval()
                alpha_sum, n_seen = None, 0
                with torch.no_grad():
                    for xc_b, xk_b, xr_b, _ in te_loader:
                        rel_b = model(xc_b.to(DEVICE), xk_b.to(DEVICE), xr_b.to(DEVICE))[2]
                        if rel_b is None or rel_b.numel() == 0:
                            continue
                        rs = rel_b.cpu().numpy().sum(axis=0)
                        alpha_sum = rs if alpha_sum is None else alpha_sum + rs
                        n_seen += rel_b.shape[0]
                if alpha_sum is not None and n_seen > 0:
                    r_te_arr = np.asarray(R_te_cur, dtype=np.float32)
                    y_te_arr = np.asarray(y_te).astype(int)
                    rows = []
                    for i_r in range(min(len(alpha_sum), r_te_arr.shape[1])):
                        m_r = r_te_arr[:, i_r] > 0.5
                        rd_r = float(y_te_arr[m_r].mean() - y_te_arr[~m_r].mean()) if (m_r.any() and (~m_r).any()) else 0.0
                        rows.append({
                            "rule_idx": i_r,
                            "rule": " AND ".join(f"{feat_names[f]} {op} {v:.4f}" for f, op, v in data_mined_rules[i_r]),
                            "n_conditions": len(data_mined_rules[i_r]),
                            "sign": float(rule_signs_cur[i_r]),
                            "mean_alpha_test": float(alpha_sum[i_r]) / float(n_seen),
                            "coverage_test": float(m_r.mean()),
                            "risk_diff_test": rd_r,
                        })
                    pd.DataFrame(rows).to_csv(
                        os.path.join(output_dir, f"{ds_name}_rule_weight_mean_fold{fold}.csv"), index=False
                    )

            calib_method = "cv_isotonic" if (ds_name.upper().startswith(("NHANES_DIABETES", "NHANES_CKD")) and len(y_cal) >= 200) else "auto"
            p_cal_cal = crossfit_calibrated(y_cal, p_cal, method=calib_method)
            p_te_cal = calibrate_probs(y_cal, p_cal, p_te, method=calib_method)
            t_star = bootstrap_best_threshold(y_cal, p_cal_cal, n=1000, seed=SEED)

            mt = metrics_dict(y_te, p_te_cal, threshold=t_star, y_prob_raw=p_te)
            mt.update(compute_rule_alignment(p_te_cal, t_star, R_te_datamined, rule_signs_datamined))
            fold_metrics[mname].append(mt)
            per_fold_records.append({"Dataset": ds_name, "Fold": fold, "Model": mname, **mt})
            p_te_flat = np.asarray(p_te_cal).ravel()
            for j, ridx in enumerate(te_idx):
                oof_records.append({
                    "Dataset": ds_name, "Fold": fold, "RowIndex": int(ridx),
                    "Model": mname, "y_true": int(y_te[j]),
                    "p": float(p_te_flat[j]), "threshold": float(t_star),
                })
            print(f"{mname} TEST: AUC {mt['AUC']:.4f} | F1 {mt['F1']:.4f} | BAcc {mt['BalancedAcc']:.4f}")

            # Fold-1 Decision Curve Analysis (comparators selected strictly on calibration AUC)
            if mname == "Transformer_KARNet" and fold == 1 and cal_probs_fold1:
                cal_auc = {
                    nm: roc_auc_score(y_cal, cal_probs_fold1[nm])
                    for nm in cal_probs_fold1
                    if len(np.unique(y_cal)) > 1
                }
                best_alt = sorted(cal_auc, key=cal_auc.get, reverse=True)[:3]
                alt_te_dict = {nm: te_probs_fold1[nm] for nm in best_alt}
                plot_single_dca(
                    y_true=y_te,
                    kar_probs=p_te_cal,
                    alt_probs_dict=alt_te_dict,
                    output_path=os.path.join(output_dir, f"Fig_DCA_{ds_name}.png"),
                    save_csv_path=os.path.join(output_dir, f"{ds_name}_dca_fold1.csv"),
                )

    # Save out-of-fold predictions, per-fold metrics, summary CIs, and Wilcoxon/FDR statistics
    if oof_records:
        pd.DataFrame(oof_records).to_csv(os.path.join(output_dir, f"{ds_name}_oof_predictions.csv"), index=False)
    if per_fold_records:
        pd.DataFrame(per_fold_records).to_csv(os.path.join(output_dir, f"{ds_name}_per_fold_metrics.csv"), index=False)

    avg = {}
    for m, folds in fold_metrics.items():
        valid = [d for d in folds if isinstance(d, dict) and d]
        if not valid:
            continue
        dfm = pd.DataFrame(valid)
        rows = {}
        for metric in dfm.columns:
            if dfm[metric].dtype == object:
                continue
            mean, lo, hi = compute_ci(dfm[metric].values)
            rows[metric] = "N/A" if np.isnan(mean) else f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"
        avg[m] = rows

    if avg:
        avg_df = pd.DataFrame(avg).T
        avg_df.to_csv(os.path.join(output_dir, f"{ds_name}_summary_ci.csv"))
        print(f"\n=== RESULTS SUMMARY FOR {ds_name} ===")
        print(avg_df.to_string())

        main_model = "Transformer_KARNet"
        stats_rows = []
        for met in ("AUC", "F1", "BalancedAcc", "ECE", "NLL"):
            main_vals = [d.get(met, np.nan) for d in fold_metrics.get(main_model, []) if np.isfinite(d.get(met, np.nan))]
            if len(main_vals) < 2:
                continue
            pvals, models = [], []
            for m, folds in fold_metrics.items():
                if m == main_model or not folds:
                    continue
                ov = [d.get(met, np.nan) for d in folds if np.isfinite(d.get(met, np.nan))]
                if len(ov) == len(main_vals) and len(ov) >= 2:
                    pvals.append(safe_wilcoxon(main_vals, ov))
                    models.append(m)
            if pvals:
                adj = fdr_bh(pvals)
                for mdl, pr, pa in zip(models, pvals, adj):
                    stats_rows.append({"Metric": met, "Model": mdl, "p": pr, "FDR": pa})
        if stats_rows:
            pd.DataFrame(stats_rows).to_csv(os.path.join(output_dir, f"{ds_name}_wilcoxon_fdr.csv"), index=False)

    # Export and plot illustrative symbolic audit record (Figure 3)
    if all_case_candidates:
        true_rescues = [c for c in all_case_candidates if c["p_kar"] < 0.5]
        pool = true_rescues if true_rescues else all_case_candidates
        pool.sort(key=lambda x: x["gap"], reverse=True)
        winner = pool[0]

        active_idx = np.where(winner["xr"][0] > 0)[0]
        winner_rules = winner["fold_rules"]
        rows = []
        for i_r in active_idx:
            w = float(winner["weights"][i_r]) if len(winner["weights"]) > i_r else 1.0
            if i_r < len(winner_rules):
                raw_str = rule_to_clinical_str(
                    simplify_rule(winner_rules[i_r]),
                    feat_names,
                    winner.get("fold_units"),
                    winner.get("fold_cat_maps"),
                    prec=2,
                    scaled_prec=3,
                )
                rows.append({
                    "Patient_ID": winner["pid"],
                    "Fold": winner["fold"],
                    "Baseline_Prob": round(winner["p_base"], 4),
                    "KARNet_Prob": round(winner["p_kar"], 4),
                    "Status": "SUPPRESSED" if w < 0.5 else "ACTIVE",
                    "Weight": w,
                    "Rule": raw_str,
                    "Label": format_rule_human_readable(raw_str),
                })

        if rows:
            df_case = pd.DataFrame(rows).sort_values("Weight")
            df_case.to_csv(os.path.join(output_dir, f"{ds_name}_case_study_rules.csv"), index=False)
            plot_case_study_figure(
                df_case,
                fold=winner["fold"],
                p_base=winner["p_base"],
                p_kar=winner["p_kar"],
                output_path=os.path.join(output_dir, f"Fig_CaseStudy_{ds_name}.png"),
            )

    return per_fold_records
