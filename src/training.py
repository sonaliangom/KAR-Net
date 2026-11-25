"""Self-supervised pre-training, three-stage optimization, checkpoint averaging, and invariant checks."""

from contextlib import nullcontext
import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from src.config import DEVICE, GRAD_CLIP, SEED, USE_AMP_CUDA
from src.models import (
    SelfSupervisedLoss,
    pairwise_auc_loss,
    ranking_preservation_loss,
    safe_tensor,
    selective_soft_consistency_loss,
)


def get_autocast():
    """Return CUDA float16 autocast only when explicitly enabled (disabled for supervised stages)."""
    if DEVICE.type == "cuda" and USE_AMP_CUDA:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def unpack_batch(batch):
    if isinstance(batch, (list, tuple)):
        if len(batch) == 4:
            xc, xk, xr, yb = batch
            return xc, xk, xr, yb, None
        if len(batch) == 5:
            return batch
    raise ValueError(f"Expected 4-tuple or 5-tuple batch, received {type(batch)}")


def make_aligned_tensor_dataset(xc, xk, xr, y):
    if xr.dim() == 1:
        xr = xr.view(-1, 1)
    n = min(xc.shape[0], xk.shape[0], xr.shape[0], y.shape[0])
    if n == 0:
        raise ValueError("No samples remain after tensor alignment.")
    return TensorDataset(xc[:n], xk[:n], xr[:n], y[:n])


def pretrain_model(
    model,
    X_tr_np: np.ndarray,
    cont_idx,
    cat_idx,
    cat_cards,
    mask_prob: float = 0.15,
    pretrain_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
):
    """Self-supervised denoising pre-training (SSP) on the 64% model-training partition."""
    model.to(DEVICE)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=pretrain_epochs, eta_min=1e-5)
    criterion = SelfSupervisedLoss(cat_cards, len(cont_idx)).to(DEVICE)

    x_cont_all = torch.from_numpy(X_tr_np[:, cont_idx]).float()
    x_cat_all = torch.from_numpy(X_tr_np[:, cat_idx]).long()

    amp_enabled = torch.cuda.is_available()
    try:
        from torch.cuda.amp import GradScaler, autocast
    except Exception:
        amp_enabled = False
        autocast = lambda enabled=False: nullcontext()
        GradScaler = None
    scaler = GradScaler(enabled=amp_enabled) if GradScaler is not None else None

    for epoch in range(1, pretrain_epochs + 1):
        model.train()
        x_cont_masked = x_cont_all.clone()
        x_cat_masked = x_cat_all.clone()

        mask_cont = torch.rand(x_cont_all.shape[0], x_cont_all.shape[1]) < mask_prob
        mask_cat = torch.rand(x_cat_all.shape[0], x_cat_all.shape[1]) < mask_prob
        x_cont_masked[mask_cont] = 0.0
        x_cat_masked[mask_cat] = 0

        x_cont_masked, x_cat_masked = x_cont_masked.to(DEVICE), x_cat_masked.to(DEVICE)
        x_cont_true, x_cat_true = x_cont_all.to(DEVICE), x_cat_all.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        dummy_rules = torch.zeros(x_cont_masked.size(0), 0, device=x_cont_masked.device)

        with autocast(enabled=amp_enabled):
            reconstructed = model(x_cont_masked, x_cat_masked, dummy_rules, pretraining=True)
            loss = criterion(reconstructed, x_cont_true, x_cat_true)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            print(f"[SSP] Epoch {epoch:2d}/{pretrain_epochs} | Reconstruction Loss: {loss.item():.4f}")

    return model


def assert_finite_training_components(prefix, stage, epoch, **tensors):
    """Abort immediately with component diagnostics if any output or loss is non-finite."""
    invalid = []
    for name, value in tensors.items():
        if value is None or not torch.is_tensor(value) or value.numel() == 0:
            continue
        finite_mask = torch.isfinite(value)
        if not bool(finite_mask.all().item()):
            invalid.append({
                "name": name,
                "shape": tuple(value.shape),
                "finite_fraction": float(finite_mask.float().mean().cpu().item()),
            })
    if invalid:
        raise RuntimeError(
            f"{prefix}: non-finite component at stage={stage}, epoch={epoch}:\n"
            f"{json.dumps(invalid, indent=2)}"
        )


def assert_finite_gradients(model, prefix, stage, epoch):
    """Abort immediately if any trainable parameter receives non-finite gradients."""
    invalid = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if not bool(torch.isfinite(param.grad).all().item()):
                invalid.append({"parameter": name, "shape": tuple(param.grad.shape)})
    if invalid:
        raise RuntimeError(
            f"{prefix}: non-finite gradient at stage={stage}, epoch={epoch}:\n"
            f"{json.dumps(invalid, indent=2)}"
        )


def initialize_global_checkpoint(model):
    model._global_best_auc = -np.inf
    model._global_best_stage = None
    model._global_best_epoch = None
    model._global_best_state = None
    model._stage_snapshots = {}
    model._deploy_states = []


def update_global_checkpoint(model, validation_auc, stage, epoch, state):
    if np.isfinite(validation_auc) and validation_auc > model._global_best_auc:
        model._global_best_auc = float(validation_auc)
        model._global_best_stage = str(stage)
        model._global_best_epoch = int(epoch)
        model._global_best_state = {k: v.detach().cpu().clone() for k, v in state.items()}


def restore_global_checkpoint(model, prefix: str = "", n_checkpoints: int = 3):
    """Restore the best validation state and select top-3 checkpoints for probability averaging."""
    if model._global_best_state is None:
        raise RuntimeError(f"{prefix}: no global checkpoint recorded.")
    model.load_state_dict({k: v.to(DEVICE) for k, v in model._global_best_state.items()}, strict=False)
    model.symbolic_enabled = True

    stage_states = list(getattr(model, "_stage_snapshots", {}).get(model._global_best_stage, []))
    stage_states.sort(key=lambda item: (-item[0], item[1]))
    deploy_states = [
        (float(auc), int(ep), {k: v.detach().cpu().clone() for k, v in st.items()})
        for auc, ep, st in stage_states[: int(n_checkpoints)]
    ]
    if not any(ep == model._global_best_epoch for _, ep, _ in deploy_states):
        deploy_states = [(
            model._global_best_auc,
            model._global_best_epoch,
            {k: v.detach().cpu().clone() for k, v in model._global_best_state.items()},
        )] + [s for s in deploy_states if s[1] != model._global_best_epoch]
        deploy_states = deploy_states[: int(n_checkpoints)]

    model._deploy_states = deploy_states
    model._snapshots = list(deploy_states)
    print(
        f"[CHECKPOINT] {prefix} | stage={model._global_best_stage} | "
        f"epoch={model._global_best_epoch} | val_AUC={model._global_best_auc:.4f} | "
        f"top-{len(deploy_states)} epochs={[ep for _, ep, _ in deploy_states]}"
    )
    return model


def predict_deployed_proba(model, loader) -> np.ndarray:
    """Average sigmoid probabilities across the deployed top-3 validation checkpoints."""
    model.eval()
    states = getattr(model, "_deploy_states", None) or [(
        float("nan"),
        0,
        {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    )]
    if getattr(model, "rule_head", None) is not None:
        model.symbolic_enabled = True

    probabilities = None
    with torch.no_grad():
        for _, _, state in states:
            model.load_state_dict({k: v.to(DEVICE) for k, v in state.items()}, strict=False)
            model.eval()
            run = []
            for batch in loader:
                xc, xk, xr, _, _ = unpack_batch(batch)
                output = model(xc.to(DEVICE), xk.to(DEVICE), xr.to(DEVICE))
                run.extend(torch.sigmoid(output[0]).cpu().numpy().ravel())
            run = np.asarray(run, dtype=float)
            probabilities = run if probabilities is None else probabilities + run

    model.load_state_dict({k: v.to(DEVICE) for k, v in states[0][2].items()}, strict=False)
    return probabilities / float(len(states))


def predict_deployed_single(model, xc, xk, xr):
    """Return checkpoint-averaged probability and contextual reliability vector for one instance."""
    states = getattr(model, "_deploy_states", None) or []
    if not states:
        return None, None
    if getattr(model, "rule_head", None) is not None:
        model.symbolic_enabled = True
    model.eval()
    param_device = next(model.parameters()).device
    probabilities, reliabilities = [], []
    with torch.no_grad():
        for _, _, state in states:
            model.load_state_dict({k: v.to(param_device) for k, v in state.items()}, strict=False)
            model.eval()
            output = model(
                safe_tensor(xc).to(param_device),
                torch.from_numpy(xk).long().to(param_device),
                safe_tensor(xr).to(param_device),
                return_components=True,
            )
            probabilities.append(float(torch.sigmoid(output["logit"]).item()))
            reliabilities.append(output["reliability"].detach().cpu().numpy()[0])

    model.load_state_dict({k: v.to(param_device) for k, v in states[0][2].items()}, strict=False)
    return float(np.mean(probabilities)), np.mean(np.stack(reliabilities), axis=0)


def set_requires_grad(module, requires_grad: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = requires_grad


def configure_training_stage(model, stage: str) -> None:
    """Configure frozen/trainable submodules for Stage 1 (backbone), Stage 2 (rule), and Stage 3 (joint)."""
    if stage not in {"backbone", "rule", "joint"}:
        raise ValueError(f"Unknown training stage: {stage}")

    if getattr(model, "rule_head", None) is not None:
        model.symbolic_enabled = stage == "rule"
        if hasattr(model.rule_head, "residual_scale"):
            model.rule_head.residual_scale.fill_(0.25 if stage == "rule" else 0.0)
    else:
        model.symbolic_enabled = True

    for param in model.parameters():
        param.requires_grad = False

    if stage in {"backbone", "joint"}:
        set_requires_grad(getattr(model, "tok", None), True)
        if getattr(model, "cls_token", None) is not None:
            model.cls_token.requires_grad = True
        set_requires_grad(getattr(model, "tr", None), True)
        set_requires_grad(getattr(model, "head", None), True)
        set_requires_grad(getattr(model, "rules_token", None), True)
    elif stage == "rule":
        set_requires_grad(getattr(model, "rule_head", None), True)


def build_stage_optimizer(model, stage: str, config: dict) -> torch.optim.Optimizer:
    weight_decay = config["weight_decay"]

    if stage == "backbone":
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=config["stage1_lr"], weight_decay=weight_decay)

    if stage == "rule":
        params = [p for p in model.rule_head.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=config["stage2_rule_lr"], weight_decay=weight_decay)

    backbone_params = list(model.tok.parameters())
    if getattr(model, "cls_token", None) is not None:
        backbone_params.append(model.cls_token)
    if getattr(model, "tr", None) is not None:
        backbone_params += list(model.tr.parameters())
    backbone_params = [p for p in backbone_params if p.requires_grad]

    head_params = [p for p in model.head.parameters() if p.requires_grad] if getattr(model, "head", None) is not None else []
    rule_params = [p for p in model.rule_head.parameters() if p.requires_grad] if getattr(model, "rule_head", None) is not None else []
    concat_params = [p for p in model.rules_token.parameters() if p.requires_grad] if getattr(model, "rules_token", None) is not None else []

    groups = [{"params": backbone_params, "lr": config["stage3_backbone_lr"]}]
    if head_params:
        groups.append({"params": head_params, "lr": config["stage3_head_lr"]})
    if concat_params:
        groups.append({"params": concat_params, "lr": config["stage3_rule_lr"]})
    if rule_params:
        groups.append({"params": rule_params, "lr": config["stage3_rule_lr"]})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def train_eval(
    model,
    tr_loader,
    va_loader,
    optimizer,
    scheduler,
    criterion,
    patience: int,
    epochs: int,
    rule_lambda: float = 0.05,
    sparsity_lambda: float = 0.01,
    reliability_lambda: float = 0.05,
    consistency_lambda: float = 0.10,
    lambda_rank: float = 0.10,
    safeguard_lambda: float = 0.05,
    rank_temperature: float = 1.0,
    rank_max_pairs: int = 4096,
    consistency_threshold: float = 0.25,
    consistency_temperature: float = 1.0,
    safeguard_backbone_margin: float = 0.50,
    safeguard_final_margin: float = 0.10,
    prefix: str = "",
    stage: str = "joint",
):
    """Train one curriculum stage with early stopping on validation AUC and top-3 snapshot retention."""
    best_auc = -np.inf
    best_ep = 0
    bad = 0
    best_state = None

    model._snapshots = []
    if not hasattr(model, "_global_best_state"):
        initialize_global_checkpoint(model)

    for ep in range(1, epochs + 1):
        model.train()
        valid_batches = 0

        for batch in tr_loader:
            xc, xcat, xr, yb, _ = unpack_batch(batch)
            xc = xc.to(DEVICE)
            xcat = xcat.to(DEVICE)
            xr = xr.to(DEVICE)
            yb = yb.to(DEVICE).float().reshape(-1)

            optimizer.zero_grad(set_to_none=True)

            with get_autocast():
                output = model(xc, xcat, xr, return_components=True)
                logit = output["logit"]
                data_logit = output["data_logit"]
                delta = output["delta"]
                l1 = output["rule_l1"]
                rel = output["reliability"]
                rel_logits = output["reliability_logits"]

                loss_data_cls = criterion(data_logit, yb)
                loss_data_rank = pairwise_auc_loss(data_logit, yb, max_pairs=rank_max_pairs, temperature=rank_temperature)

                logit_hybrid = (0.35 * data_logit.detach() + delta) if stage == "rule" else (data_logit.detach() + delta)
                loss_rule_cls = criterion(logit_hybrid, yb)
                devices = [xc.device] if xc.device.type == "cuda" else []
                with torch.random.fork_rng(devices=devices):
                    loss_rule_rank = pairwise_auc_loss(logit_hybrid, yb, max_pairs=rank_max_pairs, temperature=rank_temperature)

                if stage in ("backbone", "joint"):
                    classification_loss = loss_data_cls
                    ranking_loss = loss_data_rank
                else:
                    classification_loss = loss_rule_cls
                    ranking_loss = loss_rule_rank

                reliability_loss = logit.new_zeros(())
                consistency_loss = logit.new_zeros(())
                safeguard_loss = logit.new_zeros(())

                rule_head = getattr(model, "rule_head", None)
                if rule_head is not None and rel_logits.numel() > 0 and stage == "rule":
                    active = (xr > 0.5).to(logit.dtype)
                    y_column = yb.view(-1, 1)
                    positive_rule = (rule_head.rule_signs.view(1, -1) > 0).to(logit.dtype)
                    hard_target = torch.where(positive_rule > 0.5, y_column, 1.0 - y_column)
                    prior_q = torch.clamp(rule_head.initial_global_quality.view(1, -1).to(logit.dtype), 0.05, 0.95)
                    target_reliability = 0.70 * prior_q + 0.30 * hard_target

                    per_rule_bce = F.binary_cross_entropy_with_logits(rel_logits, target_reliability, reduction="none")
                    bounded_rule_raw = torch.clamp(rule_head.rule_raw, min=-20.0, max=10.0)
                    global_quality = torch.clamp(F.softplus(bounded_rule_raw), min=1e-4, max=1.0)
                    prior_anchor = (global_quality.view(1, -1) - prior_q).pow(2).mean()
                    reliability_loss = (per_rule_bce * active).sum() / (active.sum() + 1e-6) + 0.5 * prior_anchor

                    consistency_loss = selective_soft_consistency_loss(
                        logits=logit_hybrid,
                        x_rules=xr,
                        rule_signs=rule_head.rule_signs,
                        global_quality=global_quality,
                        contextual_reliability=rel,
                        coherence_threshold=consistency_threshold,
                        temperature=consistency_temperature,
                        data_logits=data_logit,
                        targets=yb,
                    )

                    with torch.random.fork_rng(devices=devices):
                        safeguard_loss = ranking_preservation_loss(
                            data_logits=data_logit.detach(),
                            final_logits=logit_hybrid,
                            targets=yb,
                            backbone_margin=safeguard_backbone_margin,
                            safe_margin=safeguard_final_margin,
                            max_pairs=rank_max_pairs,
                        )

                total = (
                    classification_loss
                    + lambda_rank * ranking_loss
                    + rule_lambda * sparsity_lambda * l1
                    + reliability_lambda * reliability_loss
                    + consistency_lambda * consistency_loss
                    + safeguard_lambda * safeguard_loss
                )

            assert_finite_training_components(
                prefix=prefix,
                stage=stage,
                epoch=ep,
                logit=logit,
                data_logit=data_logit,
                delta=delta,
                l1=l1,
                rel=rel,
                rel_logits=rel_logits,
                classification_loss=classification_loss,
                ranking_loss=ranking_loss,
                reliability_loss=reliability_loss,
                consistency_loss=consistency_loss,
                safeguard_loss=safeguard_loss,
                total=total,
            )

            total.backward()
            assert_finite_gradients(model, prefix=prefix, stage=stage, epoch=ep)

            rule_params = (
                [p for p in model.rule_head.parameters() if p.requires_grad and p.grad is not None]
                if getattr(model, "rule_head", None) is not None
                else []
            )
            rule_param_ids = {id(p) for p in rule_params}
            base_params = [
                p for p in model.parameters()
                if p.requires_grad and p.grad is not None and id(p) not in rule_param_ids
            ]
            grad_norm = torch.tensor(0.0, device=DEVICE)
            if base_params:
                grad_norm = torch.nn.utils.clip_grad_norm_(base_params, GRAD_CLIP, error_if_nonfinite=True)
            if rule_params:
                r_norm = torch.nn.utils.clip_grad_norm_(rule_params, GRAD_CLIP, error_if_nonfinite=True)
                if not base_params:
                    grad_norm = r_norm

            if not torch.isfinite(torch.as_tensor(grad_norm)):
                raise RuntimeError(f"{prefix}: non-finite gradient norm at stage={stage}, epoch={ep}")

            optimizer.step()
            valid_batches += 1

        if valid_batches == 0:
            raise RuntimeError(f"{prefix}: no valid training batches.")

        model.eval()
        val_logits, val_targets = [], []
        with torch.no_grad():
            for batch in va_loader:
                xc, xcat, xr, yb, _ = unpack_batch(batch)
                out = model(xc.to(DEVICE), xcat.to(DEVICE), xr.to(DEVICE))
                val_logits.extend((out[0] if isinstance(out, tuple) else out).cpu().numpy().reshape(-1))
                val_targets.extend(yb.numpy().reshape(-1))

        if not val_targets:
            continue

        val_targets_arr = np.asarray(val_targets, dtype=int)
        val_logits_arr = np.asarray(val_logits, dtype=float)
        va_auc = roc_auc_score(val_targets_arr, val_logits_arr) if len(np.unique(val_targets_arr)) > 1 else 0.5

        if scheduler is not None:
            scheduler.step(va_auc)

        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        update_global_checkpoint(model, va_auc, stage, ep, state)

        model._snapshots.append((float(va_auc), int(ep), state))
        model._snapshots.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        model._snapshots = model._snapshots[:3]
        if not hasattr(model, "_stage_snapshots"):
            model._stage_snapshots = {}
        model._stage_snapshots[stage] = list(model._snapshots)

        if va_auc > best_auc and np.isfinite(va_auc):
            best_auc, best_ep, best_state, bad = float(va_auc), int(ep), state, 0
        else:
            bad += 1

        if ep % 10 == 0 or ep == epochs:
            print(f"{prefix} | stage={stage} | epoch={ep:3d} | val AUC={va_auc:.4f}")
        if bad >= patience:
            print(f"{prefix}: early stopping at epoch {ep}")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model, best_ep, best_auc


def build_deployed_hybrid_ensemble(model, val_loader, model_name: str, fold: int) -> None:
    """Pair top-3 backbone snapshots with top-3 Stage-2 rule snapshots and select residual scale eta^(f)."""
    if getattr(model, "rule_head", None) is None:
        return
    snaps = getattr(model, "_stage_snapshots", {})
    bb_snaps = list(snaps.get("backbone", []))
    jt_snaps = list(snaps.get("joint", []))
    rl_snaps = list(snaps.get("rule", []))
    if not bb_snaps or not rl_snaps:
        return

    peak_bb = max((s[0] for s in bb_snaps), default=-np.inf)
    peak_jt = max((s[0] for s in jt_snaps), default=-np.inf)
    best_base_stage = "joint" if (jt_snaps and peak_jt > peak_bb) else "backbone"
    base_list = jt_snaps if best_base_stage == "joint" else bb_snaps

    rule_pool = sorted(rl_snaps, key=lambda x: (-x[0], x[1]))
    k_ens = min(3, len(base_list), len(rule_pool))
    paired_states, d_logits_list, u_deltas_list = [], [], []
    y_v = None

    for idx_k in range(k_ens):
        b_auc, b_ep, b_st = base_list[idx_k]
        _, _, r_st = rule_pool[idx_k]
        merged_st = {k: v.detach().cpu().clone() for k, v in b_st.items()}
        for k, v in r_st.items():
            if k.startswith("rule_head."):
                merged_st[k] = v.detach().cpu().clone()
        merged_st["rule_head.residual_scale"] = torch.tensor(1.0, dtype=torch.float32)
        merged_st["rule_head.coherence_boost"] = torch.tensor(1.0, dtype=torch.float32)

        model.load_state_dict({k: v.to(DEVICE) for k, v in merged_st.items()}, strict=False)
        model.symbolic_enabled = True
        model.eval()
        dl_k, dt_k, y_run = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                xc, xk, xr, yb, _ = unpack_batch(batch)
                out = model(xc.to(DEVICE), xk.to(DEVICE), xr.to(DEVICE), return_components=True)
                dl_k.append(out["data_logit"].cpu().numpy().reshape(-1))
                dt_k.append(out["delta"].cpu().numpy().reshape(-1))
                if y_v is None:
                    y_run.append(yb.numpy().reshape(-1))
        if y_v is None:
            y_v = np.concatenate(y_run).astype(int)
        d_logits_list.append(np.concatenate(dl_k))
        u_deltas_list.append(np.concatenate(dt_k))
        paired_states.append([b_auc, int(b_ep), merged_st])

    if model_name.endswith("_Lam0"):
        best_eta = 0.55
        ens_prob = np.mean([
            1.0 / (1.0 + np.exp(-np.clip(dl_k + best_eta * dt_k, -10.0, 10.0)))
            for dl_k, dt_k in zip(d_logits_list, u_deltas_list)
        ], axis=0)
        best_raw_auc = float(roc_auc_score(y_v, ens_prob))
    else:
        rng_b = np.random.RandomState(SEED + int(fold))
        pos_v = np.flatnonzero(y_v == 1)
        neg_v = np.flatnonzero(y_v == 0)
        boot_indices = []
        if len(pos_v) > 1 and len(neg_v) > 1:
            for _ in range(50):
                bp = rng_b.choice(pos_v, size=len(pos_v), replace=True)
                bn = rng_b.choice(neg_v, size=len(neg_v), replace=True)
                boot_indices.append(np.concatenate([bp, bn]))

        eta_grid = np.linspace(0.04, 0.55, 35)
        best_eta, best_score, best_raw_auc = float(eta_grid[0]), -np.inf, -np.inf
        for eta in eta_grid:
            ens_prob = np.zeros_like(d_logits_list[0], dtype=np.float64)
            for dl_k, dt_k in zip(d_logits_list, u_deltas_list):
                logit_k = np.clip(dl_k + float(eta) * dt_k, -10.0, 10.0)
                ens_prob += 1.0 / (1.0 + np.exp(-logit_k))
            ens_prob /= float(k_ens)
            full_auc = float(roc_auc_score(y_v, ens_prob))
            full_ap = float(average_precision_score(y_v, ens_prob))
            med_auc = float(np.median([roc_auc_score(y_v[bi], ens_prob[bi]) for bi in boot_indices])) if boot_indices else full_auc
            score = 0.45 * full_auc + 0.40 * med_auc + 0.15 * full_ap - 0.0025 * (float(eta) ** 2)
            if score > best_score:
                best_score, best_eta, best_raw_auc = score, float(eta), full_auc

    for idx_k in range(k_ens):
        paired_states[idx_k][2]["rule_head.residual_scale"] = torch.tensor(best_eta, dtype=torch.float32)
        paired_states[idx_k][2]["rule_head.coherence_boost"] = torch.tensor(1.0, dtype=torch.float32)
        paired_states[idx_k][0] = float(best_raw_auc) - 1e-6 * idx_k

    model._stage_snapshots[best_base_stage] = [tuple(x) for x in paired_states]
    model._global_best_auc = float(best_raw_auc)
    model._global_best_stage = best_base_stage
    model._global_best_epoch = int(paired_states[0][1])
    model._global_best_state = {k: v.detach().cpu().clone() for k, v in paired_states[0][2].items()}
    print(f"[RESIDUAL SCALE] {model_name} | base_stage={best_base_stage} | eta*={best_eta:.3f} | ens_val_AUC={best_raw_auc:.4f}")


def assert_neutral_rule_invariant(model, loader, atol: float = 1e-7) -> float:
    """Verify exact neutral-rule identity: R_i = 0 => final_logit == data_logit (Eq. 20)."""
    if getattr(model, "rule_head", None) is None:
        return 0.0
    model.eval()
    max_error = 0.0
    with torch.no_grad():
        for batch in loader:
            xc, xcat, xr, _, _ = unpack_batch(batch)
            zero_rules = torch.zeros_like(xr.to(DEVICE))
            out = model(xc.to(DEVICE), xcat.to(DEVICE), zero_rules, return_components=True)
            err = (out["logit"] - out["data_logit"]).abs().max().item()
            max_error = max(max_error, err)
    if max_error > atol:
        raise AssertionError(f"Neutral-rule invariant failed: max_abs_diff={max_error:.12g}")
    print(f"[INVARIANT PASSED] Neutral-rule max_abs_diff={max_error:.12g}")
    return max_error
