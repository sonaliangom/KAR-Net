"""KAR-Net architecture modules, feature tokenizers, and training objectives."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import DEVICE, MAX_EMB


def safe_tensor(arr) -> torch.Tensor:
    """Convert array-like input to a finite float32 tensor clipped to [-10, 10]."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.size == 0:
        return torch.zeros((arr.shape[0], 0), dtype=torch.float32)
    cleaned = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
    return torch.from_numpy(np.clip(cleaned, -10.0, 10.0)).float()


def quality_to_raw_parameter(quality, eps: float = 1e-4) -> torch.Tensor:
    """Inverse softplus mapping for initializing the learnable global rule-quality vector."""
    q = torch.as_tensor(quality, dtype=torch.float32)
    q = q.clamp(min=eps, max=1.0)
    return torch.log(torch.expm1(q).clamp_min(eps))


def make_pos_weight(y, device) -> torch.Tensor:
    """Compute inverse-prevalence positive class weight capped at 10.0."""
    pos = float(y.sum())
    neg = float(len(y) - pos)
    return torch.tensor([min(neg / max(pos, 1e-6), 10.0)], device=device)


class StableFeatureTokenizer(nn.Module):
    """Per-feature linear numerical and embedding categorical tokenizer (used by TabTransformer baseline)."""

    def __init__(self, n_cont: int, cat_cards, d_emb: int):
        super().__init__()
        self.n_cont = n_cont
        self.d_emb = d_emb
        if n_cont:
            self.cont_w = nn.Parameter(torch.empty(n_cont, d_emb))
            self.cont_b = nn.Parameter(torch.zeros(n_cont, d_emb))
            nn.init.xavier_uniform_(self.cont_w)
            self.cont_norm = nn.LayerNorm(d_emb)
        else:
            self.register_parameter("cont_w", None)
            self.register_parameter("cont_b", None)
            self.cont_norm = None
        self.cat_embs = nn.ModuleList([nn.Embedding(c, d_emb) for c in cat_cards]) if cat_cards else None
        self.cat_norm = nn.LayerNorm(d_emb) if self.cat_embs else None

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        toks = []
        if self.n_cont and x_cont.numel() > 0:
            xc = x_cont.unsqueeze(-1) * self.cont_w.unsqueeze(0) + self.cont_b.unsqueeze(0)
            toks.append(self.cont_norm(xc))
        if self.cat_embs and x_cat.numel() > 0:
            cat_toks = torch.stack(
                [emb(x_cat[:, i].clamp(0, emb.num_embeddings - 1)) for i, emb in enumerate(self.cat_embs)],
                dim=1,
            )
            toks.append(self.cat_norm(cat_toks))
        if not toks:
            bsz = x_cont.shape[0] if x_cont.numel() > 0 else (x_cat.shape[0] if x_cat.numel() > 0 else 1)
            return torch.zeros(bsz, 1, self.d_emb, device=DEVICE)
        return torch.cat(toks, dim=1)


class PeriodicFeatureTokenizer(nn.Module):
    """Feature-specific periodic numerical tokenizer with a gated linear residual (Eq. 5)."""

    def __init__(
        self,
        n_cont: int,
        cat_cards,
        d_emb: int,
        n_frequencies=None,
        frequency_sigma: float = 0.10,
        frequency_floor: float = 1e-4,
    ):
        super().__init__()
        self.n_cont = int(n_cont)
        self.d_emb = int(d_emb)
        self.n_frequencies = int(n_frequencies or max(8, self.d_emb // 2))
        self.frequency_floor = float(frequency_floor)
        if self.n_cont > 0:
            initial = torch.empty(self.n_cont, self.n_frequencies)
            nn.init.normal_(initial, mean=0.0, std=float(frequency_sigma))
            initial = initial.abs().clamp_min(self.frequency_floor)
            self.frequency_raw = nn.Parameter(
                torch.log(torch.expm1(initial).clamp_min(self.frequency_floor))
            )
            self.periodic_weight = nn.Parameter(
                torch.empty(self.n_cont, 2 * self.n_frequencies, self.d_emb)
            )
            self.periodic_bias = nn.Parameter(torch.zeros(self.n_cont, self.d_emb))
            self.linear_weight = nn.Parameter(torch.empty(self.n_cont, self.d_emb))
            self.linear_gate = nn.Parameter(torch.tensor(0.0))
            nn.init.xavier_uniform_(self.periodic_weight)
            nn.init.xavier_uniform_(self.linear_weight)
            self.cont_norm = nn.LayerNorm(self.d_emb)
        else:
            for name in ("frequency_raw", "periodic_weight", "periodic_bias", "linear_weight", "linear_gate"):
                self.register_parameter(name, None)
            self.cont_norm = None
        self.cat_embs = (
            nn.ModuleList([nn.Embedding(int(c), self.d_emb) for c in cat_cards])
            if cat_cards
            else None
        )
        self.cat_norm = nn.LayerNorm(self.d_emb) if self.cat_embs else None

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        toks = []
        if self.n_cont > 0 and x_cont.numel() > 0:
            frequencies = F.softplus(self.frequency_raw) + self.frequency_floor
            phase = 2.0 * math.pi * x_cont.unsqueeze(-1) * frequencies.unsqueeze(0)
            periodic = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
            periodic_token = (
                torch.einsum("bfk,fkd->bfd", periodic, self.periodic_weight)
                + self.periodic_bias.unsqueeze(0)
            )
            linear_token = x_cont.unsqueeze(-1) * self.linear_weight.unsqueeze(0)
            xc = 0.25 * periodic_token + torch.sigmoid(self.linear_gate + 1.5) * linear_token
            toks.append(self.cont_norm(xc))
        if self.cat_embs and x_cat.numel() > 0:
            cat_toks = torch.stack(
                [emb(x_cat[:, i].clamp(0, emb.num_embeddings - 1)) for i, emb in enumerate(self.cat_embs)],
                dim=1,
            )
            toks.append(self.cat_norm(cat_toks))
        if not toks:
            bsz = x_cont.shape[0] if x_cont.numel() > 0 else x_cat.shape[0]
            return torch.zeros(bsz, 1, self.d_emb, device=x_cont.device, dtype=x_cont.dtype)
        return torch.cat(toks, dim=1)


class MultiScaleTransformer(nn.Module):
    """Multi-scale tabular Transformer combining a token-wise projection and pre-norm MHSA (Eq. 6)."""

    def __init__(self, emb: int, heads: int, layers: int):
        super().__init__()
        heads = min(heads, max(1, emb // 4))
        self.local = nn.Sequential(nn.Linear(emb, emb), nn.LayerNorm(emb), nn.GELU(), nn.Dropout(0.1))
        self.global_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=emb,
                nhead=heads,
                dim_feedforward=emb * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=layers,
            norm=nn.LayerNorm(emb),
        )
        self.out_norm = nn.LayerNorm(emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        return self.out_norm(self.local(x) + self.global_enc(x))


class AdaptiveRuleHead(nn.Module):
    """Prior-anchored contextual reliability gating, signed evidence, and bounded FiLM residual (Eqs. 15-18)."""

    def __init__(self, d_emb: int, n_rules: int, rule_signs=None, global_rule_quality=None, delta_max: float = 1.0):
        super().__init__()
        self.n_rules = int(n_rules)
        self.delta_max = float(delta_max)
        self.register_buffer("residual_scale", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("coherence_boost", torch.tensor(1.0, dtype=torch.float32))

        if self.n_rules <= 0:
            self.reliability_net = None
            self.trust_gate = None
            return

        signs = np.asarray(rule_signs if rule_signs is not None else np.ones(self.n_rules), dtype=np.float32)
        quality = np.asarray(
            global_rule_quality if global_rule_quality is not None else np.ones(self.n_rules),
            dtype=np.float32,
        )
        quality = np.clip(quality, 1e-4, 1.0)

        self.register_buffer("rule_signs", torch.from_numpy(signs))
        self.register_buffer("initial_global_quality", torch.from_numpy(quality))
        self.rule_raw = nn.Parameter(quality_to_raw_parameter(quality))

        self.reliability_net = nn.Sequential(
            nn.Linear(d_emb, d_emb), nn.GELU(), nn.LayerNorm(d_emb), nn.Dropout(0.10),
            nn.Linear(d_emb, self.n_rules),
        )
        nn.init.zeros_(self.reliability_net[-1].weight)
        nn.init.zeros_(self.reliability_net[-1].bias)

        self.temp = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.rule_encoder = nn.Sequential(
            nn.Linear(self.n_rules, d_emb, bias=False),
            nn.LayerNorm(d_emb),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.data_enc = nn.Sequential(
            nn.Linear(d_emb, d_emb),
            nn.LayerNorm(d_emb),
            nn.GELU(),
            nn.Dropout(0.10),
        )

        self.film_gamma = nn.Linear(d_emb, d_emb, bias=False)
        self.film_beta = nn.Linear(d_emb, d_emb, bias=False)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_beta.weight)

        self.joint_mlp = nn.Sequential(
            nn.Linear(d_emb * 2, d_emb), nn.GELU(), nn.LayerNorm(d_emb), nn.Dropout(0.10),
            nn.Linear(d_emb, d_emb // 2), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(d_emb // 2, 1),
        )
        nn.init.zeros_(self.joint_mlp[-1].weight)
        nn.init.zeros_(self.joint_mlp[-1].bias)

        self.rule_bypass_raw = nn.Parameter(torch.full((1, self.n_rules), -2.5, dtype=torch.float32))
        self.sym_gain = nn.Parameter(torch.tensor(0.55, dtype=torch.float32))
        self.forest_mix = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

        self.trust_gate = nn.Sequential(
            nn.Linear(d_emb, d_emb // 2), nn.GELU(), nn.LayerNorm(d_emb // 2), nn.Dropout(0.10),
            nn.Linear(d_emb // 2, 1),
        )
        nn.init.zeros_(self.trust_gate[-1].weight)
        nn.init.zeros_(self.trust_gate[-1].bias)

    def forward(self, x_rules: torch.Tensor, deep_feat: torch.Tensor):
        batch_size = deep_feat.size(0)
        device = deep_feat.device

        if self.reliability_net is None or x_rules is None or x_rules.numel() == 0:
            zero_rel = torch.zeros(batch_size, 0, device=device, dtype=deep_feat.dtype)
            zero_gate = torch.zeros(batch_size, 1, device=device, dtype=deep_feat.dtype)
            return zero_rel, zero_rel, zero_gate, zero_rel, zero_gate

        prior_q = torch.clamp(self.initial_global_quality.to(deep_feat.dtype), min=0.05, max=0.95)
        prior_logit = torch.logit(prior_q).unsqueeze(0)
        rel_logits = prior_logit + 1.5 * torch.tanh(self.reliability_net(deep_feat) / 1.5)
        reliability = torch.sigmoid(rel_logits)
        trust_gate = torch.sigmoid(self.trust_gate(deep_feat))

        active = (x_rules > 0.5).to(deep_feat.dtype)
        bounded_rule_raw = torch.clamp(self.rule_raw, min=-20.0, max=10.0)
        global_q = torch.clamp(F.softplus(bounded_rule_raw), min=1e-4, max=1.0).unsqueeze(0)
        signs = self.rule_signs.unsqueeze(0)

        pos_mask = (signs > 0).to(deep_feat.dtype)
        neg_mask = (signs < 0).to(deep_feat.dtype)
        pos_scale = 0.50 / torch.clamp(pos_mask.mean(), min=0.10)
        neg_scale = 0.50 / torch.clamp(neg_mask.mean(), min=0.10)
        pol_balance = pos_mask * pos_scale + neg_mask * neg_scale

        weights = active * reliability * global_q * pol_balance
        signed_evidence = weights * signs

        prior_weights = active * (prior_q.unsqueeze(0) ** 2) * pol_balance
        prior_signed = prior_weights * signs
        prior_forest = prior_signed.sum(dim=1, keepdim=True) / (prior_weights.sum(dim=1, keepdim=True) + 0.50)
        vote_margin = signed_evidence.sum(dim=1, keepdim=True) / (weights.sum(dim=1, keepdim=True) + 0.75)
        forest_margin = signed_evidence.sum(dim=1, keepdim=True) / math.sqrt(max(self.n_rules, 1))
        mix = torch.sigmoid(self.forest_mix)
        sym_margin = 0.55 * prior_forest + 0.45 * ((1.0 - mix) * vote_margin + mix * forest_margin)

        temp = torch.clamp(self.temp.abs(), 0.2, 5.0)
        r_emb = self.rule_encoder(signed_evidence / temp)
        d_repr = self.data_enc(deep_feat)
        modulated_rule_context = d_repr * torch.tanh(self.film_gamma(r_emb)) + self.film_beta(r_emb)
        joint_repr = torch.cat([d_repr, r_emb + modulated_rule_context], dim=1)

        pos_bypass_w = F.softplus(self.rule_bypass_raw) / float(max(self.n_rules, 1))
        bypass_term = F.linear(signed_evidence, pos_bypass_w)
        mlp_term = 0.15 * torch.tanh(self.joint_mlp(joint_repr))

        raw_residual = self.coherence_boost * (
            F.softplus(self.sym_gain) * sym_margin + bypass_term + mlp_term
        )
        return reliability, rel_logits, trust_gate, signed_evidence, raw_residual


class KARNet(nn.Module):
    """Knowledge-Adaptive Reasoning Network (supports KAR-Net, Lam0, No Rules, and ConCat Rules)."""

    def __init__(
        self,
        cat_cards,
        n_cont: int,
        n_rules: int,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        rule_signs=None,
        global_rule_quality=None,
        delta_max: float = 1.0,
        run_abl=None,
    ):
        super().__init__()
        self.run_abl = run_abl or {}
        self.symbolic_enabled = True
        self.backbone_type = "transformer"
        embed_dim = min(embed_dim, MAX_EMB)

        self.tok = PeriodicFeatureTokenizer(n_cont, cat_cards, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.tr = nn.Identity() if self.run_abl.get("no_attention") else MultiScaleTransformer(embed_dim, n_heads, n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.LayerNorm(embed_dim * 2), nn.Dropout(dropout * 0.5),
            nn.Linear(embed_dim * 2, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Dropout(dropout * 0.3),
            nn.Linear(embed_dim // 2, 1),
        )
        self.rule_head = (
            None
            if (self.run_abl.get("no_rules") or self.run_abl.get("concat_rules"))
            else AdaptiveRuleHead(
                embed_dim,
                n_rules,
                rule_signs=rule_signs,
                global_rule_quality=global_rule_quality,
                delta_max=delta_max,
            )
        )
        self.rules_token = (
            nn.Linear(n_rules, embed_dim, bias=False)
            if (self.run_abl.get("concat_rules") and n_rules > 0)
            else None
        )

        self.n_cont = n_cont
        self.cat_cards = cat_cards
        self.reconstruction_head = nn.Linear(embed_dim, self.n_cont + sum(self.cat_cards))

    def forward(self, x_cont, x_cat, x_rules, pretraining: bool = False, return_components: bool = False):
        if x_cat.numel() > 0:
            x_cat = x_cat.long()
        if x_rules is not None and x_rules.numel() > 0:
            x_rules = torch.clamp(x_rules, 0.0, 1.0)

        toks = self.tok(x_cont, x_cat)
        if self.rules_token is not None and x_rules is not None and x_rules.numel() > 0:
            toks = torch.cat([toks, self.rules_token(x_rules).unsqueeze(1)], dim=1)

        if isinstance(self.tr, nn.Identity):
            feat = toks.mean(dim=1)
        else:
            cls = self.cls_token.expand(toks.size(0), 1, -1)
            tr_out = self.tr(torch.cat([cls, toks], dim=1))
            if pretraining:
                feat = tr_out[:, 0]
            else:
                tok_attn = torch.softmax(
                    tr_out[:, 1:].norm(dim=-1) / math.sqrt(float(toks.size(-1))),
                    dim=-1,
                ).unsqueeze(-1)
                feat = tr_out[:, 0] + 0.50 * (tr_out[:, 1:] * tok_attn).sum(dim=1) + 0.30 * toks.mean(dim=1)

        if pretraining:
            return self.reconstruction_head(feat)

        base_data_logit = self.head(feat)

        delta = torch.zeros_like(base_data_logit)
        rel = torch.zeros(feat.size(0), 0, device=feat.device, dtype=feat.dtype)
        rel_logits = torch.zeros_like(rel)
        trust_gate = torch.zeros_like(base_data_logit)
        signed_evidence = torch.zeros_like(rel)
        l1 = torch.zeros((), device=feat.device, dtype=feat.dtype)

        if self.symbolic_enabled and self.rule_head is not None and x_rules is not None and x_rules.numel() > 0:
            rel, rel_logits, trust_gate, signed_evidence, raw_residual = self.rule_head(x_rules, feat.detach())

            # Neutral-rule identity guard (Eq. 20)
            active_rules = (x_rules > 0.5).to(base_data_logit.dtype)
            has_active_rule = (active_rules.sum(dim=1, keepdim=True) > 0).to(base_data_logit.dtype)

            # Bell-shaped data-confidence factor centered on mini-batch median (Section 4.4.2)
            centered_logit = base_data_logit.detach() - base_data_logit.detach().median()
            p_base = torch.sigmoid(centered_logit)
            dynamic_scale = (4.0 * p_base * (1.0 - p_base)).pow(2)

            dummy_link = (rel_logits.sum() * 0.0) if rel_logits.numel() > 0 else 0.0
            res_scale = getattr(self.rule_head, "residual_scale", 1.0)
            delta = (
                self.rule_head.delta_max
                * res_scale
                * torch.tanh(raw_residual)
                * trust_gate
                * dynamic_scale
                * has_active_rule
                + dummy_link
            )

            bounded_rule_raw = torch.clamp(self.rule_head.rule_raw, min=-20.0, max=10.0)
            l1 = torch.clamp(F.softplus(bounded_rule_raw), min=1e-4, max=1.0).mean()

        final_logit = torch.clamp(base_data_logit + delta, -10.0, 10.0)

        if return_components:
            return {
                "logit": final_logit,
                "data_logit": base_data_logit,
                "modulated_data_logit": base_data_logit,
                "delta": delta,
                "reliability": rel,
                "rule_l1": l1,
                "reliability_logits": rel_logits,
                "trust_gate": trust_gate,
                "signed_evidence": signed_evidence,
                "film_gamma": torch.ones_like(feat),
                "film_beta": torch.zeros_like(feat),
            }

        return final_logit, base_data_logit, rel, l1, rel_logits


class SelfSupervisedLoss(nn.Module):
    """Denoising full-profile reconstruction objective for SSP (Eq. 8)."""

    def __init__(self, cat_cards, n_cont: int, cont_lambda: float = 1.0):
        super().__init__()
        self.cat_cards = cat_cards
        self.n_cont = n_cont
        self.cont_lambda = cont_lambda
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, y_pred: torch.Tensor, y_cont_true: torch.Tensor, y_cat_true: torch.Tensor) -> torch.Tensor:
        pred_cont = y_pred[:, :self.n_cont]
        pred_cat_logits = y_pred[:, self.n_cont:]
        cont_loss = self.mse_loss(pred_cont, y_cont_true)
        cat_loss = 0.0
        current_idx = 0
        for i, card in enumerate(self.cat_cards):
            logits = pred_cat_logits[:, current_idx : current_idx + card]
            target = y_cat_true[:, i]
            cat_loss = cat_loss + self.ce_loss(logits, target)
            current_idx += card
        return (self.cont_lambda * cont_loss) + cat_loss


class BalancedFocalLoss(nn.Module):
    """Class-weighted binary focal loss with label smoothing."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight=None, label_smoothing: float = 0.01):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.ls = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.clamp(inputs, -10.0, 10.0).squeeze(-1)
        raw_targets = targets
        if self.ls > 0:
            targets = targets * (1.0 - self.ls) + 0.5 * self.ls
        bce_raw = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_raw)
        fw = (1.0 - pt) ** self.gamma
        if self.pos_weight is not None:
            pw = torch.clamp(torch.sqrt(self.pos_weight.to(inputs.dtype)), min=1.0, max=3.0)
            cw = torch.where(raw_targets > 0.5, pw, torch.ones_like(inputs))
        else:
            cw = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (cw * fw * bce_raw).mean()


def pairwise_auc_loss(logits: torch.Tensor, targets: torch.Tensor, max_pairs: int = 4096, temperature: float = 1.0) -> torch.Tensor:
    """Smooth pairwise logistic ranking loss over positive-negative pairs."""
    logits = logits.reshape(-1)
    targets = targets.reshape(-1).long()
    positive = logits[targets == 1]
    negative = logits[targets == 0]
    if positive.numel() == 0 or negative.numel() == 0:
        return logits.new_zeros(())
    differences = (positive[:, None] - negative[None, :]).reshape(-1)
    max_pairs = max(int(max_pairs), 1)
    if differences.numel() > max_pairs:
        indices = torch.randperm(differences.numel(), device=differences.device)[:max_pairs]
        differences = differences[indices]
    temperature = max(float(temperature), 1e-6)
    return F.softplus(-differences / temperature).mean()


def selective_soft_consistency_loss(
    logits: torch.Tensor,
    x_rules: torch.Tensor,
    rule_signs: torch.Tensor,
    global_quality: torch.Tensor,
    contextual_reliability: torch.Tensor,
    coherence_threshold: float = 0.25,
    temperature: float = 1.0,
    data_logits=None,
    targets=None,
) -> torch.Tensor:
    """Directional consistency and label-discordant rule suppression objective."""
    if x_rules is None or x_rules.numel() == 0 or contextual_reliability is None or contextual_reliability.numel() == 0:
        return logits.new_zeros(())

    active = (x_rules > 0.5).to(logits.dtype)
    weights = active * global_quality.unsqueeze(0) * contextual_reliability
    evidence = (weights * rule_signs.unsqueeze(0)).sum(dim=1) / (weights.sum(dim=1) + 1.0)

    if targets is not None:
        y_sign = 2.0 * targets.reshape(-1).to(logits.dtype) - 1.0
        rule_agrees = (rule_signs.unsqueeze(0) * y_sign.unsqueeze(1)) > 0
        contradicting_active = active * (~rule_agrees).to(logits.dtype)
        suppression_loss = (contradicting_active * contextual_reliability.pow(2)).sum() / (contradicting_active.sum() + 1.0)
        coherent = (evidence * y_sign) > (0.5 * float(coherence_threshold))
        if not bool(coherent.any().item()):
            return suppression_loss
        confidence = evidence.abs().clamp(0.0, 1.0).detach()
        probability_margin = (0.5 + 0.25 * confidence).clamp(0.500001, 0.749999)
        margin = torch.logit(probability_margin) / max(float(temperature), 1e-6)
        violation = F.relu(margin - y_sign * logits.reshape(-1))
        margin_loss = (confidence * violation.pow(2) * coherent.to(logits.dtype)).sum() / (coherent.to(logits.dtype).sum() + 1e-6)
        return suppression_loss + margin_loss

    coherent = evidence.abs() > coherence_threshold
    if not bool(coherent.any().item()):
        return logits.new_zeros(())
    confidence = evidence.abs().clamp(0.0, 1.0).detach()
    direction = torch.sign(evidence).detach()
    probability_margin = (0.5 + 0.25 * confidence).clamp(0.500001, 0.749999)
    margin = torch.logit(probability_margin) / max(float(temperature), 1e-6)
    violation = F.relu(margin - direction * logits.reshape(-1))
    return (confidence * violation.pow(2) * coherent.to(logits.dtype)).sum() / (coherent.to(logits.dtype).sum() + 1e-6)


def ranking_preservation_loss(
    data_logits: torch.Tensor,
    final_logits: torch.Tensor,
    targets: torch.Tensor,
    backbone_margin: float = 0.50,
    safe_margin: float = 0.10,
    max_pairs: int = 4096,
) -> torch.Tensor:
    """Safeguard objective preventing strongly separated data-only rankings from reversing."""
    data_logits = data_logits.reshape(-1)
    final_logits = final_logits.reshape(-1)
    targets = targets.reshape(-1).long()

    pos_data = data_logits[targets == 1]
    neg_data = data_logits[targets == 0]
    pos_final = final_logits[targets == 1]
    neg_final = final_logits[targets == 0]
    if pos_data.numel() == 0 or neg_data.numel() == 0:
        return final_logits.new_zeros(())

    data_diff = (pos_data[:, None] - neg_data[None, :]).reshape(-1)
    final_diff = (pos_final[:, None] - neg_final[None, :]).reshape(-1)
    protected_idx = torch.nonzero(data_diff > float(backbone_margin), as_tuple=False).flatten()
    if protected_idx.numel() == 0:
        return final_logits.new_zeros(())

    max_pairs = max(int(max_pairs), 1)
    if protected_idx.numel() > max_pairs:
        chosen = torch.randperm(protected_idx.numel(), device=protected_idx.device)[:max_pairs]
        protected_idx = protected_idx[chosen]

    return F.relu(float(safe_margin) - final_diff[protected_idx]).mean()
