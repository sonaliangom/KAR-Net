"""External tabular comparator models (TabTransformer, Robust-GBDT, MPCFIN, TabM, ExcelFormer)."""

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.config import DEVICE, SEED
from src.models import BalancedFocalLoss, MultiScaleTransformer, StableFeatureTokenizer, make_pos_weight


class TabTransformerBaseline(nn.Module):
    """Data-only TabTransformer baseline."""

    def __init__(self, cat_cards, n_cont: int, embed_dim: int = 128, n_heads: int = 4, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.tok = StableFeatureTokenizer(n_cont, cat_cards, embed_dim)
        self.tr = MultiScaleTransformer(embed_dim, n_heads, n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        toks = self.tok(x_cont, x_cat)
        feat = self.tr(toks).mean(dim=1)
        return torch.clamp(self.head(feat), -10.0, 10.0)


class TTWrapper(nn.Module):
    """Interface adapter allowing TabTransformerBaseline to use the shared trainer."""

    def __init__(self, core_model: TabTransformerBaseline):
        super().__init__()
        self.core = core_model

    def forward(self, x_cont, x_cat, x_rules=None, return_components: bool = False):
        logit = self.core(x_cont, x_cat)
        if return_components:
            zero_scalar = logit.new_zeros(())
            zero_rules = logit.new_zeros((logit.shape[0], 0))
            return {
                "logit": logit,
                "data_logit": logit,
                "delta": torch.zeros_like(logit),
                "reliability": zero_rules,
                "rule_l1": zero_scalar,
                "reliability_logits": zero_rules,
                "trust_gate": torch.zeros_like(logit),
                "signed_evidence": zero_rules,
            }
        return logit


def _sanitize(*arrs):
    out = []
    for a in arrs:
        a = np.asarray(a, dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=1e3, neginf=-1e3)
        out.append(np.clip(a, -1e3, 1e3))
    return out[0] if len(out) == 1 else out


def _rfl_grad_hess(y_true, z, gamma, alpha_pos, alpha_neg):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    z = np.clip(np.asarray(z, dtype=float).reshape(-1), -20.0, 20.0)
    m = 2.0 * y_true - 1.0
    p = np.clip(1.0 / (1.0 + np.exp(-m * z)), 1e-7, 1.0 - 1e-7)
    logp = np.log(p)
    g = (1.0 - p) ** gamma
    grad_u = g * (gamma * p * logp - (1.0 - p))
    hess_u = p * g * (-gamma * (gamma * p * logp - (1.0 - p)) + (1.0 - p) * (gamma * logp + gamma + 1.0))
    alpha = np.where(y_true == 1.0, alpha_pos, alpha_neg)
    return alpha * m * grad_u, alpha * hess_u


def train_robust_gbdt(X_trm, y_trm, X_val, y_val, X_cal, X_te, gamma: float = 2.0, rounds: int = 800, eta: float = 0.05, early_stop: int = 100):
    """Robust-GBDT baseline with focal-style objective."""
    import xgboost as xgb

    X_trm, X_val, X_cal, X_te = _sanitize(X_trm, X_val, X_cal, X_te)
    alpha_pos = (len(y_trm) - y_trm.sum()) / len(y_trm)
    alpha_neg = y_trm.sum() / len(y_trm)
    params = {
        "tree_method": "hist",
        "max_depth": 5,
        "eta": eta,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 2,
        "eval_metric": "auc",
        "seed": SEED,
        "nthread": -1,
    }
    dtr = xgb.DMatrix(X_trm, label=y_trm)
    dva = xgb.DMatrix(X_val, label=y_val)

    def _obj(preds, dtrain):
        g, h = _rfl_grad_hess(dtrain.get_label(), preds, gamma, alpha_pos, alpha_neg)
        return g, np.maximum(h, 1e-6)

    bst = xgb.train(
        params,
        dtr,
        num_boost_round=rounds,
        obj=_obj,
        evals=[(dtr, "train"), (dva, "val")],
        early_stopping_rounds=early_stop,
        verbose_eval=False,
    )
    it = (0, bst.best_iteration + 1) if bst.best_iteration is not None else None

    def _probs(z):
        return np.clip(1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=float), -20.0, 20.0))), 1e-6, 1.0 - 1e-6)

    return _probs(bst.predict(xgb.DMatrix(X_cal), iteration_range=it)), _probs(bst.predict(xgb.DMatrix(X_te), iteration_range=it))


def build_multiplex_graph(X_trm, corr_threshold: float = 0.30, knn: int = 3):
    corr = np.nan_to_num(np.abs(np.corrcoef(X_trm.T)), nan=0.0)
    n_feat = X_trm.shape[1]
    a1 = (corr >= corr_threshold).astype(float)
    a2 = np.zeros((n_feat, n_feat), dtype=float)
    np.fill_diagonal(corr, 0.0)
    for i in range(n_feat):
        a2[i, np.argsort(-corr[i])[:knn]] = 1.0
    adjs = []
    for adj in (a1, a2):
        adj = adj + np.eye(n_feat)
        adj = adj / (adj.sum(axis=1, keepdims=True) + 1e-8)
        adjs.append(torch.from_numpy(adj.astype(np.float32)))
    return adjs


def _fit_bin_edges(Xc, n_bins: int = 4):
    edges = []
    for j in range(Xc.shape[1]):
        x = Xc[:, j]
        x = x[np.isfinite(x)]
        if np.unique(x).size < n_bins:
            q = np.quantile(x, [0.5])
        else:
            q = np.quantile(x, np.linspace(1.0 / n_bins, 1.0 - 1.0 / n_bins, n_bins - 1))
        edges.append(np.unique(np.asarray(q, dtype=np.float32)))
    return edges


class _GCNLayer(nn.Module):
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.lin = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, H, A):
        out = F.gelu(self.norm(self.lin(torch.bmm(A, H))))
        return self.drop(out) + H


class _MPCFIN(nn.Module):
    def __init__(self, cat_cards, n_cont, bin_cards, embed_dim: int = 64, n_types: int = 2, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.cont_embs = nn.ModuleList([nn.Embedding(c, embed_dim) for c in bin_cards])
        self.cat_embs = nn.ModuleList([nn.Embedding(c, embed_dim) for c in cat_cards])
        self.gnns = nn.ModuleList([
            nn.ModuleList([_GCNLayer(embed_dim, dropout) for _ in range(layers)])
            for _ in range(n_types)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, xb, xk, adjs):
        bsz = xb.size(0)
        feats = [emb(xb[:, j].clamp(0, emb.num_embeddings - 1)) for j, emb in enumerate(self.cont_embs)]
        feats += [emb(xk[:, j].clamp(0, emb.num_embeddings - 1)) for j, emb in enumerate(self.cat_embs)]
        h0 = torch.stack(feats, dim=1)
        outs = []
        for t, stack in enumerate(self.gnns):
            h = h0
            a = adjs[t].unsqueeze(0).expand(bsz, -1, -1)
            for layer in stack:
                h = layer(h, a)
            outs.append(h)
        h_mean = torch.stack(outs, dim=0).mean(dim=0)
        return self.head(h_mean.mean(dim=1)).squeeze(-1)


def train_mpcfin(
    X_trm, y_trm, X_val, y_val, X_cal, X_te, cont_idx, cat_idx, cat_cards,
    embed_dim: int = 64, n_types: int = 2, layers: int = 2, dropout: float = 0.1,
    epochs: int = 120, patience: int = 30, lr: float = 1e-3, wd: float = 1e-5, bs: int = 256,
):
    """MPCFIN multiplex feature-interaction GNN baseline."""
    X_trm, X_val, X_cal, X_te = _sanitize(X_trm, X_val, X_cal, X_te)
    bin_edges = _fit_bin_edges(X_trm[:, cont_idx])
    bin_cards = [len(e) + 1 for e in bin_edges]
    adjs = [a.to(DEVICE) for a in build_multiplex_graph(X_trm)]

    def to_bins(X):
        Xc = X[:, cont_idx]
        out = np.zeros((Xc.shape[0], len(cont_idx)), dtype=np.int64)
        for j, e in enumerate(bin_edges):
            out[:, j] = np.clip(np.searchsorted(e, Xc[:, j], side="right"), 0, len(e))
        return out

    xk_fn = lambda X: X[:, cat_idx].astype(np.int64)
    model = _MPCFIN(cat_cards, len(cont_idx), bin_cards, embed_dim=embed_dim, n_types=n_types, layers=layers, dropout=dropout).to(DEVICE)
    crit = BalancedFocalLoss(pos_weight=make_pos_weight(y_trm, DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max", patience=10, factor=0.5, min_lr=1e-5)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(to_bins(X_trm)), torch.from_numpy(xk_fn(X_trm)), torch.from_numpy(y_trm.astype(np.float32))),
        batch_size=bs, shuffle=True,
    )
    xb_v = torch.from_numpy(to_bins(X_val)).to(DEVICE)
    xk_v = torch.from_numpy(xk_fn(X_val)).to(DEVICE)

    best_auc, best_state, bad = 0.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, xk, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb.to(DEVICE), xk.to(DEVICE), adjs), yb.to(DEVICE))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va_auc = roc_auc_score(y_val, torch.sigmoid(model(xb_v, xk_v, adjs)).cpu().numpy()) if len(np.unique(y_val)) > 1 else 0.5
        sched.step(va_auc)
        if va_auc > best_auc and np.isfinite(va_auc):
            best_auc, best_state, bad = va_auc, {k: v.detach().cpu() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= patience:
            print(f"    MPCFIN early stopping at epoch {ep} (val AUC {best_auc:.4f})")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()

    def predict(xb, xk):
        with torch.no_grad():
            return torch.sigmoid(model(torch.from_numpy(xb).to(DEVICE), torch.from_numpy(xk).to(DEVICE), adjs)).cpu().numpy().ravel()

    return (
        np.clip(predict(to_bins(X_cal), xk_fn(X_cal)), 1e-6, 1.0 - 1e-6),
        np.clip(predict(to_bins(X_te), xk_fn(X_te)), 1e-6, 1.0 - 1e-6),
    )


class _TabM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 256, n_ens: int = 16, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = [nn.Linear(n_features, hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)]
        self.backbone = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(n_ens)])

    def forward(self, x):
        h = self.backbone(x)
        return torch.stack([hd(h) for hd in self.heads], dim=1)


def train_tabm(
    X_trm, y_trm, X_val, y_val, X_cal, X_te, n_features: int,
    hidden: int = 256, n_ens: int = 16, n_layers: int = 2, dropout: float = 0.1,
    epochs: int = 120, patience: int = 25, lr: float = 1e-3, wd: float = 1e-5, bs: int = 256,
):
    """TabM parameter-efficient MLP ensemble baseline."""
    X_trm, X_val, X_cal, X_te = _sanitize(X_trm, X_val, X_cal, X_te)
    model = _TabM(n_features, hidden=hidden, n_ens=n_ens, n_layers=n_layers, dropout=dropout).to(DEVICE)
    crit = BalancedFocalLoss(pos_weight=make_pos_weight(y_trm, DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max", patience=10, factor=0.5, min_lr=1e-5)
    loader = DataLoader(TensorDataset(torch.from_numpy(X_trm), torch.from_numpy(y_trm.astype(np.float32))), batch_size=bs, shuffle=True)
    xv = torch.from_numpy(X_val).to(DEVICE)

    best_auc, best_state, bad = 0.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb.to(DEVICE)).squeeze(-1).mean(1), yb.to(DEVICE))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va_auc = roc_auc_score(y_val, torch.sigmoid(model(xv).squeeze(-1).mean(1)).cpu().numpy()) if len(np.unique(y_val)) > 1 else 0.5
        sched.step(va_auc)
        if va_auc > best_auc and np.isfinite(va_auc):
            best_auc, best_state, bad = va_auc, {k: v.detach().cpu() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= patience:
            print(f"    TabM early stopping at epoch {ep} (val AUC {best_auc:.4f})")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()

    def predict(X):
        with torch.no_grad():
            return torch.sigmoid(model(torch.from_numpy(X).to(DEVICE)).squeeze(-1).mean(1)).cpu().numpy().ravel()

    return np.clip(predict(X_cal), 1e-6, 1.0 - 1e-6), np.clip(predict(X_te), 1e-6, 1.0 - 1e-6)


class _ExcelFormerLayer(nn.Module):
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.d = d
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.attn_out = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, d))
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), F.gelu(self.v(x))
        logits = q @ k.transpose(-1, -2) / (self.d ** 0.5)
        logits = logits - logits.max(dim=-1, keepdim=True).values
        s = torch.exp(logits.clamp(-30.0, 30.0))
        s = s / (s.sum(-1, keepdim=True) + 1e-8)
        x = self.norm1(x + self.drop(self.attn_out(s @ v)))
        return self.norm2(x + self.drop(self.mlp(x)))


class _ExcelFormer(nn.Module):
    def __init__(self, n_features: int, hidden: int = 256, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.feat_w = nn.Parameter(torch.empty(n_features, hidden))
        self.feat_b = nn.Parameter(torch.zeros(n_features, hidden))
        nn.init.xavier_uniform_(self.feat_w)
        self.layers = nn.ModuleList([_ExcelFormerLayer(hidden, dropout) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        h = x.unsqueeze(-1) * self.feat_w.unsqueeze(0) + self.feat_b.unsqueeze(0)
        for layer in self.layers:
            h = layer(h)
        return self.head(h.mean(dim=1)).squeeze(-1)


def train_excelformer(
    X_trm, y_trm, X_val, y_val, X_cal, X_te, n_features: int,
    hidden: int = 256, n_layers: int = 4, dropout: float = 0.1,
    epochs: int = 120, patience: int = 25, lr: float = 1e-3, wd: float = 1e-5, bs: int = 256,
):
    """ExcelFormer attention-based tabular baseline."""
    X_trm, X_val, X_cal, X_te = _sanitize(X_trm, X_val, X_cal, X_te)
    model = _ExcelFormer(n_features, hidden=hidden, n_layers=n_layers, dropout=dropout).to(DEVICE)
    crit = BalancedFocalLoss(pos_weight=make_pos_weight(y_trm, DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max", patience=10, factor=0.5, min_lr=1e-5)
    loader = DataLoader(TensorDataset(torch.from_numpy(X_trm), torch.from_numpy(y_trm.astype(np.float32))), batch_size=bs, shuffle=True)
    xv = torch.from_numpy(X_val).to(DEVICE)

    best_auc, best_state, bad = 0.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb.to(DEVICE)), yb.to(DEVICE))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va_auc = roc_auc_score(y_val, torch.sigmoid(model(xv)).cpu().numpy()) if len(np.unique(y_val)) > 1 else 0.5
        sched.step(va_auc)
        if va_auc > best_auc and np.isfinite(va_auc):
            best_auc, best_state, bad = va_auc, {k: v.detach().cpu() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= patience:
            print(f"    ExcelFormer early stopping at epoch {ep} (val AUC {best_auc:.4f})")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()

    def predict(X):
        with torch.no_grad():
            return torch.sigmoid(model(torch.from_numpy(X).to(DEVICE))).cpu().numpy().ravel()

    return np.clip(predict(X_cal), 1e-6, 1.0 - 1e-6), np.clip(predict(X_te), 1e-6, 1.0 - 1e-6)
