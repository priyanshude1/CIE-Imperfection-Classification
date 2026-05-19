#!/usr/bin/env python3
"""
FINAL TRAINING (FULL DATASET): DeepSets + SetTransformer (ISAB + PMA)

What this script does
- Reads a *full-dataset* manifest (no split required).
- Loads score normalizer JSON (mu/sigma) and applies: s_norm = (log1p(score) - mu)/sigma.
- Trains TWO models sequentially on the same dataset:
    1) DeepSetsClassifier  (with pooling options incl. mean/max/std + gated attention)
    2) SetTransformerClassifier (ISAB blocks + PMA)
- Saves:
    - model weights (.pt) for each model
    - metrics JSON for each model
    - training history CSV for each model
Optionally:
    - create an internal val split (val_frac > 0) for checkpointing / reporting.

Expected manifest columns
- Required:
    - events_csv_path
    - label_id
- Optional:
    - split  (if you already have it; values train/val/test)
    - any other metadata columns are ignored.

Run example (train both on ALL data, no internal val):
  python 05_train_full_models.py --manifest dataset_events_manifest.csv --normalizer score_normalizer.json

Run example (train both with a small internal val split for sanity-check):
  python 05_train_full_models.py --manifest dataset_events_manifest.csv --normalizer score_normalizer.json --val_frac 0.15 --seed 1

"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report


# -------------------------
# Repro / utils
# -------------------------

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(x.dtype)
    denom = m.sum(dim=dim).clamp_min(1.0)
    return (x * m).sum(dim=dim) / denom


def masked_var(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mu = masked_mean(x, mask, dim=dim)
    m = mask.unsqueeze(-1).to(x.dtype)
    denom = m.sum(dim=dim).clamp_min(1.0)
    var = ((x - mu.unsqueeze(dim)) ** 2 * m).sum(dim=dim) / denom
    return var


def masked_max(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    neg_inf = torch.finfo(x.dtype).min
    m = mask.unsqueeze(-1).bool()
    x_masked = x.masked_fill(~m, neg_inf)
    return x_masked.max(dim=dim).values


# -------------------------
# Data
# -------------------------

@dataclass
class Normalizer:
    mu: float
    sigma: float

    def apply(self, rel_strength: np.ndarray) -> np.ndarray:
        s_raw = np.log1p(np.maximum(rel_strength, 0.0))
        return (s_raw - self.mu) / (self.sigma + 1e-12)


class EventDataset(Dataset):
    """
    Reads per-sample events csv.
    Returns variable-length event tensor and label.

    Expected event columns (defaults):
      x_n, y_n, z_n, t_n, score (or fallback score column)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        normalizer: Normalizer,
        path_col: str = "events_csv_path",
        label_col: str = "label_id",
        x_col: str = "x_n",
        y_col: str = "y_n",
        z_col: str = "z_n",
        t_col: str = "t_n",
        score_col: str = "score",
        score_fallbacks: Tuple[str, ...] = ("rel_strength", "raw_score", "s_rel", "equiv_score"),
        max_events_cap: Optional[int] = None,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.normalizer = normalizer

        for req in [path_col, label_col]:
            if req not in self.df.columns:
                raise RuntimeError(f"Manifest missing required column '{req}'. Columns={list(self.df.columns)}")

        self.path_col = path_col
        self.label_col = label_col
        self.x_col, self.y_col, self.z_col, self.t_col = x_col, y_col, z_col, t_col
        self.score_col = score_col
        self.score_fallbacks = score_fallbacks
        self.max_events_cap = max_events_cap

    def __len__(self) -> int:
        return int(self.df.shape[0])

    def _detect_score_col(self, cols: List[str]) -> str:
        if self.score_col in cols:
            return self.score_col
        for c in self.score_fallbacks:
            if c in cols:
                return c
        raise RuntimeError(
            f"Could not find score column. Tried '{self.score_col}' and fallbacks {self.score_fallbacks}. "
            f"Available columns: {cols}"
        )

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[idx]
        path = str(row[self.path_col])
        y = int(row[self.label_col])

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Events file not found: {path}")

        ev = pd.read_csv(path)

        for c in [self.x_col, self.y_col, self.z_col, self.t_col]:
            if c not in ev.columns:
                raise RuntimeError(f"Events file missing column '{c}': {path}")

        score_col = self._detect_score_col(ev.columns.tolist())

        Xxyz_t = ev[[self.x_col, self.y_col, self.z_col, self.t_col]].to_numpy(dtype=np.float32)
        rel_strength = ev[score_col].to_numpy(dtype=np.float32)

        s_norm = self.normalizer.apply(rel_strength).astype(np.float32)
        feats = np.concatenate([Xxyz_t, s_norm.reshape(-1, 1)], axis=1)

        if self.max_events_cap is not None and feats.shape[0] > self.max_events_cap:
            feats = feats[: self.max_events_cap]

        return {
            "events": torch.from_numpy(feats),  # (K, 5)
            "label": torch.tensor(y, dtype=torch.long),
            "length": torch.tensor(feats.shape[0], dtype=torch.long),
        }


def collate_events(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
    lengths = torch.stack([b["length"] for b in batch], dim=0)  # (B,)
    labels = torch.stack([b["label"] for b in batch], dim=0)    # (B,)
    max_len = int(lengths.max().item())
    D = int(batch[0]["events"].shape[1])

    events_padded = torch.zeros((len(batch), max_len, D), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_len), dtype=torch.float32)

    for i, b in enumerate(batch):
        ev = b["events"]
        L = int(ev.shape[0])
        events_padded[i, :L, :] = ev
        mask[i, :L] = 1.0

    return {"events": events_padded, "mask": mask, "labels": labels, "lengths": lengths}


# -------------------------
# Models
# -------------------------

class DeepSetsClassifier(nn.Module):
    """
    DeepSets:
      h_i = phi(x_i)
      pooled = pool(h_i)  (masked)
      logits = rho(pooled)

    Pool options:
      mean, max, meanmax, meanmaxstd,
      gated, gated_meanmax, gated_meanmaxstd
    Adds K_norm scalar (+1 dim) to pooled vector.
    """
    def __init__(
        self,
        in_dim: int = 5,
        embed_dim: int = 128,
        hidden: int = 128,
        dropout: float = 0.15,
        n_classes: int = 6,
        pool: str = "gated_meanmaxstd",
        gate_hidden: int = 64,
    ):
        super().__init__()
        allowed = {
            "mean", "max", "meanmax", "meanmaxstd",
            "gated", "gated_meanmax", "gated_meanmaxstd",
        }
        if pool not in allowed:
            raise ValueError(f"pool must be one of {sorted(allowed)}")
        self.pool = pool

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
        )

        self.gate = nn.Sequential(
            nn.Linear(embed_dim, gate_hidden),
            nn.Tanh(),
            nn.Linear(gate_hidden, 1),
        )

        if pool in {"mean", "max", "gated"}:
            pooled_dim = embed_dim
        elif pool in {"meanmax", "gated_meanmax"}:
            pooled_dim = 2 * embed_dim
        elif pool in {"meanmaxstd", "gated_meanmaxstd"}:
            pooled_dim = 3 * embed_dim
        else:
            raise RuntimeError("Unexpected pool option")

        pooled_dim += 1  # K_norm

        self.head = nn.Sequential(
            nn.Linear(pooled_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, events: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # events: (B,L,in_dim) mask: (B,L)
        valid = (mask > 0.0) if mask.dtype != torch.bool else mask
        H = self.encoder(events)  # (B,L,E)

        # stats
        mu = masked_mean(H, valid.to(H.dtype), dim=1)                   # (B,E)
        mx = masked_max(H, valid.to(H.dtype), dim=1)                    # (B,E)
        var = masked_var(H, valid.to(H.dtype), dim=1)                   # (B,E)
        sd = torch.sqrt(var.clamp_min(1e-12))                           # (B,E)

        # attention / gated pooling
        if self.pool.startswith("gated"):
            # logits: (B,L,1) -> mask invalid to -inf -> softmax over L
            logits = self.gate(H).squeeze(-1)                           # (B,L)
            neg_inf = torch.finfo(H.dtype).min
            logits = logits.masked_fill(~valid, neg_inf)
            alpha = torch.softmax(logits, dim=1).unsqueeze(-1)           # (B,L,1)
            gated = (H * alpha).sum(dim=1)                               # (B,E)
            if self.pool == "gated":
                pooled = gated
            elif self.pool == "gated_meanmax":
                pooled = torch.cat([gated, mx], dim=1)
            elif self.pool == "gated_meanmaxstd":
                pooled = torch.cat([gated, mx, sd], dim=1)
            else:
                raise RuntimeError("Unexpected gated pool option")
        else:
            if self.pool == "mean":
                pooled = mu
            elif self.pool == "max":
                pooled = mx
            elif self.pool == "meanmax":
                pooled = torch.cat([mu, mx], dim=1)
            elif self.pool == "meanmaxstd":
                pooled = torch.cat([mu, mx, sd], dim=1)
            else:
                raise RuntimeError("Unexpected pool option")

        # K_norm
        K = valid.to(torch.float32).sum(dim=1, keepdim=True)            # (B,1)
        K_norm = (K - 1000.0) / (10000.0 - 1000.0)
        K_norm = K_norm.clamp(0.0, 1.0)

        feats = torch.cat([pooled, K_norm], dim=1)
        return self.head(feats)


class MAB(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, key_padding_mask_K: Optional[torch.Tensor] = None) -> torch.Tensor:
        # key_padding_mask: True for PAD positions
        attn_out, _ = self.attn(Q, K, K, key_padding_mask=key_padding_mask_K, need_weights=False)
        H = self.ln1(Q + attn_out)
        H2 = self.ff(H)
        return self.ln2(H + H2)


class ISAB(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_inds: int, dropout: float):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, num_inds, dim) * 0.02)
        self.mab1 = MAB(dim, num_heads, dropout)
        self.mab2 = MAB(dim, num_heads, dropout)

    def forward(self, X: torch.Tensor, key_padding_mask_X: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = X.size(0)
        I = self.I.expand(B, -1, -1)
        H = self.mab1(I, X, key_padding_mask_K=key_padding_mask_X)
        return self.mab2(X, H, key_padding_mask_K=None)


class PMA(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_seeds: int, dropout: float):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, num_seeds, dim) * 0.02)
        self.mab = MAB(dim, num_heads, dropout)

    def forward(self, X: torch.Tensor, key_padding_mask_X: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = X.size(0)
        S = self.S.expand(B, -1, -1)
        return self.mab(S, X, key_padding_mask_K=key_padding_mask_X)


class SetTransformerClassifier(nn.Module):
    """
    SetTransformer:
      encoder -> ISAB * num_isab -> PMA(num_seeds) -> head
    Adds optional K_norm scalar feature (+1 dim).
    """
    def __init__(
        self,
        in_dim: int = 5,
        dim: int = 32,
        num_heads: int = 2,
        num_inds: int = 16,
        num_isab: int = 1,
        num_seeds: int = 1,
        hidden: int = 128,
        dropout: float = 0.0,
        n_classes: int = 6,
        use_k_norm: bool = True,
    ):
        super().__init__()
        self.use_k_norm = use_k_norm

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, dim),
        )

        self.isabs = nn.ModuleList(
            [ISAB(dim=dim, num_heads=num_heads, num_inds=num_inds, dropout=dropout) for _ in range(num_isab)]
        )
        self.pma = PMA(dim=dim, num_heads=num_heads, num_seeds=num_seeds, dropout=dropout)

        feat_dim = num_seeds * dim + (1 if use_k_norm else 0)

        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, events: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = (mask > 0.0) if mask.dtype != torch.bool else mask
        key_padding_mask = ~valid  # True for PAD

        X = self.encoder(events)  # (B,L,dim)
        for blk in self.isabs:
            X = blk(X, key_padding_mask_X=key_padding_mask)

        pooled = self.pma(X, key_padding_mask_X=key_padding_mask)  # (B,num_seeds,dim)
        pooled = pooled.reshape(pooled.size(0), -1)                # (B,num_seeds*dim)

        if self.use_k_norm:
            K = valid.to(torch.float32).sum(dim=1, keepdim=True)
            K_norm = (K - 1000.0) / (10000.0 - 1000.0)
            K_norm = K_norm.clamp(0.0, 1.0)
            pooled = torch.cat([pooled, K_norm], dim=1)

        return self.head(pooled)


# -------------------------
# Train / eval
# -------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, object]:
    model.eval()
    all_y, all_pred = [], []
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        events = batch["events"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(events, mask)
        loss = F.cross_entropy(logits, labels, reduction="sum")

        preds = logits.argmax(dim=1)
        all_y.append(labels.cpu().numpy())
        all_pred.append(preds.cpu().numpy())
        total_loss += float(loss.item())
        total_n += labels.numel()

    y = np.concatenate(all_y, axis=0) if all_y else np.array([], dtype=np.int64)
    p = np.concatenate(all_pred, axis=0) if all_pred else np.array([], dtype=np.int64)

    if y.size == 0:
        return {"loss": float("nan"), "acc": float("nan"), "f1_macro": float("nan"),
                "confusion_matrix": None, "y_true": y, "y_pred": p}

    acc = accuracy_score(y, p)
    f1m = f1_score(y, p, average="macro")
    cm = confusion_matrix(y, p)

    return {"loss": total_loss / max(total_n, 1), "acc": acc, "f1_macro": f1m,
            "confusion_matrix": cm, "y_true": y, "y_pred": p}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        events = batch["events"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)

        optim.zero_grad(set_to_none=True)
        logits = model(events, mask)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optim.step()

        total_loss += float(loss.item()) * labels.numel()
        total_n += labels.numel()

    return total_loss / max(total_n, 1)


def _device_from_arg(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


def _load_normalizer(normalizer_path: Path) -> Normalizer:
    norm_j = json.loads(normalizer_path.read_text(encoding="utf-8"))
    mu = norm_j.get("mu", norm_j.get("Fitted mu", None))
    sigma = norm_j.get("sigma", norm_j.get("Fitted sigma", None))
    if mu is None or sigma is None:
        raise RuntimeError(f"Could not parse mu/sigma from normalizer JSON: {normalizer_path}")
    return Normalizer(mu=float(mu), sigma=float(sigma))


def _make_internal_split(df: pd.DataFrame, val_frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if val_frac <= 0.0:
        return df.copy(), df.iloc[0:0].copy()  # train=all, val=empty

    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_val = max(1, int(round(val_frac * len(df))))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def _fit_and_save_one(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    outdir: Path,
    seed: int,
    epochs: int,
    lr: float,
    wd: float,
    save_prefix: str,
) -> None:
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    history_rows: List[Dict[str, float]] = []
    best_state = None
    best_metric = -1e9  # maximize val_f1_macro if val exists else maximize -train_loss

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optim, device)

        if val_loader is not None:
            val = evaluate(model, val_loader, device)
            val_loss, val_acc, val_f1 = val["loss"], val["acc"], val["f1_macro"]
            sel_metric = float(val_f1)  # selection metric
        else:
            val_loss, val_acc, val_f1 = float("nan"), float("nan"), float("nan")
            sel_metric = float(-tr_loss)

        history_rows.append({
            "epoch": epoch,
            "train_loss": float(tr_loss),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_f1_macro": float(val_f1),
        })

        if sel_metric > best_metric:
            best_metric = sel_metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if val_loader is not None:
            print(f"{model_name} | Epoch {epoch:03d} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | val_f1m={val_f1:.4f}")
        else:
            print(f"{model_name} | Epoch {epoch:03d} | train_loss={tr_loss:.4f}")

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final eval
    train_eval = evaluate(model, train_loader, device)
    val_eval = evaluate(model, val_loader, device) if val_loader is not None else None

    # Save model
    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / f"{save_prefix}_{model_name}_seed{seed}.pt"
    torch.save({"model_name": model_name, "state_dict": model.state_dict()}, model_path)

    # Save history
    hist_path = outdir / f"{save_prefix}_{model_name}_history_seed{seed}.csv"
    pd.DataFrame(history_rows).to_csv(hist_path, index=False)

    # Save metrics
    metrics_path = outdir / f"{save_prefix}_{model_name}_metrics_seed{seed}.json"
    payload = {
        "seed": seed,
        "device": str(device),
        "epochs": epochs,
        "lr": lr,
        "weight_decay": wd,
        "selection": "val_f1_macro" if val_loader is not None else "train_loss",
        "train": {"loss": train_eval["loss"], "acc": train_eval["acc"], "f1_macro": train_eval["f1_macro"]},
        "val": None if val_eval is None else {"loss": val_eval["loss"], "acc": val_eval["acc"], "f1_macro": val_eval["f1_macro"]},
        "notes": "If val_frac=0, val is empty and selection reverts to minimizing train_loss.",
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Save confusion matrices (train + optional val)
    def _save_cm(cm: np.ndarray, tag: str) -> None:
        cm_path = outdir / f"{save_prefix}_{model_name}_cm_{tag}_seed{seed}.csv"
        pd.DataFrame(cm).to_csv(cm_path, index=False)

    if train_eval["confusion_matrix"] is not None:
        _save_cm(train_eval["confusion_matrix"], "train")
    if val_eval is not None and val_eval["confusion_matrix"] is not None:
        _save_cm(val_eval["confusion_matrix"], "val")

    print(f"\n[{model_name}] Saved:")
    print(f"  model:   {model_path}")
    print(f"  history: {hist_path}")
    print(f"  metrics: {metrics_path}")


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    # Inputs
    ap.add_argument("--manifest", required=True, type=str, help="Full dataset manifest CSV (split column optional).")
    ap.add_argument("--normalizer", required=True, type=str, help="score_normalizer.json with mu/sigma.")

    # Data cols
    ap.add_argument("--path_col", type=str, default="events_csv_path")
    ap.add_argument("--label_col", type=str, default="label_id")
    ap.add_argument("--x_col", type=str, default="x_n")
    ap.add_argument("--y_col", type=str, default="y_n")
    ap.add_argument("--z_col", type=str, default="z_n")
    ap.add_argument("--t_col", type=str, default="t_n")
    ap.add_argument("--score_col", type=str, default="score")
    ap.add_argument("--max_events_cap", type=int, default=0, help="0 means no cap; else truncate events to this length.")

    # Training
    ap.add_argument("--outdir", type=str, default=None, help="Output directory (default: manifest directory).")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--val_frac", type=float, default=0.0, help="Internal val fraction if manifest has no split. 0 = no val.")

    # Which models to train
    ap.add_argument("--train_ds", action="store_true", help="Train DeepSets (also set --train_st for SetTransformer).")
    ap.add_argument("--train_st", action="store_true", help="Train SetTransformer (also set --train_ds for DeepSets).")

    # Shared head params
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.0)

    # DeepSets params
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--pool", type=str, default="gated_meanmaxstd",
                    choices=["mean", "max", "meanmax", "meanmaxstd", "gated", "gated_meanmax", "gated_meanmaxstd"])
    ap.add_argument("--gate_hidden", type=int, default=64)

    # SetTransformer params
    ap.add_argument("--st_dim", type=int, default=32)
    ap.add_argument("--st_heads", type=int, default=2)
    ap.add_argument("--st_inds", type=int, default=16)
    ap.add_argument("--st_isab", type=int, default=1)
    ap.add_argument("--st_seeds", type=int, default=1)
    ap.add_argument("--st_no_k_norm", action="store_true")

    args = ap.parse_args()

    # default: train both if user didn't specify flags
    if not args.train_ds and not args.train_st:
        args.train_ds = True
        args.train_st = True

    set_seed(args.seed)
    device = _device_from_arg(args.device)

    manifest_path = Path(args.manifest).resolve()
    normalizer_path = Path(args.normalizer).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else manifest_path.parent.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    normalizer = _load_normalizer(normalizer_path)

    df = pd.read_csv(manifest_path)
    if args.path_col not in df.columns or args.label_col not in df.columns:
        raise RuntimeError(
            f"Manifest must contain '{args.path_col}' and '{args.label_col}'. Columns={list(df.columns)}"
        )

    # If manifest already has split=train/val/test, use train as train and val as val.
    # Otherwise create internal split from all rows using val_frac.
    has_split = "split" in df.columns
    if has_split:
        df["split"] = df["split"].astype(str).str.strip().str.lower()
        train_df = df[df["split"] == "train"].copy()
        val_df = df[df["split"] == "val"].copy()
        # If you pass a full-data manifest but forgot to set split, fallback to "all".
        if train_df.empty:
            train_df = df.copy()
            val_df = df.iloc[0:0].copy()
    else:
        train_df, val_df = _make_internal_split(df, val_frac=float(args.val_frac), seed=int(args.seed))

    cap = None if int(args.max_events_cap) == 0 else int(args.max_events_cap)

    train_ds = EventDataset(
        df=train_df,
        normalizer=normalizer,
        path_col=args.path_col,
        label_col=args.label_col,
        x_col=args.x_col, y_col=args.y_col, z_col=args.z_col, t_col=args.t_col,
        score_col=args.score_col,
        max_events_cap=cap,
    )
    val_ds = None
    if len(val_df) > 0:
        val_ds = EventDataset(
            df=val_df,
            normalizer=normalizer,
            path_col=args.path_col,
            label_col=args.label_col,
            x_col=args.x_col, y_col=args.y_col, z_col=args.z_col, t_col=args.t_col,
            score_col=args.score_col,
            max_events_cap=cap,
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_events, pin_memory=(device.type == "cuda")
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_events, pin_memory=(device.type == "cuda")
        )

    print(f"\nDevice: {device}")
    print(f"Train size: {len(train_ds)} | Val size: {0 if val_ds is None else len(val_ds)}")
    print(f"Normalizer: mu={normalizer.mu:.6f}, sigma={normalizer.sigma:.6f}")
    print(f"batch_size={args.batch_size}, epochs={args.epochs}, lr={args.lr}, wd={args.wd}")
    if cap is not None:
        print(f"max_events_cap={cap}")

    save_prefix = "final_full"

    # Train DeepSets
    if args.train_ds:
        ds_model = DeepSetsClassifier(
            in_dim=5,
            embed_dim=args.embed_dim,
            hidden=args.hidden,
            dropout=args.dropout,
            n_classes=6,
            pool=args.pool,
            gate_hidden=args.gate_hidden,
        ).to(device)

        _fit_and_save_one(
            model_name="deepsets",
            model=ds_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            outdir=outdir,
            seed=args.seed,
            epochs=args.epochs,
            lr=args.lr,
            wd=args.wd,
            save_prefix=save_prefix,
        )

    # Train SetTransformer
    if args.train_st:
        if args.st_dim % args.st_heads != 0:
            raise RuntimeError(f"Invalid ST config: st_dim={args.st_dim} must be divisible by st_heads={args.st_heads}")

        st_model = SetTransformerClassifier(
            in_dim=5,
            dim=args.st_dim,
            num_heads=args.st_heads,
            num_inds=args.st_inds,
            num_isab=args.st_isab,
            num_seeds=args.st_seeds,
            hidden=args.hidden,
            dropout=args.dropout,
            n_classes=6,
            use_k_norm=(not args.st_no_k_norm),
        ).to(device)

        _fit_and_save_one(
            model_name="settransformer",
            model=st_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            outdir=outdir,
            seed=args.seed,
            epochs=args.epochs,
            lr=args.lr,
            wd=args.wd,
            save_prefix=save_prefix,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
