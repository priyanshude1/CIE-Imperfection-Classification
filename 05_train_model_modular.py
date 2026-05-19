#!/usr/bin/env python3
"""
Baseline 0: Set statistics + MLP classification (6 classes).

Pipeline in one script:
1) Load split manifest: dataset_events_manifest_split_seed*.csv
2) Load score normalizer: score_normalizer_seed*.json
3) Dataset reads events_topk.csv per sample
4) Collate pads to max_len in batch and returns mask
5) Model computes masked set stats -> fixed vector -> MLP classifier
6) Train + evaluate + confusion matrix + classification report

Designed to be modular:
- EventDataset + collate_fn are reusable for DeepSets / SetTransformer later.
- Only the model block changes later.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import itertools
import random
from copy import deepcopy


from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)

# -------------------------
# Utilities
# -------------------------

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    # x: (B, L, D), mask: (B, L) with 1 for valid
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
    # set invalid positions to -inf so they don't win max
    neg_inf = torch.finfo(x.dtype).min
    m = mask.unsqueeze(-1).bool()
    x_masked = x.masked_fill(~m, neg_inf)
    return x_masked.max(dim=dim).values

def _get_label_names(args, n_classes: int = 6):
    if args.label_names and len(args.label_names) == n_classes:
        return list(args.label_names)
    return [str(i) for i in range(n_classes)]


def plot_training_curves(history_df, out_png, dpi=200):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 4))

    # Loss
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    ax1.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("Loss over epochs")
    ax1.legend()

    # Acc/F1
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(history_df["epoch"], history_df["val_acc"], label="val_acc")
    ax2.plot(history_df["epoch"], history_df["val_f1_macro"], label="val_f1_macro")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("metric")
    ax2.set_title("Validation metrics over epochs")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def plot_confusion_matrix(cm, labels, out_png, title="Confusion Matrix", dpi=200, normalize=None):
    """
    normalize:
      None -> raw counts
      'true' -> row-normalized (recall view)
      'pred' -> col-normalized (precision view)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    cm = cm.astype(float)
    if normalize == "true":
        denom = cm.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        cm = cm / denom
    elif normalize == "pred":
        denom = cm.sum(axis=0, keepdims=True)
        denom[denom == 0] = 1.0
        cm = cm / denom

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if normalize is None:
                txt = f"{int(val)}"
            else:
                txt = f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


# -------------------------
# Data
# -------------------------

@dataclass
class Normalizer:
    mu: float
    sigma: float

    def apply(self, rel_strength: np.ndarray) -> np.ndarray:
        """
        Your normalization definition (as used in your pipeline):
          s_raw  = log(1 + rel_strength)
          s_norm = (s_raw - mu) / sigma
        """
        s_raw = np.log1p(np.maximum(rel_strength, 0.0))
        return (s_raw - self.mu) / (self.sigma + 1e-12)


class EventDataset(Dataset):
    """
    Reads per-sample events_topk.csv.
    Returns variable-length event tensor and label.

    Expected event columns (defaults):
      x_norm, y_norm, z_norm, t_norm, score

    Where `score` is your raw rel_strength (pre log1p, pre z-score).
    """
    def __init__(
        self,
        manifest_csv: str,
        split: str,
        normalizer: Normalizer,
        path_col: str = "events_csv_path",
        label_col: str = "label_id",
        # event columns:
        x_col: str = "x_n",
        y_col: str = "y_n",
        z_col: str = "z_n",
        t_col: str = "t_n",
        score_col: str = "score",
        # fallbacks for score column:
        score_fallbacks: Tuple[str, ...] = ("rel_strength", "raw_score", "s_rel", "equiv_score"),
        max_events_cap: Optional[int] = None,  # if you want an emergency cap (None = no cap)
    ):
        self.df = pd.read_csv(manifest_csv)
        if "split" not in self.df.columns:
            raise RuntimeError("Manifest must contain a 'split' column with values train/val/test.")
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        for req in [path_col, label_col]:
            if req not in self.df.columns:
                raise RuntimeError(f"Manifest missing required column '{req}'. Columns={list(self.df.columns)}")

        self.split = split
        self.normalizer = normalizer

        self.path_col = path_col
        self.label_col = label_col

        self.x_col, self.y_col, self.z_col, self.t_col = x_col, y_col, z_col, t_col
        self.score_col = score_col
        self.score_fallbacks = score_fallbacks
        self.max_events_cap = max_events_cap

    def __len__(self) -> int:
        return len(self.df)

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

        # Validate required columns
        for c in [self.x_col, self.y_col, self.z_col, self.t_col]:
            if c not in ev.columns:
                raise RuntimeError(f"Events file missing column '{c}': {path}")

        score_col = self._detect_score_col(ev.columns.tolist())

        Xxyz_t = ev[[self.x_col, self.y_col, self.z_col, self.t_col]].to_numpy(dtype=np.float32)
        rel_strength = ev[score_col].to_numpy(dtype=np.float32)

        # Apply your train-fitted normalizer (log1p + z-score)
        s_norm = self.normalizer.apply(rel_strength).astype(np.float32)

        # Final per-event feature vector: [x,y,z,t,s_norm]
        feats = np.concatenate([Xxyz_t, s_norm.reshape(-1, 1)], axis=1)

        if self.max_events_cap is not None and feats.shape[0] > self.max_events_cap:
            feats = feats[: self.max_events_cap]

        return {
            "events": torch.from_numpy(feats),   # (K, 5)
            "label": torch.tensor(y, dtype=torch.long),
            "length": torch.tensor(feats.shape[0], dtype=torch.long),
            "path": path,
        }


def collate_events(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
    """
    Pads events to max length in batch.
    Returns:
      events_padded: (B, L, D)
      mask:         (B, L) 1=valid 0=pad
      labels:       (B,)
      lengths:      (B,)
    """
    lengths = torch.stack([b["length"] for b in batch], dim=0)  # (B,)
    labels = torch.stack([b["label"] for b in batch], dim=0)    # (B,)
    max_len = int(lengths.max().item())
    D = batch[0]["events"].shape[1]

    events_padded = torch.zeros((len(batch), max_len, D), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_len), dtype=torch.float32)

    for i, b in enumerate(batch):
        ev = b["events"]
        L = ev.shape[0]
        events_padded[i, :L, :] = ev
        mask[i, :L] = 1.0

    return {
        "events": events_padded,
        "mask": mask,
        "labels": labels,
        "lengths": lengths,
    }


def parse_backbones(backbone_strs: List[str]) -> List[Dict[str, int]]:
    out = []
    for s in backbone_strs:
        parts = s.split(",")
        if len(parts) != 3:
            raise ValueError(f"Bad backbone '{s}'. Expected 'dim,heads,inds' (e.g., 48,3,32)")
        d = int(parts[0])
        h = int(parts[1])
        m = int(parts[2])
        if d % h != 0:
            raise ValueError(f"Invalid backbone '{s}': st_dim {d} not divisible by st_heads {h}")
        out.append({"st_dim": d, "st_heads": h, "st_inds": m})
    return out


# -------------------------
# Model: Baseline 0
# -------------------------

class StatsMLP(nn.Module):
    """
    Baseline 0:
      - compute masked mean/std/max of the event features across set dimension
      - concatenate K (normalized) as an extra scalar
      - classify with MLP

    Input:
      events: (B, L, 5)
      mask:   (B, L)
    """
    def __init__(self, in_dim: int = 5, hidden: int = 128, dropout: float = 0.15, n_classes: int = 6):
        super().__init__()
        # stats produce 3*in_dim + 1 (for K)
        feat_dim = 3 * in_dim + 1
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, events: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # events: (B, L, D), mask: (B, L)
        mu = masked_mean(events, mask, dim=1)                 # (B, D)
        var = masked_var(events, mask, dim=1)                 # (B, D)
        sd = torch.sqrt(var.clamp_min(1e-12))                 # (B, D)
        mx = masked_max(events, mask, dim=1)                  # (B, D)
        K = mask.sum(dim=1, keepdim=True)                     # (B,1)

        # optional: normalize K to roughly [0,1] using known bounds (1000..10000)
        K_norm = (K - 1000.0) / (10000.0 - 1000.0)
        K_norm = K_norm.clamp(0.0, 1.0)

        feats = torch.cat([mu, sd, mx, K_norm], dim=1)        # (B, 3D+1)
        return self.net(feats)

# -------------------------
# Model: Baseline 1 (Deep Sets)
# -------------------------

class DeepSetsClassifier(nn.Module):
    """
    Deep Sets baseline with enhanced pooling options:
      - mean / max / meanmax / meanmaxstd
      - gated (attention pooling) / gated_meanmax / gated_meanmaxstd
    Adds K_norm as an extra scalar feature.
    """
    def __init__(
        self,
        in_dim: int = 5,
        embed_dim: int = 128,
        hidden: int = 128,
        dropout: float = 0.15,
        n_classes: int = 6,
        pool: str = "meanmax",  # mean|max|meanmax|meanmaxstd|gated|gated_meanmax|gated_meanmaxstd
        gate_hidden: int = 64,  # hidden size for gating MLP
    ):
        super().__init__()

        allowed = {
            "mean", "max", "meanmax", "meanmaxstd",
            "gated", "gated_meanmax", "gated_meanmaxstd",
        }
        if pool not in allowed:
            raise ValueError(f"pool must be one of {sorted(allowed)}")
        self.pool = pool

        # Per-event encoder f_theta
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
        )

        # Optional gating network for attention pooling (scalar logit per event)
        # alpha_j = softmax(gate(h_j))
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, gate_hidden),
            nn.Tanh(),
            nn.Linear(gate_hidden, 1),
        )

        # Determine pooled dimension
        # Base pooled dims from embeddings
        pooled_dim = 0
        if pool in {"mean", "gated"}:
            pooled_dim = embed_dim
        elif pool in {"max"}:
            pooled_dim = embed_dim
        elif pool in {"meanmax", "gated_meanmax"}:
            pooled_dim = 2 * embed_dim
        elif pool in {"meanmaxstd", "gated_meanmaxstd"}:
            pooled_dim = 3 * embed_dim
        else:
            raise RuntimeError("Unexpected pool option")

        # +1 for K_norm feature
        pooled_dim = pooled_dim + 1

        self.head = nn.Sequential(
            nn.Linear(pooled_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def _masked_softmax(self, logits: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """
        logits: (B, L)
        mask:   (B, L) float {0,1} or bool
        Returns alpha: (B, L) where padded positions are 0 and rows sum to 1 over valid positions.
        """
        if mask.dtype != torch.bool:
            mask_b = mask > 0.0
        else:
            mask_b = mask

        # set padded logits to -inf then softmax
        neg_inf = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~mask_b, neg_inf)
        alpha = torch.softmax(logits, dim=dim)

        # ensure exact zeros on padding
        alpha = alpha.masked_fill(~mask_b, 0.0)

        # safeguard: if a row is entirely padding (should not happen), avoid NaNs
        row_sums = alpha.sum(dim=dim, keepdim=True).clamp_min(1e-12)
        alpha = alpha / row_sums
        return alpha

    def forward(self, events: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        events: (B, L, in_dim)
        mask:   (B, L) float {0,1} or bool
        """
        h = self.encoder(events)  # (B, L, embed_dim)

        # K_norm (same as baseline0): normalize approximately [0,1] using known bounds (1000..10000)
        if mask.dtype == torch.bool:
            K = mask.to(torch.float32).sum(dim=1, keepdim=True)
        else:
            K = mask.sum(dim=1, keepdim=True)
        K_norm = (K - 1000.0) / (10000.0 - 1000.0)
        K_norm = K_norm.clamp(0.0, 1.0)

        # Standard pool components
        h_mean = masked_mean(h, mask, dim=1)  # (B, E)
        h_max = masked_max(h, mask, dim=1)    # (B, E)
        h_std = torch.sqrt(masked_var(h, mask, dim=1).clamp_min(1e-12))  # (B, E)

        # Gated attention pooling: weighted sum of embeddings
        # logits: (B, L)
        gate_logits = self.gate(h).squeeze(-1)
        alpha = self._masked_softmax(gate_logits, mask, dim=1)  # (B, L)
        # weighted sum: (B, E)
        h_gated = (h * alpha.unsqueeze(-1)).sum(dim=1)

        # Choose pooled representation based on pool mode
        if self.pool == "mean":
            pooled = h_mean
        elif self.pool == "max":
            pooled = h_max
        elif self.pool == "meanmax":
            pooled = torch.cat([h_mean, h_max], dim=1)
        elif self.pool == "meanmaxstd":
            pooled = torch.cat([h_mean, h_max, h_std], dim=1)
        elif self.pool == "gated":
            pooled = h_gated
        elif self.pool == "gated_meanmax":
            pooled = torch.cat([h_gated, h_max], dim=1)
        elif self.pool == "gated_meanmaxstd":
            pooled = torch.cat([h_gated, h_max, h_std], dim=1)
        else:
            raise RuntimeError(f"Unhandled pool mode: {self.pool}")

        # Append K_norm
        pooled = torch.cat([pooled, K_norm], dim=1)

        return self.head(pooled)
class MAB(nn.Module):
    """
    Multihead Attention Block (Set Transformer).
    Uses batch_first=True so tensors are (B, L, D).
    """
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(
        self,
        Q: torch.Tensor,                  # (B, Lq, D)
        K: torch.Tensor,                  # (B, Lk, D)
        key_padding_mask: torch.Tensor,   # (B, Lk) bool, True=PAD
    ) -> torch.Tensor:
        # Attention + residual + LN
        A, _ = self.attn(Q, K, K, key_padding_mask=key_padding_mask, need_weights=False)
        H = self.ln1(Q + A)
        # FFN + residual + LN
        H2 = self.ff(H)
        return self.ln2(H + H2)


class SAB(nn.Module):
    """Self-Attention Block: MAB(X, X)."""
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.mab = MAB(dim, num_heads, dropout)

    def forward(self, X: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        return self.mab(X, X, key_padding_mask)


class ISAB(nn.Module):
    """
    Induced Set Attention Block:
      H = MAB(I, X)
      Y = MAB(X, H)
    where I are learned inducing points.
    """
    def __init__(self, dim: int, num_heads: int, num_inds: int, dropout: float = 0.0):
        super().__init__()
        self.num_inds = num_inds
        self.I = nn.Parameter(torch.randn(1, num_inds, dim) * 0.02)  # (1, m, D)
        self.mab1 = MAB(dim, num_heads, dropout)
        self.mab2 = MAB(dim, num_heads, dropout)

    def forward(self, X: torch.Tensor, key_padding_mask_X: torch.Tensor) -> torch.Tensor:
        B = X.size(0)
        I = self.I.expand(B, -1, -1)  # (B, m, D)

        # For inducing points, there is no padding in Q.
        # But keys are X so we pass X padding mask.
        H = self.mab1(I, X, key_padding_mask=key_padding_mask_X)  # (B, m, D)

        # Now query is X; keys are H. H has no padding -> mask all False.
        key_padding_mask_H = torch.zeros((B, H.size(1)), dtype=torch.bool, device=X.device)
        Y = self.mab2(X, H, key_padding_mask=key_padding_mask_H)  # (B, L, D)
        return Y


class PMA(nn.Module):
    """
    Pooling by Multihead Attention:
      Y = MAB(S, X), where S are learned seed vectors (k seeds).
    Output: (B, k, D)
    """
    def __init__(self, dim: int, num_heads: int, num_seeds: int = 1, dropout: float = 0.0):
        super().__init__()
        self.num_seeds = num_seeds
        self.S = nn.Parameter(torch.randn(1, num_seeds, dim) * 0.02)  # (1, k, D)
        self.mab = MAB(dim, num_heads, dropout)

    def forward(self, X: torch.Tensor, key_padding_mask_X: torch.Tensor) -> torch.Tensor:
        B = X.size(0)
        S = self.S.expand(B, -1, -1)  # (B, k, D)
        return self.mab(S, X, key_padding_mask=key_padding_mask_X)  # (B, k, D)


class SetTransformerClassifier(nn.Module):
    """
    Set Transformer classifier:
      - Event encoder -> dim D
      - ISAB blocks (L times)
      - PMA pooling (k seeds)
      - Optionally concatenate K_norm
      - MLP head to n_classes

    Input:
      events: (B, L, 5)
      mask:   (B, L) 1=valid 0=pad  (float) OR bool mask
    """
    def __init__(
        self,
        in_dim: int = 5,
        dim: int = 64,
        num_heads: int = 4,
        num_inds: int = 32,
        num_isab: int = 2,
        num_seeds: int = 1,
        hidden: int = 128,
        dropout: float = 0.1,
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

        self.isabs = nn.ModuleList([
            ISAB(dim=dim, num_heads=num_heads, num_inds=num_inds, dropout=dropout)
            for _ in range(num_isab)
        ])

        self.pma = PMA(dim=dim, num_heads=num_heads, num_seeds=num_seeds, dropout=dropout)

        # pooled dim: num_seeds * dim (+1 if K_norm)
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
        # Convert mask -> key_padding_mask (True for PAD)
        if mask.dtype == torch.bool:
            valid = mask
        else:
            valid = mask > 0.0
        key_padding_mask = ~valid  # True for padding positions

        X = self.encoder(events)  # (B, L, D)

        for blk in self.isabs:
            X = blk(X, key_padding_mask_X=key_padding_mask)

        pooled = self.pma(X, key_padding_mask_X=key_padding_mask)  # (B, k, D)
        pooled = pooled.reshape(pooled.size(0), -1)                # (B, k*D)

        if self.use_k_norm:
            # normalize approximately [0,1] using known bounds (1000..10000)
            K = valid.to(torch.float32).sum(dim=1, keepdim=True)   # (B,1)
            K_norm = (K - 1000.0) / (10000.0 - 1000.0)
            K_norm = K_norm.clamp(0.0, 1.0)
            pooled = torch.cat([pooled, K_norm], dim=1)

        return self.head(pooled)

def make_model(
    model_name: str,
    in_dim: int,
    n_classes: int,
    hidden: int,
    dropout: float,
    # existing args you already pass:
    embed_dim: int = 128,
    pool: str = "meanmax",
    gate_hidden: int = 64,
    # set-transformer args:
    st_dim: int = 64,
    st_heads: int = 4,
    st_inds: int = 32,
    st_isab: int = 2,
    st_seeds: int = 1,
    st_use_k_norm: bool = True,
):
    name = model_name.lower()

    if name == "baseline0":
        return StatsMLP(in_dim=in_dim, hidden=hidden, dropout=dropout, n_classes=n_classes)

    if name == "deepsets":
        return DeepSetsClassifier(
            in_dim=in_dim,
            embed_dim=embed_dim,
            hidden=hidden,
            dropout=dropout,
            n_classes=n_classes,
            pool=pool,
            gate_hidden=gate_hidden,
        )

    if name in {"settransformer", "set_transformer", "st"}:
        return SetTransformerClassifier(
            in_dim=in_dim,
            dim=st_dim,
            num_heads=st_heads,
            num_inds=st_inds,
            num_isab=st_isab,
            num_seeds=st_seeds,
            hidden=hidden,
            dropout=dropout,
            n_classes=n_classes,
            use_k_norm=st_use_k_norm,
        )

    raise ValueError(f"Unknown model_name: {model_name}")

# -------------------------
# Train / Eval
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

    y = np.concatenate(all_y, axis=0)
    p = np.concatenate(all_pred, axis=0)

    acc = accuracy_score(y, p)
    f1m = f1_score(y, p, average="macro")
    cm = confusion_matrix(y, p)

    return {
        "loss": total_loss / max(total_n, 1),
        "acc": acc,
        "f1_macro": f1m,
        "confusion_matrix": cm,
        "y_true": y,
        "y_pred": p,
    }

def dict_product(grid: Dict[str, List[object]]) -> List[Dict[str, object]]:
    """Cartesian product of dict of lists -> list of dict configs."""
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    out = []
    for combo in itertools.product(*vals):
        out.append({k: combo[i] for i, k in enumerate(keys)})
    return out


def sample_random_configs(grid: Dict[str, List[object]], n: int, seed: int) -> List[Dict[str, object]]:
    """Randomly sample n configs from a discrete grid (with replacement)."""
    rng = random.Random(seed)
    keys = list(grid.keys())
    out = []
    for _ in range(n):
        cfg = {k: rng.choice(grid[k]) for k in keys}
        out.append(cfg)
    return out


def run_one_config(
    args,
    device: torch.device,
    normalizer: Normalizer,
    cfg: Dict[str, object],
) -> Dict[str, object]:
    """
    Runs training for a single hyperparameter config and returns summary metrics.
    Uses your existing Dataset/DataLoader pipeline.
    """
    set_seed(int(cfg["seed"]))

    cap = None if args.max_events_cap == 0 else int(args.max_events_cap)

    train_ds = EventDataset(
        manifest_csv=args.manifest, split="train", normalizer=normalizer,
        path_col=args.path_col, label_col=args.label_col,
        x_col=args.x_col, y_col=args.y_col, z_col=args.z_col, t_col=args.t_col,
        score_col=args.score_col, max_events_cap=cap,
    )
    val_ds = EventDataset(
        manifest_csv=args.manifest, split="val", normalizer=normalizer,
        path_col=args.path_col, label_col=args.label_col,
        x_col=args.x_col, y_col=args.y_col, z_col=args.z_col, t_col=args.t_col,
        score_col=args.score_col, max_events_cap=cap,
    )
    test_ds = EventDataset(
        manifest_csv=args.manifest, split="test", normalizer=normalizer,
        path_col=args.path_col, label_col=args.label_col,
        x_col=args.x_col, y_col=args.y_col, z_col=args.z_col, t_col=args.t_col,
        score_col=args.score_col, max_events_cap=cap,
    )

    train_loader = DataLoader(
        train_ds, batch_size=int(cfg["batch_size"]), shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_events, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg["batch_size"]), shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_events, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_ds, batch_size=int(cfg["batch_size"]), shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_events, pin_memory=(device.type == "cuda")
    )

    # Build model based on cfg
    model = make_model(
        model_name=str(cfg["model"]),
        in_dim=5,
        n_classes=6,
        hidden=int(cfg["hidden"]),
        dropout=float(cfg["dropout"]),
        embed_dim=int(cfg.get("embed_dim", args.embed_dim)),
        pool=str(cfg.get("pool", getattr(args, "pool", "meanmax"))),
        gate_hidden=int(cfg.get("gate_hidden", getattr(args, "gate_hidden", 64))),
        st_dim=int(cfg.get("st_dim", args.st_dim)),
        st_heads=int(cfg.get("st_heads", args.st_heads)),
        st_inds=int(cfg.get("st_inds", args.st_inds)),
        st_isab=int(cfg.get("st_isab", args.st_isab)),
        st_seeds=int(cfg.get("st_seeds", args.st_seeds)),
        st_use_k_norm=bool(cfg.get("st_use_k_norm", not args.st_no_k_norm)),
    ).to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["wd"]),
    )

    best_val = -1.0
    best_state = None

    epochs = int(cfg["epochs"])
    for _ in range(epochs):
        _ = train_one_epoch(model, train_loader, optim, device)
        val = evaluate(model, val_loader, device)
        if val["f1_macro"] > best_val:
            best_val = float(val["f1_macro"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    val = evaluate(model, val_loader, device)
    test = evaluate(model, test_loader, device)

    return {
        "val_acc": float(val["acc"]),
        "val_f1": float(val["f1_macro"]),
        "test_acc": float(test["acc"]),
        "test_f1": float(test["f1_macro"]),
    }

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


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=str, help="Split manifest CSV with a 'split' column.")
    ap.add_argument("--normalizer", required=True, type=str, help="score_normalizer_seedX.json (mu/sigma).")
    ap.add_argument("--model", type=str, default="stats",
                choices=["stats", "deepsets", "settransformer"],
                help="Model type: stats (baseline0) or deepsets (baseline1)")
    ap.add_argument("--embed_dim", type=int, default=128,
                    help="Deep Sets embedding dimension")
    ap.add_argument("--pool", type=str, default="meanmax",
                    choices=["mean", "max", "meanmax", "meanmaxstd",
                            "gated", "gated_meanmax", "gated_meanmaxstd"],
                    help="DeepSets pooling mode")

    ap.add_argument("--gate_hidden", type=int, default=64,
                    help="Hidden size for gated pooling network")
    
    ap.add_argument("--st_dim", type=int, default=64, help="SetTransformer embedding dim (D). Try 64 then 128.")
    ap.add_argument("--st_heads", type=int, default=4, help="SetTransformer num attention heads.")
    ap.add_argument("--st_inds", type=int, default=32, help="ISAB inducing points (m). 32 or 64 are typical.")
    ap.add_argument("--st_isab", type=int, default=2, help="Number of ISAB blocks.")
    ap.add_argument("--st_seeds", type=int, default=1, help="PMA seed vectors (k). Use 1 for classification.")
    ap.add_argument("--st_no_k_norm", action="store_true", help="Disable concatenation of K_norm feature.")

    ap.add_argument("--tune", action="store_true", help="Run hyperparameter tuning loop.")
    ap.add_argument("--tune_mode", type=str, default="grid", choices=["grid", "random"])
    ap.add_argument("--trials", type=int, default=30, help="Random trials if tune_mode=random.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 30, 42], help="Seeds to evaluate each config.")
    ap.add_argument("--select_metric", type=str, default="test_f1_mean",
                choices=["val_f1_mean", "val_acc_mean", "test_f1_mean", "test_acc_mean"],
                help="Metric used to rank configs.")
    ap.add_argument("--save_top", type=int, default=20, help="Write top-N configs to CSV.")
    ap.add_argument("--tune_out", type=str, default="", help="Optional output CSV path.")

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--path_col", type=str, default="events_csv_path")
    ap.add_argument("--label_col", type=str, default="label_id")
    ap.add_argument("--x_col", type=str, default="x_n")
    ap.add_argument("--y_col", type=str, default="y_n")
    ap.add_argument("--z_col", type=str, default="z_n")
    ap.add_argument("--t_col", type=str, default="t_n")
    ap.add_argument("--score_col", type=str, default="score")
    ap.add_argument("--max_events_cap", type=int, default=0, help="0 means no cap; else truncate events to this length.")
    # Set Transformer search space (safe)
    ap.add_argument(
        "--st_backbones",
        type=str,
        nargs="+",
        default=["32,2,16", "48,3,32", "64,4,32"],
        help="List of set-transformer backbones as 'dim,heads,inds'. Example: 32,2,16 48,3,32"
    )
    ap.add_argument("--save_plots", action="store_true", help="Save training curves + confusion matrix figures to disk.")
    ap.add_argument("--run_tag", type=str, default="", help="Optional run tag appended to output filenames.")
    ap.add_argument("--plot_dpi", type=int, default=200, help="DPI for saved figures.")
    ap.add_argument("--label_names", type=str, nargs="+", default=[],
                    help="Optional 6 label names in order (else uses 0..5). Example: --label_names ip0 ip1 ip2 ip3 ip4 ip5")

    ap.add_argument("--st_isab_list", type=int, nargs="+", default=[1])
    ap.add_argument("--dropout_list", type=float, nargs="+", default=[0.0, 0.1])
    ap.add_argument("--lr_list", type=float, nargs="+", default=[3e-4, 5e-4])
    ap.add_argument("--wd_list", type=float, nargs="+", default=[0.0, 1e-4])
    ap.add_argument("--batch_size_list", type=int, nargs="+", default=[4])
    ap.add_argument("--epochs_list", type=int, nargs="+", default=[60])
    ap.add_argument("--hidden_list", type=int, nargs="+", default=[128])
    ap.add_argument("--model_list", type=str, nargs="+", default=["settransformer"], help="Models to tune.")

    args = ap.parse_args()

    set_seed(args.seed)

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load normalizer
    norm_j = json.loads(Path(args.normalizer).read_text(encoding="utf-8"))
    # Accept either {"mu":..., "sigma":...} or {"Fitted mu":..., "Fitted sigma":...}
    mu = norm_j.get("mu", norm_j.get("Fitted mu", None))
    sigma = norm_j.get("sigma", norm_j.get("Fitted sigma", None))
    if mu is None or sigma is None:
        raise RuntimeError(f"Could not parse mu/sigma from normalizer JSON: {args.normalizer}")
    normalizer = Normalizer(mu=float(mu), sigma=float(sigma))

    cap = None if args.max_events_cap == 0 else int(args.max_events_cap)
    if args.tune:
        # ------------------------------------------------------------
        # Structured grid: tune only VALID (dim, heads, inds) backbones
        # instead of the Cartesian product st_dim × st_heads × st_inds.
        # ------------------------------------------------------------

        # 1) Define (or parse) backbone triplets
        # Option A (recommended): parse from CLI strings like "32,2,16"
        backbones = parse_backbones(args.st_backbones)  # returns list of dicts: {"st_dim":..,"st_heads":..,"st_inds":..}

        # 2) Enumerate candidates (grid or random)
        candidates = []

        if args.tune_mode == "grid":
            for bb in backbones:
                for drop in args.dropout_list:
                    for lr in args.lr_list:
                        for wd in args.wd_list:
                            for bs in args.batch_size_list:
                                for ep in args.epochs_list:
                                    for hidden in args.hidden_list:
                                        candidates.append({
                                            "model": "settransformer",
                                            "st_dim": bb["st_dim"],
                                            "st_heads": bb["st_heads"],
                                            "st_inds": bb["st_inds"],
                                            "st_isab": 1,                # FIXED (fast + stable)
                                            "st_seeds": 1,               # FIXED
                                            "st_use_k_norm": True,       # FIXED
                                            "dropout": float(drop),
                                            "lr": float(lr),
                                            "wd": float(wd),
                                            "batch_size": int(bs),
                                            "epochs": int(ep),
                                            "hidden": int(hidden),
                                        })
        else:
            # Random: sample from the *structured* options rather than full cartesian product
            rng = np.random.default_rng(args.seed)
            for _ in range(int(args.trials)):
                bb = backbones[int(rng.integers(0, len(backbones)))]
                candidates.append({
                    "model": "settransformer",
                    "st_dim": bb["st_dim"],
                    "st_heads": bb["st_heads"],
                    "st_inds": bb["st_inds"],
                    "st_isab": 1,
                    "st_seeds": 1,
                    "st_use_k_norm": True,
                    "dropout": float(rng.choice(args.dropout_list)),
                    "lr": float(rng.choice(args.lr_list)),
                    "wd": float(rng.choice(args.wd_list)),
                    "batch_size": int(rng.choice(args.batch_size_list)),
                    "epochs": int(rng.choice(args.epochs_list)),
                    "hidden": int(rng.choice(args.hidden_list)),
                })

        # 3) (Optional) keep your existing filtering safety net
        filtered = []
        for c in candidates:
            if c["st_dim"] % c["st_heads"] != 0:
                continue
            head_dim = c["st_dim"] // c["st_heads"]
            if head_dim < 8:
                continue
            filtered.append(c)

        candidates = filtered
        print(f"[TUNE] candidates after filtering: {len(candidates)}")



        rows = []
        for i, c in enumerate(candidates, start=1):
            # Evaluate across seeds
            per_seed = []
            for s in args.seeds:
                cc = deepcopy(c)
                cc["seed"] = int(s)
                # Map tuning cfg to model factory expected args
                cc["model"] = str(cc["model"])
                cc["embed_dim"] = getattr(args, "embed_dim", 128)
                cc["pool"] = getattr(args, "pool", "meanmax")
                cc["gate_hidden"] = getattr(args, "gate_hidden", 64)

                metrics = run_one_config(args, device, normalizer, cc)
                per_seed.append(metrics)

            # Aggregate
            def agg(key: str):
                vals = [m[key] for m in per_seed]
                return float(np.mean(vals)), float(np.std(vals))

            val_acc_mean, val_acc_std = agg("val_acc")
            val_f1_mean, val_f1_std = agg("val_f1")
            test_acc_mean, test_acc_std = agg("test_acc")
            test_f1_mean, test_f1_std = agg("test_f1")

            row = {
                "idx": i,
                **{k: c[k] for k in c.keys()},
                "val_acc_mean": val_acc_mean, "val_acc_std": val_acc_std,
                "val_f1_mean": val_f1_mean,   "val_f1_std": val_f1_std,
                "test_acc_mean": test_acc_mean, "test_acc_std": test_acc_std,
                "test_f1_mean": test_f1_mean,   "test_f1_std": test_f1_std,
            }
            rows.append(row)

            if i % 10 == 0 or i == 1:
                print(f"[TUNE] {i:4d}/{len(candidates)} done | best {args.select_metric} so far...")

        df = pd.DataFrame(rows)
        df = df.sort_values(by=args.select_metric, ascending=False).reset_index(drop=True)

        outdir = Path(args.manifest).resolve().parent
        out_csv = Path(args.tune_out) if args.tune_out else (outdir / f"tune_results_{'_'.join(args.model_list)}.csv")
        df.to_csv(out_csv, index=False)

        best = df.iloc[0].to_dict()
        out_best = outdir / f"best_config_{'_'.join(args.model_list)}.json"
        out_best.write_text(json.dumps(best, indent=2), encoding="utf-8")

        print(f"\n[TUNE] Wrote: {out_csv}")
        print(f"[TUNE] Wrote: {out_best}")
        print("[TUNE] Top 5 configs:")
        print(df.head(5)[["st_dim","st_heads","st_inds","st_isab","dropout","lr","wd","batch_size","epochs",
                        "val_f1_mean","test_f1_mean","val_acc_mean","test_acc_mean"]])

        return

    # Datasets
    train_ds = EventDataset(
        manifest_csv=args.manifest, split="train", normalizer=normalizer,
        path_col=args.path_col, label_col=args.label_col,
        x_col=args.x_col, y_col=args.y_col, z_col=args.z_col, t_col=args.t_col,
        score_col=args.score_col, max_events_cap=cap,
    )
    val_ds = EventDataset(
        manifest_csv=args.manifest, split="val", normalizer=normalizer,
        path_col=args.path_col, label_col=args.label_col,
        x_col=args.x_col, y_col=args.y_col, z_col=args.z_col, t_col=args.t_col,
        score_col=args.score_col, max_events_cap=cap,
    )
    test_ds = EventDataset(
        manifest_csv=args.manifest, split="test", normalizer=normalizer,
        path_col=args.path_col, label_col=args.label_col,
        x_col=args.x_col, y_col=args.y_col, z_col=args.z_col, t_col=args.t_col,
        score_col=args.score_col, max_events_cap=cap,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_events, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_events, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_events, pin_memory=(device.type == "cuda")
    )

    # Model
    model = make_model(
        model_name=args.model,
        in_dim=5,
        n_classes=6,
        hidden=args.hidden,
        dropout=args.dropout,
        embed_dim=args.embed_dim,
        pool=getattr(args, "pool", "meanmax"),
        gate_hidden=getattr(args, "gate_hidden", 64),
        st_dim=args.st_dim,
        st_heads=args.st_heads,
        st_inds=args.st_inds,
        st_isab=args.st_isab,
        st_seeds=args.st_seeds,
        st_use_k_norm=(not args.st_no_k_norm),
    ).to(device)
    model_tag = args.model.lower()

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    print(f"\nDevice: {device}")
    print(f"Train/Val/Test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(f"Normalizer: mu={normalizer.mu:.6f}, sigma={normalizer.sigma:.6f}")
    print(f"Batch size: {args.batch_size}, epochs: {args.epochs}")
    if cap is not None:
        print(f"Max events cap: {cap}")

    history = []  # epoch-wise logs

    # Training loop with best-val checkpoint in memory
    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optim, device)
        val = evaluate(model, val_loader, device)

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_f1_macro": val["f1_macro"],
        })

        if val["acc"] > best_val_acc:
            best_val_acc = val["acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} | val_loss={val['loss']:.4f} | val_acc={val['acc']:.4f} | val_f1m={val['f1_macro']:.4f}")

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)

    print("\n=== Final Evaluation (BEST VAL MODEL) ===")
    val = evaluate(model, val_loader, device)
    test = evaluate(model, test_loader, device)


    print(f"\nVAL:  loss={val['loss']:.4f}  acc={val['acc']:.4f}  f1_macro={val['f1_macro']:.4f}")
    print("VAL Confusion matrix:\n", val["confusion_matrix"])

    print(f"\nTEST: loss={test['loss']:.4f} acc={test['acc']:.4f} f1_macro={test['f1_macro']:.4f}")
    print("TEST Confusion matrix:\n", test["confusion_matrix"])

    # Full classification report (test)
    print("\nTEST classification report:")
    print(classification_report(test["y_true"], test["y_pred"], digits=4))

    # Save outputs next to manifest
    outdir = Path(args.manifest).resolve().parent
    run_name = args.model.lower()
    tag = f"_{args.run_tag}" if args.run_tag else ""
    prefix = outdir / f"{run_name}_seed{args.seed}{tag}"
    out_metrics = outdir / f"{run_name}_metrics_seed{args.seed}.json"
    out_cm = outdir / f"{run_name}_confusion_matrix_seed{args.seed}.csv"
    hist_df = pd.DataFrame(history)
    hist_csv = Path(str(prefix) + "_train_history.csv")
    hist_df.to_csv(hist_csv, index=False)
    print(f"Wrote: {hist_csv}")
    # 4.2 Save per-class metrics table (TEST)
    labels = _get_label_names(args, n_classes=6)
    rep = classification_report(
        test["y_true"], test["y_pred"],
        target_names=labels,
        digits=4,
        output_dict=True,
        zero_division=0
    )
    rep_df = pd.DataFrame(rep).T
    rep_csv = Path(str(prefix) + "_test_classification_report.csv")
    rep_df.to_csv(rep_csv, index=True)
    print(f"Wrote: {rep_csv}")

    # 4.3 Save confusion matrix CSVs (raw + recall-normalized)
    cm_test = test["confusion_matrix"]
    cm_test_csv = Path(str(prefix) + "_test_confusion_matrix_raw.csv")
    pd.DataFrame(cm_test, index=labels, columns=labels).to_csv(cm_test_csv)
    print(f"Wrote: {cm_test_csv}")

    # 4.4 Save plots (curves + confusion matrices)
    if args.save_plots:
        curves_png = Path(str(prefix) + "_curves.png")
        plot_training_curves(hist_df, curves_png, dpi=args.plot_dpi)
        print(f"Wrote: {curves_png}")

        cm_raw_png = Path(str(prefix) + "_cm_test_raw.png")
        plot_confusion_matrix(cm_test, labels, cm_raw_png, title="TEST Confusion Matrix (counts)", dpi=args.plot_dpi, normalize=None)
        print(f"Wrote: {cm_raw_png}")

        cm_recall_png = Path(str(prefix) + "_cm_test_recall.png")
        plot_confusion_matrix(cm_test, labels, cm_recall_png, title="TEST Confusion Matrix (row-normalized = recall)", dpi=args.plot_dpi, normalize="true")
        print(f"Wrote: {cm_recall_png}")

        cm_prec_png = Path(str(prefix) + "_cm_test_precision.png")
        plot_confusion_matrix(cm_test, labels, cm_prec_png, title="TEST Confusion Matrix (col-normalized = precision)", dpi=args.plot_dpi, normalize="pred")
        print(f"Wrote: {cm_prec_png}")

    metrics_payload = {
        "seed": args.seed,
        "device": str(device),
        "normalizer_mu": normalizer.mu,
        "normalizer_sigma": normalizer.sigma,
        "val": {"loss": val["loss"], "acc": val["acc"], "f1_macro": val["f1_macro"]},
        "test": {"loss": test["loss"], "acc": test["acc"], "f1_macro": test["f1_macro"]},
        "model": {
            "name": args.model,
            "hidden": args.hidden,
            "embed_dim": getattr(args, "embed_dim", None),
            "pool": getattr(args, "pool", None),
            "dropout": args.dropout,
        },
        "train": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.wd},
    }
    out_metrics.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    pd.DataFrame(test["confusion_matrix"]).to_csv(out_cm, index=False)

    print(f"\nWrote: {out_metrics}")
    print(f"Wrote: {out_cm}")
    print("Done.")


if __name__ == "__main__":
    main()
