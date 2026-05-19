#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Helper Functions
# -------------------------

def load_score_normalizer(path: str) -> Tuple[float, float]:
    j = json.loads(Path(path).read_text(encoding="utf-8"))
    return float(j["mu"]), float(j["sigma"])

def load_coord_scaler(path: str) -> Dict[str, float]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def load_node_lookup(tsv_path: str, scaler: Dict) -> Dict[int, Tuple[float, float, float]]:
    df = pd.read_csv(tsv_path, sep="\t", engine="python")
    df.columns = [c.strip() for c in df.columns]
    
    id_col = "Node ID"
    x_col, y_col, z_col = "X(mm)", "Y(mm)", "Z(mm)"
    
    # Filter valid nodes
    df = df.dropna(subset=[id_col])
    df[id_col] = pd.to_numeric(df[id_col], errors='coerce')
    df = df[df[id_col].between(1, scaler["max_node_id"])].copy()
    
    # Min-Max Normalize coordinates
    xn = (df[x_col].values - scaler["x_min"]) / (scaler["x_max"] - scaler["x_min"])
    yn = (df[y_col].values - scaler["y_min"]) / (scaler["y_max"] - scaler["y_min"])
    zn = (df[z_col].values - scaler["z_min"]) / (scaler["z_max"] - scaler["z_min"])
    
    return {int(nid): (float(x), float(y), float(z)) 
            for nid, x, y, z in zip(df[id_col], xn, yn, zn)}

# -------------------------
# Set Transformer Model (Architecture Match)
# -------------------------

class MAB(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.ln2 = nn.LayerNorm(dim)
    def forward(self, Q, K, key_padding_mask_K=None):
        out, _ = self.attn(Q, K, K, key_padding_mask=key_padding_mask_K)
        H = self.ln1(Q + out)
        return self.ln2(H + self.ff(H))

class ISAB(nn.Module):
    def __init__(self, dim, num_heads, num_inds, dropout):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, num_inds, dim) * 0.02)
        self.mab1, self.mab2 = MAB(dim, num_heads, dropout), MAB(dim, num_heads, dropout)
    def forward(self, X, key_padding_mask_X=None):
        H = self.mab1(self.I.expand(X.size(0), -1, -1), X, key_padding_mask_K=key_padding_mask_X)
        return self.mab2(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, dropout):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, num_seeds, dim) * 0.02)
        self.mab = MAB(dim, num_heads, dropout)
    def forward(self, X, key_padding_mask_X=None):
        return self.mab(self.S.expand(X.size(0), -1, -1), X, key_padding_mask_K=key_padding_mask_X)

class SetTransformerClassifier(nn.Module):
    def __init__(self, in_dim=5, dim=32, num_heads=2, num_inds=16, num_isab=1, num_seeds=1, hidden=128, n_classes=6):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.0), nn.Linear(64, dim))
        self.isabs = nn.ModuleList([ISAB(dim, num_heads, num_inds, 0.0) for _ in range(num_isab)])
        self.pma = PMA(dim, num_heads, num_seeds, 0.0)
        self.head = nn.Sequential(
            nn.Linear(dim*num_seeds + 1, hidden), nn.ReLU(), nn.Dropout(0.0),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.0), nn.Linear(hidden, n_classes)
        )
    def forward(self, X):
        mask = torch.zeros(X.size(0), X.size(1), device=X.device).bool() 
        H = self.encoder(X)
        for blk in self.isabs: H = blk(H, key_padding_mask_X=mask)
        P = self.pma(H, key_padding_mask_X=mask).reshape(X.size(0), -1)
        k_norm = torch.tensor([[(X.size(1)-1000)/9000.0]], device=X.device).clamp(0,1)
        return self.head(torch.cat([P, k_norm], dim=1))

# -------------------------
# Event Logic & CSV Output
# -------------------------

def process_events(args, node_lookup, mu, sigma):
    df = pd.read_csv(args.residual_csv)
    res_cols = [c for c in df.columns if c.startswith("Residual_")]
    
    scaler = json.loads(Path(args.stress_scaler).read_text())
    reg_stats = scaler["per_regime"][args.regime_key]["columns"]
    
    df = df[df["Node Number"].isin(node_lookup.keys())].reset_index(drop=True)
    if df.empty: return None, None
    
    node_ids = df["Node Number"].values
    R = df[res_cols].values
    meds = np.array([reg_stats[c]["median"] for c in res_cols])
    scas = np.array([reg_stats[c]["scale_used"] for c in res_cols])
    
    z = np.abs(R - meds) / (scas + 1e-12)
    mask = (z >= args.tau_rel) | (np.abs(R) >= args.tau_abs)
    idx = np.argwhere(mask)
    if len(idx) == 0: return None, None
    
    rel_s = z[idx[:,0], idx[:,1]]
    K = max(args.k_min, min(int(round(args.p * len(idx))), args.k_max))
    top_idx = np.argsort(-rel_s)[:K]
    sel = idx[top_idx]
    
    # Feature extraction
    coords = np.array([node_lookup[node_ids[i]] for i in sel[:,0]])
    t_n = (sel[:,1] / max(len(res_cols)-1, 1)).reshape(-1, 1)
    s_norm = ((np.log1p(np.maximum(rel_s[top_idx], 0)) - mu) / sigma).reshape(-1, 1)
    
    # Prepare CSV for output
    event_df = pd.DataFrame({
        "node_id": node_ids[sel[:,0]],
        "x_n": coords[:,0], "y_n": coords[:,1], "z_n": coords[:,2],
        "t_n": t_n.flatten(),
        "rel_strength": rel_s[top_idx],
        "s_norm": s_norm.flatten()
    })
    
    if args.save_events_csv:
        event_df.to_csv(args.save_events_csv, index=False)
        print(f"Events saved to: {args.save_events_csv}")

    feat_tensor = torch.from_numpy(np.hstack([coords, t_n, s_norm])).float().unsqueeze(0)
    return feat_tensor, len(event_df)

# -------------------------
# Main Execution
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual_csv", required=True)
    parser.add_argument("--regime_key", required=True)
    parser.add_argument("--stress_scaler", required=True)
    parser.add_argument("--score_normalizer", required=True)
    parser.add_argument("--node_coords_tsv", required=True)
    parser.add_argument("--coord_scaler_json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save_events_csv", help="Path to save extracted events CSV")
    parser.add_argument("--tau_rel", type=float, default=100.0)
    parser.add_argument("--tau_abs", type=float, default=10000.0)
    parser.add_argument("--p", type=float, default=0.03)
    parser.add_argument("--k_min", type=int, default=1000)
    parser.add_argument("--k_max", type=int, default=10000)
    args = parser.parse_args()

    mu, sigma = load_score_normalizer(args.score_normalizer)
    c_scaler = load_coord_scaler(args.coord_scaler_json)
    lookup = load_node_lookup(args.node_coords_tsv, c_scaler)
    
    events, count = process_events(args, lookup, mu, sigma)
    if events is None:
        print("No events detected.")
        return

    print(f"Extracted {count} events.")

    # Load and Predict
    model = SetTransformerClassifier()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(events)
        probs = F.softmax(logits, dim=1).numpy()[0]
        print(f"Predicted Class: {np.argmax(probs)}")
        print(f"Probabilities: {np.round(probs, 4)}")

if __name__ == "__main__":
    main()