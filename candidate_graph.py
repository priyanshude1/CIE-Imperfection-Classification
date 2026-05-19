#!/usr/bin/env python3
"""
Count candidate residual "events" per imperfect simulation using stress_score_scaler.json,
and plot the distribution.

Candidate rule (default):
  abs_strength = log1p(|Δ|) > TAU_ABS
  rel_strength = |(Δ - median)/(scale_used + eps)| >= TAU_REL

Outputs:
- <OUTDIR>/candidate_counts_tauRelX_tauAbsY.csv
- Histogram + sorted line plot

Run:
  python count_candidates.py --outdir "C:/path/to/outdir" --scaler "C:/path/to/outdir/stress_score_scaler.CAPPEDx10.Q0.75.json"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RES_PREFIX = "Residual_"


def load_scaler(path: Path) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(f"Scaler JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def detect_residual_columns(residual_csv: Path, col_stats: Dict) -> List[str]:
    hdr = pd.read_csv(residual_csv, nrows=0)
    cols = [c for c in hdr.columns if c.startswith(RES_PREFIX) and c in col_stats]
    return cols


def count_candidates_for_residual_csv(
    residual_csv: Path,
    regime_key: str,
    scaler: Dict,
    tau_rel: float,
    tau_abs: float,
    chunksize: int,
) -> int:
    per_regime = scaler.get("per_regime", {})
    if regime_key not in per_regime:
        raise KeyError(f"regime_key '{regime_key}' not found in scaler.")
    col_stats = per_regime[regime_key]["columns"]
    eps = float(scaler.get("eps", 1e-12))

    residual_cols = detect_residual_columns(residual_csv, col_stats)
    if not residual_cols:
        return 0

    # Pre-vectorize medians/scales in file column order
    med = np.array([float(col_stats[c]["median"]) for c in residual_cols], dtype=np.float64)
    scl = np.array([float(col_stats[c]["scale_used"]) for c in residual_cols], dtype=np.float64) + eps

    total = 0
    for chunk in pd.read_csv(residual_csv, usecols=residual_cols, chunksize=chunksize):
        X = chunk.to_numpy(dtype=np.float64, copy=False)

        # abs_strength = log1p(|Δ|)
        abs_strength = np.log1p(np.abs(X))

        # rel_strength = |(Δ - median) / (scale_used + eps)|
        rel_strength = np.abs((X - med) / scl)

        # Candidate rule
        cand = (abs_strength > tau_abs) & (rel_strength >= tau_rel)
        total += int(cand.sum())

    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, required=True, help="Outdir containing residual_index.csv and residuals_stress/")
    ap.add_argument("--scaler", type=str, required=True, help="Path to stress_score_scaler.json (prefer your capped file)")
    ap.add_argument("--tau_rel", type=float, default=10.0, help="Relative threshold (default 3.0)")
    ap.add_argument("--tau_abs", type=float, default=10.0, help="Absolute/log threshold on log1p(|Δ|) (default 0.0)")
    ap.add_argument("--chunksize", type=int, default=5000, help="CSV read chunksize (rows) (default 5000)")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    index_csv = outdir / "residual_index.csv"
    if not index_csv.is_file():
        raise FileNotFoundError(f"residual_index.csv not found at: {index_csv}")

    scaler_path = Path(args.scaler).resolve()
    scaler = load_scaler(scaler_path)

    idx = pd.read_csv(index_csv)

    # Filter to imperfect and OK
    idx = idx[idx["status"].astype(str) == "OK"].copy()
    idx = idx[idx["case_type"].astype(str).str.upper() == "IMPERFECT"].copy()

    # Ignore if residual path doesn't exist (your rule)
    idx["residual_exists"] = idx["residual_csv_path"].apply(lambda p: Path(str(p)).is_file())
    idx = idx[idx["residual_exists"]].copy()

    results = []
    errors = []

    for row in idx.itertuples(index=False):
        residual_path = Path(getattr(row, "residual_csv_path"))
        regime_key = getattr(row, "regime_key")
        case_name = getattr(row, "case_name", "")

        try:
            n = count_candidates_for_residual_csv(
                residual_csv=residual_path,
                regime_key=regime_key,
                scaler=scaler,
                tau_rel=float(args.tau_rel),
                tau_abs=float(args.tau_abs),
                chunksize=int(args.chunksize),
            )
            results.append(
                {
                    "regime_key": regime_key,
                    "case_name": case_name,
                    "residual_csv_path": str(residual_path),
                    "candidate_count": int(n),
                }
            )
        except Exception as e:
            errors.append(
                {
                    "regime_key": regime_key,
                    "case_name": case_name,
                    "residual_csv_path": str(residual_path),
                    "error": str(e),
                }
            )

    df = pd.DataFrame(results).sort_values("candidate_count", ascending=False).reset_index(drop=True)

    out_csv = outdir / f"candidate_counts_tauRel{args.tau_rel:g}_tauAbs{args.tau_abs:g}.csv"
    df.to_csv(out_csv, index=False)

    print(f"Scaler:     {scaler_path}")
    print(f"Index:      {index_csv}")
    print(f"Processed:  {len(df)} imperfect simulations")
    print(f"Errors:     {len(errors)}")
    print(f"Saved CSV:  {out_csv}")

    if errors:
        err_csv = outdir / "candidate_count_errors.csv"
        pd.DataFrame(errors).to_csv(err_csv, index=False)
        print(f"Saved errors CSV: {err_csv}")

    # --- Plot 1: Histogram (distribution) ---
    plt.figure()
    plt.hist(df["candidate_count"].to_numpy(), bins=30)
    plt.title(f"Candidate counts across imperfect simulations (tau_rel={args.tau_rel}, tau_abs={args.tau_abs})")
    plt.xlabel("Candidate count (cells passing thresholds)")
    plt.ylabel("Number of simulations")
    plt.show()

    # --- Plot 2: Sorted counts (shape / elbow) ---
    plt.figure()
    plt.plot(np.arange(len(df)), df["candidate_count"].to_numpy())
    plt.title(f"Sorted candidate counts (tau_rel={args.tau_rel}, tau_abs={args.tau_abs})")
    plt.xlabel("Simulation rank (sorted desc)")
    plt.ylabel("Candidate count")
    plt.show()


if __name__ == "__main__":
    main()
