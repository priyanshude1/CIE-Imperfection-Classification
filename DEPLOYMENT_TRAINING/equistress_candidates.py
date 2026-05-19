#!/usr/bin/env python3
"""
Count candidate "events" per imperfect simulation, but ONLY using EquivalentStress residual columns.

Candidate rule:
  abs_strength = log1p(|Δ|) > TAU_ABS
  rel_strength = |(Δ - median) / (scale_used + eps)| >= TAU_REL

This version:
- Loads scaler JSON (per-regime medians/scales)
- Reads residual_index.csv
- For each IMPERFECT residual CSV, counts candidates ONLY over:
    Residual_EquivalentStress_t0p1 ... Residual_EquivalentStress_t3p0
- Saves summary CSV and plots distribution.

Run (example):
  python count_candidates_equiv_only.py --outdir "C:/path/to/outdir" --scaler "C:/path/to/stress_score_scaler.CAPPED.json" --tau_rel 5 --tau_abs 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RES_PREFIX = "Residual_"
EQ_STRESS_COLS: List[str] = [
    "EquivalentStress_t0p1","EquivalentStress_t0p5","EquivalentStress_t0p9","EquivalentStress_t1p1","EquivalentStress_t1p5",
    "EquivalentStress_t1p9","EquivalentStress_t2p1","EquivalentStress_t2p5","EquivalentStress_t2p9","EquivalentStress_t3p0",
]
EQ_RES_COLS: List[str] = [f"{RES_PREFIX}{c}" for c in EQ_STRESS_COLS]


def load_json(path: Path) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(f"JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def get_regime_col_stats(scaler: Dict, regime_key: str) -> Dict[str, Dict]:
    per_regime = scaler.get("per_regime", {})
    if regime_key not in per_regime:
        raise KeyError(f"regime_key '{regime_key}' not found in scaler['per_regime'].")
    return per_regime[regime_key].get("columns", {})


def intersect_equiv_cols_in_file(residual_csv: Path, col_stats: Dict[str, Dict]) -> List[str]:
    hdr = pd.read_csv(residual_csv, nrows=0)
    present = set(hdr.columns.tolist())

    cols = []
    for c in EQ_RES_COLS:
        if c in present and c in col_stats:
            cols.append(c)

    return cols


def count_candidates_for_residual_csv_equiv_only(
    residual_csv: Path,
    regime_key: str,
    scaler: Dict,
    tau_rel: float,
    tau_abs: float,
    chunksize: int,
) -> int:
    col_stats = get_regime_col_stats(scaler, regime_key)
    eps = float(scaler.get("eps", 1e-12))

    use_cols = intersect_equiv_cols_in_file(residual_csv, col_stats)
    if not use_cols:
        return 0

    # Pre-vectorize medians/scales in the column order we will read
    med = np.array([float(col_stats[c]["median"]) for c in use_cols], dtype=np.float64)
    scl = np.array([float(col_stats[c]["scale_used"]) for c in use_cols], dtype=np.float64) + eps

    total = 0
    for chunk in pd.read_csv(residual_csv, usecols=use_cols, chunksize=chunksize):
        X = chunk.to_numpy(dtype=np.float64, copy=False)

        abs_strength = np.log1p(np.abs(X))
        rel_strength = np.abs((X - med) / scl)

        cand = (abs_strength > tau_abs) & (rel_strength >= tau_rel)
        total += int(cand.sum())

    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, required=True, help="Outdir containing residual_index.csv and residual CSV paths.")
    ap.add_argument("--scaler", type=str, required=True, help="Path to stress_score_scaler.json (prefer capped).")
    ap.add_argument("--index", type=str, default=None, help="Optional path to residual_index.csv (default: <outdir>/residual_index.csv)")
    ap.add_argument("--tau_rel", type=float, default=5.0)
    ap.add_argument("--tau_abs", type=float, default=0.0)
    ap.add_argument("--chunksize", type=int, default=5000)
    ap.add_argument("--out", type=str, default=None, help="Output CSV name (default auto).")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    scaler_path = Path(args.scaler).resolve()
    index_path = Path(args.index).resolve() if args.index else (outdir / "residual_index.csv")

    if not index_path.is_file():
        raise FileNotFoundError(f"residual_index.csv not found: {index_path}")

    scaler = load_json(scaler_path)
    idx = pd.read_csv(index_path)

    # Filter to usable imperfect runs
    idx = idx[idx["status"].astype(str) == "OK"].copy()
    idx = idx[idx["case_type"].astype(str).str.upper() == "IMPERFECT"].copy()

    # Keep only existing residual paths
    idx["residual_exists"] = idx["residual_csv_path"].apply(lambda p: Path(str(p)).is_file())
    idx = idx[idx["residual_exists"]].copy()

    results = []
    errors = []

    for row in idx.itertuples(index=False):
        residual_path = Path(getattr(row, "residual_csv_path"))
        regime_key = getattr(row, "regime_key")
        case_name = getattr(row, "case_name", "")

        try:
            n = count_candidates_for_residual_csv_equiv_only(
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
                    "candidate_count_equivalent_only": int(n),
                    "tau_rel": float(args.tau_rel),
                    "tau_abs": float(args.tau_abs),
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

    df = pd.DataFrame(results).sort_values("candidate_count_equivalent_only", ascending=False).reset_index(drop=True)

    out_name = args.out or f"candidate_counts_equiv_only_tauRel{args.tau_rel:g}_tauAbs{args.tau_abs:g}.csv"
    out_csv = outdir / out_name
    df.to_csv(out_csv, index=False)

    print(f"Scaler:     {scaler_path}")
    print(f"Index:      {index_path}")
    print(f"Processed:  {len(df)} imperfect simulations")
    print(f"Errors:     {len(errors)}")
    print(f"Saved CSV:  {out_csv}")

    if errors:
        err_csv = outdir / "candidate_count_equiv_only_errors.csv"
        pd.DataFrame(errors).to_csv(err_csv, index=False)
        print(f"Saved errors CSV: {err_csv}")

    # Plot distribution
    if not df.empty:
        plt.figure()
        plt.hist(df["candidate_count_equivalent_only"].to_numpy(), bins=30)
        plt.title(f"Candidate counts (EquivalentStress only) (tau_rel={args.tau_rel}, tau_abs={args.tau_abs})")
        plt.xlabel("Candidate count (cells passing thresholds)")
        plt.ylabel("Number of simulations")
        plt.show()

        plt.figure()
        plt.plot(np.arange(len(df)), df["candidate_count_equivalent_only"].to_numpy())
        plt.title(f"Sorted candidate counts (EquivalentStress only) (tau_rel={args.tau_rel}, tau_abs={args.tau_abs})")
        plt.xlabel("Simulation rank (sorted desc)")
        plt.ylabel("Candidate count")
        plt.show()


if __name__ == "__main__":
    main()
