#!/usr/bin/env python3
"""
Rank the 40 residual stress columns using stress_score_scaler.json.

Produces:
1) Global ranking across regimes:
   - mean_n_nonzero_absdev (average across regimes for each column)
   - mean_scale_used (average across regimes for each column)
   - plus min/max for both (useful sanity)
2) Per-regime ranking (optional console print):
   - ranks the 40 columns inside each regime by n_nonzero_absdev

Run:
  python rank_stress_columns.py --scaler "C:/path/to/stress_score_scaler.json" --out "column_rankings.csv"

Optional:
  --print_per_regime
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def load_scaler(path: Path) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(f"Scaler JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaler", type=str, required=True, help="Path to stress_score_scaler.json")
    ap.add_argument("--out", type=str, default="column_rankings.csv", help="Output CSV for global rankings")
    ap.add_argument(
        "--print_per_regime",
        action="store_true",
        help="If set, prints per-regime rankings to console (40 lines per regime).",
    )
    args = ap.parse_args()

    scaler_path = Path(args.scaler).resolve()
    scaler = load_scaler(scaler_path)

    if "per_regime" not in scaler or not isinstance(scaler["per_regime"], dict):
        raise RuntimeError("Scaler JSON missing 'per_regime' dictionary.")

    regimes = sorted(scaler["per_regime"].keys())
    if not regimes:
        raise RuntimeError("No regimes found under scaler['per_regime'].")

    # Build long-form table: one row per (regime_key, column)
    rows: List[Dict] = []
    for rk in regimes:
        reg = scaler["per_regime"][rk]
        cols = reg.get("columns", {})
        if not isinstance(cols, dict) or not cols:
            continue

        for col_name, stats in cols.items():
            # These keys are from your scaler builder
            n_nz = stats.get("n_nonzero_absdev", None)
            scale_used = stats.get("scale_used", None)

            # Be defensive: skip if missing
            if n_nz is None or scale_used is None:
                continue

            rows.append(
                {
                    "regime_key": rk,
                    "column": col_name,
                    "n_nonzero_absdev": int(n_nz),
                    "scale_used": float(scale_used),
                    "mad_raw": float(stats.get("mad_raw", np.nan)),
                    "median": float(stats.get("median", np.nan)),
                    "was_floored": bool(stats.get("was_floored", False)),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No usable rows extracted from scaler JSON. Check keys/format.")

    # ---- Global aggregation across regimes (per column) ----
    agg = (
        df.groupby("column", as_index=False)
        .agg(
            mean_n_nonzero_absdev=("n_nonzero_absdev", "mean"),
            median_n_nonzero_absdev=("n_nonzero_absdev", "median"),
            min_n_nonzero_absdev=("n_nonzero_absdev", "min"),
            max_n_nonzero_absdev=("n_nonzero_absdev", "max"),
            mean_scale_used=("scale_used", "mean"),
            median_scale_used=("scale_used", "median"),
            min_scale_used=("scale_used", "min"),
            max_scale_used=("scale_used", "max"),
            floored_regime_count=("was_floored", "sum"),
            regimes_present=("regime_key", "nunique"),
        )
    )

    # Rank primarily by mean_n_nonzero_absdev (desc), then by mean_scale_used (desc)
    agg["rank_by_mean_nonzero_absdev"] = (
        agg[["mean_n_nonzero_absdev", "mean_scale_used"]]
        .apply(tuple, axis=1)
        .rank(method="min", ascending=False)
        .astype(int)
    )

    # Also provide separate rank by mean scale only
    agg["rank_by_mean_scale_used"] = agg["mean_scale_used"].rank(method="min", ascending=False).astype(int)

    # Sort for output
    agg = agg.sort_values(
        ["mean_n_nonzero_absdev", "mean_scale_used", "column"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    out_path = Path(args.out).resolve()
    agg.to_csv(out_path, index=False)

    # ---- Console summary (top 15) ----
    print(f"Loaded scaler: {scaler_path}")
    print(f"Regimes found: {len(regimes)}")
    print(f"Saved global column rankings to: {out_path}\n")

    print("Top 15 columns by mean_n_nonzero_absdev (then mean_scale_used):")
    show = agg.head(15)[
        [
            "column",
            "mean_n_nonzero_absdev",
            "mean_scale_used",
            "floored_regime_count",
            "regimes_present",
        ]
    ].copy()
    # prettier formatting
    show["mean_n_nonzero_absdev"] = show["mean_n_nonzero_absdev"].map(lambda x: f"{x:.2f}")
    show["mean_scale_used"] = show["mean_scale_used"].map(lambda x: f"{x:.6g}")
    print(show.to_string(index=False))

    # ---- Optional: per-regime ranking by n_nonzero_absdev ----
    if args.print_per_regime:
        print("\n\n=== Per-regime rankings by n_nonzero_absdev (desc) ===")
        for rk in regimes:
            sub = df[df["regime_key"] == rk].copy()
            if sub.empty:
                continue
            sub = sub.sort_values(["n_nonzero_absdev", "scale_used", "column"], ascending=[False, False, True])
            print(f"\n[{rk}]")
            for i, r in enumerate(sub.itertuples(index=False), start=1):
                print(f"{i:02d}. {r.column}  n_nonzero_absdev={r.n_nonzero_absdev}  scale_used={r.scale_used:.6g}")


if __name__ == "__main__":
    main()
