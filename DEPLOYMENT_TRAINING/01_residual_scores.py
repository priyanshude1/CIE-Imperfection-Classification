#!/usr/bin/env python3
"""
Build per-regime, per-stress-column robust scalers for residual scoring,
adapted for the NEW residual_index CSV (naming inconsistencies).

Core logic (unchanged):
- Fit per regime_key and per column:
    median_all = median(x) over all values
    MAD_raw = median(|x - median_all|) computed over NON-ZERO abs deviations only
- Per-regime MAD floor:
    mad_floor_regime = q-quantile of non-zero MAD_raw across columns in that regime
- scale_used = max(MAD_raw, mad_floor_regime, global_floor)
- Write stress_score_scaler.json + scaler_build_report.csv

Differences vs old script:
- Index schema is flexible (auto-detects path columns)
- No dependency on metadata/status columns
- Filters "perfect" by path/case_name heuristics
- Can restrict to EquivalentStress residual columns

Run (recommended):
  python 01_build_stress_score_scaler_from_new_index.py \
    --outdir "C:/.../DEPLOYMENT_TRAINING" \
    --index  "C:/.../residuals_stress_index_clean.csv" \
    --only_equivstress

If you want explicit columns (most robust):
  python ... --cols Residual_EquivalentStress_t0p1 Residual_EquivalentStress_t0p5 ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# -------------------------
# Helpers
# -------------------------

def _path_exists(p: str | Path) -> bool:
    try:
        return Path(p).is_file()
    except Exception:
        return False


def _infer_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    if required:
        raise RuntimeError(f"Could not find any of columns {candidates} in index. Available={list(df.columns)}")
    return None


def robust_median_and_mad_nonzero(vals: np.ndarray) -> Tuple[float, float, int]:
    """
    Returns (median_all, mad_raw_nonzero, n_nonzero_absdev)
    - median_all computed over all values (including zeros)
    - MAD computed as median(|x - median|) over NON-ZERO absolute deviations only
    """
    vals = vals.astype(np.float64, copy=False)
    if vals.size == 0:
        return float("nan"), 0.0, 0

    med = float(np.median(vals))
    absdev = np.abs(vals - med)
    nz = absdev[absdev > 0.0]
    n_nz = int(nz.size)
    if n_nz == 0:
        return med, 0.0, 0
    mad = float(np.median(nz))
    return med, mad, n_nz


def detect_columns_from_csv(csv_path: Path, only_equivstress: bool) -> List[str]:
    """
    Detect candidate residual columns from one CSV header.

    Supports:
    - "Residual_..." prefix (old format)
    - plain "EquivalentStress_t..." (if residual file stored already as residual values without prefix)
    - "Residual_EquivalentStress_t..." (hybrid)
    """
    cols = pd.read_csv(csv_path, nrows=0).columns.tolist()

    if only_equivstress:
        # accept any of these patterns
        keep = []
        for c in cols:
            cl = c.lower()
            if "equivalentstress" not in cl:
                continue
            # allow both residual-prefixed and non-prefixed
            # and avoid coordinate columns etc
            # keep only time-step series columns: contain "_t"
            if "_t" in cl:
                keep.append(c)
        if not keep:
            raise RuntimeError(f"No EquivalentStress time-step columns detected in: {csv_path}")
        return keep

    # otherwise detect residual-like series columns
    keep = []
    for c in cols:
        if c.startswith("Residual_"):
            keep.append(c)
    if keep:
        return keep

    # fallback: try stress series columns with "_t"
    for c in cols:
        cl = c.lower()
        if "_t" in cl and ("stress" in cl):
            keep.append(c)
    if not keep:
        raise RuntimeError(f"No residual/stress series columns detected in: {csv_path}")
    return keep


def is_perfect_row(path_str: str, case_name: Optional[str]) -> bool:
    s = path_str.lower()
    if "perfect" in s:
        return True
    if case_name is not None and "perfect" in str(case_name).lower():
        return True
    return False


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, required=True, help="Output directory for scaler json/report.")
    ap.add_argument("--index", type=str, required=True, help="Path to NEW residual index CSV.")
    ap.add_argument("--eps", type=float, default=1e-12, help="Epsilon added to scale at scoring time.")
    ap.add_argument("--global_floor", type=float, default=1e-12, help="Minimum scale floor.")
    ap.add_argument("--mad_floor_quantile", type=float, default=0.05, help="Per-regime floor quantile.")
    ap.add_argument(
        "--only_equivstress",
        action="store_true",
        help="If set, fit scalers only for EquivalentStress time-step columns.",
    )
    ap.add_argument(
        "--cols",
        nargs="*",
        default=None,
        help="Optional explicit list of columns to scale (overrides auto-detection).",
    )
    ap.add_argument(
        "--exclude_ignore",
        action="store_true",
        help="If set, skip rows whose residual_csv_path contains '[IGNORE]'.",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    index_path = Path(args.index).resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"Index not found: {index_path}")

    df = pd.read_csv(index_path)

    # Required: regime_key and residual path
    regime_col = _infer_col(df, ["regime_key", "regime", "rk"], required=True)
    path_col = _infer_col(df, ["residual_csv_path", "residual_path", "csv_path", "path", "file_path"], required=True)
    case_col = _infer_col(df, ["case_name", "case", "name"], required=False)

    # Filter: path exists
    df["__path_exists__"] = df[path_col].astype(str).apply(_path_exists)
    df = df[df["__path_exists__"]].copy()

    # Optional: skip [IGNORE]
    if args.exclude_ignore:
        df = df[~df[path_col].astype(str).str.contains(r"\[IGNORE\]", regex=True)].copy()

    # Filter: remove perfect rows
    if case_col is not None:
        df["__is_perfect__"] = [
            is_perfect_row(p, cn) for p, cn in zip(df[path_col].astype(str), df[case_col].astype(str))
        ]
    else:
        df["__is_perfect__"] = df[path_col].astype(str).str.lower().str.contains("perfect")
    df = df[~df["__is_perfect__"]].copy()

    if df.empty:
        raise RuntimeError("No rows left after filtering path existence / perfect / ignore rules.")

    # Determine columns to scale
    if args.cols and len(args.cols) > 0:
        cols_to_scale = list(args.cols)
    else:
        first_csv = Path(df.iloc[0][path_col]).resolve()
        cols_to_scale = detect_columns_from_csv(first_csv, only_equivstress=bool(args.only_equivstress))

    regimes = sorted(df[regime_col].astype(str).unique().tolist())

    eps = float(args.eps)
    global_floor = float(args.global_floor)
    q = float(args.mad_floor_quantile)

    scaler: Dict[str, Dict] = {
        "scaler_type": "per_regime_per_column",
        "fit_cases": "INDEX_ROWS_MINUS_PERFECT",  # because new index may not have case_type
        "mad_definition": "MAD_raw = median(|x - median_all|) over non-zero absolute deviations only",
        "mad_floor_strategy": f"per_regime_quantile_nonzero_mad(q={q})",
        "eps": eps,
        "global_floor": global_floor,
        "columns_scaled": cols_to_scale,
        "per_regime": {},
    }

    report_rows: List[Dict] = []

    for rk in regimes:
        df_r = df[df[regime_col].astype(str) == rk].copy()
        paths = [Path(p).resolve() for p in df_r[path_col].astype(str).tolist()]
        paths = [p for p in paths if p.is_file()]
        if not paths:
            continue

        per_col_vals: Dict[str, List[np.ndarray]] = {c: [] for c in cols_to_scale}
        total_files_used = 0

        for p in paths:
            try:
                arr = pd.read_csv(p, usecols=cols_to_scale).to_numpy(dtype=np.float64)
            except Exception:
                # skip unreadable/corrupt file
                continue

            if not np.isfinite(arr).all():
                continue

            for j, c in enumerate(cols_to_scale):
                per_col_vals[c].append(arr[:, j])

            total_files_used += 1

        if total_files_used == 0:
            continue

        col_stats: Dict[str, Dict] = {}
        mad_list_nonzero = []

        for c in cols_to_scale:
            vals = np.concatenate(per_col_vals[c], axis=0) if per_col_vals[c] else np.array([], dtype=np.float64)
            med, mad_raw, n_nz = robust_median_and_mad_nonzero(vals)

            if mad_raw > 0.0:
                mad_list_nonzero.append(mad_raw)

            col_stats[c] = {
                "median": float(med),
                "mad_raw": float(mad_raw),
                "n_samples": int(vals.size),
                "n_nonzero_absdev": int(n_nz),
            }

        mad_floor_regime = float(np.quantile(np.array(mad_list_nonzero, dtype=np.float64), q)) if mad_list_nonzero else 0.0

        n_floored = 0
        for c in cols_to_scale:
            mad_raw = col_stats[c]["mad_raw"]
            scale_used = max(float(mad_raw), float(mad_floor_regime), global_floor)
            was_floored = (scale_used > mad_raw)
            if was_floored:
                n_floored += 1

            col_stats[c].update(
                {
                    "mad_floor_regime": float(mad_floor_regime),
                    "scale_used": float(scale_used),
                    "was_floored": bool(was_floored),
                }
            )

            report_rows.append(
                {
                    "regime_key": rk,
                    "column": c,
                    "median": col_stats[c]["median"],
                    "mad_raw": col_stats[c]["mad_raw"],
                    "mad_floor_regime": float(mad_floor_regime),
                    "scale_used": col_stats[c]["scale_used"],
                    "was_floored": col_stats[c]["was_floored"],
                    "n_samples": col_stats[c]["n_samples"],
                    "n_nonzero_absdev": col_stats[c]["n_nonzero_absdev"],
                    "files_used_in_regime": total_files_used,
                }
            )

        scaler["per_regime"][rk] = {
            "files_used": int(total_files_used),
            "mad_floor_regime": float(mad_floor_regime),
            "columns": col_stats,
        }

        print(
            f"[{rk}] files_used={total_files_used}  mad_floor_regime={mad_floor_regime:.6g}  "
            f"floored_cols={n_floored}/{len(cols_to_scale)}"
        )

    scaler_path = outdir / "stress_score_scaler.json"
    report_path = outdir / "scaler_build_report.csv"

    scaler_path.write_text(json.dumps(scaler, indent=2), encoding="utf-8")
    pd.DataFrame(report_rows).to_csv(report_path, index=False)

    print("\nWrote scaler:", scaler_path)
    print("Wrote report:", report_path)
    print("Done.")


if __name__ == "__main__":
    main()
