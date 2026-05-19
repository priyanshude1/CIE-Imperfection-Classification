#!/usr/bin/env python3
"""
Extract top-k "events" from stress residuals using per-regime scalers.

What it does:
- Looks under: OUTDIR/residuals_stress/<regime_key>/<case_name>/stress_residual.csv
- Uses ONLY EquivalentStress residual columns (10 time steps):
    Residual_EquivalentStress_t0p1 ... Residual_EquivalentStress_t3p0
- Applies per-regime scaling from stress_score_scaler*.json:
    abs_strength = log1p(|Δ|)
    rel_strength = |(Δ - median)/(scale_used + eps)|
- Candidate mask:
    (rel_strength >= tau_rel) & (abs_strength > tau_abs)
- For each simulation:
    n_candidates = count(mask)
    k = clamp(round(p * n_candidates), k_min, k_max)
    rank candidates by rel_strength desc, abs_strength desc
    take top-k
- Adds normalized coordinates and normalized time:
    coord norm: min-max to [0,1] computed from full coordinate file
    time norm: (t - 0.1)/(3.0 - 0.1) to [0,1]
- Writes:
    OUTDIR/events_equivstress/<regime_key>/<case_name>/events_topk.csv
- Writes a summary CSV:
    OUTDIR/events_equivstress_index.csv

Usage:
  python extract_events_equivstress.py --outdir "C:/path/to/outdir" ^
      --scaler "C:/path/to/outdir/stress_score_scaler.CAPPEDx10.Q0.75.json" ^
      --coords "C:/path/to/root/nnode_coordinates.txt" ^
      --tau_rel 10 --tau_abs 10 --p 0.03 --kmin 100 --kmax 10000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


RESIDUALS_DIRNAME = "residuals_stress"
RESIDUAL_FILENAME = "stress_residual.csv"
RES_PREFIX = "Residual_"
NODE_COL = "Node Number"

# Equivalent stress time columns in the original CSV naming convention (10 steps)
EQ_COLS = [
    "EquivalentStress_t0p1",
    "EquivalentStress_t0p5",
    "EquivalentStress_t0p9",
    "EquivalentStress_t1p1",
    "EquivalentStress_t1p5",
    "EquivalentStress_t1p9",
    "EquivalentStress_t2p1",
    "EquivalentStress_t2p5",
    "EquivalentStress_t2p9",
    "EquivalentStress_t3p0",
]
EQ_RESID_COLS = [f"{RES_PREFIX}{c}" for c in EQ_COLS]


def load_json(path: Path) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(f"JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def parse_time_from_col(col: str) -> float:
    """
    Input: 'Residual_EquivalentStress_t0p1' or 'EquivalentStress_t0p1'
    Output: 0.1, 0.5, ..., 3.0
    """
    m = re.search(r"_t(\d+p\d+)$", col)
    if not m:
        raise ValueError(f"Cannot parse time from column: {col}")
    token = m.group(1)  # e.g., '0p1'
    return float(token.replace("p", "."))


def time_to_unit_interval(t: float, t_min: float = 0.1, t_max: float = 3.0) -> float:
    if t_max <= t_min:
        raise ValueError("Invalid time normalization bounds.")
    return (t - t_min) / (t_max - t_min)


def load_node_coordinates_tsv(coords_path: Path) -> pd.DataFrame:
    """
    Robustly parse your nnode_coordinates.txt TSV.

    Observed line pattern snippet:
      'Mesh Node 4252\t39\t58.666\t21.479\t4252\tSYS\\Solid5'

    We will:
    - Read as TSV with no header.
    - Keep rows where col0 contains 'Mesh Node'
    - Interpret:
        x = col1, y = col2, z = col3, node_id = col4
      If node_id missing/unparseable, fallback to extracting digits from col0.
    """
    if not coords_path.is_file():
        raise FileNotFoundError(f"Coordinate TSV not found: {coords_path}")

    df = pd.read_csv(coords_path, sep="\t", header=None, engine="python", dtype=str, on_bad_lines="skip")
    if df.shape[1] < 4:
        raise RuntimeError(f"Coordinate file seems to have too few columns: {coords_path}")

    # Filter likely rows
    col0 = df.iloc[:, 0].astype(str)
    mask = col0.str.contains(r"Mesh\s+Node", case=False, regex=True)
    df = df[mask].copy()
    if df.empty:
        raise RuntimeError("No 'Mesh Node' rows found in coordinate file. Check format.")

    # Try the common format: [0]=label, [1]=x, [2]=y, [3]=z, [4]=node_id
    def to_float_safe(s: str) -> float:
        try:
            return float(s)
        except Exception:
            return float("nan")

    # Extract node_id
    node_id = None
    if df.shape[1] >= 5:
        node_id = pd.to_numeric(df.iloc[:, 4], errors="coerce")
    if node_id is None or node_id.isna().all():
        # fallback: extract digits from label
        extracted = df.iloc[:, 0].astype(str).str.extract(r"(\d+)", expand=False)
        node_id = pd.to_numeric(extracted, errors="coerce")

    x = df.iloc[:, 1].map(to_float_safe)
    y = df.iloc[:, 2].map(to_float_safe)
    z = df.iloc[:, 3].map(to_float_safe)

    out = pd.DataFrame({"node_id": node_id, "x": x, "y": y, "z": z}).dropna()
    out["node_id"] = out["node_id"].astype(int)

    # Deduplicate: keep first occurrence
    out = out.drop_duplicates(subset=["node_id"], keep="first").reset_index(drop=True)

    if out.empty:
        raise RuntimeError("Parsed coordinate table is empty after cleaning.")

    return out


def minmax_normalize_coords(coords_df: pd.DataFrame) -> Tuple[Dict[int, Tuple[float, float, float]], Dict[str, float]]:
    """
    Returns:
      coord_map[node_id] = (x_n, y_n, z_n) in [0,1]
      stats dict for debugging
    """
    for c in ["x", "y", "z"]:
        if c not in coords_df.columns:
            raise RuntimeError(f"Missing coordinate column: {c}")

    x_min, x_max = float(coords_df["x"].min()), float(coords_df["x"].max())
    y_min, y_max = float(coords_df["y"].min()), float(coords_df["y"].max())
    z_min, z_max = float(coords_df["z"].min()), float(coords_df["z"].max())

    def norm(v: float, vmin: float, vmax: float) -> float:
        if vmax <= vmin:
            return 0.0
        return (v - vmin) / (vmax - vmin)

    coord_map: Dict[int, Tuple[float, float, float]] = {}
    for row in coords_df.itertuples(index=False):
        nid = int(getattr(row, "node_id"))
        xn = norm(float(getattr(row, "x")), x_min, x_max)
        yn = norm(float(getattr(row, "y")), y_min, y_max)
        zn = norm(float(getattr(row, "z")), z_min, z_max)
        coord_map[nid] = (xn, yn, zn)

    stats = {
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "z_min": z_min, "z_max": z_max,
        "n_nodes": int(coords_df.shape[0]),
    }
    return coord_map, stats


def get_regime_key_from_path(residual_csv: Path, outdir: Path) -> str:
    """
    residual_csv:
      <OUTDIR>/residuals_stress/<regime_key>/<case_name>/stress_residual.csv
    """
    rel = residual_csv.relative_to(outdir)
    # rel parts: ['residuals_stress', regime_key, case_name, 'stress_residual.csv']
    if len(rel.parts) < 4 or rel.parts[0] != RESIDUALS_DIRNAME:
        raise ValueError(f"Unexpected residual path layout: {residual_csv}")
    return rel.parts[1]


def case_name_from_path(residual_csv: Path, outdir: Path) -> str:
    rel = residual_csv.relative_to(outdir)
    return rel.parts[2]


def is_perfect_case(case_name: str) -> bool:
    return case_name.strip().lower().replace("-", "_") in {"perfect_structure", "perfect_structure ", "perfect_structure"}


def detect_eq_residual_cols_in_file(residual_csv: Path) -> List[str]:
    hdr = pd.read_csv(residual_csv, nrows=0)
    cols = [c for c in hdr.columns if c in EQ_RESID_COLS]
    # enforce exact 10 if present (but tolerate missing by warning later)
    return cols


def extract_topk_events_for_sim(
    residual_csv: Path,
    regime_key: str,
    scaler: Dict,
    coord_map: Dict[int, Tuple[float, float, float]],
    tau_rel: float,
    tau_abs: float,
    p: float,
    kmin: int,
    kmax: int,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Returns:
      events_df with columns: rank,node_number,x_n,y_n,z_n,t_n,score
      meta dict (counts, k, etc.)
    """
    per_regime = scaler.get("per_regime", {})
    if regime_key not in per_regime:
        raise KeyError(f"regime_key '{regime_key}' not found in scaler JSON.")

    col_stats = per_regime[regime_key]["columns"]
    eps = float(scaler.get("eps", 1e-12))

    eq_cols_in_file = detect_eq_residual_cols_in_file(residual_csv)
    if not eq_cols_in_file:
        return pd.DataFrame(), {"n_candidates": 0, "k": 0, "reason": "no_equivalentstress_columns"}

    # Build median/scale arrays aligned to file column order
    med = np.array([float(col_stats[c]["median"]) for c in eq_cols_in_file], dtype=np.float64)
    scl = np.array([float(col_stats[c]["scale_used"]) for c in eq_cols_in_file], dtype=np.float64) + eps

    usecols = [NODE_COL] + eq_cols_in_file
    df = pd.read_csv(residual_csv, usecols=usecols)

    if NODE_COL not in df.columns:
        raise RuntimeError(f"Missing '{NODE_COL}' in residual CSV: {residual_csv}")

    nodes = df[NODE_COL].to_numpy(dtype=np.int64, copy=False)
    X = df[eq_cols_in_file].to_numpy(dtype=np.float64, copy=False)

    # scores
    abs_strength = np.log1p(np.abs(X))
    rel_strength = np.abs((X - med) / scl)

    cand = (rel_strength >= float(tau_rel)) & (abs_strength > float(tau_abs))
    n_candidates = int(cand.sum())
    if n_candidates <= 0:
        return pd.DataFrame(), {"n_candidates": 0, "k": 0, "reason": "no_candidates"}

    # determine k
    k = clamp_int(int(round(float(p) * n_candidates)), int(kmin), int(kmax))
    k = min(k, n_candidates)

    # Flatten candidate cells
    cand_idx = np.argwhere(cand)  # rows: [i_node, j_time]
    # Gather strengths for ranking
    rel_vals = rel_strength[cand_idx[:, 0], cand_idx[:, 1]]
    abs_vals = abs_strength[cand_idx[:, 0], cand_idx[:, 1]]

    # Select top-k by rel_vals (primary)
    if k < n_candidates:
        topk_rel_cut = np.argpartition(rel_vals, -k)[-k:]
        cand_idx = cand_idx[topk_rel_cut]
        rel_vals = rel_vals[topk_rel_cut]
        abs_vals = abs_vals[topk_rel_cut]

    # Now sort selected by rel desc, abs desc
    order = np.lexsort((-abs_vals, -rel_vals))
    cand_idx = cand_idx[order]
    rel_vals = rel_vals[order]

    # Time normalization per column
    times = np.array([parse_time_from_col(c.replace(RES_PREFIX, "")) for c in eq_cols_in_file], dtype=np.float64)
    t_norm = np.array([time_to_unit_interval(t) for t in times], dtype=np.float64)

    # Build events
    out_rows = []
    missing_coords = 0
    for rank0, (i_node, j_time) in enumerate(cand_idx, start=1):
        node_id = int(nodes[i_node])
        coord = coord_map.get(node_id)
        if coord is None:
            missing_coords += 1
            continue
        xn, yn, zn = coord
        tn = float(t_norm[j_time])
        score = float(rel_vals[rank0 - 1])  # rel_strength as the single score
        out_rows.append(
            {
                "rank": int(rank0),
                "node_number": int(node_id),
                "x_n": float(xn),
                "y_n": float(yn),
                "z_n": float(zn),
                "t_n": float(tn),
                "score": float(score),
            }
        )

    events_df = pd.DataFrame(out_rows)

    meta = {
        "n_candidates": int(n_candidates),
        "k": int(k),
        "n_events_written": int(events_df.shape[0]),
        "n_missing_coord_nodes_skipped": int(missing_coords),
        "tau_rel": float(tau_rel),
        "tau_abs": float(tau_abs),
        "p": float(p),
        "kmin": int(kmin),
        "kmax": int(kmax),
    }
    return events_df, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, required=True, help="Directory containing residuals_stress/")
    ap.add_argument("--scaler", type=str, required=True, help="Path to stress_score_scaler*.json (use your capped one)")
    ap.add_argument("--coords", type=str, required=True, help="Path to nnode_coordinates.txt (TSV)")
    ap.add_argument("--tau_rel", type=float, default=3.0, help="Relative threshold on scaled residual")
    ap.add_argument("--tau_abs", type=float, default=0.0, help="Absolute threshold on log1p(|Δ|)")
    ap.add_argument("--p", type=float, default=0.03, help="k = clamp(round(p * n_candidates), kmin, kmax)")
    ap.add_argument("--kmin", type=int, default=1000, help="Minimum k per simulation")
    ap.add_argument("--kmax", type=int, default=10000, help="Maximum k per simulation")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    residual_root = outdir / RESIDUALS_DIRNAME
    if not residual_root.is_dir():
        raise FileNotFoundError(f"Expected residual directory not found: {residual_root}")

    scaler = load_json(Path(args.scaler).resolve())

    coords_df = load_node_coordinates_tsv(Path(args.coords).resolve())
    coord_map, coord_stats = minmax_normalize_coords(coords_df)

    events_root = outdir / "events_equivstress"
    events_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    n_processed = 0
    n_skipped = 0

    # Walk all residual CSVs
    for dirpath, _, filenames in os.walk(residual_root):
        if RESIDUAL_FILENAME not in filenames:
            continue

        residual_csv = Path(dirpath) / RESIDUAL_FILENAME

        # Rename-safe: skip if missing
        if not residual_csv.is_file():
            n_skipped += 1
            continue

        try:
            regime_key = get_regime_key_from_path(residual_csv, outdir)
            case_name = case_name_from_path(residual_csv, outdir)
        except Exception:
            n_skipped += 1
            continue

        # Skip IGNORE folders
        if case_name.strip().startswith("[IGNORE]"):
            n_skipped += 1
            continue

        # Skip perfect
        if is_perfect_case(case_name):
            n_skipped += 1
            continue

        # Extract
        try:
            events_df, meta = extract_topk_events_for_sim(
                residual_csv=residual_csv,
                regime_key=regime_key,
                scaler=scaler,
                coord_map=coord_map,
                tau_rel=float(args.tau_rel),
                tau_abs=float(args.tau_abs),
                p=float(args.p),
                kmin=int(args.kmin),
                kmax=int(args.kmax),
            )

            # Output path mirrors residual layout
            out_case_dir = events_root / regime_key / case_name
            out_case_dir.mkdir(parents=True, exist_ok=True)
            out_events_csv = out_case_dir / "events_topk.csv"

            if not events_df.empty:
                events_df.to_csv(out_events_csv, index=False)

            summary_rows.append(
                {
                    "regime_key": regime_key,
                    "case_name": case_name,
                    "residual_csv_path": str(residual_csv),
                    "events_csv_path": str(out_events_csv) if not events_df.empty else "",
                    **meta,
                    "status": "OK" if meta.get("n_candidates", 0) > 0 else "NO_CANDIDATES",
                }
            )
            n_processed += 1

        except Exception as e:
            summary_rows.append(
                {
                    "regime_key": regime_key,
                    "case_name": case_name,
                    "residual_csv_path": str(residual_csv),
                    "events_csv_path": "",
                    "status": "ERROR",
                    "error": str(e),
                }
            )
            n_processed += 1

    summary_df = pd.DataFrame(summary_rows)
    summary_path = outdir / "events_equivstress_index.csv"
    summary_df.to_csv(summary_path, index=False)

    # Console summary
    print(f"OUTDIR:          {outdir}")
    print(f"Scaler:          {Path(args.scaler).resolve()}")
    print(f"Coords:          {Path(args.coords).resolve()}")
    print(f"Coord stats:     {coord_stats}")
    print(f"Events root:     {events_root}")
    print(f"Processed sims:  {n_processed}")
    print(f"Skipped sims:    {n_skipped}")
    print(f"Wrote summary:   {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
