#!/usr/bin/env python3
"""
Fit a global score normalizer (TRAIN only) and print diagnostics for train/val/test.

This script implements steps 1–3 in one run:
1) Read a split manifest (e.g., dataset_events_manifest_split_seed42.csv).
2) Fit score normalizer on TRAIN only:
      s_raw = log1p(score)
      s_norm = (s_raw - mu) / sigma
3) Print per-split diagnostics and write a score_normalizer JSON artifact.

Assumptions:
- Split manifest has columns:
    - split  (train/val/test)
    - events_csv_path
    - label_id, label_name (optional but useful)
- Each events CSV has a column:
    - score  (your rel_strength)
- Coordinates/time are already normalized in the events CSVs (not used here).

Outputs (in outdir; default = manifest directory):
- score_normalizer_seed<SEED>.json   (or score_normalizer.json if no seed inferred)
- score_normalizer_diagnostics_seed<SEED>.csv

Usage:
  python 04_fit_score_normalizer.py --manifest "dataset_events_manifest_split_seed42.csv"

Optional:
  python 04_fit_score_normalizer.py --manifest "dataset_events_manifest_split_seed42.csv" --outdir "." --max_events_per_file 0
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


REQUIRED_MANIFEST_COLS_BASE = ["events_csv_path"]
SCORE_COL = "score"


def infer_seed_from_filename(path: Path) -> int | None:
    m = re.search(r"seed(\d+)", path.name)
    return int(m.group(1)) if m else None


def require_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}\nFound: {list(df.columns)}")


def safe_read_scores(events_csv_path: Path, max_events: int = 0) -> np.ndarray:
    """
    Read score column as float64. Optionally subsample rows to cap memory:
      - max_events == 0 => read all rows
      - max_events > 0  => take a deterministic head(max_events)
    """
    if not events_csv_path.is_file():
        raise FileNotFoundError(f"events file not found: {events_csv_path}")

    # Read only the score column for efficiency
    df = pd.read_csv(events_csv_path, usecols=[SCORE_COL])
    if df.empty:
        return np.empty((0,), dtype=np.float64)

    if max_events and max_events > 0 and df.shape[0] > max_events:
        df = df.iloc[:max_events].copy()

    scores = df[SCORE_COL].to_numpy(dtype=np.float64, copy=False)
    # Filter non-finite / negatives (should not happen, but guard)
    scores = scores[np.isfinite(scores)]
    scores = scores[scores >= 0.0]
    return scores


def log1p_transform(x: np.ndarray) -> np.ndarray:
    # x is >=0; safe
    return np.log1p(x)


def fit_zscore(x: np.ndarray) -> Tuple[float, float]:
    if x.size == 0:
        raise RuntimeError("Cannot fit normalizer: no training events found.")
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=0))
    # Guard against degenerate variance
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise RuntimeError(f"Degenerate sigma={sigma}. Check score distribution or input data.")
    return mu, sigma


def summarize(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {
            "n_events": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p01": float("nan"),
            "p50": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    p01, p50, p99 = np.percentile(arr, [1, 50, 99])
    return {
        "n_events": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "p01": float(p01),
        "p50": float(p50),
        "p99": float(p99),
        "max": float(np.max(arr)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True, help="Split manifest CSV (with split column).")
    ap.add_argument("--outdir", type=str, default=None, help="Output directory (default: manifest directory).")
    ap.add_argument(
        "--max_events_per_file",
        type=int,
        default=0,
        help="If >0, cap events read per events_topk.csv (deterministic head). Use 0 to read all.",
    )
    ap.add_argument(
        "--fit_on",
        type=str,
        default="train",
        choices=["train", "all"],
        help="Fit mu/sigma on 'train' split (default) or on all rows ('all').",
    )

    args = ap.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    outdir = Path(args.outdir).resolve() if args.outdir else manifest_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    seed = infer_seed_from_filename(manifest_path)

    df = pd.read_csv(manifest_path)
    require_columns(df, REQUIRED_MANIFEST_COLS_BASE, "manifest")

    has_split = "split" in df.columns
    if args.fit_on == "train":
        if not has_split:
            raise RuntimeError("fit_on=train requires a 'split' column in the manifest.")
        df["split"] = df["split"].astype(str).str.strip().str.lower()
        valid_splits = {"train", "val", "test"}
        bad = sorted(set(df["split"].unique().tolist()) - valid_splits)
        if bad:
            raise RuntimeError(f"Unexpected split values found: {bad}. Expected only {sorted(valid_splits)}.")


    # Convert events paths to Path
    df["events_csv_path"] = df["events_csv_path"].astype(str)
    df["events_csv_path_path"] = df["events_csv_path"].apply(lambda p: Path(p))

    # --- Step 2: Fit normalizer on TRAIN only ---
    if args.fit_on == "train":
        fit_rows = df[df["split"] == "train"].copy()
        fit_split_name = "train"
    else:
        fit_rows = df.copy()
        fit_split_name = "all"

    if fit_rows.empty:
        raise RuntimeError(f"No rows found for fit_on={args.fit_on}.")
    

    train_raw_list: List[np.ndarray] = []
    n_fit_files = 0
    for p in fit_rows["events_csv_path_path"].tolist():
        scores = safe_read_scores(p, max_events=int(args.max_events_per_file))
        if scores.size == 0:
            continue
        train_raw_list.append(scores)
        n_fit_files += 1

    if not train_raw_list:
        raise RuntimeError("No usable training scores found (empty files or missing score column).")

    train_scores = np.concatenate(train_raw_list)
    train_sraw = log1p_transform(train_scores)
    mu, sigma = fit_zscore(train_sraw)

    # --- Step 3: Diagnostics for train/val/test ---
    diag_rows = []
    if has_split:
        splits_to_report = ["train", "val", "test"]
    else:
        splits_to_report = ["all"]
        df = df.copy()
        df["split"] = "all"

    for split in splits_to_report:
        sub = df[df["split"] == split]
        all_scores_list: List[np.ndarray] = []

        n_files = 0

        for p in sub["events_csv_path_path"].tolist():
            scores = safe_read_scores(p, max_events=int(args.max_events_per_file))
            if scores.size == 0:
                continue
            all_scores_list.append(scores)
            n_files += 1

        if all_scores_list:
            scores_all = np.concatenate(all_scores_list)
            sraw = log1p_transform(scores_all)
            snorm = (sraw - mu) / sigma
        else:
            snorm = np.empty((0,), dtype=np.float64)

        stats = summarize(snorm)
        stats.update(
            {
                "split": split,
                "n_files": int(n_files),
            }
        )
        diag_rows.append(stats)

    diag_df = pd.DataFrame(diag_rows)

    # Write artifacts
    suffix = f"_seed{seed}" if seed is not None else ""
    tag = "full" if args.fit_on == "all" else "train"

    normalizer_name = f"score_normalizer_{tag}{suffix}.json"
    diag_name = f"score_normalizer_diagnostics_{tag}{suffix}.csv"
    

    payload = {
        "transform": "zscore_log1p",
        "score_column": SCORE_COL,
        "fitted_on": args.fit_on,          # <-- key change
        "fitted_on_split": fit_split_name, # "train" or "all"
        "mu": mu,
        "sigma": sigma,
        "seed": seed,
        "max_events_per_file": int(args.max_events_per_file),
        "manifest_path": str(manifest_path),
        "n_fit_rows": int(len(fit_rows)),
        "notes": "Compute s_raw = log1p(score), then s_norm = (s_raw - mu)/sigma using fitted mu/sigma.",
    }
    normalizer_path = outdir / normalizer_name
    diag_path = outdir / diag_name
    diag_df.to_csv(diag_path, index=False)
    normalizer_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Console output
    print(f"Manifest:     {manifest_path}")
    print(f"Outdir:       {outdir}")
    print(f"Seed:         {seed}")
    print(f"Fitted mu:    {mu:.6f}")
    print(f"Fitted sigma: {sigma:.6f}")
    print(f"Wrote:        {normalizer_path}")
    print(f"Wrote:        {diag_path}\n")
    print("Diagnostics (s_norm):")
    print(diag_df.to_string(index=False))


if __name__ == "__main__":
    main()
