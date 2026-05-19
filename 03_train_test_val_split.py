#!/usr/bin/env python3
"""
Regime-aware stratified splitting for the labelled event dataset.

Goal:
- Keep exact per-class counts (default: train=12, val=2, test=2 per label_id)
- But choose WHICH samples go to splits such that splits are balanced across regime attributes:
    - group_id  (1..4)
    - season    (S/W)
    - train_size (H/L)

Approach:
- For each seed:
  - Generate N candidate splits (default 1000)
  - Each candidate:
      for each label_id, randomly pick 2 val and 2 test (rest train)
  - Score each candidate based on:
      (A) global balance per split across attributes (chi-square-like deviation from expected)
      (B) per-label train coverage penalties (missing any group/season/train_size in train)
  - Pick best (lowest score)
  - Write:
      dataset_events_manifest_split_seed<seed>.csv
      split_balance_report_seed<seed>.csv
      split_balance_best_seed<seed>.json

Usage:
  python 03_train_test_val_split_balanced.py --manifest dataset_events_manifest.csv --seeds 30 42
Optional:
  --candidates 1000
  --train 12 --val 2 --test 2
  --check_paths
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLS = ["sample_id", "label_id", "events_csv_path", "regime_key"]


def _require_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} is missing required columns: {missing}\nFound: {list(df.columns)}")


def _validate_events_paths(df: pd.DataFrame) -> None:
    missing = [p for p in df["events_csv_path"].astype(str).tolist() if not Path(p).is_file()]
    if missing:
        preview = "\n".join(missing[:10])
        raise RuntimeError(
            f"Some events_csv_path files do not exist on disk (showing up to 10):\n{preview}\n"
            f"Total missing: {len(missing)}"
        )


@dataclass(frozen=True)
class RegimeParts:
    group_id: int
    season: str      # S or W
    train_size: str  # H or L


def parse_regime_key(regime_key: str) -> RegimeParts:
    """
    Accepts strings like:
      "G4_W_H" or "G1_S_L"
    """
    rk = str(regime_key).strip()
    parts = rk.split("_")
    if len(parts) != 3:
        raise ValueError(f"Invalid regime_key '{rk}' (expected format 'G<id>_<S/W>_<H/L>')")
    g, season, ts = parts
    if not g.upper().startswith("G"):
        raise ValueError(f"Invalid regime_key '{rk}' (group must start with 'G')")
    gid = int(g[1:])
    season = season.upper()
    ts = ts.upper()
    if season not in {"S", "W"}:
        raise ValueError(f"Invalid season '{season}' in regime_key '{rk}' (expected S/W)")
    if ts not in {"H", "L"}:
        raise ValueError(f"Invalid train_size '{ts}' in regime_key '{rk}' (expected H/L)")
    return RegimeParts(group_id=gid, season=season, train_size=ts)


def add_regime_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parsed = out["regime_key"].apply(parse_regime_key)
    out["group_id"] = parsed.apply(lambda r: r.group_id).astype(int)
    out["season"] = parsed.apply(lambda r: r.season).astype(str)
    out["train_size"] = parsed.apply(lambda r: r.train_size).astype(str)
    return out


def _expected_distribution(values: List) -> Dict:
    """Uniform expected distribution over observed categories."""
    cats = sorted(set(values))
    p = 1.0 / max(len(cats), 1)
    return {c: p for c in cats}


def _chi_like_score(counts: Dict, expected_probs: Dict, total: int) -> float:
    """
    Chi-square-like deviation:
      sum_c ( (obs - exp)^2 / (exp + eps) )
    where exp = total * expected_probs[c]
    """
    eps = 1e-9
    s = 0.0
    for c, prob in expected_probs.items():
        exp = total * prob
        obs = float(counts.get(c, 0))
        s += (obs - exp) ** 2 / (exp + eps)
    return s


def _counts(series: pd.Series) -> Dict:
    return series.value_counts().to_dict()


def score_candidate(
    df: pd.DataFrame,
    split_assignments: pd.Series,
    n_train: int,
    n_val: int,
    n_test: int,
    w_global: float = 1.0,
    w_coverage: float = 2.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Scores a candidate split using:
      A) Global balance per split for group_id/season/train_size
      B) Coverage penalties per label for TRAIN split (missing categories)

    Returns total_score and a breakdown.
    """
    tmp = df.copy()
    tmp["split"] = split_assignments.values

    breakdown: Dict[str, float] = {}

    # ---- A) Global balance
    total_global = 0.0
    for split in ["train", "val", "test"]:
        sub = tmp[tmp["split"] == split]
        if sub.empty:
            # should never happen
            total_global += 1e9
            continue

        # expected distributions based on ALL data categories (fixed)
        exp_gid = _expected_distribution(tmp["group_id"].tolist())
        exp_sea = _expected_distribution(tmp["season"].tolist())
        exp_ts = _expected_distribution(tmp["train_size"].tolist())

        total = int(sub.shape[0])
        total_global += _chi_like_score(_counts(sub["group_id"]), exp_gid, total)
        total_global += _chi_like_score(_counts(sub["season"]), exp_sea, total)
        total_global += _chi_like_score(_counts(sub["train_size"]), exp_ts, total)

    breakdown["global_balance"] = total_global

    # ---- B) Per-label TRAIN coverage penalties
    # For each label, check that train contains:
    #   - at least one of each season (S,W)
    #   - at least one of each train_size (H,L)
    #   - and ideally coverage across group_id 1..4 (penalize missing groups)
    cov_pen = 0.0
    labels = sorted(tmp["label_id"].unique().tolist())
    all_groups = sorted(tmp["group_id"].unique().tolist())
    for lid in labels:
        sub = tmp[(tmp["label_id"] == lid) & (tmp["split"] == "train")]
        if sub.shape[0] != n_train:
            cov_pen += 50.0  # should never happen if candidate generation correct
            continue

        seasons_present = set(sub["season"].tolist())
        ts_present = set(sub["train_size"].tolist())
        groups_present = set(sub["group_id"].tolist())

        # season coverage (expect both)
        if "S" not in seasons_present:
            cov_pen += 5.0
        if "W" not in seasons_present:
            cov_pen += 5.0

        # train_size coverage (expect both)
        if "H" not in ts_present:
            cov_pen += 5.0
        if "L" not in ts_present:
            cov_pen += 5.0

        # group coverage (expect as many as possible)
        # missing any group adds 1 penalty (train has 12, so should often include all 4)
        missing_groups = [g for g in all_groups if g not in groups_present]
        cov_pen += 1.0 * len(missing_groups)

    breakdown["train_coverage_penalty"] = cov_pen

    total_score = w_global * total_global + w_coverage * cov_pen
    breakdown["total_score"] = total_score
    return total_score, breakdown


def make_candidate_split(
    df: pd.DataFrame,
    seed: int,
    attempt: int,
    n_train: int,
    n_val: int,
    n_test: int,
) -> pd.Series:
    """
    Create one candidate split:
      For each label_id:
        - deterministically shuffle indices with RNG(seed, attempt, label_id)
        - take first n_val as val, next n_test as test, rest train
    """
    out_split = pd.Series([""] * len(df), index=df.index, dtype=str)

    labels = sorted(df["label_id"].unique().tolist())
    expected = n_train + n_val + n_test

    for lid in labels:
        sub = df[df["label_id"] == lid]
        n = int(sub.shape[0])
        if n != expected:
            raise RuntimeError(
                f"Label {lid} has {n} rows, but requested split sizes sum to {expected} "
                f"(train={n_train}, val={n_val}, test={n_test})."
            )

        # RNG: depends on seed, attempt, and label to keep determinism
        rng = np.random.default_rng(seed + 10_000 * int(lid) + 1_000_000 * int(attempt))
        idx = sub.index.to_numpy()
        rng.shuffle(idx)

        val_idx = idx[:n_val]
        test_idx = idx[n_val:n_val + n_test]
        train_idx = idx[n_val + n_test:]

        out_split.loc[train_idx] = "train"
        out_split.loc[val_idx] = "val"
        out_split.loc[test_idx] = "test"

    if (out_split == "").any():
        raise RuntimeError("Unassigned rows exist; candidate generation failed.")
    return out_split


def summarize_balance(df_split: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a compact report table of counts for group_id/season/train_size by split.
    """
    rows = []

    for split in ["train", "val", "test"]:
        sub = df_split[df_split["split"] == split]
        rows.append({"split": split, "attribute": "group_id", **sub["group_id"].value_counts().to_dict()})
        rows.append({"split": split, "attribute": "season", **sub["season"].value_counts().to_dict()})
        rows.append({"split": split, "attribute": "train_size", **sub["train_size"].value_counts().to_dict()})

    rep = pd.DataFrame(rows).fillna(0).sort_values(["split", "attribute"]).reset_index(drop=True)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True, help="Path to dataset_events_manifest.csv")
    ap.add_argument("--outdir", type=str, default=None, help="Output directory (default: manifest directory)")
    ap.add_argument("--seeds", type=int, nargs="+", required=True, help="One or more integer seeds")
    ap.add_argument("--train", type=int, default=12, help="Train samples per class")
    ap.add_argument("--val", type=int, default=2, help="Val samples per class")
    ap.add_argument("--test", type=int, default=2, help="Test samples per class")
    ap.add_argument("--candidates", type=int, default=1000, help="Number of candidate splits per seed")
    ap.add_argument("--check_paths", action="store_true", help="Verify events_csv_path exists on disk (slower)")

    # scoring weights (advanced)
    ap.add_argument("--w_global", type=float, default=1.0, help="Weight for global balance score")
    ap.add_argument("--w_coverage", type=float, default=2.0, help="Weight for per-label train coverage penalty")

    args = ap.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    outdir = Path(args.outdir).resolve() if args.outdir else manifest_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    _require_columns(df, REQUIRED_COLS, "dataset_events_manifest.csv")

    # normalize types
    df["label_id"] = df["label_id"].astype(int)

    # add regime columns (group_id/season/train_size)
    df = add_regime_columns(df)

    # optional integrity check
    if args.check_paths:
        _validate_events_paths(df)

    # for stable sort later (only if these exist)
    sort_cols = [c for c in ["label_id", "split", "group_id", "season", "train_size", "regime_key", "case_name", "sample_id"] if c in df.columns]

    # Generate best split per seed
    for seed in args.seeds:
        best_score = float("inf")
        best_split = None
        best_breakdown = None

        for attempt in range(1, int(args.candidates) + 1):
            split_assign = make_candidate_split(
                df=df,
                seed=int(seed),
                attempt=int(attempt),
                n_train=int(args.train),
                n_val=int(args.val),
                n_test=int(args.test),
            )
            score, breakdown = score_candidate(
                df=df,
                split_assignments=split_assign,
                n_train=int(args.train),
                n_val=int(args.val),
                n_test=int(args.test),
                w_global=float(args.w_global),
                w_coverage=float(args.w_coverage),
            )
            if score < best_score:
                best_score = score
                best_split = split_assign
                best_breakdown = breakdown

        if best_split is None:
            raise RuntimeError("Failed to generate any candidate splits.")

        out = df.copy()
        out["split"] = best_split.values
        out["split_seed"] = int(seed)
        out["split_strategy"] = f"balanced_regime_search_{int(args.candidates)}cands_{args.train}_{args.val}_{args.test}"
        out["split_version"] = "v2_balanced"

        if sort_cols:
            out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

        out_path = outdir / f"dataset_events_manifest_split_seed{int(seed)}.csv"
        out.to_csv(out_path, index=False)

        # Write balance report
        rep = summarize_balance(out)
        rep_path = outdir / f"split_balance_report_seed{int(seed)}.csv"
        rep.to_csv(rep_path, index=False)

        # Write json with score breakdown
        meta = {
            "seed": int(seed),
            "candidates": int(args.candidates),
            "train_per_class": int(args.train),
            "val_per_class": int(args.val),
            "test_per_class": int(args.test),
            "weights": {"w_global": float(args.w_global), "w_coverage": float(args.w_coverage)},
            "best": best_breakdown,
        }
        meta_path = outdir / f"split_balance_best_seed{int(seed)}.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Console summary
        ctab = out.groupby(["label_id", "split"]).size().unstack(fill_value=0)
        print(f"\nWrote: {out_path}")
        print(f"Wrote: {rep_path}")
        print(f"Wrote: {meta_path}")
        print(f"Seed: {seed} | Best total_score={best_score:.4f} | Strategy: {out['split_strategy'].iloc[0]}")
        print(ctab)

    print("\nDone.")


if __name__ == "__main__":
    main()
