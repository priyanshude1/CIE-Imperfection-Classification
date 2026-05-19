#!/usr/bin/env python3
"""
Build a labelled dataset manifest from events_equivstress_index.csv.

Changes vs older version:
- Removes the hard requirement of 16 samples per class.
- Adds optional constraints:
    --min_per_class (default: 1)   -> require at least this many samples in each class
    --allow_missing_classes        -> if set, do NOT require exactly 6 classes (rarely needed)
- Still uses case_name as authoritative class name and assigns labels 0..(n_classes-1)
  deterministically by alphabetical order.

Outputs (in the same directory as the index by default):
- dataset_events_manifest.csv
- label_map_events.json

Run:
  python build_dataset_manifest.py --index "events_equivstress_index.csv"

Optional:
  python build_dataset_manifest.py --index "events_equivstress_index.csv" --min_per_class 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


REQUIRED_COLS = [
    "regime_key",
    "case_name",
    "events_csv_path",
    "n_candidates",
    "k",
    "n_events_written",
    "status",
]


def _require_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} is missing required columns: {missing}\nFound: {list(df.columns)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=str, required=True, help="Path to events_equivstress_index.csv")
    ap.add_argument("--outdir", type=str, default=None, help="Output directory (default: index file directory)")
    ap.add_argument(
        "--min_per_class",
        type=int,
        default=1,
        help="Minimum required samples per class. Default=1 (no strict balancing).",
    )
    ap.add_argument(
        "--allow_missing_classes",
        action="store_true",
        help="If set, do not enforce exactly 6 unique case_name values.",
    )
    args = ap.parse_args()

    index_path = Path(args.index).resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"Index CSV not found: {index_path}")

    outdir = Path(args.outdir).resolve() if args.outdir else index_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(index_path)
    _require_columns(df, REQUIRED_COLS, "events_equivstress_index.csv")

    # Keep only OK rows with a usable events path
    df = df[df["status"].astype(str).str.upper() == "OK"].copy()
    df["events_exists"] = df["events_csv_path"].apply(lambda p: Path(str(p)).is_file())
    df = df[df["events_exists"]].copy()

    if df.empty:
        raise RuntimeError("No usable rows found (status==OK and events_csv_path exists).")

    # Class discovery
    case_names = sorted(df["case_name"].astype(str).str.strip().unique().tolist())
    n_classes = len(case_names)

    if not args.allow_missing_classes and n_classes != 6:
        raise RuntimeError(
            f"Expected exactly 6 unique case_name values (imperfection classes), but found {n_classes}:\n"
            f"{case_names}\n"
            f"If this is intentional (e.g., you removed entire classes), re-run with --allow_missing_classes."
        )

    # Deterministic label assignment
    label_map: Dict[str, int] = {name: i for i, name in enumerate(case_names)}

    df["label_name"] = df["case_name"].astype(str).str.strip()
    df["label_id"] = df["label_name"].map(label_map).astype(int)

    # Relaxed count validation (NEW)
    counts = df.groupby("label_id").size().to_dict()
    min_per_class = int(args.min_per_class)

    # Ensure every discovered class meets minimum count
    bad = {lid: n for lid, n in counts.items() if n < min_per_class}
    if bad:
        msg = f"Some classes have fewer than min_per_class={min_per_class} samples:\n"
        # label_id order is aligned to case_names ordering
        for lid in sorted(bad.keys()):
            msg += f"  label_id={lid} ({case_names[lid]}): n={counts[lid]}\n"
        raise RuntimeError(msg)

    # Create final manifest with stable ordering
    manifest = df.copy()
    manifest = manifest.sort_values(["label_id", "regime_key", "case_name"]).reset_index(drop=True)
    manifest.insert(0, "sample_id", range(1, len(manifest) + 1))

    keep_cols = [
        "sample_id",
        "label_id",
        "label_name",
        "regime_key",
        "case_name",
        "events_csv_path",
        "n_events_written",
        "n_candidates",
        "k",
        # optional provenance:
        "tau_rel",
        "tau_abs",
        "p",
        "kmin",
        "kmax",
        "residual_csv_path",
    ]
    keep_cols = [c for c in keep_cols if c in manifest.columns]
    manifest = manifest[keep_cols].copy()

    manifest_path = outdir / "dataset_events_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    label_map_path = outdir / "label_map_events.json"
    label_map_payload = {
        "n_classes": int(n_classes),
        "labeling_rule": "label_id assigned by alphabetical order of unique case_name values from events_equivstress_index.csv",
        "label_map": label_map,  # case_name -> label_id
        "min_per_class_enforced": int(min_per_class),
        "counts": {case_names[lid]: int(counts.get(lid, 0)) for lid in range(n_classes)},
    }
    label_map_path.write_text(json.dumps(label_map_payload, indent=2), encoding="utf-8")

    # Console summary
    print(f"Index:           {index_path}")
    print(f"Output dir:      {outdir}")
    print(f"Wrote manifest:  {manifest_path}")
    print(f"Wrote label map: {label_map_path}")
    print("Label distribution:")
    for name in case_names:
        lid = label_map[name]
        print(f"  {lid} -> {name}  (n={counts.get(lid, 0)})")
    print("Done.")


if __name__ == "__main__":
    main()
