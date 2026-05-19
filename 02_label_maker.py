#!/usr/bin/env python3
"""
Build a labelled dataset manifest from events_equivstress_index.csv.

Assumptions (per your instructions):
- events_equivstress_index.csv is authoritative.
- Column 'case_name' uniquely identifies the imperfection type (6 unique values total).
- Labels should be integers 0..5.
- No renaming/moving of event files.
- Splitting is done later (this script does NOT assign train/val/test).

Outputs (in the same directory as the index by default):
- dataset_events_manifest.csv
- label_map_events.json

Run:
  python build_dataset_manifest.py --index "events_equivstress_index.csv"

Optional:
  python build_dataset_manifest.py --index "events_equivstress_index.csv" --outdir "C:/path/to/outdir"
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

    # Basic integrity checks
    if df.empty:
        raise RuntimeError("No usable rows found (status==OK and events_csv_path exists).")

    # There should be exactly 6 unique imperfection names (case_name)
    # We treat case_name as the class name exactly (no parsing).
    case_names = sorted(df["case_name"].astype(str).str.strip().unique().tolist())
    n_classes = len(case_names)
    if n_classes != 6:
        raise RuntimeError(
            f"Expected exactly 6 unique case_name values (imperfection classes), but found {n_classes}:\n"
            f"{case_names}"
        )

    # Deterministic label assignment: alphabetical order of case_name
    label_map: Dict[str, int] = {name: i for i, name in enumerate(case_names)}

    df["label_name"] = df["case_name"].astype(str).str.strip()
    df["label_id"] = df["label_name"].map(label_map).astype(int)

    # Validate expected sample counts:
    # - total 96 rows (6 classes × 16 regimes)
    # - 16 per class
    # (If your dataset differs, you can relax these checks.)
    total = int(df.shape[0])
    counts = df.groupby("label_id").size().to_dict()

    # Strong checks (you told me each class has 16 samples across 16 regimes)
    expected_per_class = 16
    bad = {k: v for k, v in counts.items() if v != expected_per_class}
    if bad:
        msg = "Class counts are not 16 each. Found:\n"
        for lid in sorted(counts.keys()):
            msg += f"  label_id={lid} ({case_names[lid]}): {counts[lid]}\n"
        raise RuntimeError(msg)

    expected_total = expected_per_class * 6
    if total != expected_total:
        raise RuntimeError(f"Expected total {expected_total} samples, but found {total}.")

    # Create final manifest with stable, training-friendly fields
    # Keep key provenance fields for traceability.
    manifest = df.copy()

    # Assign deterministic sample_id (stable ordering)
    manifest = manifest.sort_values(["label_id", "regime_key", "case_name"]).reset_index(drop=True)
    manifest.insert(0, "sample_id", range(1, len(manifest) + 1))

    # Keep/select columns (include useful diagnostics)
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

    # Write outputs
    manifest_path = outdir / "dataset_events_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    label_map_path = outdir / "label_map_events.json"
    label_map_payload = {
        "n_classes": 6,
        "labeling_rule": "label_id assigned by alphabetical order of unique case_name values from events_equivstress_index.csv",
        "label_map": label_map,  # case_name -> label_id
    }
    label_map_path.write_text(json.dumps(label_map_payload, indent=2), encoding="utf-8")

    # Console summary
    print(f"Index:           {index_path}")
    print(f"Output dir:      {outdir}")
    print(f"Wrote manifest:  {manifest_path}")
    print(f"Wrote label map: {label_map_path}")
    print("Label distribution:")
    for name in case_names:
        print(f"  {label_map[name]} -> {name}  (n={expected_per_class})")
    print("Done.")


if __name__ == "__main__":
    main()
