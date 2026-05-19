#!/usr/bin/env python3
"""
Build residuals_stress_index.csv from a residuals_stress directory.

Assumptions:
- residuals_stress/
    ├── G1_S_H/
    │     ├── case_x/
    │     │     └── residuals.csv
    │     ├── case_y/
    │     │     └── residuals.csv
    ├── G2_W_L/
    │     └── ...

Rules:
- Ignore paths containing '[IGNORE]'
- Ignore perfect cases
- Ignore missing / unreadable CSVs
- One row per residual CSV
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def is_ignored(path: Path) -> bool:
    return "[IGNORE]" in str(path)


def is_perfect(path: Path) -> bool:
    s = str(path).lower()
    return "perfect" in s


def discover_residual_csvs(root: Path) -> list[Path]:
    return list(root.rglob("*.csv"))


def infer_regime_key(csv_path: Path, root: Path) -> str:
    """
    Assumes structure:
      residuals_stress/<REGIME_KEY>/.../residual.csv
    """
    rel = csv_path.relative_to(root)
    if len(rel.parts) < 2:
        raise RuntimeError(f"Cannot infer regime_key from path: {csv_path}")
    return rel.parts[0]


def infer_case_name(csv_path: Path) -> str:
    """
    Uses immediate parent folder as case name.
    """
    return csv_path.parent.name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        required=True,
        type=str,
        help="Path to residuals_stress directory"
    )
    ap.add_argument(
        "--out",
        default=None,
        type=str,
        help="Output CSV path (default: <root>/residuals_stress_index.csv)"
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print skipped / included files"
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise RuntimeError(f"Not a directory: {root}")

    out_csv = (
        Path(args.out).resolve()
        if args.out
        else root / "residuals_stress_index.csv"
    )

    rows = []
    all_csvs = discover_residual_csvs(root)

    if args.verbose:
        print(f"[DISCOVER] Found {len(all_csvs)} CSV files")

    for csv_path in all_csvs:
        try:
            if is_ignored(csv_path):
                if args.verbose:
                    print(f"[SKIP][IGNORE] {csv_path}")
                continue

            if is_perfect(csv_path):
                if args.verbose:
                    print(f"[SKIP][PERFECT] {csv_path}")
                continue

            regime_key = infer_regime_key(csv_path, root)
            case_name = infer_case_name(csv_path)

            rows.append({
                "regime_key": regime_key,
                "case_type": case_name,
                "residual_csv_path": str(csv_path),
            })

            if args.verbose:
                print(f"[ADD] {regime_key} | {case_name}")

        except Exception as e:
            if args.verbose:
                print(f"[SKIP][ERROR] {csv_path} -> {e}")
            continue

    if not rows:
        raise RuntimeError("No valid residual CSVs found.")

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["regime_key", "case_type", "residual_csv_path"]
    ).reset_index(drop=True)

    df.to_csv(out_csv, index=False)

    print(f"\nWrote residual index:")
    print(f"  {out_csv}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
