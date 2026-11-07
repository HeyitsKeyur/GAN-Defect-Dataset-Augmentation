#!/usr/bin/env python3
"""Execute Phase 1: Group-aware, leakage-safe dataset splitting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_paths import default_data_root, resolve_layout
from src.data.splits import (
    DEFAULT_SEED,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    build_manifest,
    save_manifests,
    validate_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate group-aware leakage-safe train/val/test splits."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_data_root(PROJECT_ROOT),
        help="Path to the Ceramic-Defect data directory containing 'original image', 'image', 'lables'.",
    )
    parser.add_argument(
        "--audit-report",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "audit" / "phase0" / "audit_report.json",
        help="Path to Phase 0 audit_report.json (used to load confirmed exact MD5 duplicate pairs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "splits" / "phase1",
        help="Directory to save split manifests and summaries.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help=f"Group-level train ratio (default: {DEFAULT_TRAIN_RATIO}).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help=f"Group-level validation ratio (default: {DEFAULT_VAL_RATIO}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for shuffling groups (default: {DEFAULT_SEED}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.dataset_root
    if not data_root.exists():
        print(f"Error: dataset root not found at {data_root}", file=sys.stderr)
        return 1

    layout = resolve_layout(data_root)
    print(f"Loaded dataset layout from: {layout.data_root}")
    print(f"Audit report reference: {args.audit_report}")

    print(
        f"Generating splits with ratios train={args.train_ratio}, val={args.val_ratio}, seed={args.seed}..."
    )
    manifest = build_manifest(
        layout=layout,
        audit_report_path=args.audit_report,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    violations = validate_manifest(manifest)
    if violations:
        print("ERROR: Split validation failed with violations:", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 2

    print("Integrity check passed. No group leakage detected.")
    saved = save_manifests(manifest, args.output_dir)
    print(f"Successfully saved manifests to {args.output_dir}:")
    for key, path in saved.items():
        print(f"  - {key}: {path.name}")

    print("\nSummary of split counts:")
    for split_name, count in manifest.split_counts.items():
        print(f"  {split_name}: {count} images")

    print(f"Total groups: {manifest.total_groups}")
    print(f"Vendor leakage groups corrected: {manifest.leakage_candidate_groups_in_vendor_split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
