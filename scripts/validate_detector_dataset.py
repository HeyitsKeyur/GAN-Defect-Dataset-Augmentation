#!/usr/bin/env python3
"""CLI script to validate preprocessed SolDef detector datasets prior to training preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.validation.detector_dataset_validator import DetectorDatasetValidator


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate SolDef detector dataset integrity before training preflight."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprocessing" / "soldef" / "phase_d" / "detector",
        help="Path to preprocessed detector dataset root (containing train, val, test subdirs).",
    )
    parser.add_argument(
        "--split-manifest-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "splits" / "soldef" / "phase1",
        help="Path to frozen Phase 1 split manifests.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional destination path to write JSON validation report.",
    )
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Deep verify image files via PIL open/verify (slower).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to validate.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SOLDEF DETECTOR DATASET PREFLIGHT INTEGRITY VALIDATION")
    print("=" * 70)
    print(f"Dataset root:         {args.data_dir}")
    print(f"Split manifests dir:  {args.split_manifest_dir}")
    print(f"Splits to validate:   {args.splits}")
    print(f"Verify image reads:   {args.verify_images}")
    print("-" * 70)

    validator = DetectorDatasetValidator(
        data_root=args.data_dir,
        split_manifest_dir=args.split_manifest_dir,
        verify_image_reads=args.verify_images,
    )

    report = validator.validate(splits=args.splits)

    print(report.to_summary_markdown())

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\n[OK] Machine-readable report saved to: {args.report_json}")

    print("=" * 70)
    if report.is_valid:
        print("[PREFLIGHT SUCCESS] All dataset integrity and anti-leakage checks passed.")
        return 0
    else:
        print("[PREFLIGHT FAILED] Critical integrity issues detected! Halting training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
