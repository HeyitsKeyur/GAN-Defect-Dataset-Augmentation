#!/usr/bin/env python3
"""Run the Phase 0 Ceramic-Defect dataset audit without modifying originals."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.audit import run_audit, write_markdown_summary
from src.data.dataset_paths import default_data_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the local Ceramic-Defect dataset.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_data_root(PROJECT_ROOT),
        help="Path to the discovered data/data directory (images + lables).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "audit" / "phase0",
        help="Directory for audit JSON/CSV/Markdown/visualizations.",
    )
    parser.add_argument("--max-visualizations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-markdown",
        action="store_true",
        help="Write machine-readable outputs only. Markdown can be filled after tests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.dataset_root
    if not data_root.exists():
        print(
            "Dataset root not found:\n"
            f"  {data_root}\n"
            "Provide --dataset-root pointing at the folder that contains "
            "'original image', 'image', and 'lables'. Do not download another dataset."
        )
        return 2
    report = run_audit(
        data_root,
        args.output_dir,
        max_visualizations=args.max_visualizations,
        seed=args.seed,
    )
    if not args.skip_markdown:
        write_markdown_summary(
            report,
            args.output_dir / "audit_summary.md",
            suitability={
                "verdict": "See the Phase 0 chat report / updated Markdown after qualitative review.",
                "reasoning": (
                    "Numeric audit values are in this folder. The research suitability "
                    "verdict is written after reviewing those values and license files."
                ),
            },
            license_status={
                "status": "Unverified in automated pass",
                "detail": (
                    "Automated search results are in audit_report.json under "
                    "license_files_found_in_dataset_tree."
                ),
            },
            tests={"command": "pytest", "result": "Run separately; see test_results.txt"},
            commands=[
                f"python scripts/audit_ceramic_defect.py --dataset-root {data_root} --output-dir {args.output_dir}"
            ],
        )
    print(f"Wrote audit outputs to {args.output_dir}")
    print(f"Unique stems: {report['file_counts']['unique_stems']}")
    print(f"JSON instances: {report['per_class']['instance_total_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
