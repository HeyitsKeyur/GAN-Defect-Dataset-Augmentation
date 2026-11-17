#!/usr/bin/env python3
"""CLI script to execute the Phase 2 smoke test (Colab & Local compatible)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_config
from src.utils.smoke_test import run_smoke_test, save_smoke_test_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 non-training smoke test.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional dataset root path override.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save smoke test reports (default: <output_dir>/smoke_test).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = get_config(dataset_root_override=args.dataset_root)

    print(f"============================================================")
    print(f"Executing Phase 2 Smoke Test on: {config.environment_name.upper()}")
    print(f"Project root: {config.project_root}")
    print(f"Dataset root: {config.dataset_root}")
    print(f"Split dir:    {config.split_manifest_dir}")
    print(f"============================================================")

    result = run_smoke_test(config)

    out_dir = args.output_dir or (config.experiment_output_dir / "smoke_test")
    json_path, md_path = save_smoke_test_report(result, out_dir)

    print(f"\n[Result] Overall Status: {result.overall_status}")
    print(f"[Result] GPU Available:  {result.gpu_available} ({result.gpu_name or 'None'})")
    print(f"[Result] Dataset Valid:  {result.dataset_verified}")
    print(f"[Result] Splits Valid:   {result.split_manifests_verified} (Leakage detected: {result.split_leakage_detected})")
    print(f"[Result] Tensor Compute: {result.tensor_op_success} on device {result.tensor_device}")
    print(f"\nSaved Reports:")
    print(f"  - JSON: {json_path}")
    print(f"  - MD:   {md_path}")

    if result.overall_status == "FAIL":
        print("\nERROR: Smoke test failed. Please inspect errors above.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
