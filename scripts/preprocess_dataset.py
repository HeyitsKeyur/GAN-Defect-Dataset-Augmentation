#!/usr/bin/env python3
"""CLI executable for Phase 3 Preprocessing Pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_paths import default_data_root, resolve_layout
from src.data.preprocessing.pipeline import PreprocessingConfig, run_preprocessing_pipeline
from src.utils.config import get_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute Phase 3 Preprocessing & Crop Extraction Pipeline.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Path to Ceramic-Defect dataset containing original image, image, lables.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=None,
        help="Path to Phase 1 split manifests directory (containing train.csv, val.csv, test.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "phase3",
        help="Destination directory for processed crops, full images, and manifests.",
    )
    parser.add_argument("--crop-size", type=int, default=256, help="Target crop size (default: 256).")
    parser.add_argument("--context-margin", type=float, default=0.20, help="Context margin ratio (default: 0.20).")
    parser.add_argument("--cluster-distance", type=float, default=128.0, help="Clustering distance in px (default: 128.0).")
    parser.add_argument("--clustering-mode", type=str, default="cluster", choices=["cluster", "individual"], help="Clustering mode.")
    parser.add_argument("--min-retention-ratio", type=float, default=0.30, help="Minimum retained area ratio (default: 0.30).")
    parser.add_argument("--min-clipped-area", type=float, default=9.0, help="Minimum clipped box area in px^2 (default: 9.0).")
    parser.add_argument("--full-image-size", type=int, default=640, help="Full-image letterbox size (default: 640).")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit number of source images per split (for testing/samples).")
    parser.add_argument("--no-crops", action="store_true", help="Disable crop stream.")
    parser.add_argument("--no-full", action="store_true", help="Disable full-image stream.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    proj_cfg = get_config(dataset_root_override=args.dataset_root)

    ds_root = args.dataset_root or proj_cfg.dataset_root
    man_dir = args.manifest_dir or proj_cfg.split_manifest_dir

    layout = resolve_layout(ds_root)

    print("============================================================")
    print("Executing Phase 3 Preprocessing Pipeline")
    print(f"Dataset root: {layout.data_root}")
    print(f"Manifest dir: {man_dir}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Crop size:    {args.crop_size} (Mode: {args.clustering_mode}, Cluster dist: {args.cluster_distance})")
    print(f"Full size:    {args.full_image_size}")
    if args.sample_limit:
        print(f"Sample limit: {args.sample_limit} images per split")
    print("============================================================")

    config = PreprocessingConfig(
        crop_size=args.crop_size,
        context_margin=args.context_margin,
        cluster_distance=args.cluster_distance,
        clustering_mode=args.clustering_mode,
        min_retention_ratio=args.min_retention_ratio,
        min_clipped_area=args.min_clipped_area,
        full_image_size=args.full_image_size,
        enable_crops=not args.no_crops,
        enable_full_images=not args.no_full,
    )

    summary = run_preprocessing_pipeline(
        layout=layout,
        manifest_dir=man_dir,
        output_dir=args.output_dir,
        config=config,
        sample_limit_per_split=args.sample_limit,
    )

    print("\n[SUCCESS] Preprocessing completed successfully.")
    print(f"Total source images processed: {summary['statistics']['total_source_images_processed']}")
    print(f"Total defect crops generated:  {summary['statistics']['total_crops_generated']}")
    print(f"Total full images generated:   {summary['statistics']['total_full_images_generated']}")
    for split, s_data in summary["statistics"]["processed_splits"].items():
        print(f"  Split '{split}': {s_data['source_images']} images -> {s_data['crops_generated']} crops, {s_data['full_images_generated']} full images")

    print(f"\nOutputs written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
