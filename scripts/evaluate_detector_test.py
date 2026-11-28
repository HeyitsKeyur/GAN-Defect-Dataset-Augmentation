#!/usr/bin/env python3
"""Isolated evaluation script for the held-out real-only test split.

STRICT QUARANTINE PROTOCOL:
- This script is isolated from the training workflow.
- Must only be executed on an explicitly selected final checkpoint (best.pt).
- Must NEVER be called iteratively during training or hyperparameter tuning.
- Applies the optimal confidence threshold pre-selected strictly on validation data.
"""

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
from src.models.baseline_config import BaselineExperimentConfig
from src.models.metrics_collector import MetricsCollector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained YOLOv8 checkpoint on the quarantined SolDef test split."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to trained checkpoint (e.g. best.pt).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "soldef_baseline_yolov8s.yaml",
        help="Path to baseline experiment configuration.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override path to preprocessed detector dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory for evaluation artifacts.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold pre-selected on validation split.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Compute device for evaluation (e.g. '0', 'cpu').",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 75)
    print("SOLDEF_AI HELD-OUT TEST SPLIT EVALUATION (ISOLATED ENTRY POINT)")
    print("=" * 75)
    print(f"Checkpoint Weights:     {args.weights}")
    print(f"Confidence Threshold:   {args.conf_threshold} (Frozen from Validation)")
    print(f"Target Benchmark:       Primary Side-View (Test Partition: 30 images, 62 boxes)")
    print("-" * 75)

    if not args.weights.is_file():
        print(f"[ERROR] Checkpoint weights file not found: {args.weights}", file=sys.stderr)
        return 1

    # Load configuration
    config = BaselineExperimentConfig.load(args.config)
    data_dir = (PROJECT_ROOT / config.data_dir).resolve() if not Path(config.data_dir).is_absolute() else Path(config.data_dir)
    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()

    output_dir = args.output_dir if args.output_dir else args.weights.parent.parent / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preflight check on Test split
    validator = DetectorDatasetValidator(data_root=data_dir)
    test_report = validator.validate(splits=["test"])
    if not test_report.is_valid:
        print("[ERROR] Test split failed integrity check:", file=sys.stderr)
        for err in test_report.critical_errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"[TEST SPLIT OK] 30 images, 62 target annotations verified.")

    # Generate temporary single-split test yaml for Ultralytics
    test_yaml_path = output_dir / "test_eval_dataset.yaml"
    test_imgs = (data_dir / "test" / "images").as_posix()
    test_yaml_lines = [
        f"path: {data_dir.as_posix()}",
        f"val: {test_imgs}",  # Evaluate on test split
        f"nc: {config.model.num_classes}",
        "names:",
    ]
    for cid, cname in config.model.names.items():
        test_yaml_lines.append(f"  {cid}: {cname}")
    test_yaml_path.write_text("\n".join(test_yaml_lines), encoding="utf-8")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is not installed. Please install: pip install ultralytics", file=sys.stderr)
        return 1

    print("[EVALUATION] Loading model and running inference on quarantined test split...")
    model = YOLO(str(args.weights))
    eval_results = model.val(
        data=str(test_yaml_path),
        split="val",  # Points to test split in test_eval_dataset.yaml
        imgsz=config.schedule.image_size,
        conf=args.conf_threshold,
        device=args.device,
        project=str(output_dir.parent),
        name="test_evaluation",
        exist_ok=True,
    )

    # Extract metrics
    map50 = float(eval_results.box.map50) if hasattr(eval_results, "box") else 0.0
    map50_95 = float(eval_results.box.map) if hasattr(eval_results, "box") else 0.0
    mean_p = float(eval_results.box.mp) if hasattr(eval_results, "box") else 0.0
    mean_r = float(eval_results.box.mr) if hasattr(eval_results, "box") else 0.0

    # Extract per-class metrics if available
    class_metrics = {}
    if hasattr(eval_results, "box"):
        box = eval_results.box
        p_arr = getattr(box, "p", None)
        r_arr = getattr(box, "r", None)
        ap50_arr = getattr(box, "ap50", None)
        ap_arr = getattr(box, "maps", None)

        for cid, cname in config.model.names.items():
            c_dict = {}
            if p_arr is not None and len(p_arr) > cid:
                c_dict["precision"] = float(p_arr[cid])
            if r_arr is not None and len(r_arr) > cid:
                c_dict["recall"] = float(r_arr[cid])
            if ap50_arr is not None and len(ap50_arr) > cid:
                c_dict["ap50"] = float(ap50_arr[cid])
            if ap_arr is not None and len(ap_arr) > cid:
                c_dict["ap50_95"] = float(ap_arr[cid])
            class_metrics[cname] = c_dict

    summary = MetricsCollector.build_summary_report(
        experiment_name=config.experiment_name,
        split_name="test",
        map50=map50,
        map50_95=map50_95,
        mean_p=mean_p,
        mean_r=mean_r,
        optimal_conf=args.conf_threshold,
        class_metrics=class_metrics,
    )

    summary.save_json(output_dir / "test_metrics.json")
    summary.save_csv(output_dir / "test_per_class_metrics.csv")

    print("-" * 75)
    print(f"[TEST EVALUATION COMPLETE] Results exported to: {output_dir}")
    print(f"  mAP@0.50:      {map50:.4f}")
    print(f"  mAP@0.50:0.95: {map50_95:.4f}")
    print(f"  Mean Precision:{mean_p:.4f}")
    print(f"  Mean Recall:   {mean_r:.4f}")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())
