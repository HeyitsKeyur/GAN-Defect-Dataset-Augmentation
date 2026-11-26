"""Unit tests for Stage G1 real-only detector baseline infrastructure.

Tests configuration, dataset validation, test-set quarantine, reproducibility,
metric collection, and dry-run execution on CPU with zero external network access.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
from typing import Any

from PIL import Image
import pytest

from src.data.validation.detector_dataset_validator import (
    DatasetValidationError,
    DetectorDatasetValidator,
)
from src.models.baseline_config import (
    BaselineAugmentationConfig,
    BaselineExperimentConfig,
    ModelArchitectureConfig,
    TrainingScheduleConfig,
    parse_simple_yaml,
)
from src.models.metrics_collector import MetricsCollector
from src.utils.reproducibility import collect_environment_metadata, get_git_info, set_seed


@pytest.fixture
def mock_detector_dataset(tmp_path: Path) -> Path:
    """Create a minimal mock detector dataset adhering to YOLO formatting."""
    data_root = tmp_path / "mock_detector"
    data_root.mkdir()

    for split in ["train", "val", "test"]:
        img_dir = data_root / split / "images"
        lbl_dir = data_root / split / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        # Create 2 valid images and labels
        for idx in range(2):
            stem = f"{split}_img_{idx}"
            img_path = img_dir / f"{stem}.jpg"
            lbl_path = lbl_dir / f"{stem}.txt"

            # Create dummy 640x640 JPEG
            img = Image.new("RGB", (640, 640), color=(100, 100, 100))
            img.save(img_path, format="JPEG")

            # Create valid YOLO annotations: class cx cy w h
            # Class 0: exc_solder, Class 1: spike, Class 2: poor_solder
            lbl_path.write_text(f"0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n", encoding="utf-8")

        # Create 1 empty label file (legitimate negative image)
        neg_stem = f"{split}_neg"
        neg_img = img_dir / f"{neg_stem}.jpg"
        neg_lbl = lbl_dir / f"{neg_stem}.txt"
        Image.new("RGB", (640, 640), color=(50, 50, 50)).save(neg_img, format="JPEG")
        neg_lbl.write_text("", encoding="utf-8")

    return data_root


# ==============================================================================
# 1. Configuration Tests
# ==============================================================================

def test_default_baseline_config() -> None:
    cfg = BaselineExperimentConfig()
    assert cfg.model.variant == "s"
    assert cfg.model.weights == "yolov8s.pt"
    assert cfg.schedule.image_size == 640
    assert cfg.schedule.batch_size == 16
    assert cfg.schedule.epochs == 120
    assert cfg.schedule.seed == 42
    # Verify strict real-only augmentation
    assert cfg.augmentation.mosaic == 0.0
    assert cfg.augmentation.mixup == 0.0
    assert cfg.augmentation.copy_paste == 0.0
    assert cfg.augmentation.fliplr == 0.5


def test_config_save_load_json_and_yaml(tmp_path: Path) -> None:
    cfg = BaselineExperimentConfig(
        experiment_name="test_run",
        schedule=TrainingScheduleConfig(batch_size=8, epochs=10),
    )
    json_path = tmp_path / "config.json"
    yaml_path = tmp_path / "config.yaml"

    cfg.save_json(json_path)
    cfg.save_yaml(yaml_path)

    loaded_json = BaselineExperimentConfig.load(json_path)
    loaded_yaml = BaselineExperimentConfig.load(yaml_path)

    assert loaded_json.experiment_name == "test_run"
    assert loaded_json.schedule.batch_size == 8
    assert loaded_yaml.experiment_name == "test_run"
    assert loaded_yaml.schedule.batch_size == 8


def test_test_set_quarantine_in_generated_yaml(tmp_path: Path) -> None:
    cfg = BaselineExperimentConfig(data_dir=str(tmp_path))
    dest_yaml = tmp_path / "dataset.yaml"
    cfg.generate_yolo_data_yaml(dest_yaml, abs_data_root=tmp_path)

    content = dest_yaml.read_text(encoding="utf-8")
    assert "train:" in content
    assert "val:" in content
    # Crucial quarantine check: test split must NOT be present as an evaluation key!
    assert "test:" not in content
    assert "TEST SET QUARANTINE NOTICE" in content


# ==============================================================================
# 2. Dataset Validation Tests
# ==============================================================================

def test_validator_on_mock_dataset(mock_detector_dataset: Path) -> None:
    validator = DetectorDatasetValidator(data_root=mock_detector_dataset)
    report = validator.validate(splits=["train", "val", "test"])

    # 3 images per split * 3 splits = 9 images total
    assert report.total_images == 9
    # 2 images with 2 boxes each = 4 boxes per split * 3 splits = 12 boxes total
    assert report.total_boxes == 12
    # Check that empty labels were recognized
    for split_summary in report.splits.values():
        assert split_summary.empty_label_count == 1
        assert split_summary.target_box_count == 4


def test_validator_detects_missing_image(mock_detector_dataset: Path) -> None:
    # Delete one image
    img_to_remove = mock_detector_dataset / "train" / "images" / "train_img_0.jpg"
    img_to_remove.unlink()

    validator = DetectorDatasetValidator(data_root=mock_detector_dataset)
    report = validator.validate(splits=["train"])

    assert not report.is_valid
    assert len(report.splits["train"].missing_images) > 0
    assert any("train_img_0.txt" in err for err in report.splits["train"].missing_images)


def test_validator_detects_invalid_class_id(mock_detector_dataset: Path) -> None:
    # Write invalid class ID 99
    lbl_file = mock_detector_dataset / "train" / "labels" / "train_img_0.txt"
    lbl_file.write_text("99 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    validator = DetectorDatasetValidator(data_root=mock_detector_dataset)
    report = validator.validate(splits=["train"])

    assert not report.is_valid
    assert any("Invalid class ID 99" in err for err in report.splits["train"].malformed_labels)


def test_validator_detects_invalid_coordinates(mock_detector_dataset: Path) -> None:
    # Write coordinate > 1.0 and negative width
    lbl_file = mock_detector_dataset / "train" / "labels" / "train_img_0.txt"
    lbl_file.write_text("0 1.5 0.5 -0.2 0.2\n", encoding="utf-8")

    validator = DetectorDatasetValidator(data_root=mock_detector_dataset)
    report = validator.validate(splits=["train"])

    assert not report.is_valid
    assert any("out of normalized [0, 1] range" in err for err in report.splits["train"].malformed_labels)
    assert any("non-positive" in err for err in report.splits["train"].malformed_labels)


def test_validator_detects_split_overlap(mock_detector_dataset: Path) -> None:
    # Copy a train image into validation directory (simulating data leakage)
    leaked_img = mock_detector_dataset / "val" / "images" / "train_img_0.jpg"
    leaked_lbl = mock_detector_dataset / "val" / "labels" / "train_img_0.txt"
    leaked_img.write_bytes(b"dummy")
    leaked_lbl.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    validator = DetectorDatasetValidator(data_root=mock_detector_dataset)
    report = validator.validate(splits=["train", "val"])

    assert not report.is_valid
    assert len(report.overlap_errors) > 0
    assert any("Data leakage detected!" in err for err in report.overlap_errors)


# ==============================================================================
# 3. Reproducibility & Environment Tracking Tests
# ==============================================================================

def test_set_seed_execution() -> None:
    # Verify set_seed executes without crashing
    set_seed(42, deterministic=True)


def test_collect_environment_metadata() -> None:
    meta = collect_environment_metadata()
    assert "timestamp_utc" in meta
    assert "python" in meta
    assert "platform" in meta
    assert "git" in meta
    assert "gpu" in meta
    assert meta["python"]["version"] != ""


# ==============================================================================
# 4. Metrics & Evaluation Collector Tests
# ==============================================================================

def test_metrics_collector_summary() -> None:
    summary = MetricsCollector.build_summary_report(
        experiment_name="baseline_test",
        split_name="val",
        map50=0.85,
        map50_95=0.62,
        mean_p=0.88,
        mean_r=0.82,
        optimal_conf=0.35,
        class_metrics={
            "exc_solder": {"precision": 0.90, "recall": 0.85, "ap50": 0.88},
            "spike": {"precision": 0.85, "recall": 0.80, "ap50": 0.82},
            "poor_solder": {"precision": 0.70, "recall": 0.60, "ap50": 0.65},
        },
    )

    assert summary.map50 == 0.85
    assert summary.optimal_confidence_threshold == 0.35
    assert len(summary.classes) == 3
    # Check that minority class caveat was attached to poor_solder
    assert summary.classes["poor_solder"].statistical_caveat is not None
    assert "Small sample minority class" in summary.classes["poor_solder"].statistical_caveat


def test_metrics_collector_parse_results_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train/box_loss", "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"])
        writer.writerow(["1", "1.5", "0.60", "0.55", "0.58", "0.35"])
        writer.writerow(["2", "1.2", "0.75", "0.70", "0.72", "0.45"])
        writer.writerow(["3", "1.0", "0.80", "0.78", "0.79", "0.50"])

    parsed = MetricsCollector.parse_ultralytics_results_csv(csv_path)
    assert parsed["epochs_count"] == 3
    assert parsed["best_epoch"]["epoch"] == 3.0
    assert parsed["best_epoch"]["metrics/mAP50(B)"] == 0.79


def test_optimal_confidence_threshold_selection() -> None:
    # F1 curve data: [(threshold, f1_score)]
    f1_curve = [(0.1, 0.65), (0.2, 0.78), (0.35, 0.84), (0.5, 0.75), (0.7, 0.50)]
    opt_thresh = MetricsCollector.select_optimal_confidence_threshold(f1_curve)
    assert opt_thresh == 0.35
