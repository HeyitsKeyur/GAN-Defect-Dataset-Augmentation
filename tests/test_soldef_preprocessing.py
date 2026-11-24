"""Unit and integration tests for SolDef preprocessing pipelines (Stage D)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest
from PIL import Image

from src.data.adapters.soldef import SolDefAdapter
from src.data.preprocessing.crop_geometry import (
    BoundingBox,
    calculate_crop_window,
    letterbox_calculate,
    transform_box_letterbox,
)
from src.data.preprocessing.soldef_config import SolDefPreprocessingConfig
from src.data.preprocessing.soldef_detector_pipeline import (
    inverse_transform_letterbox,
    preprocess_soldef_detector,
)
from src.data.preprocessing.soldef_patch_extractor import (
    SolDefPatchRecord,
    extract_soldef_patches,
)

PREPROCESSING_BASE = Path("outputs/preprocessing/soldef/phase_d")


class TestSolDefPreprocessingGeometry:
    """Test geometric calculations, boundary clamping, and letterboxing."""

    def test_letterbox_calculation_soldef_dimensions(self) -> None:
        """Verify letterbox calculation for 2560x1440 image targeting 640x640."""
        scale, pad_left, pad_top, new_w, new_h = letterbox_calculate(
            original_width=2560,
            original_height=1440,
            target_width=640,
            target_height=640,
        )
        assert scale == 0.25
        assert new_w == 640
        assert new_h == 360
        assert pad_left == 0
        assert pad_top == 140
        assert pad_top * 2 + new_h == 640

    def test_letterbox_box_transformation_and_inverse(self) -> None:
        """Verify forward and inverse letterbox coordinate transformations."""
        orig_box = BoundingBox(
            xmin=200.0,
            ymin=400.0,
            xmax=600.0,
            ymax=800.0,
            class_id=0,
            class_name="exc_solder",
            original_id=1,
        )

        scale, pad_left, pad_top, _, _ = letterbox_calculate(2560, 1440, 640, 640)
        det_box = transform_box_letterbox(orig_box, scale, pad_left, pad_top, 640, 640)

        # Expected: x * 0.25, y * 0.25 + 140
        assert det_box.xmin == pytest.approx(50.0, abs=1e-3)
        assert det_box.ymin == pytest.approx(240.0, abs=1e-3)
        assert det_box.xmax == pytest.approx(150.0, abs=1e-3)
        assert det_box.ymax == pytest.approx(340.0, abs=1e-3)

        # Invert
        restored = inverse_transform_letterbox(det_box, scale, pad_left, pad_top, 2560, 1440)
        assert restored.xmin == pytest.approx(orig_box.xmin, abs=1e-2)
        assert restored.ymin == pytest.approx(orig_box.ymin, abs=1e-2)
        assert restored.xmax == pytest.approx(orig_box.xmax, abs=1e-2)
        assert restored.ymax == pytest.approx(orig_box.ymax, abs=1e-2)

    def test_crop_window_boundary_clamping(self) -> None:
        """Verify boundary clamping and padding for edge/corner defects."""
        # Defect near top-left corner
        cw_corner = calculate_crop_window(
            center_x=50.0,
            center_y=50.0,
            crop_size=(256, 256),
            image_width=2560,
            image_height=1440,
            allow_padding=True,
        )
        assert cw_corner.src_x1 == 0
        assert cw_corner.src_y1 == 0
        assert cw_corner.src_x2 == 256
        assert cw_corner.src_y2 == 256
        assert cw_corner.crop_width == 256
        assert cw_corner.crop_height == 256

        # Defect near bottom-right corner
        cw_br = calculate_crop_window(
            center_x=2540.0,
            center_y=1420.0,
            crop_size=(256, 256),
            image_width=2560,
            image_height=1440,
            allow_padding=True,
        )
        assert cw_br.src_x2 == 2560
        assert cw_br.src_y2 == 1440
        assert cw_br.src_x1 == 2560 - 256
        assert cw_br.src_y1 == 1440 - 256


class TestSolDefPipelineOutputs:
    """Test generated Stage D artifacts, split discipline, and preservation."""

    def test_patch_manifest_and_image_counts(self) -> None:
        """Verify patch manifest exists and contains 256 patches matching train defect count."""
        manifest_p = PREPROCESSING_BASE / "metadata" / "patch_manifest.csv"
        assert manifest_p.exists(), f"Patch manifest missing at {manifest_p}"

        with open(manifest_p, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        # Expected: exactly 256 patches from the 140 training images
        assert len(rows) == 256, f"Expected 256 patches, got {len(rows)}"

        classes = [r["target_class"] for r in rows]
        assert classes.count("exc_solder") == 128
        assert classes.count("spike") == 96
        assert classes.count("poor_solder") == 32
        assert "good" not in classes

        # Check split isolation: 100% of patches from Train
        splits = {r["split"] for r in rows}
        assert splits == {"train"}

    def test_patch_image_dimensions_and_labels(self) -> None:
        """Verify sample patch dimensions and YOLO label formats."""
        manifest_p = PREPROCESSING_BASE / "metadata" / "patch_manifest.csv"
        with open(manifest_p, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        sample_row = rows[0]
        crop_id = sample_row["crop_id"]
        img_p = PREPROCESSING_BASE / "patches" / "images" / f"{crop_id}.jpg"
        lbl_p = PREPROCESSING_BASE / "patches" / "labels" / f"{crop_id}.txt"

        assert img_p.exists()
        assert lbl_p.exists()

        with Image.open(img_p) as im:
            assert im.size == (256, 256)

        # Check YOLO label lines
        lines = lbl_p.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 1
        for line in lines:
            parts = line.split()
            assert len(parts) == 5
            cid, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            assert cid in (0, 1, 2)
            assert 0.0 <= cx <= 1.0
            assert 0.0 <= cy <= 1.0
            assert 0.0 < w <= 1.0
            assert 0.0 < h <= 1.0

    def test_detector_preprocessed_outputs(self) -> None:
        """Verify detector letterboxed outputs for train, val, and test."""
        det_dir = PREPROCESSING_BASE / "detector"
        for split, expected_count in [("train", 140), ("val", 30), ("test", 30)]:
            split_imgs = list((det_dir / split / "images").glob("*.jpg"))
            split_lbls = list((det_dir / split / "labels").glob("*.txt"))
            assert len(split_imgs) == expected_count, f"{split} images count mismatch"
            assert len(split_lbls) == expected_count, f"{split} labels count mismatch"

            # Check image size is 640x640
            with Image.open(split_imgs[0]) as im:
                assert im.size == (640, 640)

    def test_qc_reports_and_visualizations_exist(self) -> None:
        """Verify visual QC images and metrics reports exist."""
        qc_dir = PREPROCESSING_BASE / "qc" / "visualizations"
        qc_imgs = list(qc_dir.glob("*.png"))
        assert len(qc_imgs) >= 12, f"Expected at least 12 QC panels, found {len(qc_imgs)}"

        report_json = PREPROCESSING_BASE / "reports" / "qc_metrics_report.json"
        report_md = PREPROCESSING_BASE / "reports" / "qc_metrics_report.md"
        assert report_json.exists()
        assert report_md.exists()

        data = json.loads(report_json.read_text(encoding="utf-8"))
        assert data["pipeline_a_patch_extraction"]["total_extracted_patches"] == 256
        assert data["pipeline_b_detector_preprocessing"]["split_summaries"]["train"]["image_count"] == 140
