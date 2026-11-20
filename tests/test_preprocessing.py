"""Comprehensive unit tests for Phase 3 preprocessing, crop geometry, and transformations."""

from __future__ import annotations

import pytest
from PIL import Image

from src.data.preprocessing.crop_geometry import (
    BoundingBox,
    CropWindow,
    calculate_crop_window,
    transform_box_to_crop,
    box_to_yolo_normalized,
    yolo_to_box,
    letterbox_calculate,
    transform_box_letterbox,
)
from src.data.preprocessing.defect_cluster import (
    BoundingBox as ClusterBox,
    DefectCluster,
    cluster_defects,
)
from src.data.preprocessing.crop_extractor import extract_crops_from_image
from src.data.preprocessing.full_image_pipeline import preprocess_full_image


def test_calculate_crop_window_normal():
    # Defect at center (500, 500) on 1000x1000 image, crop_size=256
    cw = calculate_crop_window(500, 500, 256, 1000, 1000)
    assert cw.src_x1 == 500 - 128
    assert cw.src_x2 == 500 + 128
    assert cw.src_y1 == 500 - 128
    assert cw.src_y2 == 500 + 128
    assert cw.pad_left == 0 and cw.pad_right == 0
    assert cw.crop_width == 256
    assert cw.crop_height == 256


def test_calculate_crop_window_edge_clamping():
    # Defect near top-left corner (20, 20) on 1000x1000 image, crop_size=256
    # Should clamp src_x1=0, src_x2=256 without unnecessary padding
    cw = calculate_crop_window(20, 20, 256, 1000, 1000)
    assert cw.src_x1 == 0
    assert cw.src_x2 == 256
    assert cw.src_y1 == 0
    assert cw.src_y2 == 256
    assert cw.crop_width == 256


def test_calculate_crop_window_oversized_requires_padding():
    # Crop size (512x512) larger than source image (300x400)
    cw = calculate_crop_window(150, 200, 512, 300, 400)
    assert cw.src_x1 == 0
    assert cw.src_x2 == 300
    assert cw.src_y1 == 0
    assert cw.src_y2 == 400
    assert cw.pad_left + cw.pad_right == 512 - 300
    assert cw.pad_top + cw.pad_bottom == 512 - 400
    assert cw.crop_width == 512
    assert cw.crop_height == 512


def test_transform_box_inside_crop():
    crop = CropWindow(src_x1=100, src_y1=100, src_x2=356, src_y2=356)
    box = BoundingBox(xmin=120, ymin=130, xmax=160, ymax=180, class_id=2, class_name="spot", original_id=1)
    res = transform_box_to_crop(box, crop)
    assert res.retained is True
    assert res.box.xmin == 20.0
    assert res.box.ymin == 30.0
    assert res.box.xmax == 60.0
    assert res.box.ymax == 80.0
    assert res.box.class_id == 2
    assert res.retention_ratio == 1.0


def test_transform_box_partially_clipped():
    crop = CropWindow(src_x1=100, src_y1=100, src_x2=200, src_y2=200)
    # Box spans across right boundary x=150 to x=250 (width=100), inside is 150 to 200 (width=50) -> 50% retained
    box = BoundingBox(xmin=150, ymin=120, xmax=250, ymax=180, class_id=1, class_name="scar", original_id=2)
    res = transform_box_to_crop(box, crop, min_retention_ratio=0.30)
    assert res.retained is True
    assert res.box.xmin == 50.0
    assert res.box.xmax == 100.0
    assert abs(res.retention_ratio - 0.50) < 1e-4


def test_transform_box_excluded_low_retention():
    crop = CropWindow(src_x1=100, src_y1=100, src_x2=200, src_y2=200)
    # Box spans 190 to 300 (width 110), only 10px inside (9% retained)
    box = BoundingBox(xmin=190, ymin=120, xmax=300, ymax=180, class_id=1, class_name="scar", original_id=3)
    res = transform_box_to_crop(box, crop, min_retention_ratio=0.30)
    assert res.retained is False
    assert "low_retention_ratio" in (res.exclusion_reason or "")


def test_yolo_normalization_roundtrip():
    box = BoundingBox(xmin=50, ymin=100, xmax=150, ymax=200, class_id=0, class_name="adhesion")
    cx, cy, w, h = box_to_yolo_normalized(box, image_width=500, image_height=400)
    assert cx == 100 / 500
    assert cy == 150 / 400
    assert w == 100 / 500
    assert h == 100 / 400

    recovered = yolo_to_box(cx, cy, w, h, class_id=0, class_name="adhesion", image_width=500, image_height=400)
    assert abs(recovered.xmin - box.xmin) < 1e-4
    assert abs(recovered.ymin - box.ymin) < 1e-4
    assert abs(recovered.xmax - box.xmax) < 1e-4
    assert abs(recovered.ymax - box.ymax) < 1e-4


def test_letterbox_calculate_and_transform():
    # Letterbox 1000x500 to 640x640: aspect ratio 2:1 -> scaled to 640x320 with pad_top=160
    scale, pad_left, pad_top, new_w, new_h = letterbox_calculate(1000, 500, 640, 640)
    assert scale == 0.64
    assert new_w == 640
    assert new_h == 320
    assert pad_left == 0
    assert pad_top == 160

    box = BoundingBox(xmin=100, ymin=100, xmax=200, ymax=200, class_id=3, class_name="uneven")
    lbox = transform_box_letterbox(box, scale, pad_left, pad_top, 640, 640)
    assert abs(lbox.xmin - (100 * 0.64)) < 1e-4
    assert abs(lbox.ymin - (100 * 0.64 + 160)) < 1e-4


def test_cluster_defects_grouping():
    # Three boxes: two close together (<50px), one distant (300px away)
    b1 = BoundingBox(xmin=10, ymin=10, xmax=30, ymax=30, class_id=2, class_name="spot", original_id=1)
    b2 = BoundingBox(xmin=25, ymin=25, xmax=45, ymax=45, class_id=2, class_name="spot", original_id=2)
    b3 = BoundingBox(xmin=400, ymin=400, xmax=420, ymax=420, class_id=2, class_name="spot", original_id=3)

    clusters = cluster_defects([b1, b2, b3], cluster_distance_threshold=50.0, mode="cluster")
    assert len(clusters) == 2
    assert clusters[0].count == 2
    assert clusters[1].count == 1


def test_extract_crops_from_image_synthetic():
    img = Image.new("RGB", (1000, 1000), color=(128, 128, 128))
    b1 = BoundingBox(xmin=100, ymin=100, xmax=130, ymax=130, class_id=2, class_name="spot", original_id=1)
    b2 = BoundingBox(xmin=120, ymin=120, xmax=140, ymax=140, class_id=2, class_name="spot", original_id=2)

    crops = extract_crops_from_image(
        image=img,
        source_stem="test_synthetic",
        group_id="group_1",
        split="train",
        all_image_boxes=[b1, b2],
        crop_size=256,
        cluster_distance=100.0,
    )
    assert len(crops) == 1
    crop = crops[0]
    assert crop.crop_id.startswith("test_synthetic_c00_")
    assert crop.image.size == (256, 256)
    assert len(crop.retained_boxes) == 2
    assert crop.split == "train"
    assert crop.group_id == "group_1"


def test_preprocess_full_image_synthetic():
    img = Image.new("RGB", (1000, 500), color=(200, 200, 200))
    b = BoundingBox(xmin=200, ymin=100, xmax=400, ymax=300, class_id=1, class_name="scar", original_id=1)
    full = preprocess_full_image(img, "test_full", "group_1", "val", [b], target_size=640)
    assert full.image.size == (640, 640)
    assert len(full.transformed_boxes) == 1
    tbox = full.transformed_boxes[0]
    assert tbox.xmin > 0 and tbox.xmax < 640
    assert tbox.ymin > 0 and tbox.ymax < 640
