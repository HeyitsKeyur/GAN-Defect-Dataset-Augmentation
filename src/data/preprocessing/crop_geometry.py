"""Bounding-box geometry, padding, translation, clipping, and coordinate transformation utilities.

Conventions:
- Coordinates in absolute pixels: (xmin, ymin, xmax, ymax) with 0 <= xmin < xmax and 0 <= ymin < ymax.
- Top-left pixel origin is (0, 0).
- Normalized YOLO format: (cx, cy, w, h) where each coordinate is normalized to [0, 1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    """Represents a single bounding box with class label and original defect ID."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    class_id: int
    class_name: str
    original_id: int | str | None = None
    confidence: float = 1.0

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0


@dataclass(frozen=True)
class TransformedBoxResult:
    """Result of transforming a bounding box into a crop frame."""
    box: BoundingBox
    original_box: BoundingBox
    retained: bool
    retention_ratio: float
    exclusion_reason: str | None = None
    is_primary_defect: bool = False


@dataclass(frozen=True)
class CropWindow:
    """Crop window definition with source and padding geometry."""
    src_x1: int
    src_y1: int
    src_x2: int
    src_y2: int
    pad_left: int = 0
    pad_top: int = 0
    pad_right: int = 0
    pad_bottom: int = 0
    target_width: int = 0
    target_height: int = 0

    @property
    def crop_width(self) -> int:
        return (self.src_x2 - self.src_x1) + self.pad_left + self.pad_right

    @property
    def crop_height(self) -> int:
        return (self.src_y2 - self.src_y1) + self.pad_top + self.pad_bottom


def calculate_crop_window(
    center_x: float,
    center_y: float,
    crop_size: int | tuple[int, int],
    image_width: int,
    image_height: int,
    allow_padding: bool = True,
) -> CropWindow:
    """Calculate crop window boundaries safely handling boundaries and oversized crops.

    Handles:
    - Centers crop at (center_x, center_y).
    - If crop_size <= image dimension: clamps to image boundaries [0, W - crop_w] without padding.
    - If crop_size > image dimension: takes entire image dimension and symmetrically pads.
    - If edge/corner defect cannot center without padding and allow_padding is True,
      correct symmetric/asymmetric padding is applied.
    """
    if isinstance(crop_size, int):
        target_w, target_h = crop_size, crop_size
    else:
        target_w, target_h = crop_size

    # Horizontal coordinate calculation
    if target_w <= image_width:
        ideal_x1 = int(round(center_x - target_w / 2.0))
        # Clamp within [0, image_width - target_w]
        x1 = max(0, min(ideal_x1, image_width - target_w))
        x2 = x1 + target_w
        pad_l, pad_r = 0, 0
    else:
        # Crop is wider than image
        x1 = 0
        x2 = image_width
        diff = target_w - image_width
        pad_l = diff // 2
        pad_r = diff - pad_l

    # Vertical coordinate calculation
    if target_h <= image_height:
        ideal_y1 = int(round(center_y - target_h / 2.0))
        y1 = max(0, min(ideal_y1, image_height - target_h))
        y2 = y1 + target_h
        pad_t, pad_b = 0, 0
    else:
        # Crop is taller than image
        y1 = 0
        y2 = image_height
        diff = target_h - image_height
        pad_t = diff // 2
        pad_b = diff - pad_t

    return CropWindow(
        src_x1=x1,
        src_y1=y1,
        src_x2=x2,
        src_y2=y2,
        pad_left=pad_l,
        pad_top=pad_t,
        pad_right=pad_r,
        pad_bottom=pad_b,
        target_width=target_w,
        target_height=target_h,
    )


def transform_box_to_crop(
    box: BoundingBox,
    crop: CropWindow,
    min_retention_ratio: float = 0.30,
    min_clipped_area: float = 9.0,
    is_primary: bool = False,
) -> TransformedBoxResult:
    """Transform bounding box from original image space to crop space with padding and clipping.

    Math:
    1. Translate to crop window:
       x_crop = x_orig - src_x1 + pad_left
       y_crop = y_orig - src_y1 + pad_top
    2. Clip to crop boundaries [0, crop_width] and [0, crop_height]:
       x_clipped1 = max(0, min(x_crop1, crop_width))
    3. Evaluate retention ratio = clipped_area / original_area.
    4. Reject box if retention_ratio < min_retention_ratio or clipped_area < min_clipped_area.
    """
    orig_area = box.area
    if orig_area <= 0:
        return TransformedBoxResult(
            box=box,
            original_box=box,
            retained=False,
            retention_ratio=0.0,
            exclusion_reason="original_zero_area",
            is_primary_defect=is_primary,
        )

    # 1. Translate
    tx1 = box.xmin - crop.src_x1 + crop.pad_left
    ty1 = box.ymin - crop.src_y1 + crop.pad_top
    tx2 = box.xmax - crop.src_x1 + crop.pad_left
    ty2 = box.ymax - crop.src_y1 + crop.pad_top

    # 2. Clip to crop boundaries
    c_w = float(crop.crop_width)
    c_h = float(crop.crop_height)
    cx1 = max(0.0, min(tx1, c_w))
    cy1 = max(0.0, min(ty1, c_h))
    cx2 = max(0.0, min(tx2, c_w))
    cy2 = max(0.0, min(ty2, c_h))

    clipped_w = max(0.0, cx2 - cx1)
    clipped_h = max(0.0, cy2 - cy1)
    clipped_area = clipped_w * clipped_h

    retention_ratio = clipped_area / orig_area

    # Check intersection
    if clipped_area <= 0.0:
        return TransformedBoxResult(
            box=BoundingBox(cx1, cy1, cx2, cy2, box.class_id, box.class_name, box.original_id),
            original_box=box,
            retained=False,
            retention_ratio=0.0,
            exclusion_reason="outside_crop_window",
            is_primary_defect=is_primary,
        )

    # Check retention thresholds
    if retention_ratio < min_retention_ratio:
        return TransformedBoxResult(
            box=BoundingBox(cx1, cy1, cx2, cy2, box.class_id, box.class_name, box.original_id),
            original_box=box,
            retained=False,
            retention_ratio=retention_ratio,
            exclusion_reason=f"low_retention_ratio ({retention_ratio:.3f} < {min_retention_ratio})",
            is_primary_defect=is_primary,
        )

    if clipped_area < min_clipped_area:
        return TransformedBoxResult(
            box=BoundingBox(cx1, cy1, cx2, cy2, box.class_id, box.class_name, box.original_id),
            original_box=box,
            retained=False,
            retention_ratio=retention_ratio,
            exclusion_reason=f"clipped_area_too_small ({clipped_area:.1f} < {min_clipped_area})",
            is_primary_defect=is_primary,
        )

    transformed = BoundingBox(
        xmin=cx1,
        ymin=cy1,
        xmax=cx2,
        ymax=cy2,
        class_id=box.class_id,
        class_name=box.class_name,
        original_id=box.original_id,
        confidence=box.confidence,
    )

    return TransformedBoxResult(
        box=transformed,
        original_box=box,
        retained=True,
        retention_ratio=retention_ratio,
        exclusion_reason=None,
        is_primary_defect=is_primary,
    )


def box_to_yolo_normalized(box: BoundingBox, image_width: float, image_height: float) -> tuple[float, float, float, float]:
    """Convert absolute bounding box to normalized YOLO format (cx, cy, w, h)."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image dimensions: {image_width}x{image_height}")
    cx = (box.xmin + box.xmax) / (2.0 * image_width)
    cy = (box.ymin + box.ymax) / (2.0 * image_height)
    w = (box.xmax - box.xmin) / image_width
    h = (box.ymax - box.ymin) / image_height
    # Clamp to [0, 1]
    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, w)),
        max(0.0, min(1.0, h)),
    )


def yolo_to_box(
    cx: float, cy: float, w: float, h: float,
    class_id: int, class_name: str,
    image_width: float, image_height: float,
    original_id: int | str | None = None,
) -> BoundingBox:
    """Convert normalized YOLO format (cx, cy, w, h) to absolute BoundingBox."""
    box_w = w * image_width
    box_h = h * image_height
    xmin = (cx * image_width) - (box_w / 2.0)
    ymin = (cy * image_height) - (box_h / 2.0)
    xmax = xmin + box_w
    ymax = ymin + box_h
    return BoundingBox(
        xmin=max(0.0, xmin),
        ymin=max(0.0, ymin),
        xmax=min(image_width, xmax),
        ymax=min(image_height, ymax),
        class_id=class_id,
        class_name=class_name,
        original_id=original_id,
    )


def letterbox_calculate(
    original_width: int,
    original_height: int,
    target_width: int,
    target_height: int,
) -> tuple[float, int, int, int, int]:
    """Calculate scale factor and padding offsets for aspect-ratio-preserving letterboxing.

    Returns:
    (scale, pad_left, pad_top, new_unpadded_width, new_unpadded_height)
    """
    scale = min(target_width / original_width, target_height / original_height)
    new_w = int(round(original_width * scale))
    new_h = int(round(original_height * scale))
    pad_w = target_width - new_w
    pad_h = target_height - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    return scale, pad_left, pad_top, new_w, new_h


def transform_box_letterbox(
    box: BoundingBox,
    scale: float,
    pad_left: int,
    pad_top: int,
    target_width: int,
    target_height: int,
) -> BoundingBox:
    """Transform bounding box coordinates under aspect-preserving letterboxing."""
    xmin = (box.xmin * scale) + pad_left
    ymin = (box.ymin * scale) + pad_top
    xmax = (box.xmax * scale) + pad_left
    ymax = (box.ymax * scale) + pad_top
    return BoundingBox(
        xmin=max(0.0, min(float(target_width), xmin)),
        ymin=max(0.0, min(float(target_height), ymin)),
        xmax=max(0.0, min(float(target_width), xmax)),
        ymax=max(0.0, min(float(target_height), ymax)),
        class_id=box.class_id,
        class_name=box.class_name,
        original_id=box.original_id,
        confidence=box.confidence,
    )
