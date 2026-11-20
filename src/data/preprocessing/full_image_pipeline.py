"""Full-image preprocessing stream for baseline detector workflows.

Performs:
- Aspect-ratio-preserving resize with letterbox padding.
- Exact bounding-box coordinate scaling and padding offset adjustments.
- Normalized YOLO export and validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from PIL import Image

from src.data.preprocessing.crop_geometry import (
    BoundingBox,
    letterbox_calculate,
    transform_box_letterbox,
)


@dataclass
class PreprocessedFullImage:
    """Result of preprocessing a full substrate image."""
    stem: str
    group_id: str
    split: str
    original_width: int
    original_height: int
    target_width: int
    target_height: int
    scale: float
    pad_left: int
    pad_top: int
    image: Image.Image
    transformed_boxes: list[BoundingBox] = field(default_factory=list)


def preprocess_full_image(
    image: Image.Image,
    stem: str,
    group_id: str,
    split: str,
    boxes: Sequence[BoundingBox],
    target_size: int | tuple[int, int] = 640,
    pad_color: tuple[int, int, int] = (114, 114, 114),
    resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> PreprocessedFullImage:
    """Preprocess a full image preserving aspect ratio with letterboxing.

    1. Computes scaling factor scale = min(target_w / orig_w, target_h / orig_h).
    2. Resizes original image using high-quality interpolation.
    3. Pastes resized image into centered target canvas.
    4. Adjusts all bounding box coordinates: (x * scale + pad_left, y * scale + pad_top).
    """
    if isinstance(target_size, int):
        target_w, target_h = target_size, target_size
    else:
        target_w, target_h = target_size

    orig_w, orig_h = image.size
    scale, pad_left, pad_top, new_w, new_h = letterbox_calculate(
        original_width=orig_w,
        original_height=orig_h,
        target_width=target_w,
        target_height=target_h,
    )

    # Resize
    resized_patch = image.resize((new_w, new_h), resample=resample)

    # Canvas
    canvas = Image.new("RGB", (target_w, target_h), color=pad_color)
    canvas.paste(resized_patch, (pad_left, pad_top))

    # Transform all boxes
    transformed_boxes = [
        transform_box_letterbox(b, scale, pad_left, pad_top, target_w, target_h)
        for b in boxes
    ]

    return PreprocessedFullImage(
        stem=stem,
        group_id=group_id,
        split=split,
        original_width=orig_w,
        original_height=orig_h,
        target_width=target_w,
        target_height=target_h,
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        image=canvas,
        transformed_boxes=transformed_boxes,
    )
