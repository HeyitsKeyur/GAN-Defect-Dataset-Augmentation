"""Defect-centred crop extraction engine.

Extracts defect crops with:
- Configurable crop dimensions (fixed or context-scaled).
- Safe boundary handling and padding.
- Complete bounding-box tracking (primary defect, retained neighbours, clipped/excluded defects).
- Deterministic crop IDs and full source metadata provenance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from PIL import Image

from src.data.preprocessing.crop_geometry import (
    BoundingBox,
    CropWindow,
    TransformedBoxResult,
    box_to_yolo_normalized,
    calculate_crop_window,
    transform_box_to_crop,
)
from src.data.preprocessing.defect_cluster import DefectCluster, cluster_defects


@dataclass
class ExtractedCrop:
    """Represents a generated crop with transformed annotations and provenance metadata."""
    crop_id: str
    source_stem: str
    group_id: str
    split: str
    cluster_id: int
    crop_window: CropWindow
    image: Image.Image
    retained_boxes: list[BoundingBox] = field(default_factory=list)
    retained_results: list[TransformedBoxResult] = field(default_factory=list)
    excluded_results: list[TransformedBoxResult] = field(default_factory=list)
    primary_classes: list[str] = field(default_factory=list)
    all_classes: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


def extract_crops_from_image(
    image: Image.Image,
    source_stem: str,
    group_id: str,
    split: str,
    all_image_boxes: Sequence[BoundingBox],
    crop_size: int = 256,
    context_margin: float = 0.20,
    cluster_distance: float = 128.0,
    clustering_mode: str = "cluster",  # "individual" | "cluster"
    min_retention_ratio: float = 0.30,
    min_clipped_area: float = 9.0,
    pad_color: tuple[int, int, int] = (114, 114, 114),
) -> list[ExtractedCrop]:
    """Extract all defect-centred crops from an image based on clusters or individual defects.

    Process:
    1. Cluster defects using spatial proximity.
    2. For each cluster:
       a. Calculate cluster center (center_x, center_y).
       b. Determine crop size (accounting for context_margin if cluster extent is large).
       c. Compute CropWindow handling boundaries and padding.
       d. Slice image patch from source and apply padding if required.
       e. Transform ALL boxes from the source image into this crop frame.
       f. Separate into retained vs. excluded boxes with full reason tracking.
       g. Package into ExtractedCrop with deterministic crop_id.
    """
    img_w, img_h = image.size
    clusters = cluster_defects(all_image_boxes, cluster_distance_threshold=cluster_distance, mode=clustering_mode)

    crops: list[ExtractedCrop] = []

    for cluster in clusters:
        # Determine center
        c_x, c_y = cluster.center
        ext_w, ext_h = cluster.extent_dimensions

        # Calculate crop size incorporating context margin if needed
        # If cluster extent exceeds default crop_size, adapt crop size with margin
        margin_w = ext_w * (1.0 + 2.0 * context_margin)
        margin_h = ext_h * (1.0 + 2.0 * context_margin)
        needed_size = max(crop_size, int(math.ceil(max(margin_w, margin_h))))
        # Round up to multiple of 32 for neural net friendliness if adapted
        if needed_size > crop_size:
            target_dim = int(math.ceil(needed_size / 32.0) * 32)
        else:
            target_dim = crop_size

        crop_window = calculate_crop_window(
            center_x=c_x,
            center_y=c_y,
            crop_size=(target_dim, target_dim),
            image_width=img_w,
            image_height=img_h,
            allow_padding=True,
        )

        # Slice image patch
        cropped_patch = image.crop((
            crop_window.src_x1,
            crop_window.src_y1,
            crop_window.src_x2,
            crop_window.src_y2,
        ))

        # Apply padding if crop window extends outside source image
        if (crop_window.pad_left > 0 or crop_window.pad_top > 0 or
                crop_window.pad_right > 0 or crop_window.pad_bottom > 0):
            final_img = Image.new("RGB", (crop_window.crop_width, crop_window.crop_height), color=pad_color)
            final_img.paste(cropped_patch, (crop_window.pad_left, crop_window.pad_top))
        else:
            final_img = cropped_patch

        # Identify which boxes are in this cluster (primary) vs neighbours
        cluster_box_ids = {b.original_id for b in cluster.boxes if b.original_id is not None}

        retained_boxes: list[BoundingBox] = []
        retained_results: list[TransformedBoxResult] = []
        excluded_results: list[TransformedBoxResult] = []

        for box in all_image_boxes:
            is_prim = (box.original_id in cluster_box_ids) or (box in cluster.boxes)
            res = transform_box_to_crop(
                box=box,
                crop=crop_window,
                min_retention_ratio=min_retention_ratio,
                min_clipped_area=min_clipped_area,
                is_primary=is_prim,
            )
            if res.retained:
                retained_boxes.append(res.box)
                retained_results.append(res)
            else:
                excluded_results.append(res)

        # Build unique deterministic crop_id
        crop_id = f"{source_stem}_c{cluster.cluster_id:02d}_{target_dim}"

        primary_classes = sorted({b.class_name for b in cluster.boxes})
        all_classes = sorted({b.class_name for b in retained_boxes})

        crops.append(
            ExtractedCrop(
                crop_id=crop_id,
                source_stem=source_stem,
                group_id=group_id,
                split=split,
                cluster_id=cluster.cluster_id,
                crop_window=crop_window,
                image=final_img,
                retained_boxes=retained_boxes,
                retained_results=retained_results,
                excluded_results=excluded_results,
                primary_classes=primary_classes,
                all_classes=all_classes,
                metadata={
                    "cluster_size": cluster.count,
                    "target_dim": target_dim,
                    "padding_applied": (
                        crop_window.pad_left + crop_window.pad_top +
                        crop_window.pad_right + crop_window.pad_bottom
                    ) > 0,
                    "retained_count": len(retained_boxes),
                    "excluded_count": len(excluded_results),
                },
            )
        )

    return crops
