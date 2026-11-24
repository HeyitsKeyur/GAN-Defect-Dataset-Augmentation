"""Pipeline A: Defect-centered patch extraction for SolDef_AI (GAN preparation).

Extracts 256x256 defect patches strictly from the frozen TRAIN partition.
Preserves complete provenance, handles boundary padding, and retains overlapping annotations.
"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from src.data.adapters.soldef import SolDefAdapter
from src.data.preprocessing.crop_geometry import (
    BoundingBox,
    CropWindow,
    TransformedBoxResult,
    box_to_yolo_normalized,
    calculate_crop_window,
    transform_box_to_crop,
)
from src.data.preprocessing.soldef_config import SolDefPreprocessingConfig


@dataclass
class SolDefPatchRecord:
    """Rich metadata record for an extracted defect patch."""
    crop_id: str
    dataset_name: str
    benchmark_name: str
    source_stem: str
    source_image_path: str
    group_id: str
    split: str
    target_annotation_id: int
    target_class: str
    target_class_id: int
    orig_xmin: float
    orig_ymin: float
    orig_xmax: float
    orig_ymax: float
    orig_width: float
    orig_height: float
    crop_src_x1: int
    crop_src_y1: int
    crop_src_x2: int
    crop_src_y2: int
    crop_extent: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    scale_factor: float
    target_patch_size: int
    resize_method: str
    patch_xmin: float
    patch_ymin: float
    patch_xmax: float
    patch_ymax: float
    patch_width: float
    patch_height: float
    patch_bbox_area_ratio: float
    yolo_cx: float
    yolo_cy: float
    yolo_w: float
    yolo_h: float
    is_boundary_crop: bool
    overlapping_annotations_count: int
    neighbor_classes: list[str]
    extraction_status: str
    qc_status: str


def extract_soldef_patches(
    config: SolDefPreprocessingConfig,
    sample_limit: int | None = None,
) -> tuple[list[SolDefPatchRecord], dict[str, Any]]:
    """Extract defect-centered patches from frozen Train partition.

    Returns:
        (patch_records, summary_stats)
    """
    out_dir = config.paths.output_base_dir / "patches"
    img_out_dir = out_dir / "images"
    lbl_out_dir = out_dir / "labels"
    meta_out_dir = config.paths.output_base_dir / "metadata"

    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)
    meta_out_dir.mkdir(parents=True, exist_ok=True)

    adapter = SolDefAdapter(benchmark_type=config.benchmark_name, defect_only=False)

    train_manifest_csv = config.paths.split_manifest_dir / f"{config.patches.source_split}.csv"
    if not train_manifest_csv.exists():
        raise FileNotFoundError(f"Train split manifest not found at: {train_manifest_csv}")

    with open(train_manifest_csv, "r", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))

    if sample_limit is not None:
        train_rows = train_rows[:sample_limit]

    records: list[SolDefPatchRecord] = []
    class_counts = Counter()
    boundary_crops = 0
    total_eligible = 0

    resample_filter = Image.Resampling.BILINEAR
    if config.patches.resize_resample == "bicubic":
        resample_filter = Image.Resampling.BICUBIC

    target_cls_set = set(config.target_classes)

    for row in train_rows:
        stem = row["stem"]
        group_id = row["group_id"]
        split = config.patches.source_split
        img_path = Path(row["image_path"])
        ann_path = Path(row["annotation_path"])

        if not img_path.exists() or not ann_path.exists():
            continue

        with Image.open(img_path) as src_im:
            src_im = src_im.convert("RGB")
            img_w, img_h = src_im.size

            # Load all boxes from image
            all_boxes = adapter.load_annotations(ann_path, defect_only=False)

            # Filter for target defects
            target_boxes = [b for b in all_boxes if b.class_name in target_cls_set]

            for t_idx, target_box in enumerate(target_boxes):
                total_eligible += 1
                t_cls = target_box.class_name
                t_cid = config.class_to_id.get(t_cls, target_box.class_id)
                t_cx, t_cy = target_box.center
                t_w, t_h = target_box.width, target_box.height

                # Compute crop window geometry
                if config.patches.crop_strategy == "context_crop_resize":
                    margin_w = t_w * (1.0 + 2.0 * config.patches.context_margin)
                    margin_h = t_h * (1.0 + 2.0 * config.patches.context_margin)
                    needed_size = max(config.patches.min_crop_size, int(math.ceil(max(margin_w, margin_h))))
                    # Round up to even number
                    crop_dim = needed_size if needed_size % 2 == 0 else needed_size + 1
                else:
                    crop_dim = config.patches.target_size

                crop_window = calculate_crop_window(
                    center_x=t_cx,
                    center_y=t_cy,
                    crop_size=(crop_dim, crop_dim),
                    image_width=img_w,
                    image_height=img_h,
                    allow_padding=config.patches.allow_padding,
                )

                # Check if boundary was touched or padded
                is_boundary = (
                    crop_window.src_x1 == 0 or crop_window.src_y1 == 0 or
                    crop_window.src_x2 == img_w or crop_window.src_y2 == img_h or
                    crop_window.pad_left > 0 or crop_window.pad_top > 0 or
                    crop_window.pad_right > 0 or crop_window.pad_bottom > 0
                )
                if is_boundary:
                    boundary_crops += 1

                # Slice image
                patch_slice = src_im.crop((
                    crop_window.src_x1,
                    crop_window.src_y1,
                    crop_window.src_x2,
                    crop_window.src_y2,
                ))

                # Apply padding if required
                if (crop_window.pad_left > 0 or crop_window.pad_top > 0 or
                        crop_window.pad_right > 0 or crop_window.pad_bottom > 0):
                    canvas = Image.new(
                        "RGB",
                        (crop_window.crop_width, crop_window.crop_height),
                        color=config.patches.pad_color,
                    )
                    canvas.paste(patch_slice, (crop_window.pad_left, crop_window.pad_top))
                    intermediate_patch = canvas
                else:
                    intermediate_patch = patch_slice

                # Resize to target patch size (256x256)
                target_sz = config.patches.target_size
                if intermediate_patch.size != (target_sz, target_sz):
                    final_patch = intermediate_patch.resize((target_sz, target_sz), resample=resample_filter)
                    scale = target_sz / float(crop_window.crop_width)
                else:
                    final_patch = intermediate_patch
                    scale = 1.0

                # Transform all boxes from source image into patch space
                # 1. Transform into crop window (translation + padding)
                retained_patch_boxes: list[BoundingBox] = []
                neighbor_classes: list[str] = []

                for b in all_boxes:
                    is_primary = (b.original_id == target_box.original_id)
                    # Translate to unscaled crop space
                    tx1 = b.xmin - crop_window.src_x1 + crop_window.pad_left
                    ty1 = b.ymin - crop_window.src_y1 + crop_window.pad_top
                    tx2 = b.xmax - crop_window.src_x1 + crop_window.pad_left
                    ty2 = b.ymax - crop_window.src_y1 + crop_window.pad_top

                    # Scale to final patch size
                    sx1 = max(0.0, min(float(target_sz), tx1 * scale))
                    sy1 = max(0.0, min(float(target_sz), ty1 * scale))
                    sx2 = max(0.0, min(float(target_sz), tx2 * scale))
                    sy2 = max(0.0, min(float(target_sz), ty2 * scale))

                    if sx2 > sx1 and sy2 > sy1:
                        clipped_area = (sx2 - sx1) * (sy2 - sy1)
                        orig_scaled_area = ((tx2 - tx1) * scale) * ((ty2 - ty1) * scale)
                        retention_ratio = clipped_area / max(1e-6, orig_scaled_area)

                        if is_primary or (retention_ratio >= config.patches.min_retention_ratio and
                                          clipped_area >= config.patches.min_clipped_area):
                            b_cid = config.class_to_id.get(b.class_name, b.class_id)
                            retained_box = BoundingBox(
                                xmin=sx1,
                                ymin=sy1,
                                xmax=sx2,
                                ymax=sy2,
                                class_id=b_cid,
                                class_name=b.class_name,
                                original_id=b.original_id,
                            )
                            retained_patch_boxes.append(retained_box)
                            if not is_primary:
                                neighbor_classes.append(b.class_name)

                # Find transformed target box
                primary_patch_box = next(
                    (b for b in retained_patch_boxes if b.original_id == target_box.original_id),
                    retained_patch_boxes[0] if retained_patch_boxes else None
                )
                if primary_patch_box is None:
                    continue

                p_xmin, p_ymin = primary_patch_box.xmin, primary_patch_box.ymin
                p_xmax, p_ymax = primary_patch_box.xmax, primary_patch_box.ymax
                p_w = p_xmax - p_xmin
                p_h = p_ymax - p_ymin
                p_area_ratio = (p_w * p_h) / (float(target_sz) * float(target_sz))

                # Compute normalized YOLO for primary defect
                ycx = (p_xmin + p_xmax) / (2.0 * target_sz)
                ycy = (p_ymin + p_ymax) / (2.0 * target_sz)
                yw = p_w / float(target_sz)
                yh = p_h / float(target_sz)

                crop_id = f"{stem}_d{target_box.original_id:02d}_{t_cls}"

                # Save patch image
                patch_img_path = img_out_dir / f"{crop_id}.jpg"
                final_patch.save(patch_img_path, quality=95)

                # Save YOLO labels for target defect classes present in patch
                patch_lbl_path = lbl_out_dir / f"{crop_id}.txt"
                yolo_lines = []
                for pb in retained_patch_boxes:
                    if pb.class_name in target_cls_set:
                        pcx = (pb.xmin + pb.xmax) / (2.0 * target_sz)
                        pcy = (pb.ymin + pb.ymax) / (2.0 * target_sz)
                        pw = (pb.xmax - pb.xmin) / float(target_sz)
                        ph = (pb.ymax - pb.ymin) / float(target_sz)
                        cid = config.class_to_id[pb.class_name]
                        yolo_lines.append(f"{cid} {pcx:.6f} {pcy:.6f} {pw:.6f} {ph:.6f}")
                patch_lbl_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

                class_counts[t_cls] += 1

                record = SolDefPatchRecord(
                    crop_id=crop_id,
                    dataset_name=config.dataset_name,
                    benchmark_name=config.benchmark_name,
                    source_stem=stem,
                    source_image_path=str(img_path.as_posix()),
                    group_id=group_id,
                    split=split,
                    target_annotation_id=int(target_box.original_id or t_idx),
                    target_class=t_cls,
                    target_class_id=t_cid,
                    orig_xmin=round(target_box.xmin, 2),
                    orig_ymin=round(target_box.ymin, 2),
                    orig_xmax=round(target_box.xmax, 2),
                    orig_ymax=round(target_box.ymax, 2),
                    orig_width=round(t_w, 2),
                    orig_height=round(t_h, 2),
                    crop_src_x1=crop_window.src_x1,
                    crop_src_y1=crop_window.src_y1,
                    crop_src_x2=crop_window.src_x2,
                    crop_src_y2=crop_window.src_y2,
                    crop_extent=crop_dim,
                    pad_left=crop_window.pad_left,
                    pad_top=crop_window.pad_top,
                    pad_right=crop_window.pad_right,
                    pad_bottom=crop_window.pad_bottom,
                    scale_factor=round(scale, 4),
                    target_patch_size=target_sz,
                    resize_method=config.patches.resize_resample,
                    patch_xmin=round(p_xmin, 2),
                    patch_ymin=round(p_ymin, 2),
                    patch_xmax=round(p_xmax, 2),
                    patch_ymax=round(p_ymax, 2),
                    patch_width=round(p_w, 2),
                    patch_height=round(p_h, 2),
                    patch_bbox_area_ratio=round(p_area_ratio, 4),
                    yolo_cx=round(ycx, 6),
                    yolo_cy=round(ycy, 6),
                    yolo_w=round(yw, 6),
                    yolo_h=round(yh, 6),
                    is_boundary_crop=is_boundary,
                    overlapping_annotations_count=len(retained_patch_boxes) - 1,
                    neighbor_classes=sorted(list(set(neighbor_classes))),
                    extraction_status="SUCCESS",
                    qc_status="PENDING_INSPECTION",
                )
                records.append(record)

    # Write manifests
    csv_path = meta_out_dir / "patch_manifest.csv"
    jsonl_path = meta_out_dir / "patch_manifest.jsonl"

    if records:
        fieldnames = list(asdict(records[0]).keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                d = asdict(r)
                d["neighbor_classes"] = ";".join(d["neighbor_classes"])
                writer.writerow(d)

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(asdict(r)) + "\n")

    summary_stats = {
        "total_source_train_images": len(train_rows),
        "total_eligible_target_defects": total_eligible,
        "total_extracted_patches": len(records),
        "patch_counts_by_class": dict(class_counts),
        "unique_parent_images": len(set(r.source_stem for r in records)),
        "unique_grouping_units": len(set(r.group_id for r in records)),
        "boundary_crops_count": boundary_crops,
        "boundary_crops_percentage": round(100.0 * boundary_crops / max(1, len(records)), 2),
        "patch_dimensions": f"{config.patches.target_size}x{config.patches.target_size}",
    }

    return records, summary_stats
