"""Pipeline B: Full-image detector preprocessing for SolDef_AI.

Performs aspect-ratio-preserving letterboxing to 640x640 with padding and exact
coordinate transformations across Train, Validation, and Test partitions.
"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from PIL import Image

from src.data.adapters.soldef import SolDefAdapter
from src.data.preprocessing.crop_geometry import (
    BoundingBox,
    box_to_yolo_normalized,
    letterbox_calculate,
    transform_box_letterbox,
)
from src.data.preprocessing.soldef_config import SolDefPreprocessingConfig


@dataclass
class SolDefDetectorRecord:
    """Metadata record for a preprocessed full detector image."""
    stem: str
    split: str
    group_id: str
    subset_or_cs: str
    component_type: str
    view: str
    orig_width: int
    orig_height: int
    target_width: int
    target_height: int
    scale: float
    pad_left: int
    pad_top: int
    num_target_boxes: int
    class_counts: dict[str, int]
    processed_image_path: str
    processed_label_path: str


def inverse_transform_letterbox(
    det_box: BoundingBox,
    scale: float,
    pad_left: int,
    pad_top: int,
    orig_width: int = 2560,
    orig_height: int = 1440,
) -> BoundingBox:
    """Transform bounding box from 640x640 letterbox space back to original image space.

    Formula:
        x_orig = (x_letterbox - pad_left) / scale
        y_orig = (y_letterbox - pad_top) / scale
    """
    if scale <= 0:
        raise ValueError(f"Invalid letterbox scale: {scale}")

    oxmin = (det_box.xmin - pad_left) / scale
    oymin = (det_box.ymin - pad_top) / scale
    oxmax = (det_box.xmax - pad_left) / scale
    oymax = (det_box.ymax - pad_top) / scale

    return BoundingBox(
        xmin=max(0.0, min(float(orig_width), oxmin)),
        ymin=max(0.0, min(float(orig_height), oymin)),
        xmax=max(0.0, min(float(orig_width), oxmax)),
        ymax=max(0.0, min(float(orig_height), oymax)),
        class_id=det_box.class_id,
        class_name=det_box.class_name,
        original_id=det_box.original_id,
        confidence=det_box.confidence,
    )


def preprocess_soldef_detector(
    config: SolDefPreprocessingConfig,
    sample_limit_per_split: int | None = None,
) -> tuple[dict[str, list[SolDefDetectorRecord]], dict[str, Any]]:
    """Execute Pipeline B across all configured splits (train, val, test).

    Returns:
        (records_by_split, summary_stats)
    """
    out_base = config.paths.output_base_dir / "detector"
    meta_out_dir = config.paths.output_base_dir / "metadata"
    meta_out_dir.mkdir(parents=True, exist_ok=True)

    adapter = SolDefAdapter(benchmark_type=config.benchmark_name, defect_only=True)
    resample_filter = Image.Resampling.BILINEAR
    if config.detector.resize_resample == "bicubic":
        resample_filter = Image.Resampling.BICUBIC

    all_records_by_split: dict[str, list[SolDefDetectorRecord]] = {}
    split_stats: dict[str, Any] = {}

    target_cls_set = set(config.target_classes)
    target_dim = config.detector.target_size

    for split in config.detector.splits_to_process:
        manifest_csv = config.paths.split_manifest_dir / f"{split}.csv"
        if not manifest_csv.exists():
            continue

        with open(manifest_csv, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        if sample_limit_per_split is not None:
            rows = rows[:sample_limit_per_split]

        split_img_dir = out_base / split / "images"
        split_lbl_dir = out_base / split / "labels"
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        split_records: list[SolDefDetectorRecord] = []
        split_class_counts = Counter()
        total_transformed_boxes = 0

        for row in rows:
            stem = row["stem"]
            group_id = row["group_id"]
            img_path = Path(row["image_path"])
            ann_path = Path(row["annotation_path"])

            if not img_path.exists() or not ann_path.exists():
                continue

            with Image.open(img_path) as src_im:
                src_im = src_im.convert("RGB")
                orig_w, orig_h = src_im.size

                scale, pad_left, pad_top, new_w, new_h = letterbox_calculate(
                    original_width=orig_w,
                    original_height=orig_h,
                    target_width=target_dim,
                    target_height=target_dim,
                )

                # Resize with letterboxing
                resized = src_im.resize((new_w, new_h), resample=resample_filter)
                canvas = Image.new("RGB", (target_dim, target_dim), color=config.detector.pad_color)
                canvas.paste(resized, (pad_left, pad_top))

                # Load target defect boxes
                boxes = adapter.load_annotations(ann_path, defect_only=True)
                boxes = [b for b in boxes if b.class_name in target_cls_set]

                transformed_boxes: list[BoundingBox] = []
                img_class_counts = Counter()
                yolo_lines = []

                for b in boxes:
                    tb = transform_box_letterbox(
                        box=b,
                        scale=scale,
                        pad_left=pad_left,
                        pad_top=pad_top,
                        target_width=target_dim,
                        target_height=target_dim,
                    )
                    transformed_boxes.append(tb)
                    img_class_counts[tb.class_name] += 1
                    split_class_counts[tb.class_name] += 1

                    cid = config.class_to_id[tb.class_name]
                    cx, cy, bw, bh = box_to_yolo_normalized(tb, target_dim, target_dim)
                    yolo_lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                # Save processed letterboxed image
                out_img_path = split_img_dir / f"{stem}.jpg"
                canvas.save(out_img_path, quality=95)

                # Save YOLO label file (even if empty for negative/background images)
                out_lbl_path = split_lbl_dir / f"{stem}.txt"
                out_lbl_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

                total_transformed_boxes += len(transformed_boxes)

                rec = SolDefDetectorRecord(
                    stem=stem,
                    split=split,
                    group_id=group_id,
                    subset_or_cs=row.get("subset_or_cs", ""),
                    component_type=row.get("component_type", ""),
                    view=row.get("view", ""),
                    orig_width=orig_w,
                    orig_height=orig_h,
                    target_width=target_dim,
                    target_height=target_dim,
                    scale=round(scale, 6),
                    pad_left=pad_left,
                    pad_top=pad_top,
                    num_target_boxes=len(transformed_boxes),
                    class_counts=dict(img_class_counts),
                    processed_image_path=str(out_img_path.as_posix()),
                    processed_label_path=str(out_lbl_path.as_posix()),
                )
                split_records.append(rec)

        # Write split manifest CSV
        split_csv = meta_out_dir / f"detector_{split}_manifest.csv"
        if split_records:
            fieldnames = [
                "stem", "split", "group_id", "subset_or_cs", "component_type", "view",
                "orig_width", "orig_height", "target_width", "target_height",
                "scale", "pad_left", "pad_top", "num_target_boxes",
                "exc_solder", "spike", "poor_solder",
                "processed_image_path", "processed_label_path"
            ]
            with open(split_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in split_records:
                    row_dict = asdict(r)
                    row_dict.pop("class_counts", None)
                    for c in config.target_classes:
                        row_dict[c] = r.class_counts.get(c, 0)
                    writer.writerow(row_dict)

        all_records_by_split[split] = split_records
        split_stats[split] = {
            "image_count": len(split_records),
            "target_box_count": total_transformed_boxes,
            "class_counts": dict(split_class_counts),
            "scale_factor": 0.25,
            "pad_top_bottom": 140,
            "pad_left_right": 0,
        }

    summary = {
        "target_size": f"{target_dim}x{target_dim}",
        "aspect_ratio_preservation": "letterbox_padding",
        "pad_color": list(config.detector.pad_color),
        "split_summaries": split_stats,
    }

    return all_records_by_split, summary
