"""High-level orchestration for Phase 3 preprocessing pipeline.

Manages:
- Loading frozen Phase 1 split manifests.
- Loading image and annotation pairs (using LabelMe JSON / YOLO TXT).
- Routing splits (train, val, test) independently.
- Writing preprocessed outputs, crop images, YOLO txt labels, and traceability metadata.
- Generating summary reports.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from src.data.annotation_parsers import parse_labelme_json
from src.data.dataset_paths import DatasetLayout, resolve_layout
from src.data.preprocessing.crop_extractor import ExtractedCrop, extract_crops_from_image
from src.data.preprocessing.crop_geometry import BoundingBox, box_to_yolo_normalized
from src.data.preprocessing.full_image_pipeline import PreprocessedFullImage, preprocess_full_image
from src.data.splits import YOLO_ID_TO_CLASS

CLASS_TO_YOLO_ID = {name: cid for cid, name in YOLO_ID_TO_CLASS.items()}
# Standardize "abnormal thickness" -> "abnormal_thickness"
CLASS_TO_YOLO_ID["abnormal thickness"] = 4


@dataclass
class PreprocessingConfig:
    crop_size: int = 256
    context_margin: float = 0.20
    cluster_distance: float = 128.0
    clustering_mode: str = "cluster"
    min_retention_ratio: float = 0.30
    min_clipped_area: float = 9.0
    full_image_size: int = 640
    enable_crops: bool = True
    enable_full_images: bool = True


def load_boxes_from_labelme(json_path: Path) -> list[BoundingBox]:
    """Parse LabelMe JSON file into a list of BoundingBox objects."""
    parsed = parse_labelme_json(json_path)
    boxes: list[BoundingBox] = []
    for i, b in enumerate(parsed.boxes):
        raw_name = b.class_raw.strip().lower()
        norm_name = raw_name.replace(" ", "_")
        cid = CLASS_TO_YOLO_ID.get(norm_name, CLASS_TO_YOLO_ID.get(raw_name, 5))
        boxes.append(
            BoundingBox(
                xmin=b.xmin,
                ymin=b.ymin,
                xmax=b.xmax,
                ymax=b.ymax,
                class_id=cid,
                class_name=norm_name,
                original_id=i,
            )
        )
    return boxes


def write_yolo_label_file(output_path: Path, boxes: list[BoundingBox], img_w: float, img_h: float) -> None:
    """Write bounding boxes to normalized YOLO TXT format: class_id cx cy w h."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for b in boxes:
        cx, cy, w, h = box_to_yolo_normalized(b, img_w, img_h)
        lines.append(f"{b.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def run_preprocessing_pipeline(
    layout: DatasetLayout,
    manifest_dir: Path,
    output_dir: Path,
    config: PreprocessingConfig,
    sample_limit_per_split: int | None = None,
) -> dict[str, Any]:
    """Execute the Phase 3 preprocessing pipeline across train, val, and test splits.

    Strict Split Discipline:
    - Each split is processed strictly against its Phase 1 manifest.
    - Zero data leakage: crops maintain source stem and group_id provenance.
    - Outputs are written atomically into structured directories.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    patches_dir = output_dir / f"patches_{config.crop_size}"
    full_dir = output_dir / f"full_{config.full_image_size}"
    manifests_out_dir = output_dir / "manifests"
    manifests_out_dir.mkdir(parents=True, exist_ok=True)

    json_dir = layout.json_dir
    orig_img_dir = layout.original_image_dir

    # Pre-index images and json files for high performance O(1) lookups
    json_map = {p.stem: p for p in json_dir.rglob("*.json")}
    img_map = {p.stem: p for p in orig_img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}

    provenance_records: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "processed_splits": {},
        "total_source_images_processed": 0,
        "total_crops_generated": 0,
        "total_full_images_generated": 0,
    }

    for split_name in ("train", "val", "test"):
        split_csv = manifest_dir / f"{split_name}.csv"
        if not split_csv.is_file():
            raise FileNotFoundError(f"Missing frozen manifest: {split_csv}")

        # Read stems and group_ids from frozen manifest
        stems_info: list[tuple[str, str]] = []
        with split_csv.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stems_info.append((row["stem"], row["group_id"]))

        if sample_limit_per_split is not None:
            stems_info = stems_info[:sample_limit_per_split]

        split_crops_count = 0
        split_full_count = 0
        split_crop_manifest_rows: list[dict[str, Any]] = []

        for stem, group_id in stems_info:
            img_path = img_map.get(stem)
            if img_path is None or not img_path.is_file():
                continue

            ann_path = json_map.get(stem)
            if ann_path is None or not ann_path.is_file():
                continue

            with Image.open(img_path) as pil_img:
                pil_img.load()
                boxes = load_boxes_from_labelme(ann_path)

                # Stream 1: Defect-Centred Crops
                if config.enable_crops:
                    extracted = extract_crops_from_image(
                        image=pil_img,
                        source_stem=stem,
                        group_id=group_id,
                        split=split_name,
                        all_image_boxes=boxes,
                        crop_size=config.crop_size,
                        context_margin=config.context_margin,
                        cluster_distance=config.cluster_distance,
                        clustering_mode=config.clustering_mode,
                        min_retention_ratio=config.min_retention_ratio,
                        min_clipped_area=config.min_clipped_area,
                    )

                    for crop in extracted:
                        split_crops_count += 1
                        crop_img_out = patches_dir / split_name / "images" / f"{crop.crop_id}.jpg"
                        crop_lbl_out = patches_dir / split_name / "labels" / f"{crop.crop_id}.txt"

                        crop_img_out.parent.mkdir(parents=True, exist_ok=True)
                        crop.image.save(crop_img_out, format="JPEG", quality=95)

                        w_c, h_c = crop.image.size
                        write_yolo_label_file(crop_lbl_out, crop.retained_boxes, w_c, h_c)

                        crop_row = {
                            "crop_id": crop.crop_id,
                            "source_stem": stem,
                            "group_id": group_id,
                            "split": split_name,
                            "cluster_id": crop.cluster_id,
                            "crop_size": f"{w_c}x{h_c}",
                            "retained_boxes_count": len(crop.retained_boxes),
                            "excluded_boxes_count": len(crop.excluded_results),
                            "primary_classes": "|".join(crop.primary_classes),
                            "all_classes": "|".join(crop.all_classes),
                        }
                        split_crop_manifest_rows.append(crop_row)
                        provenance_records.append({
                            **crop_row,
                            "crop_window": asdict(crop.crop_window),
                            "excluded_details": [
                                {
                                    "original_box_id": res.original_box.original_id,
                                    "class_name": res.original_box.class_name,
                                    "retention_ratio": res.retention_ratio,
                                    "reason": res.exclusion_reason,
                                }
                                for res in crop.excluded_results
                            ],
                        })

                # Stream 2: Full-Image Resizing & Letterboxing
                if config.enable_full_images:
                    preprocessed_full = preprocess_full_image(
                        image=pil_img,
                        stem=stem,
                        group_id=group_id,
                        split=split_name,
                        boxes=boxes,
                        target_size=config.full_image_size,
                    )
                    split_full_count += 1
                    full_img_out = full_dir / split_name / "images" / f"{stem}.jpg"
                    full_lbl_out = full_dir / split_name / "labels" / f"{stem}.txt"

                    full_img_out.parent.mkdir(parents=True, exist_ok=True)
                    preprocessed_full.image.save(full_img_out, format="JPEG", quality=95)

                    w_f, h_f = preprocessed_full.image.size
                    write_yolo_label_file(full_lbl_out, preprocessed_full.transformed_boxes, w_f, h_f)

        # Write split crop manifest CSV
        if split_crop_manifest_rows:
            csv_out = manifests_out_dir / f"{split_name}_crops.csv"
            keys = list(split_crop_manifest_rows[0].keys())
            with csv_out.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(split_crop_manifest_rows)

        stats["processed_splits"][split_name] = {
            "source_images": len(stems_info),
            "crops_generated": split_crops_count,
            "full_images_generated": split_full_count,
        }
        stats["total_source_images_processed"] += len(stems_info)
        stats["total_crops_generated"] += split_crops_count
        stats["total_full_images_generated"] += split_full_count

    # Save full provenance mapping
    provenance_path = manifests_out_dir / "crop_provenance_mapping.json"
    with provenance_path.open("w", encoding="utf-8") as f:
        json.dump(provenance_records, f, indent=2)

    # Save summary report
    summary_path = output_dir / "preprocessing_summary.json"
    summary_data = {
        "config": asdict(config),
        "statistics": stats,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return summary_data
