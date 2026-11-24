"""Visual Quality Control and Metrics Generation for SolDef Preprocessing (Stage D)."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import random
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from src.data.adapters.soldef import SolDefAdapter
from src.data.preprocessing.crop_geometry import BoundingBox

CLASS_COLORS = {
    "exc_solder": (255, 69, 0),    # Red-Orange
    "spike": (30, 144, 255),       # Dodger Blue
    "poor_solder": (255, 215, 0),   # Gold
    "good": (50, 205, 50),         # Lime Green
}


def draw_bounding_box(
    draw: ImageDraw.ImageDraw,
    box: BoundingBox | tuple[float, float, float, float],
    label: str,
    color: tuple[int, int, int],
    width: int = 3,
) -> None:
    """Draw bounding box with text banner on PIL ImageDraw."""
    if isinstance(box, BoundingBox):
        xmin, ymin, xmax, ymax = box.xmin, box.ymin, box.xmax, box.ymax
    else:
        xmin, ymin, xmax, ymax = box

    draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=width)
    banner_h = 16
    draw.rectangle([xmin, max(0.0, ymin - banner_h), min(xmax, xmin + len(label) * 9 + 8), ymin], fill=color)
    draw.text((xmin + 4, max(0.0, ymin - banner_h + 2)), label, fill=(255, 255, 255))


def generate_multipanel_qc_image(
    patch_record: dict[str, Any],
    adapter: SolDefAdapter,
    output_path: Path,
) -> None:
    """Generate a high-detail 4-panel visual QC comparison for a defect patch."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    src_img_path = Path(patch_record["source_image_path"])
    ann_path = Path(src_img_path.parent / f"{patch_record['source_stem']}.json")

    with Image.open(src_img_path) as im:
        orig_img = im.convert("RGB")

    orig_w, orig_h = orig_img.size
    all_boxes = adapter.load_annotations(ann_path, defect_only=False)

    target_ann_id = int(patch_record["target_annotation_id"])
    target_box = next((b for b in all_boxes if b.original_id == target_ann_id), None)
    if target_box is None:
        target_box = BoundingBox(
            float(patch_record["orig_xmin"]),
            float(patch_record["orig_ymin"]),
            float(patch_record["orig_xmax"]),
            float(patch_record["orig_ymax"]),
            class_id=int(patch_record["target_class_id"]),
            class_name=patch_record["target_class"],
            original_id=target_ann_id,
        )

    # Panel 1: Full original image scaled to 640x360 with annotations
    p1 = orig_img.resize((640, 360), resample=Image.Resampling.BILINEAR)
    draw1 = ImageDraw.Draw(p1)
    s_x = 640.0 / orig_w
    s_y = 360.0 / orig_h

    # Draw crop window on Panel 1
    cw_x1 = int(patch_record["crop_src_x1"]) * s_x
    cw_y1 = int(patch_record["crop_src_y1"]) * s_y
    cw_x2 = int(patch_record["crop_src_x2"]) * s_x
    cw_y2 = int(patch_record["crop_src_y2"]) * s_y
    draw1.rectangle([cw_x1, cw_y1, cw_x2, cw_y2], outline=(255, 0, 255), width=2)

    for b in all_boxes:
        c = CLASS_COLORS.get(b.class_name, (200, 200, 200))
        is_target = (b.original_id == target_ann_id)
        w = 3 if is_target else 1
        scaled_box = (b.xmin * s_x, b.ymin * s_y, b.xmax * s_x, b.ymax * s_y)
        draw_bounding_box(draw1, scaled_box, f"#{b.original_id} {b.class_name}", c, width=w)

    # Panel 2: Cropped region on original image
    c_x1 = int(patch_record["crop_src_x1"])
    c_y1 = int(patch_record["crop_src_y1"])
    c_x2 = int(patch_record["crop_src_x2"])
    c_y2 = int(patch_record["crop_src_y2"])
    p2_crop = orig_img.crop((c_x1, c_y1, c_x2, c_y2))
    p2 = p2_crop.resize((256, 256), resample=Image.Resampling.BILINEAR)

    # Panel 3: Extracted patch raw
    patch_img_path = Path("outputs/preprocessing/soldef/phase_d/patches/images") / f"{patch_record['crop_id']}.jpg"
    if patch_img_path.exists():
        p3 = Image.open(patch_img_path).convert("RGB")
    else:
        p3 = p2.copy()

    # Panel 4: Extracted patch with transformed boxes
    p4 = p3.copy()
    draw4 = ImageDraw.Draw(p4)
    # Draw primary target box
    p_box = (
        float(patch_record["patch_xmin"]),
        float(patch_record["patch_ymin"]),
        float(patch_record["patch_xmax"]),
        float(patch_record["patch_ymax"]),
    )
    t_col = CLASS_COLORS.get(patch_record["target_class"], (255, 0, 0))
    draw_bounding_box(draw4, p_box, f"TARGET: {patch_record['target_class']}", t_col, width=2)

    # Assemble composite canvas: (Width: 920, Height: 440)
    canvas = Image.new("RGB", (920, 440), color=(30, 30, 30))
    draw_canvas = ImageDraw.Draw(canvas)

    # Header banner
    banner_text = (
        f"Crop ID: {patch_record['crop_id']} | Class: {patch_record['target_class']} | "
        f"Group: {patch_record['group_id']} | Extent: {patch_record['crop_extent']}px -> 256px | "
        f"Boundary: {patch_record['is_boundary_crop']}"
    )
    draw_canvas.text((15, 10), banner_text, fill=(255, 255, 255))

    # Paste panels
    canvas.paste(p1, (15, 40))
    draw_canvas.text((15, 405), "Panel 1: Original Image (2560x1440) with Crop Window (Magenta)", fill=(200, 200, 200))

    canvas.paste(p4, (665, 40))
    draw_canvas.text((665, 305), "Panel 2: Extracted 256x256 Patch", fill=(200, 200, 200))
    detail_text = (
        f"Target Box: [{patch_record['patch_xmin']}, {patch_record['patch_ymin']}, "
        f"{patch_record['patch_xmax']}, {patch_record['patch_ymax']}]\n"
        f"Area Ratio: {float(patch_record['patch_bbox_area_ratio'])*100:.1f}%\n"
        f"YOLO (cx,cy,w,h): {patch_record['yolo_cx']}, {patch_record['yolo_cy']}, "
        f"{patch_record['yolo_w']}, {patch_record['yolo_h']}"
    )
    draw_canvas.text((665, 330), detail_text, fill=(180, 180, 180))

    canvas.save(output_path, quality=95)


def run_visual_qc(
    manifest_csv: Path,
    output_dir: Path,
    num_samples: int = 12,
    seed: int = 0,
) -> list[str]:
    """Run stratified visual QC sampling covering all defect types, boundaries, and overlaps."""
    if not manifest_csv.exists():
        return []

    with open(manifest_csv, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    adapter = SolDefAdapter(benchmark_type="primary_side_view", defect_only=False)

    # Stratify selection to cover:
    # 1. exc_solder
    # 2. spike
    # 3. poor_solder
    # 4. overlapping spike
    # 5. boundary crop
    # 6. small bbox
    # 7. large bbox
    selected: list[dict[str, Any]] = []

    # Category 1: Overlapping spikes
    overlapping = [r for r in records if r["target_class"] == "spike" and int(r["overlapping_annotations_count"]) > 0]
    if overlapping:
        selected.append(overlapping[0])

    # Category 2: Minority class poor_solder
    poor = [r for r in records if r["target_class"] == "poor_solder"]
    if poor:
        selected.append(poor[0])
        if len(poor) > 1:
            selected.append(poor[len(poor) // 2])

    # Category 3: Boundary crops
    boundary = [r for r in records if r["is_boundary_crop"] == "True"]
    if boundary:
        selected.append(boundary[0])

    # Category 4: Small bbox (spike or other)
    sorted_by_area = sorted(records, key=lambda x: float(x["orig_width"]) * float(x["orig_height"]))
    selected.append(sorted_by_area[0])  # smallest

    # Category 5: Large bbox
    selected.append(sorted_by_area[-1])  # largest

    # Category 6: exc_solder
    exc = [r for r in records if r["target_class"] == "exc_solder"]
    if exc:
        selected.append(exc[0])
        if len(exc) > 1:
            selected.append(exc[len(exc) // 2])

    # Fill remaining from diverse groups
    rng = random.Random(seed)
    remaining = [r for r in records if r["crop_id"] not in {s["crop_id"] for s in selected}]
    rng.shuffle(remaining)
    while len(selected) < num_samples and remaining:
        selected.append(remaining.pop())

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_paths = []

    for idx, rec in enumerate(selected):
        out_name = f"qc_{idx+1:02d}_{rec['target_class']}_{rec['crop_id']}.png"
        out_path = output_dir / out_name
        generate_multipanel_qc_image(rec, adapter, out_path)
        generated_paths.append(str(out_path.as_posix()))

    return generated_paths
