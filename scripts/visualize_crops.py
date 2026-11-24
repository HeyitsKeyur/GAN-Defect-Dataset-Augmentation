#!/usr/bin/env python3
"""Side-by-side Quality Control Visualizer for Preprocessing Validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_paths import resolve_layout
from src.data.preprocessing.crop_extractor import extract_crops_from_image
from src.data.preprocessing.crop_geometry import BoundingBox
from src.data.preprocessing.pipeline import load_boxes_from_labelme
from src.utils.config import get_config

CLASS_COLORS = {
    0: (0, 255, 0),       # adhesion - green
    1: (255, 165, 0),     # scar - orange
    2: (255, 0, 0),       # spot - red
    3: (255, 255, 0),     # uneven - yellow
    4: (147, 112, 219),   # abnormal thickness - purple
    5: (0, 255, 255),     # defect - cyan
}


def draw_boxes(draw: ImageDraw.ImageDraw, boxes: list[BoundingBox], width: int = 2) -> None:
    for b in boxes:
        color = CLASS_COLORS.get(b.class_id, (255, 255, 255))
        draw.rectangle([b.xmin, b.ymin, b.xmax, b.ymax], outline=color, width=width)
        label = f"{b.class_name}:{b.original_id}"
        draw.text((b.xmin + 2, max(0, b.ymin - 12)), label, fill=color)


def generate_qc_figure(
    orig_img: Image.Image,
    orig_boxes: list[BoundingBox],
    stem: str,
    output_path: Path,
    crop_size: int = 256,
) -> None:
    """Generate a multi-panel QC comparison:
    Panel 1: Original image with all defects marked.
    Panel 2: Original image with extracted crop windows highlighted.
    Panel 3: Extracted crops with transformed and clipped bounding boxes.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    w_orig, h_orig = orig_img.size

    # Extract crops
    crops = extract_crops_from_image(
        image=orig_img,
        source_stem=stem,
        group_id=stem,
        split="sample",
        all_image_boxes=orig_boxes,
        crop_size=crop_size,
    )

    # 1. Panel 1: Original with boxes
    p1 = orig_img.copy()
    d1 = ImageDraw.Draw(p1)
    draw_boxes(d1, orig_boxes, width=3)
    p1_thumb = p1.resize((512, 512), resample=Image.Resampling.BILINEAR)

    # 2. Panel 2: Original with crop windows overlaid
    p2 = orig_img.copy()
    d2 = ImageDraw.Draw(p2)
    for c in crops:
        cw = c.crop_window
        d2.rectangle([cw.src_x1, cw.src_y1, cw.src_x2, cw.src_y2], outline=(0, 255, 255), width=4)
        d2.text((cw.src_x1 + 4, cw.src_y1 + 4), c.crop_id, fill=(0, 255, 255))
    p2_thumb = p2.resize((512, 512), resample=Image.Resampling.BILINEAR)

    # 3. Panel 3: First 2 crops
    crop_thumbs: list[Image.Image] = []
    for c in crops[:2]:
        c_img = c.image.copy()
        d_c = ImageDraw.Draw(c_img)
        draw_boxes(d_c, c.retained_boxes, width=2)
        crop_thumbs.append(c_img.resize((256, 256), resample=Image.Resampling.BILINEAR))

    while len(crop_thumbs) < 2:
        blank = Image.new("RGB", (256, 256), color=(40, 40, 40))
        d_b = ImageDraw.Draw(blank)
        d_b.text((20, 120), "No further crops", fill=(200, 200, 200))
        crop_thumbs.append(blank)

    # Combine side-by-side into 1280x512 canvas
    canvas = Image.new("RGB", (1280, 512), color=(20, 20, 20))
    canvas.paste(p1_thumb, (0, 0))
    canvas.paste(p2_thumb, (512, 0))
    canvas.paste(crop_thumbs[0], (1024, 0))
    canvas.paste(crop_thumbs[1], (1024, 256))

    d_canv = ImageDraw.Draw(canvas)
    d_canv.text((10, 10), f"1. Source: {stem} ({w_orig}x{h_orig})", fill=(255, 255, 255))
    d_canv.text((522, 10), f"2. Extracted Windows ({len(crops)} crops)", fill=(255, 255, 255))
    d_canv.text((1034, 10), "3. Crop 1", fill=(255, 255, 255))
    d_canv.text((1034, 266), "4. Crop 2", fill=(255, 255, 255))

    canvas.save(output_path, format="JPEG", quality=95)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Visual Quality-Control panels.")
    parser.add_argument("--samples", type=int, default=5, help="Number of visual QC samples to generate.")
    parser.add_argument("--stems", nargs="*", default=None, help="Explicit list of image stems to generate QC for.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "qc_phase3", help="Destination folder.")
    args = parser.parse_args()

    cfg = get_config()
    layout = resolve_layout(cfg.dataset_root)
    json_dir = layout.json_dir
    orig_img_dir = layout.original_image_dir

    json_files = list(json_dir.rglob("*.json"))
    if not json_files:
        print("Error: No JSON annotations found.", file=sys.stderr)
        return 1

    if args.stems:
        stem_set = set(args.stems)
        json_files = [jf for jf in json_files if jf.stem in stem_set]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated = 0

    print(f"Generating {args.samples} visual quality-control inspection panels...")
    for jf in json_files:
        if generated >= args.samples:
            break
        stem = jf.stem
        img_path = orig_img_dir / f"{stem}.jpg"
        if not img_path.is_file():
            continue

        boxes = load_boxes_from_labelme(jf)
        if not boxes:
            continue

        with Image.open(img_path) as pil_img:
            pil_img = pil_img.convert("RGB")
            out_file = args.output_dir / f"qc_{stem}.jpg"
            generate_qc_figure(pil_img, boxes, stem, out_file)
            print(f"  [QC Generated] {out_file.name}")
            generated += 1

    print(f"\nVisual QC completed. Inspect images in: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
