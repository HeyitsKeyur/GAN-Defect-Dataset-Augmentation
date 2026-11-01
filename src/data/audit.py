"""Phase 0 Ceramic-Defect audit: counts, validation, duplicates, visualizations."""

from __future__ import annotations

import csv
import json
import platform
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from src.data.annotation_parsers import ParsedAnnotation, parse_labelme_json, parse_voc_xml, parse_yolo_txt
from src.data.box_validation import coco_size_bin, box_area
from src.data.dataset_paths import (
    IMAGE_EXTENSIONS,
    DatasetLayout,
    index_by_stem,
    iter_files,
    resolve_layout,
)
from src.data.image_utils import (
    counter_to_dict,
    file_md5,
    filename_prefix_bucket,
    hamming_distance,
    infer_group_id,
    inspect_image,
    summarize_counts,
)

NEAR_DUP_HAMMING_MAX = 5
VIS_MAX_SIDE = 1024
ISSUE_SAMPLE_LIMIT = 12
CLASS_SAMPLE_LIMIT = 2


@dataclass
class IssueRecord:
    stem: str
    source: str
    code: str
    message: str


def run_audit(
    data_root: Path,
    output_dir: Path,
    *,
    max_visualizations: int = 40,
    seed: int = 42,
) -> dict[str, Any]:
    layout = resolve_layout(data_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    orig_images = iter_files(layout.original_image_dir, IMAGE_EXTENSIONS)
    split_images = iter_files(layout.split_image_dir, IMAGE_EXTENSIONS)
    json_files = iter_files(layout.json_dir, {".json"})
    xml_files = iter_files(layout.xml_dir, {".xml"})
    txt_files = iter_files(layout.txt_dir, {".txt"})

    orig_index = index_by_stem(orig_images)
    split_index = index_by_stem(split_images)
    json_index = index_by_stem(json_files)
    xml_index = index_by_stem(xml_files)
    txt_index = index_by_stem(txt_files)

    all_stems = sorted(
        set(orig_index)
        | set(split_index)
        | set(json_index)
        | set(xml_index)
        | set(txt_index)
    )

    issues: list[IssueRecord] = []
    json_label_counts: Counter[str] = Counter()
    xml_name_counts: Counter[str] = Counter()
    yolo_id_counts: Counter[str] = Counter()
    json_labels_per_image: dict[str, set[str]] = {}
    xml_names_per_image: dict[str, set[str]] = {}
    box_widths: list[float] = []
    box_heights: list[float] = []
    box_areas: list[float] = []
    rel_areas: list[float] = []
    size_bins: Counter[str] = Counter()
    json_shape_types: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    prefix_counts: Counter[str] = Counter()
    group_members: dict[str, list[str]] = defaultdict(list)
    group_evidence: dict[str, str] = {}
    image_widths: list[int] = []
    image_heights: list[int] = []
    aspect_ratios: list[float] = []
    image_formats: Counter[str] = Counter()
    orig_md5: dict[str, str] = {}
    split_md5: dict[str, str] = {}
    orig_ahash: dict[str, int] = {}
    annotated_stems: set[str] = set()
    unreadable: list[dict[str, str]] = []
    unsupported: list[str] = []
    dim_mismatch: list[dict[str, Any]] = []
    empty_ann: list[str] = []
    format_count_mismatch: list[dict[str, Any]] = []
    duplicate_ann_entries: list[dict[str, Any]] = []
    xml_json_pair_counts: Counter[tuple[str, str]] = Counter()
    yolo_json_pair_counts: Counter[tuple[str, str]] = Counter()
    vendor_split_of: dict[str, str] = {}
    class_image_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    small_examples: list[dict[str, Any]] = []
    large_examples: list[dict[str, Any]] = []
    issue_examples: list[dict[str, Any]] = []
    orig_vs_split_hash_mismatch = 0
    orig_vs_split_compared = 0
    boxes_by_class: dict[str, list[dict[str, float]]] = defaultdict(list)

    total_stems = len(all_stems)
    for stem_index, stem in enumerate(all_stems, start=1):
        if stem_index == 1 or stem_index % 100 == 0 or stem_index == total_stems:
            print(f"Auditing {stem_index}/{total_stems}: {stem}", flush=True)
        orig_paths = orig_index.get(stem, [])
        split_paths = split_index.get(stem, [])
        json_paths = json_index.get(stem, [])
        xml_paths = xml_index.get(stem, [])
        txt_paths = txt_index.get(stem, [])

        if len(orig_paths) > 1:
            issues.append(IssueRecord(stem, "original_image", "duplicate_stem", "Multiple original images share this stem"))
        if len(split_paths) > 1:
            issues.append(IssueRecord(stem, "split_image", "duplicate_stem", "Multiple split-folder images share this stem"))
        if not orig_paths and not split_paths:
            issues.append(IssueRecord(stem, "image", "missing_image", "Annotation exists without an image file"))
        if orig_paths and not split_paths:
            issues.append(IssueRecord(stem, "split_image", "missing_image", "Present in original image/ but missing from image/{split}"))
        if split_paths and not orig_paths:
            issues.append(IssueRecord(stem, "original_image", "missing_image", "Present in image/{split} but missing from original image/"))

        image_path = orig_paths[0] if orig_paths else (split_paths[0] if split_paths else None)
        image_info = inspect_image(image_path) if image_path is not None else None
        actual_w = image_info.width if image_info else None
        actual_h = image_info.height if image_info else None

        if image_info is not None:
            if not image_info.readable:
                unreadable.append({"stem": stem, "path": str(image_info.path), "error": image_info.error or ""})
                issues.append(IssueRecord(stem, "image", "unreadable_image", image_info.error or "unreadable"))
            if image_info.unsupported_format:
                unsupported.append(str(image_info.path))
            if image_info.readable and actual_w and actual_h:
                image_widths.append(actual_w)
                image_heights.append(actual_h)
                aspect_ratios.append(actual_w / actual_h if actual_h else 0.0)
                image_formats[image_info.format_name or "unknown"] += 1
                if image_info.md5:
                    orig_md5[stem] = image_info.md5
                if image_info.average_hash is not None:
                    orig_ahash[stem] = image_info.average_hash

        if orig_paths and split_paths:
            orig_vs_split_compared += 1
            o_hash = image_info.md5 if image_info and orig_paths and image_info.path == orig_paths[0] else file_md5(orig_paths[0])
            s_hash = file_md5(split_paths[0])
            split_md5[stem] = s_hash
            if o_hash != s_hash:
                orig_vs_split_hash_mismatch += 1
                issues.append(
                    IssueRecord(
                        stem,
                        "copy_consistency",
                        "orig_split_hash_mismatch",
                        "original image/ and image/{split} files differ",
                    )
                )

        split_name = _vendor_split(split_paths, layout)
        if split_name:
            split_counts[split_name] += 1
            vendor_split_of[stem] = split_name

        prefix_counts[filename_prefix_bucket(stem)] += 1
        group_id, evidence = infer_group_id(stem)
        group_members[group_id].append(stem)
        group_evidence[group_id] = evidence

        json_ann = parse_labelme_json(json_paths[0]) if json_paths else None
        xml_ann = parse_voc_xml(xml_paths[0]) if xml_paths else None
        yolo_ann = (
            parse_yolo_txt(txt_paths[0], actual_w, actual_h)
            if txt_paths
            else None
        )

        if not json_paths:
            issues.append(IssueRecord(stem, "json", "missing_annotation", "No LabelMe JSON for this stem"))
        if not xml_paths:
            issues.append(IssueRecord(stem, "xml", "missing_annotation", "No VOC XML for this stem"))
        if not txt_paths:
            issues.append(IssueRecord(stem, "txt", "missing_annotation", "No YOLO TXT for this stem"))

        for source, parsed in (("json", json_ann), ("xml", xml_ann), ("txt", yolo_ann)):
            if parsed is None:
                continue
            for issue in parsed.issues:
                issues.append(IssueRecord(stem, source, issue.code, issue.message))
                if issue.code in {"malformed_annotation"} or not parsed.raw_ok:
                    if len(issue_examples) < ISSUE_SAMPLE_LIMIT and image_path is not None:
                        issue_examples.append(
                            {"stem": stem, "image_path": str(image_path), "code": issue.code, "source": source}
                        )

        if json_ann and json_ann.boxes:
            annotated_stems.add(stem)
            labels = [box.class_raw for box in json_ann.boxes]
            json_label_counts.update(labels)
            json_labels_per_image[stem] = set(labels)
            for box in json_ann.boxes:
                shape_type = str(box.extra.get("shape_type") or "unknown")
                json_shape_types[shape_type] += 1
                area = box_area(box.xmin, box.ymin, box.xmax, box.ymax)
                width = max(0.0, box.xmax - box.xmin)
                height = max(0.0, box.ymax - box.ymin)
                box_widths.append(width)
                box_heights.append(height)
                box_areas.append(area)
                if actual_w and actual_h and actual_w * actual_h > 0:
                    rel_areas.append(area / float(actual_w * actual_h))
                size_bin = coco_size_bin(area)
                size_bins[size_bin] += 1
                record = {
                    "stem": stem,
                    "class": box.class_raw,
                    "xmin": box.xmin,
                    "ymin": box.ymin,
                    "xmax": box.xmax,
                    "ymax": box.ymax,
                    "width": width,
                    "height": height,
                    "area": area,
                    "size_bin": size_bin,
                    "image_path": str(image_path) if image_path else "",
                }
                boxes_by_class[box.class_raw].append(record)
                if len(class_image_examples[box.class_raw]) < CLASS_SAMPLE_LIMIT:
                    class_image_examples[box.class_raw].append(record)
                if size_bin == "small" and len(small_examples) < 6:
                    small_examples.append(record)
                if size_bin == "large" and len(large_examples) < 6:
                    large_examples.append(record)
            dup_boxes = _duplicate_boxes(json_ann)
            if dup_boxes:
                duplicate_ann_entries.append({"stem": stem, "format": "json", "duplicate_groups": dup_boxes})
                issues.append(
                    IssueRecord(stem, "json", "duplicate_annotation_entry", f"{dup_boxes} duplicate box group(s)")
                )
            if json_ann.recorded_width and json_ann.recorded_height and actual_w and actual_h:
                if json_ann.recorded_width != actual_w or json_ann.recorded_height != actual_h:
                    dim_mismatch.append(
                        {
                            "stem": stem,
                            "source": "json",
                            "recorded": [json_ann.recorded_width, json_ann.recorded_height],
                            "actual": [actual_w, actual_h],
                        }
                    )
                    issues.append(IssueRecord(stem, "json", "dimension_mismatch", "JSON imageWidth/Height differ from pixels"))
        elif json_ann is not None:
            empty_ann.append(stem)

        if xml_ann:
            xml_name_counts.update(box.class_raw for box in xml_ann.boxes)
            xml_names_per_image[stem] = {box.class_raw for box in xml_ann.boxes}
            if xml_ann.recorded_width and xml_ann.recorded_height and actual_w and actual_h:
                if xml_ann.recorded_width != actual_w or xml_ann.recorded_height != actual_h:
                    dim_mismatch.append(
                        {
                            "stem": stem,
                            "source": "xml",
                            "recorded": [xml_ann.recorded_width, xml_ann.recorded_height],
                            "actual": [actual_w, actual_h],
                        }
                    )
                    issues.append(IssueRecord(stem, "xml", "dimension_mismatch", "XML size differs from pixels"))
        if yolo_ann:
            yolo_id_counts.update(str(box.class_id) for box in yolo_ann.boxes)

        counts = {
            "json": len(json_ann.boxes) if json_ann else None,
            "xml": len(xml_ann.boxes) if xml_ann else None,
            "txt": len(yolo_ann.boxes) if yolo_ann else None,
        }
        present_counts = [value for value in counts.values() if value is not None]
        if present_counts and len(set(present_counts)) > 1:
            format_count_mismatch.append({"stem": stem, **counts})
            issues.append(IssueRecord(stem, "cross_format", "instance_count_mismatch", str(counts)))

        if json_ann and xml_ann and len(json_ann.boxes) == len(xml_ann.boxes):
            for jbox, xbox in zip(json_ann.boxes, xml_ann.boxes, strict=False):
                xml_json_pair_counts[(xbox.class_raw, jbox.class_raw)] += 1
        if json_ann and yolo_ann and len(json_ann.boxes) == len(yolo_ann.boxes):
            for jbox, ybox in zip(json_ann.boxes, yolo_ann.boxes, strict=False):
                yolo_json_pair_counts[(str(ybox.class_id), jbox.class_raw)] += 1

        if json_ann and image_path and json_ann.issues:
            severe = {item.code for item in json_ann.issues} & {
                "zero_area",
                "out_of_bounds",
                "exceeds_image",
                "negative_extent",
                "malformed_annotation",
            }
            if severe and len(issue_examples) < ISSUE_SAMPLE_LIMIT:
                issue_examples.append(
                    {
                        "stem": stem,
                        "image_path": str(image_path),
                        "code": ",".join(sorted(severe)),
                        "source": "json",
                    }
                )

    xml_to_json = _dominant_mapping(xml_json_pair_counts)
    yolo_to_json = _dominant_mapping(yolo_json_pair_counts)

    exact_dups = _group_by_value(orig_md5)
    exact_dup_groups = {digest: stems for digest, stems in exact_dups.items() if len(stems) > 1}
    near_dups = _near_duplicate_groups(orig_ahash, orig_md5)

    per_class_image_counts = {
        name: sum(1 for labels in json_labels_per_image.values() if name in labels)
        for name in sorted(json_label_counts)
    }
    multiclass_images = sum(1 for labels in json_labels_per_image.values() if len(labels) > 1)
    empty_json_images = [stem for stem in all_stems if stem not in json_labels_per_image]

    grouped_specimens = {
        group: members
        for group, members in group_members.items()
        if len(members) > 1 and group_evidence.get(group, "").startswith("filename")
    }

    visualizations = _write_visualizations(
        vis_dir=vis_dir,
        class_examples=class_image_examples,
        small_examples=small_examples,
        large_examples=large_examples,
        issue_examples=issue_examples,
        json_index=json_index,
        max_visualizations=max_visualizations,
    )

    issue_counter = Counter(item.code for item in issues)
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "pillow": _package_version("PIL"),
        },
        "paths": {
            "data_root": str(layout.data_root),
            "original_image_dir": str(layout.original_image_dir),
            "split_image_dir": str(layout.split_image_dir),
            "json_dir": str(layout.json_dir),
            "xml_dir": str(layout.xml_dir),
            "txt_dir": str(layout.txt_dir),
            "output_dir": str(output_dir),
        },
        "counting_method": (
            "Image counts: unique filename stems under original image/ and image/{train,val,test}, "
            "extensions in IMAGE_EXTENSIONS. Annotation file counts: recursive files with "
            ".json/.xml/.txt under lables/. Instance counts: objects/shapes/lines parsed from "
            "each file. JSON LabelMe labels are the semantic class names used for per-class "
            "instance and image-containing-class counts. Image counts and instance counts are "
            "computed separately. Vendor train/val/test folders were recorded but not modified."
        ),
        "file_counts": {
            "original_images": len(orig_images),
            "split_images": len(split_images),
            "unique_stems": len(all_stems),
            "json_files": len(json_files),
            "xml_files": len(xml_files),
            "txt_files": len(txt_files),
            "annotated_images_json_nonempty": len(annotated_stems),
            "images_without_json_objects": len(empty_json_images),
            "vendor_split": dict(split_counts),
        },
        "annotation_formats": {
            "labelme_json": {
                "description": "LabelMe-style JSON with shapes[].points rectangles",
                "shape_types": counter_to_dict(json_shape_types),
            },
            "pascal_voc_xml": {
                "description": "Pascal VOC XML with object/bndbox xmin ymin xmax ymax",
            },
            "yolo_txt": {
                "description": "YOLO TXT class_id cx cy w h normalized to image size",
            },
        },
        "class_mapping_observed": {
            "note": (
                "Derived by pairing objects in file order when JSON/XML/TXT instance counts "
                "matched. This is observed in this archive, not an official codebook file."
            ),
            "xml_name_to_json_label": xml_to_json,
            "yolo_id_to_json_label": yolo_to_json,
            "xml_json_pair_counts": {f"{a} -> {b}": n for (a, b), n in xml_json_pair_counts.most_common()},
            "yolo_json_pair_counts": {f"{a} -> {b}": n for (a, b), n in yolo_json_pair_counts.most_common()},
        },
        "per_class": {
            "json_instance_counts": counter_to_dict(json_label_counts),
            "xml_name_instance_counts": counter_to_dict(xml_name_counts),
            "yolo_id_instance_counts": counter_to_dict(yolo_id_counts),
            "json_image_counts_containing_class": per_class_image_counts,
            "multiclass_images": multiclass_images,
            "instance_total_json": int(sum(json_label_counts.values())),
            "imbalance": _imbalance(json_label_counts),
        },
        "image_analysis": {
            "formats": counter_to_dict(image_formats),
            "width": summarize_counts(image_widths),
            "height": summarize_counts(image_heights),
            "aspect_ratio_width_over_height": summarize_counts(aspect_ratios),
            "resolution_histogram": _resolution_histogram(image_widths, image_heights),
            "unreadable_images": unreadable,
            "unsupported_formats": unsupported,
            "filename_prefix_buckets": counter_to_dict(prefix_counts),
        },
        "bounding_box_statistics": {
            "source": "labelme_json rectangles converted to axis-aligned xyxy",
            "size_bins_coco_pixel_area": {
                "definition": {
                    "small": f"area < {32*32}",
                    "medium": f"{32*32} <= area < {96*96}",
                    "large": f"area >= {96*96}",
                },
                "counts": counter_to_dict(size_bins),
            },
            "width_px": summarize_counts(box_widths),
            "height_px": summarize_counts(box_heights),
            "area_px": summarize_counts(box_areas),
            "relative_area": summarize_counts(rel_areas),
            "per_class": {
                name: {
                    "instances": len(items),
                    "width_px": summarize_counts([item["width"] for item in items]),
                    "height_px": summarize_counts([item["height"] for item in items]),
                    "area_px": summarize_counts([item["area"] for item in items]),
                    "size_bins": counter_to_dict(Counter(item["size_bin"] for item in items)),
                }
                for name, items in boxes_by_class.items()
            },
        },
        "validation": {
            "issue_counts": counter_to_dict(issue_counter),
            "dimension_mismatches": dim_mismatch,
            "format_instance_count_mismatches": format_count_mismatch,
            "duplicate_annotation_entries": duplicate_ann_entries,
            "images_with_no_json_objects": empty_json_images,
        },
        "duplicates": {
            "near_duplicate_definition": (
                f"Potential near-duplicates: 8x8 average-hash Hamming distance <= {NEAR_DUP_HAMMING_MAX}, "
                "excluding confirmed exact MD5 duplicates. These are candidates only."
            ),
            "confirmed_exact_md5_duplicate_groups": {
                digest: stems for digest, stems in exact_dup_groups.items()
            },
            "confirmed_exact_md5_duplicate_group_count": len(exact_dup_groups),
            "potential_near_duplicate_pairs": near_dups,
            "original_vs_split_copies_compared": orig_vs_split_compared,
            "original_vs_split_hash_mismatches": orig_vs_split_hash_mismatch,
            "note_on_two_image_folders": (
                "original image/ and image/{{train,val,test}} contain the same filename stems. "
                "Identical MD5 hashes mean two copies of the same file, not two distinct specimens."
            ),
        },
        "grouping": {
            "method": "Filename pattern d_d_dddd_dddd_NG with optional _partN or _full",
            "multi_member_filename_groups": len(grouped_specimens),
            "largest_groups": sorted(
                (
                    {"group_id": gid, "n": len(members), "evidence": group_evidence.get(gid, "")}
                    for gid, members in grouped_specimens.items()
                ),
                key=lambda item: item["n"],
                reverse=True,
            )[:20],
            "vendor_split_vs_group_leakage_candidates": _split_leakage(grouped_specimens, vendor_split_of),
        },
        "license_files_found_in_dataset_tree": _license_search(layout.data_root.parents[2]),
        "visualizations": visualizations,
    }

    _write_json(output_dir / "audit_report.json", report)
    _write_per_class_csv(output_dir / "per_class_statistics.csv", json_label_counts, per_class_image_counts, boxes_by_class)
    _write_image_stats_csv(output_dir / "image_dimension_statistics.csv", image_widths, image_heights, aspect_ratios)
    _write_box_stats_csv(output_dir / "bounding_box_statistics.csv", boxes_by_class)
    _write_issues_csv(output_dir / "annotation_validation_report.csv", issues)
    _write_duplicate_csv(output_dir / "duplicate_detection_report.csv", exact_dup_groups, near_dups, grouped_specimens)
    return report


def write_markdown_summary(
    report: dict[str, Any],
    output_path: Path,
    *,
    suitability: dict[str, Any],
    license_status: dict[str, Any],
    tests: dict[str, Any],
    commands: list[str],
) -> None:
    file_counts = report["file_counts"]
    per_class = report["per_class"]
    image = report["image_analysis"]
    boxes = report["bounding_box_statistics"]
    validation = report["validation"]
    dups = report["duplicates"]
    mapping = report["class_mapping_observed"]
    lines = [
        "# Phase 0 — Ceramic-Defect Dataset Audit Summary",
        "",
        f"Generated (UTC): {report['generated_at_utc']}",
        "",
        "This report was produced by counting files and parsing annotations in the local archive. "
        "It does not modify original dataset files.",
        "",
        "## Dataset location",
        "",
        f"- Data root: `{report['paths']['data_root']}`",
        f"- Original images: `{report['paths']['original_image_dir']}`",
        f"- Split images: `{report['paths']['split_image_dir']}`",
        f"- Labels (spelling as on disk: `lables`): JSON/XML/TXT under that folder",
        "",
        "## Counting method",
        "",
        report["counting_method"],
        "",
        "## File counts",
        "",
        f"- Original-folder images: **{file_counts['original_images']}**",
        f"- Split-folder images: **{file_counts['split_images']}**",
        f"- Unique stems: **{file_counts['unique_stems']}**",
        f"- JSON files: **{file_counts['json_files']}**",
        f"- XML files: **{file_counts['xml_files']}**",
        f"- TXT files: **{file_counts['txt_files']}**",
        f"- Images with at least one JSON object: **{file_counts['annotated_images_json_nonempty']}**",
        f"- Images with no JSON objects: **{file_counts['images_without_json_objects']}**",
        f"- Vendor split (recorded, not altered): `{file_counts['vendor_split']}`",
        "",
        "## Annotation formats discovered",
        "",
        "- LabelMe-style JSON rectangles (`shapes` with `shape_type=rectangle`).",
        "- Pascal VOC XML (`object/bndbox`).",
        "- YOLO TXT (`class_id cx cy w h`, normalized).",
        "- All three formats share the same filename stems as the images.",
        "",
        "## Observed class mapping",
        "",
        mapping["note"],
        "",
        f"- XML name → JSON label: `{mapping['xml_name_to_json_label']}`",
        f"- YOLO id → JSON label: `{mapping['yolo_id_to_json_label']}`",
        "",
        "## Per-class counts (JSON semantic labels)",
        "",
        "| Class | Images containing class | Bounding-box instances |",
        "|---|---:|---:|",
    ]
    for name, inst in per_class["json_instance_counts"].items():
        img_n = per_class["json_image_counts_containing_class"].get(name, 0)
        lines.append(f"| {name} | {img_n} | {inst} |")
    lines.extend(
        [
            "",
            f"- JSON instance total: **{per_class['instance_total_json']}**",
            f"- Multi-class images: **{per_class['multiclass_images']}**",
            f"- Instance imbalance (max/min): **{per_class['imbalance'].get('max_over_min')}**",
            "",
            "## Image statistics",
            "",
            f"- Formats: `{image['formats']}`",
            f"- Width: `{image['width']}`",
            f"- Height: `{image['height']}`",
            f"- Aspect ratio (W/H): `{image['aspect_ratio_width_over_height']}`",
            f"- Resolution histogram: `{image['resolution_histogram']}`",
            f"- Unreadable images: **{len(image['unreadable_images'])}**",
            f"- Unsupported formats: **{len(image['unsupported_formats'])}**",
            f"- Filename prefix buckets: `{image['filename_prefix_buckets']}`",
            "",
            "## Bounding-box statistics (from JSON)",
            "",
            f"- COCO-style size bins: `{boxes['size_bins_coco_pixel_area']}`",
            f"- Width px: `{boxes['width_px']}`",
            f"- Height px: `{boxes['height_px']}`",
            f"- Area px: `{boxes['area_px']}`",
            f"- Relative area: `{boxes['relative_area']}`",
            "",
            "## Validation issue counts",
            "",
            f"`{validation['issue_counts']}`",
            "",
            f"- Annotation/image dimension mismatches: **{len(validation['dimension_mismatches'])}**",
            f"- Cross-format instance-count mismatches: **{len(validation['format_instance_count_mismatches'])}**",
            f"- Duplicate annotation entry groups: **{len(validation['duplicate_annotation_entries'])}**",
            "",
            "## Duplicates and related views",
            "",
            dups["near_duplicate_definition"],
            "",
            f"- Confirmed exact MD5 duplicate groups (within original image/): **{dups['confirmed_exact_md5_duplicate_group_count']}**",
            f"- Potential near-duplicate pairs: **{len(dups['potential_near_duplicate_pairs'])}**",
            f"- Original vs split copies compared: **{dups['original_vs_split_copies_compared']}**",
            f"- Original vs split hash mismatches: **{dups['original_vs_split_hash_mismatches']}**",
            "",
            dups["note_on_two_image_folders"],
            "",
            f"- Filename groups with multiple members: **{report['grouping']['multi_member_filename_groups']}**",
            f"- Vendor-split leakage candidates (same filename group in more than one vendor split): "
            f"**{len(report['grouping']['vendor_split_vs_group_leakage_candidates'])}**",
            "",
            "## Visualizations",
            "",
        ]
    )
    for item in report.get("visualizations", []):
        lines.append(f"- `{item}`")
    lines.extend(
        [
            "",
            "## License status",
            "",
            f"- Status: **{license_status.get('status', 'Unverified')}**",
            f"- Detail: {license_status.get('detail', '')}",
            "",
            "## Suitability verdict",
            "",
            f"**{suitability.get('verdict')}**",
            "",
            suitability.get("reasoning", ""),
            "",
            "## Tests",
            "",
            f"- Command: `{tests.get('command')}`",
            f"- Result: {tests.get('result')}",
            "",
            "## Commands executed",
            "",
        ]
    )
    for command in commands:
        lines.append(f"- `{command}`")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _vendor_split(paths: list[Path], layout: DatasetLayout) -> str | None:
    for path in paths:
        try:
            rel = path.relative_to(layout.split_image_dir)
        except ValueError:
            continue
        parts = rel.parts
        if parts:
            return parts[0]
    return None


def _duplicate_boxes(parsed: ParsedAnnotation) -> int:
    counter: Counter[tuple] = Counter()
    for box in parsed.boxes:
        key = (
            box.class_raw,
            round(box.xmin, 3),
            round(box.ymin, 3),
            round(box.xmax, 3),
            round(box.ymax, 3),
        )
        counter[key] += 1
    return sum(1 for count in counter.values() if count > 1)


def _dominant_mapping(pair_counts: Counter[tuple[str, str]]) -> dict[str, str]:
    best: dict[str, tuple[str, int]] = {}
    for (left, right), count in pair_counts.items():
        current = best.get(left)
        if current is None or count > current[1]:
            best[left] = (right, count)
    return {left: right for left, (right, _count) in sorted(best.items())}


def _imbalance(counts: Counter[str]) -> dict[str, Any]:
    if not counts:
        return {}
    values = list(counts.values())
    return {
        "min_class": min(counts, key=counts.get),  # type: ignore[arg-type]
        "max_class": max(counts, key=counts.get),  # type: ignore[arg-type]
        "min_instances": min(values),
        "max_instances": max(values),
        "max_over_min": (max(values) / min(values)) if min(values) else None,
        "share_of_largest": max(values) / sum(values) if sum(values) else None,
    }


def _resolution_histogram(widths: list[int], heights: list[int]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for width, height in zip(widths, heights, strict=False):
        counter[f"{width}x{height}"] += 1
    return dict(counter.most_common(20))


def _group_by_value(mapping: dict[str, str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for stem, value in mapping.items():
        grouped[value].append(stem)
    return dict(grouped)


def _near_duplicate_groups(
    hashes: dict[str, int],
    md5s: dict[str, str],
    max_pairs: int = 200,
) -> list[dict[str, Any]]:
    stems = [stem for stem, value in hashes.items() if value is not None]
    pairs: list[dict[str, Any]] = []
    for i, stem_a in enumerate(stems):
        for stem_b in stems[i + 1 :]:
            if md5s.get(stem_a) and md5s.get(stem_a) == md5s.get(stem_b):
                continue
            distance = hamming_distance(hashes[stem_a], hashes[stem_b])
            if distance <= NEAR_DUP_HAMMING_MAX:
                pairs.append({"stem_a": stem_a, "stem_b": stem_b, "hamming": distance, "status": "potential"})
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def _split_leakage(
    grouped_specimens: dict[str, list[str]],
    vendor_split_of: dict[str, str],
) -> list[dict[str, Any]]:
    leaked: list[dict[str, Any]] = []
    for group_id, members in grouped_specimens.items():
        splits = {vendor_split_of.get(stem) for stem in members if vendor_split_of.get(stem)}
        if len(splits) > 1:
            leaked.append({"group_id": group_id, "n_members": len(members), "splits": sorted(splits)})
    leaked.sort(key=lambda item: item["n_members"], reverse=True)
    return leaked[:50]


def _license_search(dataset_top: Path) -> list[str]:
    names = []
    if not dataset_top.exists():
        return names
    for path in dataset_top.rglob("*"):
        if not path.is_file():
            continue
        lowered = path.name.lower()
        if path.suffix.lower() in {".jpg", ".xml", ".json", ".txt"}:
            continue
        if any(token in lowered for token in ("license", "licence", "readme", "citation", "copying")):
            names.append(str(path))
    return names


def _write_visualizations(
    vis_dir: Path,
    class_examples: dict[str, list[dict[str, Any]]],
    small_examples: list[dict[str, Any]],
    large_examples: list[dict[str, Any]],
    issue_examples: list[dict[str, Any]],
    json_index: dict[str, list[Path]],
    max_visualizations: int,
) -> list[str]:
    written: list[str] = []
    planned: list[tuple[str, dict[str, Any]]] = []
    for class_name, records in sorted(class_examples.items()):
        for record in records:
            planned.append((f"class_{_slug(class_name)}", record))
    for record in small_examples:
        planned.append(("small_defect", record))
    for record in large_examples:
        planned.append(("large_defect", record))
    for record in issue_examples:
        planned.append(("issue", record))
    seen_stems: set[tuple[str, str]] = set()
    for tag, record in planned:
        stem = record.get("stem")
        if not stem:
            continue
        key = (tag, stem)
        if key in seen_stems:
            continue
        seen_stems.add(key)
        image_path = Path(record["image_path"]) if record.get("image_path") else None
        if image_path is None or not image_path.is_file():
            continue
        json_paths = json_index.get(stem, [])
        boxes = parse_labelme_json(json_paths[0]).boxes if json_paths else []
        out_name = f"{tag}_{_slug(stem)}.jpg"
        out_path = vis_dir / out_name
        try:
            _render_overlay(image_path, boxes, out_path, highlight=record)
            written.append(str(out_path))
        except OSError:
            continue
        if len(written) >= max_visualizations:
            break
    return written


def _render_overlay(image_path: Path, boxes: list, out_path: Path, highlight: dict[str, Any]) -> None:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        scale = min(1.0, VIS_MAX_SIDE / max(rgb.size))
        if scale < 1.0:
            new_size = (max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale)))
            rgb = rgb.resize(new_size, Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(rgb)
        try:
            font = ImageFont.load_default()
        except OSError:
            font = None
        colors = {
            "spot": (255, 64, 64),
            "scar": (64, 128, 255),
            "adhesion": (64, 200, 64),
            "defect": (255, 180, 0),
            "uneven": (180, 64, 255),
            "abnormal thickness": (0, 200, 200),
        }
        for box in boxes:
            xmin = box.xmin * scale
            ymin = box.ymin * scale
            xmax = box.xmax * scale
            ymax = box.ymax * scale
            color = colors.get(box.class_raw, (255, 255, 255))
            width = 3 if _same_box(box, highlight) else 2
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=width)
            label = box.class_raw
            if font is not None:
                draw.text((xmin + 2, max(0, ymin - 10)), label, fill=color, font=font)
        rgb.save(out_path, format="JPEG", quality=90)


def _same_box(box: Any, highlight: dict[str, Any]) -> bool:
    if "xmin" not in highlight:
        return False
    return (
        abs(box.xmin - float(highlight["xmin"])) < 1.5
        and abs(box.ymin - float(highlight["ymin"])) < 1.5
        and abs(box.xmax - float(highlight["xmax"])) < 1.5
        and abs(box.ymax - float(highlight["ymax"])) < 1.5
    )


def _slug(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value)
    return cleaned.strip("_")[:80]


def _package_version(name: str) -> str | None:
    try:
        module = __import__(name)
        return getattr(module, "__version__", "unknown")
    except Exception:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_per_class_csv(
    path: Path,
    instance_counts: Counter[str],
    image_counts: dict[str, int],
    boxes_by_class: dict[str, list[dict[str, Any]]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "class",
                "image_count_containing_class",
                "instance_count",
                "mean_box_width_px",
                "mean_box_height_px",
                "mean_box_area_px",
                "small",
                "medium",
                "large",
            ]
        )
        for name, inst in sorted(instance_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            items = boxes_by_class.get(name, [])
            bins = Counter(item["size_bin"] for item in items)
            writer.writerow(
                [
                    name,
                    image_counts.get(name, 0),
                    inst,
                    _mean([item["width"] for item in items]),
                    _mean([item["height"] for item in items]),
                    _mean([item["area"] for item in items]),
                    bins.get("small", 0),
                    bins.get("medium", 0),
                    bins.get("large", 0),
                ]
            )


def _write_image_stats_csv(
    path: Path,
    widths: list[int],
    heights: list[int],
    aspects: list[float],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "count", "min", "max", "mean", "median"])
        for name, values in (("width", widths), ("height", heights), ("aspect_ratio", aspects)):
            if not values:
                writer.writerow([name, 0, "", "", "", ""])
                continue
            writer.writerow(
                [
                    name,
                    len(values),
                    min(values),
                    max(values),
                    _mean(values),
                    statistics.median(values),
                ]
            )


def _write_box_stats_csv(path: Path, boxes_by_class: dict[str, list[dict[str, Any]]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "n", "min_w", "max_w", "mean_w", "min_h", "max_h", "mean_h", "min_area", "max_area", "mean_area"])
        for name, items in sorted(boxes_by_class.items()):
            widths = [item["width"] for item in items]
            heights = [item["height"] for item in items]
            areas = [item["area"] for item in items]
            writer.writerow(
                [
                    name,
                    len(items),
                    min(widths) if widths else "",
                    max(widths) if widths else "",
                    _mean(widths),
                    min(heights) if heights else "",
                    max(heights) if heights else "",
                    _mean(heights),
                    min(areas) if areas else "",
                    max(areas) if areas else "",
                    _mean(areas),
                ]
            )


def _write_issues_csv(path: Path, issues: list[IssueRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stem", "source", "code", "message"])
        for item in issues:
            writer.writerow([item.stem, item.source, item.code, item.message])


def _write_duplicate_csv(
    path: Path,
    exact_groups: dict[str, list[str]],
    near_pairs: list[dict[str, Any]],
    grouped_specimens: dict[str, list[str]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "status", "key", "members"])
        for digest, stems in exact_groups.items():
            writer.writerow(["exact_md5", "confirmed", digest, "|".join(stems)])
        for pair in near_pairs:
            writer.writerow(
                [
                    "average_hash",
                    "potential",
                    f"hamming={pair['hamming']}",
                    f"{pair['stem_a']}|{pair['stem_b']}",
                ]
            )
        for group_id, members in grouped_specimens.items():
            writer.writerow(["filename_group", "related_views_possible", group_id, "|".join(members)])


def _mean(values: list[float] | list[int]) -> float | str:
    if not values:
        return ""
    return float(sum(values) / len(values))
