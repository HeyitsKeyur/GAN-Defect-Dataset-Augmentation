"""Parsers for annotation formats discovered in the Ceramic-Defect archive.

Discovered formats (do not treat this as a claim that only these exist elsewhere):
- LabelMe-style JSON (version field observed as 2.4.4) with rectangular shapes.
- Pascal VOC XML with <object><bndbox> entries.
- YOLO TXT lines: class_id cx cy w h, values normalized to image size.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from src.data.box_validation import (
    BoxIssue,
    parse_float,
    validate_xyxy,
    validate_yolo_normalized,
    xyxy_from_points,
    yolo_to_xyxy,
)


@dataclass
class ParsedBox:
    class_raw: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    source_format: str
    class_id: int | None = None
    extra: dict[str, object] = field(default_factory=dict)


@dataclass
class ParsedAnnotation:
    path: Path
    format_name: str
    stem: str
    boxes: list[ParsedBox] = field(default_factory=list)
    issues: list[BoxIssue] = field(default_factory=list)
    image_filename: str | None = None
    recorded_width: int | None = None
    recorded_height: int | None = None
    raw_ok: bool = True


def parse_labelme_json(path: Path) -> ParsedAnnotation:
    result = ParsedAnnotation(path=path, format_name="labelme_json", stem=path.stem)
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        result.raw_ok = False
        result.issues.append(BoxIssue("malformed_annotation", f"Invalid JSON: {exc}"))
        return result
    except OSError as exc:
        result.raw_ok = False
        result.issues.append(BoxIssue("malformed_annotation", f"Cannot read JSON: {exc}"))
        return result
    if not isinstance(data, dict):
        result.raw_ok = False
        result.issues.append(BoxIssue("malformed_annotation", "JSON root is not an object"))
        return result

    result.image_filename = data.get("imagePath") if isinstance(data.get("imagePath"), str) else None
    result.recorded_width = _as_int(data.get("imageWidth"))
    result.recorded_height = _as_int(data.get("imageHeight"))
    shapes = data.get("shapes", [])
    if not isinstance(shapes, list):
        result.issues.append(BoxIssue("malformed_annotation", "JSON 'shapes' is not a list"))
        return result
    if not shapes:
        result.issues.append(BoxIssue("missing_objects", "JSON has no shapes"))

    for index, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            result.issues.append(BoxIssue("malformed_annotation", f"Shape {index} is not an object"))
            continue
        label = shape.get("label")
        if label is None or str(label).strip() == "":
            result.issues.append(BoxIssue("invalid_class_id", f"Shape {index} has no label"))
            class_raw = ""
        else:
            class_raw = str(label)
        points = shape.get("points")
        if not isinstance(points, list) or len(points) < 2:
            result.issues.append(
                BoxIssue("missing_coordinate", f"Shape {index} has insufficient points")
            )
            continue
        try:
            parsed_points = [[float(p[0]), float(p[1])] for p in points]
        except (TypeError, ValueError, IndexError):
            result.issues.append(
                BoxIssue("invalid_coordinate", f"Shape {index} has non-numeric points")
            )
            continue
        xmin, ymin, xmax, ymax = xyxy_from_points(parsed_points)
        result.boxes.append(
            ParsedBox(
                class_raw=class_raw,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                source_format="labelme_json",
                extra={
                    "shape_type": shape.get("shape_type"),
                    "n_points": len(parsed_points),
                },
            )
        )
        result.issues.extend(
            validate_xyxy(
                xmin,
                ymin,
                xmax,
                ymax,
                result.recorded_width,
                result.recorded_height,
            )
        )
    return result


def parse_voc_xml(path: Path) -> ParsedAnnotation:
    result = ParsedAnnotation(path=path, format_name="voc_xml", stem=path.stem)
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError as exc:
        result.raw_ok = False
        result.issues.append(BoxIssue("malformed_annotation", f"Invalid XML: {exc}"))
        return result
    except OSError as exc:
        result.raw_ok = False
        result.issues.append(BoxIssue("malformed_annotation", f"Cannot read XML: {exc}"))
        return result

    result.image_filename = root.findtext("filename")
    size = root.find("size")
    if size is not None:
        result.recorded_width = _as_int(size.findtext("width"))
        result.recorded_height = _as_int(size.findtext("height"))
    objects = list(root.findall("object"))
    if not objects:
        result.issues.append(BoxIssue("missing_objects", "XML has no object entries"))
    for index, obj in enumerate(objects):
        name = obj.findtext("name")
        if name is None or name.strip() == "":
            result.issues.append(BoxIssue("invalid_class_id", f"Object {index} has no name"))
            class_raw = ""
        else:
            class_raw = name
        bnd = obj.find("bndbox")
        if bnd is None:
            result.issues.append(BoxIssue("missing_coordinate", f"Object {index} has no bndbox"))
            continue
        coords: dict[str, float] = {}
        missing = False
        for field in ("xmin", "ymin", "xmax", "ymax"):
            value, issue = parse_float(bnd.findtext(field), field)
            if issue is not None:
                result.issues.append(issue)
                missing = True
            else:
                assert value is not None
                coords[field] = value
        if missing:
            continue
        result.boxes.append(
            ParsedBox(
                class_raw=class_raw,
                xmin=coords["xmin"],
                ymin=coords["ymin"],
                xmax=coords["xmax"],
                ymax=coords["ymax"],
                source_format="voc_xml",
            )
        )
        result.issues.extend(
            validate_xyxy(
                coords["xmin"],
                coords["ymin"],
                coords["xmax"],
                coords["ymax"],
                result.recorded_width,
                result.recorded_height,
            )
        )
    return result


def parse_yolo_txt(
    path: Path,
    image_width: int | None,
    image_height: int | None,
    allowed_class_ids: set[int] | None = None,
) -> ParsedAnnotation:
    result = ParsedAnnotation(path=path, format_name="yolo_txt", stem=path.stem)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        result.raw_ok = False
        result.issues.append(BoxIssue("malformed_annotation", f"Cannot read TXT: {exc}"))
        return result
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        result.issues.append(BoxIssue("missing_objects", "YOLO file has no object lines"))
    if image_width is None or image_height is None:
        result.issues.append(
            BoxIssue(
                "missing_image_size",
                "YOLO boxes cannot be converted to pixels without image size",
            )
        )
    for index, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 5:
            result.issues.append(
                BoxIssue("malformed_annotation", f"Line {index} does not have 5 fields: {line!r}")
            )
            continue
        class_token, cx_s, cy_s, w_s, h_s = parts[:5]
        parsed_vals: list[float] = []
        bad = False
        for field, token in (("cx", cx_s), ("cy", cy_s), ("width", w_s), ("height", h_s)):
            value, issue = parse_float(token, field)
            if issue is not None:
                result.issues.append(issue)
                bad = True
            else:
                assert value is not None
                parsed_vals.append(value)
        if bad:
            continue
        cx, cy, width, height = parsed_vals
        class_id: int | None
        try:
            class_id = int(class_token)
            class_raw = str(class_id)
        except ValueError:
            class_id = None
            class_raw = class_token
        result.issues.extend(
            validate_yolo_normalized(
                class_token,
                cx,
                cy,
                width,
                height,
                allowed_class_ids=allowed_class_ids,
            )
        )
        if image_width is None or image_height is None:
            continue
        xmin, ymin, xmax, ymax = yolo_to_xyxy(cx, cy, width, height, image_width, image_height)
        result.boxes.append(
            ParsedBox(
                class_raw=class_raw,
                class_id=class_id,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                source_format="yolo_txt",
            )
        )
        result.issues.extend(validate_xyxy(xmin, ymin, xmax, ymax, image_width, image_height))
    return result


def _as_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None
