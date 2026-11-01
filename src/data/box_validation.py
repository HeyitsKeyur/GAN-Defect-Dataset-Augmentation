"""Bounding-box geometry checks independent of annotation file format."""

from __future__ import annotations

from dataclasses import dataclass


# COCO-style object-size bins using absolute box area in pixels.
# Documented here so later phases do not silently change thresholds.
COCO_SMALL_AREA_MAX = 32 * 32  # 1024
COCO_MEDIUM_AREA_MAX = 96 * 96  # 9216


@dataclass(frozen=True)
class BoxIssue:
    code: str
    message: str


def parse_float(value: object, field: str) -> tuple[float | None, BoxIssue | None]:
    if value is None or value == "":
        return None, BoxIssue("missing_coordinate", f"Missing coordinate field: {field}")
    try:
        return float(value), None
    except (TypeError, ValueError):
        return None, BoxIssue(
            "invalid_coordinate",
            f"Non-numeric coordinate in field {field}: {value!r}",
        )


def xyxy_from_points(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def yolo_to_xyxy(
    cx: float,
    cy: float,
    width: float,
    height: float,
    image_width: float,
    image_height: float,
) -> tuple[float, float, float, float]:
    box_w = width * image_width
    box_h = height * image_height
    xmin = (cx * image_width) - (box_w / 2.0)
    ymin = (cy * image_height) - (box_h / 2.0)
    xmax = xmin + box_w
    ymax = ymin + box_h
    return xmin, ymin, xmax, ymax


def box_area(xmin: float, ymin: float, xmax: float, ymax: float) -> float:
    return max(0.0, xmax - xmin) * max(0.0, ymax - ymin)


def coco_size_bin(area: float) -> str:
    if area < COCO_SMALL_AREA_MAX:
        return "small"
    if area < COCO_MEDIUM_AREA_MAX:
        return "medium"
    return "large"


def validate_xyxy(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    image_width: int | None = None,
    image_height: int | None = None,
    *,
    coordinate_eps: float = 1e-6,
    bound_tol: float = 1.0,
) -> list[BoxIssue]:
    """Validate an axis-aligned box in absolute pixel coordinates.

    ``bound_tol`` allows coordinates within one pixel of the image edge so
    integer rounding from JSON floats / YOLO denormalization is not treated
    as out-of-bounds by default.
    """
    issues: list[BoxIssue] = []
    width = xmax - xmin
    height = ymax - ymin
    if width < -coordinate_eps or height < -coordinate_eps:
        issues.append(
            BoxIssue(
                "negative_extent",
                f"Negative width/height: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}",
            )
        )
    area = box_area(xmin, ymin, xmax, ymax)
    if area <= coordinate_eps:
        issues.append(
            BoxIssue(
                "zero_area",
                f"Zero-area box: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}",
            )
        )
    if image_width is None or image_height is None:
        return issues
    if image_width <= 0 or image_height <= 0:
        issues.append(
            BoxIssue(
                "invalid_image_size",
                f"Invalid image size {image_width}x{image_height}",
            )
        )
        return issues
    if xmin < -bound_tol or ymin < -bound_tol:
        issues.append(
            BoxIssue(
                "out_of_bounds",
                f"Box origin outside image: xmin={xmin}, ymin={ymin}",
            )
        )
    if xmax > image_width + bound_tol or ymax > image_height + bound_tol:
        issues.append(
            BoxIssue(
                "exceeds_image",
                (
                    f"Box exceeds image {image_width}x{image_height}: "
                    f"xmax={xmax}, ymax={ymax}"
                ),
            )
        )
    return issues


def validate_yolo_normalized(
    class_id: object,
    cx: float,
    cy: float,
    width: float,
    height: float,
    *,
    allowed_class_ids: set[int] | None = None,
) -> list[BoxIssue]:
    issues: list[BoxIssue] = []
    try:
        cid = int(class_id)
    except (TypeError, ValueError):
        issues.append(BoxIssue("invalid_class_id", f"Non-integer class id: {class_id!r}"))
        cid = None  # type: ignore[assignment]
    else:
        if cid < 0:
            issues.append(BoxIssue("invalid_class_id", f"Negative class id: {cid}"))
        if allowed_class_ids is not None and cid not in allowed_class_ids:
            issues.append(
                BoxIssue("invalid_class_id", f"Class id {cid} is not in the observed/allowed set")
            )
    for name, value in (("cx", cx), ("cy", cy), ("width", width), ("height", height)):
        if value < 0 or value > 1:
            issues.append(
                BoxIssue(
                    "invalid_normalized_coordinate",
                    f"YOLO {name}={value} is outside [0, 1]",
                )
            )
    if width <= 0 or height <= 0:
        issues.append(
            BoxIssue("zero_area", f"YOLO width/height not positive: w={width}, h={height}")
        )
    return issues
