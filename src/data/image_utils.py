"""Image inspection helpers used by the Phase 0 audit."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.data.dataset_paths import IMAGE_EXTENSIONS


@dataclass
class ImageInfo:
    path: Path
    stem: str
    suffix: str
    readable: bool
    width: int | None = None
    height: int | None = None
    format_name: str | None = None
    mode: str | None = None
    md5: str | None = None
    average_hash: int | None = None
    error: str | None = None
    unsupported_format: bool = False


SPECIMEN_RE = re.compile(
    r"^(?P<group>\d+_\d+_\d+_\d+_NG)(?:_part\d+|_full)?$",
    re.IGNORECASE,
)
CLASSLIKE_PREFIX_RE = re.compile(
    r"^(?P<prefix>adhension|adhesion|spot|scar|uneven|con|crack|dirty|defect|Abnormal thickness)",
    re.IGNORECASE,
)


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def average_hash(image: Image.Image, hash_size: int = 8) -> int:
    gray = image.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
    pixels = list(gray.getdata())
    mean = sum(pixels) / max(len(pixels), 1)
    bits = 0
    for index, value in enumerate(pixels):
        if value >= mean:
            bits |= 1 << index
    return bits


def hamming_distance(hash_a: int, hash_b: int) -> int:
    return (hash_a ^ hash_b).bit_count()


def inspect_image(path: Path) -> ImageInfo:
    info = ImageInfo(path=path, stem=path.stem, suffix=path.suffix.lower(), readable=False)
    if info.suffix not in IMAGE_EXTENSIONS:
        info.unsupported_format = True
        info.error = f"Unsupported image extension: {path.suffix}"
        return info
    try:
        info.md5 = file_md5(path)
        with Image.open(path) as image:
            image.load()
            info.width, info.height = image.size
            info.format_name = image.format
            info.mode = image.mode
            info.average_hash = average_hash(image)
        info.readable = True
        if info.format_name not in {"JPEG", "PNG", "BMP", "TIFF", "WEBP"}:
            info.unsupported_format = True
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        info.error = str(exc)
        info.readable = False
    return info


def infer_group_id(stem: str) -> tuple[str, str]:
    """Return (group_id, evidence_type) from the filename stem.

    Confirmed grouping is only claimed for the ``d_d_dddd_dddd_NG`` pattern
    including ``_part`` / ``_full`` suffixes. Other names are singleton groups.
    """
    match = SPECIMEN_RE.match(stem)
    if match:
        group = match.group("group")
        if "_part" in stem.lower():
            return group, "filename_part_suffix"
        if "_full" in stem.lower():
            return group, "filename_full_suffix"
        return group, "filename_ng_id"
    return stem, "singleton_filename"


def filename_prefix_bucket(stem: str) -> str:
    match = CLASSLIKE_PREFIX_RE.match(stem)
    if match:
        return match.group("prefix")
    if SPECIMEN_RE.match(stem):
        return "numeric_id"
    return "other"


def summarize_counts(values: list[int | float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    return {
        "count": n,
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "mean": float(sum(ordered) / n),
        "median": float(_percentile(ordered, 50)),
        "p25": float(_percentile(ordered, 25)),
        "p75": float(_percentile(ordered, 75)),
    }


def _percentile(sorted_values: list[int | float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    weight = rank - low
    return float(sorted_values[low]) * (1.0 - weight) + float(sorted_values[high]) * weight


def counter_to_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))}
