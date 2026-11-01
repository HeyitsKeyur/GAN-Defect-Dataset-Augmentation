"""Locate the Ceramic-Defect files actually present in this workspace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class DatasetLayout:
    data_root: Path
    original_image_dir: Path
    split_image_dir: Path
    json_dir: Path
    xml_dir: Path
    txt_dir: Path
    split_names: tuple[str, ...] = ("train", "val", "test")


def default_data_root(project_root: Path) -> Path:
    """Return the layout discovered in this repository, if it exists."""
    return project_root / "Dateset" / "Ceramic-defect" / "data" / "data"


def resolve_layout(data_root: Path) -> DatasetLayout:
    data_root = data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset data root not found: {data_root}")
    original_image_dir = data_root / "original image"
    split_image_dir = data_root / "image"
    labels_dir = data_root / "lables"
    json_dir = labels_dir / "json"
    xml_dir = labels_dir / "xml"
    txt_dir = labels_dir / "txt"
    missing = [
        str(path)
        for path in (original_image_dir, split_image_dir, json_dir, xml_dir, txt_dir)
        if not path.is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            "Dataset layout is incomplete. Missing directories:\n" + "\n".join(missing)
        )
    return DatasetLayout(
        data_root=data_root,
        original_image_dir=original_image_dir,
        split_image_dir=split_image_dir,
        json_dir=json_dir,
        xml_dir=xml_dir,
        txt_dir=txt_dir,
    )


def iter_files(directory: Path, suffixes: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            files.append(path)
    return sorted(files)


def index_by_stem(paths: list[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in paths:
        index.setdefault(path.stem, []).append(path)
    return index
