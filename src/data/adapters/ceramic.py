"""Adapter for the legacy Ceramic-Defect dataset."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from src.data.adapters.base import BaseDatasetAdapter, DatasetImageRecord
from src.data.annotation_parsers import parse_labelme_json
from src.data.dataset_paths import DatasetLayout, index_by_stem, iter_files, resolve_layout
from src.data.image_utils import infer_group_id
from src.data.preprocessing.crop_geometry import BoundingBox
from src.data.splits import YOLO_ID_TO_CLASS

CERAMIC_DEFAULT_ROOT = Path("Dateset/Ceramic-defect/data/data")

CERAMIC_CLASS_TO_ID: dict[str, int] = {name: cid for cid, name in YOLO_ID_TO_CLASS.items()}
CERAMIC_CLASS_TO_ID["abnormal thickness"] = 4

CERAMIC_ID_TO_CLASS: dict[int, str] = dict(YOLO_ID_TO_CLASS)


class CeramicAdapter(BaseDatasetAdapter):
    """Dataset adapter for the Ceramic-Defect proxy dataset."""

    def __init__(self, root_dir: Path | str | None = None) -> None:
        if root_dir is not None:
            self._root_dir = Path(root_dir)
        else:
            self._root_dir = CERAMIC_DEFAULT_ROOT

        if self._root_dir.exists():
            self._layout: DatasetLayout | None = resolve_layout(self._root_dir)
        else:
            self._layout = None

    @property
    def dataset_name(self) -> str:
        return "ceramic"

    @property
    def target_classes(self) -> list[str]:
        return [
            "adhesion",
            "scar",
            "spot",
            "uneven",
            "abnormal_thickness",
            "defect",
        ]

    @property
    def class_to_id(self) -> dict[str, int]:
        return dict(CERAMIC_CLASS_TO_ID)

    @property
    def id_to_class(self) -> dict[int, str]:
        return dict(CERAMIC_ID_TO_CLASS)

    def load_annotations(self, annotation_path: Path, defect_only: bool = False) -> list[BoundingBox]:
        """Parse LabelMe JSON annotations into BoundingBox objects."""
        parsed = parse_labelme_json(annotation_path)
        boxes: list[BoundingBox] = []
        for i, b in enumerate(parsed.boxes):
            raw_name = b.class_raw.strip().lower()
            norm_name = raw_name.replace(" ", "_")
            cid = self.class_to_id.get(norm_name, self.class_to_id.get(raw_name, 5))
            canonical_name = self.id_to_class.get(cid, "defect")
            boxes.append(
                BoundingBox(
                    xmin=b.xmin,
                    ymin=b.ymin,
                    xmax=b.xmax,
                    ymax=b.ymax,
                    class_id=cid,
                    class_name=canonical_name,
                    original_id=i,
                )
            )
        return boxes

    def get_image_records(self) -> list[DatasetImageRecord]:
        """Enumerate all image records for Ceramic dataset."""
        records: list[DatasetImageRecord] = []
        if self._layout is None:
            return records

        img_files = iter_files(self._layout.original_image_dir, {".jpg", ".jpeg", ".png"})
        json_files = iter_files(self._layout.json_dir, {".json"})
        json_index = index_by_stem(json_files)

        for img_path in img_files:
            stem = img_path.stem
            ann_paths = json_index.get(stem, [])
            if not ann_paths:
                continue
            ann_path = ann_paths[0]

            w, h = 0, 0
            raw_labels: list[str] = []
            try:
                with ann_path.open(encoding="utf-8") as f:
                    data = json.load(f)
                    w = int(data.get("imageWidth", 0))
                    h = int(data.get("imageHeight", 0))
                    for shape in data.get("shapes", []):
                        lbl = shape.get("label", "").strip().lower().replace(" ", "_")
                        raw_labels.append(lbl)
            except Exception:
                pass

            counts = Counter(raw_labels)
            classes_present = sorted(counts.keys())

            group_id, _ = infer_group_id(stem)

            records.append(
                DatasetImageRecord(
                    stem=stem,
                    image_path=img_path,
                    annotation_path=ann_path,
                    dataset_name=self.dataset_name,
                    group_id=group_id,
                    evidence_level="inferred",
                    specimen_identity_status="inferred",
                    rationale="Heuristic filename pattern grouping for ceramic substrate tiles",
                    subset_or_cs="ceramic_tile",
                    component_type="ceramic_substrate",
                    view="top_down",
                    width=w,
                    height=h,
                    classes_present=classes_present,
                    instance_counts=dict(counts),
                    extra_metadata={"format": "labelme_json"},
                )
            )

        return records
