"""Dataset adapter for SolDef_AI (aerospace-grade PCB solder defect dataset)."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

from PIL import Image

from src.data.adapters.base import BaseDatasetAdapter, DatasetImageRecord
from src.data.preprocessing.crop_geometry import BoundingBox

SOLDEF_DEFAULT_ROOT = Path("Dateset/SolDef_AI")

# Class configurations for different benchmark modes
PRIMARY_SIDE_VIEW_CLASSES = ["exc_solder", "spike", "poor_solder"]
PRIMARY_SIDE_VIEW_CLASS_TO_ID = {name: idx for idx, name in enumerate(PRIMARY_SIDE_VIEW_CLASSES)}

FOUR_CLASS_SIDE_VIEW_CLASSES = ["good", "exc_solder", "spike", "poor_solder"]
FOUR_CLASS_SIDE_VIEW_CLASS_TO_ID = {name: idx for idx, name in enumerate(FOUR_CLASS_SIDE_VIEW_CLASSES)}

TOP_DOWN_CLASSES = ["good", "no_good"]
TOP_DOWN_CLASS_TO_ID = {name: idx for idx, name in enumerate(TOP_DOWN_CLASSES)}

ALL_CLASSES = ["good", "exc_solder", "spike", "no_good", "poor_solder"]
ALL_CLASS_TO_ID = {name: idx for idx, name in enumerate(ALL_CLASSES)}


class SolDefAdapter(BaseDatasetAdapter):
    """Adapter for SolDef_AI dataset supporting benchmark separation and grouping provenance."""

    def __init__(
        self,
        root_dir: Path | str | None = None,
        benchmark_type: str = "primary_side_view",
        defect_only: bool = True,
    ) -> None:
        """Initialize SolDef adapter.

        Args:
            root_dir: Path to Dateset/SolDef_AI directory.
            benchmark_type: One of "primary_side_view", "secondary_top_down", "all_inclusive".
            defect_only: If True in side-view mode, maps only defect classes (exc_solder, spike, poor_solder).
        """
        self._root_dir = Path(root_dir) if root_dir is not None else SOLDEF_DEFAULT_ROOT
        self._labeled_dir = self._root_dir / "Labeled"
        self._dataset_dir = self._root_dir / "Dataset"
        self.benchmark_type = benchmark_type
        self.defect_only = defect_only

        # Configure target classes and mapping based on benchmark mode
        if self.benchmark_type == "primary_side_view":
            if self.defect_only:
                self._classes = list(PRIMARY_SIDE_VIEW_CLASSES)
                self._c2id = dict(PRIMARY_SIDE_VIEW_CLASS_TO_ID)
            else:
                self._classes = list(FOUR_CLASS_SIDE_VIEW_CLASSES)
                self._c2id = dict(FOUR_CLASS_SIDE_VIEW_CLASS_TO_ID)
        elif self.benchmark_type == "secondary_top_down":
            self._classes = list(TOP_DOWN_CLASSES)
            self._c2id = dict(TOP_DOWN_CLASS_TO_ID)
        elif self.benchmark_type == "all_inclusive":
            self._classes = list(ALL_CLASSES)
            self._c2id = dict(ALL_CLASS_TO_ID)
        else:
            raise ValueError(f"Unknown benchmark_type: {benchmark_type}")

        self._id2c = {idx: name for name, idx in self._c2id.items()}
        self._file_metadata_cache: dict[str, dict[str, Any]] | None = None

    @property
    def dataset_name(self) -> str:
        return "soldef"

    @property
    def target_classes(self) -> list[str]:
        return list(self._classes)

    @property
    def class_to_id(self) -> dict[str, int]:
        return dict(self._c2id)

    @property
    def id_to_class(self) -> dict[int, str]:
        return dict(self._id2c)

    def _build_metadata_cache(self) -> dict[str, dict[str, Any]]:
        """Index all raw dataset files to determine Case Study, Component, and View."""
        if self._file_metadata_cache is not None:
            return self._file_metadata_cache

        cache: dict[str, dict[str, Any]] = {}
        if self._dataset_dir.exists():
            for p in self._dataset_dir.rglob("*.jpg"):
                try:
                    rel_parts = p.relative_to(self._dataset_dir).parts
                    if len(rel_parts) >= 3:
                        cs = rel_parts[0]
                        comp = rel_parts[1]
                        view = rel_parts[2]
                        cache[p.name] = {
                            "case_study": cs,
                            "component_type": comp,
                            "view": view,
                            "dataset_rel_path": str(p.relative_to(self._root_dir)),
                        }
                except ValueError:
                    continue

        self._file_metadata_cache = cache
        return cache

    def load_annotations(self, annotation_path: Path, defect_only: bool | None = None) -> list[BoundingBox]:
        """Parse LabelMe polygon annotations into BoundingBox instances."""
        if defect_only is None:
            defect_only = self.defect_only

        if not annotation_path.exists():
            return []

        try:
            with annotation_path.open(encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return []

        img_w = float(data.get("imageWidth", 2560))
        img_h = float(data.get("imageHeight", 1440))

        boxes: list[BoundingBox] = []
        for idx, shape in enumerate(data.get("shapes", [])):
            raw_label = shape.get("label", "").strip()
            norm_label = raw_label.lower().replace(" ", "_")

            # Filter for defect-only benchmark if requested
            if defect_only and norm_label == "good":
                continue

            if norm_label not in self.class_to_id:
                continue

            pts = shape.get("points", [])
            if not pts or len(pts) < 2:
                continue

            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]

            xmin = max(0.0, min(min(xs), img_w))
            ymin = max(0.0, min(min(ys), img_h))
            xmax = max(0.0, min(max(xs), img_w))
            ymax = max(0.0, min(max(ys), img_h))

            if xmax <= xmin or ymax <= ymin:
                continue

            cid = self.class_to_id[norm_label]
            boxes.append(
                BoundingBox(
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    class_id=cid,
                    class_name=norm_label,
                    original_id=idx,
                )
            )

        return boxes

    def _determine_grouping(
        self,
        stem: str,
        filename: str,
        cs: str,
        comp: str,
        view: str,
        subset_files: dict[str, list[str]],
    ) -> tuple[str, str, str, str]:
        """Determine group ID, evidence level, specimen status, and rationale.

        Returns:
            (group_id, evidence_level, specimen_identity_status, rationale)
        """
        # Case 1: CS6 C0805 with explicit component numbers
        if cs == "CS6" and comp == "C0805":
            match = re.search(r"_C(\d+)$", stem)
            if match:
                c_num = int(match.group(1))
                return (
                    f"CS6_C0805_C{c_num}",
                    "verified",
                    "verified",
                    f"Explicit component suffix _C{c_num} in filename",
                )
            # Check known chronological V2.1 alignment for unlabelled C0805
            # V2.1 filenames without _C map chronologically:
            # 13_43_00 -> C7, 13_43_17 -> C8, 13_43_34 -> C9
            if "13_43_00" in stem:
                return ("CS6_C0805_C7", "inferred", "inferred", "Chronological sequence alignment to C7 in V2.1")
            elif "13_43_17" in stem:
                return ("CS6_C0805_C8", "inferred", "inferred", "Chronological sequence alignment to C8 in V2.1")
            elif "13_43_34" in stem:
                return ("CS6_C0805_C9", "inferred", "inferred", "Chronological sequence alignment to C9 in V2.1")
            else:
                return (
                    f"CS6_C0805_{stem}",
                    "inferred",
                    "inferred",
                    "Single-image specimen group in CS6 C0805",
                )

        # Case 2: Atomic fallback blocks for asymmetric multi-view runs
        if (cs == "CS2" and comp == "R0805") or (cs == "CS6" and comp == "R0805"):
            return (
                f"{cs}_{comp}_atomic_block",
                "atomic_fallback",
                "unknown",
                f"Asymmetric capture counts across views in {cs} {comp}; grouped into atomic block for zero leakage",
            )

        # Case 3: Chronologically inferred specimen groups (CS1 R0805, CS1 R1206, CS2 R0603, CS2 R1206)
        if cs in ("CS1", "CS2"):
            files_in_view = subset_files.get(view, [])
            if filename in files_in_view:
                idx = files_in_view.index(filename)
                return (
                    f"{cs}_{comp}_specimen_{idx:02d}",
                    "inferred",
                    "inferred",
                    f"Chronological sequence pairing index ({idx:02d}) across matched views",
                )

        # Case 4: Secondary top-down or fallback
        return (
            f"{cs}_{comp}_{stem}",
            "inferred",
            "inferred",
            f"Per-image group in {cs} {comp}",
        )

    def get_image_records(self) -> list[DatasetImageRecord]:
        """Enumerate all image records matching current benchmark configuration."""
        meta_cache = self._build_metadata_cache()
        if not self._labeled_dir.exists():
            return []

        # Group labeled files by (cs, comp, view) to support chronological pairing
        view_groups: dict[tuple[str, str], dict[str, list[str]]] = {}
        all_labeled_json = sorted(self._labeled_dir.glob("*.json"))

        for jf in all_labeled_json:
            img_name = f"{jf.stem}.jpg"
            meta = meta_cache.get(img_name)
            if not meta:
                continue
            cs, comp, view = meta["case_study"], meta["component_type"], meta["view"]
            key = (cs, comp)
            if key not in view_groups:
                view_groups[key] = {}
            if view not in view_groups[key]:
                view_groups[key][view] = []
            view_groups[key][view].append(img_name)

        # Sort filename lists chronologically
        for key in view_groups:
            for v in view_groups[key]:
                view_groups[key][v].sort()

        records: list[DatasetImageRecord] = []
        for jf in all_labeled_json:
            img_name = f"{jf.stem}.jpg"
            img_path = self._labeled_dir / img_name
            if not img_path.exists():
                continue

            meta = meta_cache.get(img_name)
            if not meta:
                continue

            cs = meta["case_study"]
            comp = meta["component_type"]
            view = meta["view"]

            # Filter by benchmark type
            if self.benchmark_type == "primary_side_view" and view not in ("V2", "V2.1"):
                continue
            if self.benchmark_type == "secondary_top_down" and view != "V1":
                continue

            # Load annotations
            try:
                with jf.open(encoding="utf-8") as f:
                    ann_data = json.load(f)
            except Exception:
                ann_data = {}

            all_labels = [s.get("label", "").strip().lower() for s in ann_data.get("shapes", [])]
            raw_counts = Counter(all_labels)

            group_id, ev_level, spec_status, rationale = self._determine_grouping(
                stem=jf.stem,
                filename=img_name,
                cs=cs,
                comp=comp,
                view=view,
                subset_files=view_groups.get((cs, comp), {}),
            )

            records.append(
                DatasetImageRecord(
                    stem=jf.stem,
                    image_path=img_path,
                    annotation_path=jf,
                    dataset_name=self.dataset_name,
                    group_id=group_id,
                    evidence_level=ev_level,
                    specimen_identity_status=spec_status,
                    rationale=rationale,
                    subset_or_cs=cs,
                    component_type=comp,
                    view=view,
                    width=int(ann_data.get("imageWidth", 2560)),
                    height=int(ann_data.get("imageHeight", 1440)),
                    classes_present=sorted(raw_counts.keys()),
                    instance_counts=dict(raw_counts),
                    extra_metadata={
                        "raw_dataset_path": meta["dataset_rel_path"],
                        "total_shapes": len(ann_data.get("shapes", [])),
                    },
                )
            )

        return records
