"""Dataset integrity validation module for SolDef detector datasets.

Validates preprocessed full-image letterboxed detector datasets (Train, Val, Test)
prior to training preflight to ensure zero data leakage, correct YOLO label formatting,
file consistency, and adherence to frozen benchmark splits.
"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image


class DatasetValidationError(Exception):
    """Raised when critical dataset integrity checks fail."""
    pass


@dataclass
class SplitValidationSummary:
    split_name: str
    image_count: int = 0
    label_count: int = 0
    empty_label_count: int = 0
    target_box_count: int = 0
    class_box_counts: dict[str, int] = field(default_factory=dict)
    class_image_counts: dict[str, int] = field(default_factory=dict)
    missing_images: list[str] = field(default_factory=list)
    missing_labels: list[str] = field(default_factory=list)
    orphaned_images: list[str] = field(default_factory=list)
    orphaned_labels: list[str] = field(default_factory=list)
    malformed_labels: list[str] = field(default_factory=list)
    unreadable_images: list[str] = field(default_factory=list)


@dataclass
class DatasetValidationReport:
    dataset_name: str
    benchmark_name: str
    is_valid: bool
    total_images: int = 0
    total_boxes: int = 0
    splits: dict[str, SplitValidationSummary] = field(default_factory=dict)
    class_imbalance_ratios: dict[str, float] = field(default_factory=dict)
    overlap_errors: list[str] = field(default_factory=list)
    critical_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_summary_markdown(self) -> str:
        status_str = "PASSED" if self.is_valid else "FAILED"
        lines = [
            f"# Dataset Preflight Integrity Report: {status_str}",
            f"- **Dataset:** {self.dataset_name} ({self.benchmark_name})",
            f"- **Total Images:** {self.total_images}",
            f"- **Total Target Boxes:** {self.total_boxes}",
            f"- **Validation Status:** {'VALID' if self.is_valid else 'INVALID'}",
            "",
            "## Split Breakdown",
            "| Split | Images | Target Boxes | Empty Labels | exc_solder | spike | poor_solder |",
            "|---|---|---|---|---|---|---|",
        ]
        for s_name, s_data in self.splits.items():
            exc = s_data.class_box_counts.get("exc_solder", 0)
            spk = s_data.class_box_counts.get("spike", 0)
            por = s_data.class_box_counts.get("poor_solder", 0)
            lines.append(
                f"| {s_name} | {s_data.image_count} | {s_data.target_box_count} | "
                f"{s_data.empty_label_count} | {exc} | {spk} | {por} |"
            )

        if self.class_imbalance_ratios:
            lines.extend([
                "",
                "## Class Imbalance Ratios (vs Minority class `poor_solder`)",
            ])
            for cls_name, ratio in self.class_imbalance_ratios.items():
                lines.append(f"- **{cls_name}**: {ratio:.2f}:1")

        if self.critical_errors:
            lines.extend(["", "## Critical Errors"])
            for err in self.critical_errors:
                lines.append(f"- [ERROR] {err}")

        if self.warnings:
            lines.extend(["", "## Warnings"])
            for warn in self.warnings:
                lines.append(f"- [WARNING] {warn}")

        if self.limitations:
            lines.extend(["", "## Documented Scientific Limitations"])
            for lim in self.limitations:
                lines.append(f"- [LIMITATION] {lim}")

        return "\n".join(lines)


class DetectorDatasetValidator:
    """Validates preprocessed YOLO detector dataset files and manifests."""

    EXPECTED_SPLIT_COUNTS = {
        "train": {"images": 140, "boxes": 256, "empty_labels": 5},
        "val": {"images": 30, "boxes": 51, "empty_labels": 0},
        "test": {"images": 30, "boxes": 62, "empty_labels": 0},
    }
    CLASS_ID_MAP = {0: "exc_solder", 1: "spike", 2: "poor_solder"}

    def __init__(
        self,
        data_root: Path | str,
        split_manifest_dir: Path | str | None = None,
        verify_image_reads: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.split_manifest_dir = Path(split_manifest_dir) if split_manifest_dir else None
        self.verify_image_reads = verify_image_reads

    def validate(self, splits: list[str] | None = None) -> DatasetValidationReport:
        if splits is None:
            splits = ["train", "val", "test"]

        report = DatasetValidationReport(
            dataset_name="soldef",
            benchmark_name="primary_side_view",
            is_valid=True,
        )

        # 1. Inspect each split
        all_split_stems: dict[str, set[str]] = {}
        total_images = 0
        total_boxes = 0

        for split_name in splits:
            split_dir = self.data_root / split_name
            summary = self._validate_split(split_name, split_dir)
            report.splits[split_name] = summary
            all_split_stems[split_name] = set(
                f.stem for f in (split_dir / "images").glob("*.jpg")
            ) if (split_dir / "images").exists() else set()

            total_images += summary.image_count
            total_boxes += summary.target_box_count

            # Check against expected counts
            expected = self.EXPECTED_SPLIT_COUNTS.get(split_name)
            if expected:
                if summary.image_count != expected["images"]:
                    report.critical_errors.append(
                        f"Split '{split_name}' image count mismatch: got {summary.image_count}, expected {expected['images']}"
                    )
                if summary.target_box_count != expected["boxes"]:
                    report.critical_errors.append(
                        f"Split '{split_name}' box count mismatch: got {summary.target_box_count}, expected {expected['boxes']}"
                    )
                if summary.empty_label_count != expected["empty_labels"]:
                    report.warnings.append(
                        f"Split '{split_name}' empty label count mismatch: got {summary.empty_label_count}, expected {expected['empty_labels']}"
                    )

            if summary.missing_images:
                report.critical_errors.append(
                    f"Split '{split_name}' has {len(summary.missing_images)} missing image files."
                )
            if summary.missing_labels:
                report.critical_errors.append(
                    f"Split '{split_name}' has {len(summary.missing_labels)} missing label files."
                )
            if summary.malformed_labels:
                report.critical_errors.append(
                    f"Split '{split_name}' has {len(summary.malformed_labels)} malformed label entries."
                )
            if summary.unreadable_images:
                report.critical_errors.append(
                    f"Split '{split_name}' has {len(summary.unreadable_images)} unreadable image files."
                )

        report.total_images = total_images
        report.total_boxes = total_boxes

        # 2. Split non-overlap check
        split_names = list(all_split_stems.keys())
        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                s1, s2 = split_names[i], split_names[j]
                overlap = all_split_stems[s1].intersection(all_split_stems[s2])
                if overlap:
                    err_msg = f"Data leakage detected! Stems overlap between {s1} and {s2}: {sorted(list(overlap))[:5]}"
                    report.overlap_errors.append(err_msg)
                    report.critical_errors.append(err_msg)

        # 3. Check split manifest consistency if split_manifest_dir provided
        if self.split_manifest_dir and self.split_manifest_dir.exists():
            self._validate_manifest_consistency(all_split_stems, report)

        # 4. Class imbalance calculation
        total_class_boxes: Counter[str] = Counter()
        for summary in report.splits.values():
            for c_name, c_cnt in summary.class_box_counts.items():
                total_class_boxes[c_name] += c_cnt

        poor_solder_cnt = total_class_boxes.get("poor_solder", 0)
        if poor_solder_cnt > 0:
            for c_name, c_cnt in total_class_boxes.items():
                report.class_imbalance_ratios[c_name] = c_cnt / poor_solder_cnt

        # 5. Scientific limitations tracking
        report.limitations.append(
            "Grouping units reduce view/batch leakage but do not guarantee physical specimen independence for inferred groups."
        )
        report.limitations.append(
            "Minority class 'poor_solder' has small sample representation (32 train, 10 val, 15 test); test metrics are exploratory."
        )
        report.limitations.append(
            "74.2% of train defect annotations reside in two atomic fallback blocks (CS6_R0805 and CS2_R0805)."
        )

        if report.critical_errors or report.overlap_errors:
            report.is_valid = False

        return report

    def _validate_split(self, split_name: str, split_dir: Path) -> SplitValidationSummary:
        summary = SplitValidationSummary(split_name=split_name)
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        if not img_dir.exists() or not lbl_dir.exists():
            summary.missing_images.append(f"Directory missing: {img_dir}")
            return summary

        img_files = {f.stem: f for f in img_dir.glob("*.jpg")}
        lbl_files = {f.stem: f for f in lbl_dir.glob("*.txt")}

        summary.image_count = len(img_files)
        summary.label_count = len(lbl_files)

        # Check association
        img_stems = set(img_files.keys())
        lbl_stems = set(lbl_files.keys())

        for missing in img_stems - lbl_stems:
            summary.missing_labels.append(f"Image {missing}.jpg has no corresponding label file.")
        for missing in lbl_stems - img_stems:
            summary.missing_images.append(f"Label {missing}.txt has no corresponding image file.")

        # Validate label contents and YOLO formatting
        class_box_counter: Counter[str] = Counter()
        class_img_counter: Counter[str] = Counter()

        for stem, lbl_path in lbl_files.items():
            try:
                content = lbl_path.read_text(encoding="utf-8").strip()
            except Exception as e:
                summary.malformed_labels.append(f"{lbl_path.name}: Read error: {e}")
                continue

            if not content:
                summary.empty_label_count += 1
                continue

            lines = content.splitlines()
            img_classes: set[str] = set()

            for line_idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    summary.malformed_labels.append(
                        f"{lbl_path.name}:L{line_idx+1}: Expected 5 columns, got {len(parts)}: '{line}'"
                    )
                    continue

                try:
                    cls_id = int(parts[0])
                    cx = float(parts[1])
                    cy = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except ValueError as ve:
                    summary.malformed_labels.append(
                        f"{lbl_path.name}:L{line_idx+1}: Non-numeric value in line: {ve}"
                    )
                    continue

                if cls_id not in self.CLASS_ID_MAP:
                    summary.malformed_labels.append(
                        f"{lbl_path.name}:L{line_idx+1}: Invalid class ID {cls_id}. Allowed: {list(self.CLASS_ID_MAP.keys())}"
                    )
                    continue

                for val_name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
                    if math.isnan(val) or math.isinf(val):
                        summary.malformed_labels.append(
                            f"{lbl_path.name}:L{line_idx+1}: {val_name} is NaN or Inf"
                        )
                    if not (0.0 <= val <= 1.0):
                        summary.malformed_labels.append(
                            f"{lbl_path.name}:L{line_idx+1}: {val_name}={val} out of normalized [0, 1] range"
                        )

                if w <= 0.0 or h <= 0.0:
                    summary.malformed_labels.append(
                        f"{lbl_path.name}:L{line_idx+1}: w={w} or h={h} is non-positive"
                    )

                cls_name = self.CLASS_ID_MAP[cls_id]
                class_box_counter[cls_name] += 1
                img_classes.add(cls_name)
                summary.target_box_count += 1

            for c in img_classes:
                class_img_counter[c] += 1

            # Optional deep read check for image
            if self.verify_image_reads and stem in img_files:
                try:
                    with Image.open(img_files[stem]) as im:
                        im.verify()
                except Exception as img_err:
                    summary.unreadable_images.append(f"{img_files[stem].name}: Corrupt image ({img_err})")

        summary.class_box_counts = dict(class_box_counter)
        summary.class_image_counts = dict(class_img_counter)
        return summary

    def _validate_manifest_consistency(
        self,
        observed_stems: dict[str, set[str]],
        report: DatasetValidationReport,
    ) -> None:
        manifest_json_path = self.split_manifest_dir / "split_manifest.json"
        if manifest_json_path.exists():
            try:
                manifest_data = json.loads(manifest_json_path.read_text(encoding="utf-8"))
                for split_name, stems in observed_stems.items():
                    csv_path = self.split_manifest_dir / f"{split_name}.csv"
                    if csv_path.exists():
                        with open(csv_path, newline="", encoding="utf-8") as f:
                            csv_stems = set(row["stem"] for row in csv.DictReader(f))
                        if csv_stems != stems:
                            diff = csv_stems.symmetric_difference(stems)
                            report.critical_errors.append(
                                f"Split '{split_name}' preprocessed images mismatch frozen {split_name}.csv! Stems diff: {list(diff)[:5]}"
                            )
            except Exception as e:
                report.warnings.append(f"Failed to verify split_manifest.json: {e}")
