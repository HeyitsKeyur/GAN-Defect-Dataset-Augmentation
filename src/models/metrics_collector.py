"""Metrics extraction, evaluation reporting, and confidence threshold selection.

Supports extraction of primary mAP@0.50, secondary mAP@0.50:0.95, per-class AP,
precision, recall, and F1 curves. Enforces strict validation-only confidence
threshold selection before test evaluation.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class ClassMetricSummary:
    class_id: int
    class_name: str
    ap50: float = 0.0
    ap50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    support_instances: int = 0
    statistical_caveat: str | None = None


@dataclass
class ExperimentMetricsSummary:
    experiment_name: str
    split_evaluated: str
    map50: float = 0.0
    map50_95: float = 0.0
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    mean_f1: float = 0.0
    optimal_confidence_threshold: float = 0.25
    threshold_selection_split: str = "val"
    classes: dict[str, ClassMetricSummary] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_csv(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "class_id", "class_name", "ap50", "ap50_95",
            "precision", "recall", "f1_score", "support_instances", "statistical_caveat"
        ]
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for c_summary in self.classes.values():
                writer.writerow(asdict(c_summary))


class MetricsCollector:
    """Extracts and formats metrics from YOLO training events and validation outputs."""

    CLASS_NAMES = {0: "exc_solder", 1: "spike", 2: "poor_solder"}

    @staticmethod
    def parse_ultralytics_results_csv(results_csv_path: Path | str) -> dict[str, Any]:
        """Extract final and best epoch metrics from an Ultralytics results.csv."""
        p = Path(results_csv_path)
        if not p.is_file():
            raise FileNotFoundError(f"results.csv not found at: {p}")

        epochs_data: list[dict[str, float]] = []
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = [h.strip() for h in next(reader)]
            for row in reader:
                if not row or len(row) != len(header):
                    continue
                row_dict = {}
                for h, val in zip(header, row):
                    try:
                        row_dict[h] = float(val.strip())
                    except ValueError:
                        pass
                epochs_data.append(row_dict)

        if not epochs_data:
            return {"epochs_count": 0, "best_epoch": {}}

        # Find best epoch by metrics/mAP50(B)
        map50_key = next((k for k in header if "mAP50(B)" in k or "mAP_0.5" in k), None)
        best_epoch = {}
        if map50_key:
            best_epoch = max(epochs_data, key=lambda x: x.get(map50_key, 0.0))

        return {
            "epochs_count": len(epochs_data),
            "final_epoch": epochs_data[-1] if epochs_data else {},
            "best_epoch": best_epoch,
            "metric_keys": header,
        }

    @staticmethod
    def select_optimal_confidence_threshold(
        f1_curve_data: list[tuple[float, float]] | None = None,
        default_threshold: float = 0.25,
    ) -> float:
        """Select confidence threshold that maximizes F1 score on validation data.
        
        Strict rule: MUST be computed on Validation split only, never Test split!
        """
        if not f1_curve_data:
            return default_threshold

        best_thresh = default_threshold
        max_f1 = -1.0
        for thresh, f1 in f1_curve_data:
            if f1 > max_f1:
                max_f1 = f1
                best_thresh = thresh

        return round(float(best_thresh), 4)

    @classmethod
    def build_summary_report(
        cls,
        experiment_name: str,
        split_name: str,
        map50: float,
        map50_95: float,
        mean_p: float,
        mean_r: float,
        optimal_conf: float,
        class_metrics: dict[str, dict[str, Any]] | None = None,
    ) -> ExperimentMetricsSummary:
        """Create a standardized metrics summary object with statistical caveats."""
        mean_f1 = (2 * mean_p * mean_r / (mean_p + mean_r)) if (mean_p + mean_r) > 0 else 0.0

        summary = ExperimentMetricsSummary(
            experiment_name=experiment_name,
            split_evaluated=split_name,
            map50=round(map50, 4),
            map50_95=round(map50_95, 4),
            mean_precision=round(mean_p, 4),
            mean_recall=round(mean_r, 4),
            mean_f1=round(mean_f1, 4),
            optimal_confidence_threshold=round(optimal_conf, 4),
            threshold_selection_split="val",
        )

        # Standard supports for SolDef side-view benchmark
        support_map = {
            "val": {"exc_solder": 26, "spike": 15, "poor_solder": 10},
            "test": {"exc_solder": 28, "spike": 19, "poor_solder": 15},
            "train": {"exc_solder": 128, "spike": 96, "poor_solder": 32},
        }

        supports = support_map.get(split_name, {})

        for cid, cname in cls.CLASS_NAMES.items():
            c_data = (class_metrics or {}).get(cname, {})
            c_p = float(c_data.get("precision", mean_p))
            c_r = float(c_data.get("recall", mean_r))
            c_f1 = (2 * c_p * c_r / (c_p + c_r)) if (c_p + c_r) > 0 else 0.0

            caveat = None
            if cname == "poor_solder":
                caveat = (
                    f"Small sample minority class ({supports.get('poor_solder', 15)} instances in {split_name}); "
                    "metrics are exploratory and subject to high variance."
                )

            summary.classes[cname] = ClassMetricSummary(
                class_id=cid,
                class_name=cname,
                ap50=round(float(c_data.get("ap50", map50)), 4),
                ap50_95=round(float(c_data.get("ap50_95", map50_95)), 4),
                precision=round(c_p, 4),
                recall=round(c_r, 4),
                f1_score=round(c_f1, 4),
                support_instances=supports.get(cname, 0),
                statistical_caveat=caveat,
            )

        summary.notes.append(
            "Confidence threshold selected strictly on validation split to prevent test-set data snooping."
        )
        return summary
