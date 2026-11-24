"""Base interface for dataset adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data.preprocessing.crop_geometry import BoundingBox


@dataclass
class DatasetImageRecord:
    """Standardized metadata record for an image in any dataset."""

    stem: str
    image_path: Path
    annotation_path: Path
    dataset_name: str
    group_id: str
    evidence_level: str  # "verified" | "inferred" | "atomic_fallback"
    specimen_identity_status: str  # "verified" | "inferred" | "unknown"
    rationale: str
    subset_or_cs: str
    component_type: str
    view: str
    width: int
    height: int
    classes_present: list[str] = field(default_factory=list)
    instance_counts: dict[str, int] = field(default_factory=dict)
    extra_metadata: dict[str, Any] = field(default_factory=dict)


class BaseDatasetAdapter(ABC):
    """Abstract Base Class for dataset adapters."""

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Name of the dataset."""
        ...

    @property
    @abstractmethod
    def target_classes(self) -> list[str]:
        """Class names targeted for downstream tasks."""
        ...

    @property
    @abstractmethod
    def class_to_id(self) -> dict[str, int]:
        """Mapping from semantic class name to integer class ID."""
        ...

    @property
    @abstractmethod
    def id_to_class(self) -> dict[int, str]:
        """Mapping from integer class ID to semantic class name."""
        ...

    @abstractmethod
    def get_image_records(self) -> list[DatasetImageRecord]:
        """Return all valid image records for this dataset configuration."""
        ...

    @abstractmethod
    def load_annotations(self, annotation_path: Path, defect_only: bool = False) -> list[BoundingBox]:
        """Parse and return BoundingBox instances from the annotation file."""
        ...
