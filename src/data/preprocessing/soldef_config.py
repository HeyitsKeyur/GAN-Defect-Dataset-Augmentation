"""Configuration schema and loader for SolDef_AI preprocessing pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_JSON = Path("configs/soldef_preprocessing.json")
DEFAULT_CONFIG_YAML = Path("configs/soldef_preprocessing.yaml")


@dataclass
class SolDefPatchConfig:
    target_size: int = 256
    crop_strategy: str = "context_crop_resize"  # "context_crop_resize" | "fixed_window_crop"
    context_margin: float = 0.20
    min_crop_size: int = 256
    allow_padding: bool = True
    pad_color: tuple[int, int, int] = (114, 114, 114)
    resize_resample: str = "bilinear"
    min_retention_ratio: float = 0.30
    min_clipped_area: float = 9.0
    source_split: str = "train"


@dataclass
class SolDefDetectorConfig:
    target_size: int = 640
    pad_color: tuple[int, int, int] = (114, 114, 114)
    resize_resample: str = "bilinear"
    splits_to_process: list[str] = field(default_factory=lambda: ["train", "val", "test"])


@dataclass
class SolDefQCConfig:
    num_visual_samples: int = 12
    random_seed: int = 0


@dataclass
class SolDefPathsConfig:
    split_manifest_dir: Path = Path("outputs/splits/soldef/phase1")
    raw_labeled_dir: Path = Path("Dateset/SolDef_AI/Labeled")
    output_base_dir: Path = Path("outputs/preprocessing/soldef/phase_d")


@dataclass
class SolDefPreprocessingConfig:
    dataset_name: str = "soldef"
    benchmark_name: str = "primary_side_view"
    views_included: list[str] = field(default_factory=lambda: ["V2", "V2.1"])
    target_classes: list[str] = field(default_factory=lambda: ["exc_solder", "spike", "poor_solder"])
    class_to_id: dict[str, int] = field(default_factory=lambda: {"exc_solder": 0, "spike": 1, "poor_solder": 2})
    patches: SolDefPatchConfig = field(default_factory=SolDefPatchConfig)
    detector: SolDefDetectorConfig = field(default_factory=SolDefDetectorConfig)
    qc: SolDefQCConfig = field(default_factory=SolDefQCConfig)
    paths: SolDefPathsConfig = field(default_factory=SolDefPathsConfig)

    @classmethod
    def from_file(cls, config_path: Path | str | None = None) -> SolDefPreprocessingConfig:
        """Load configuration from JSON or YAML file, or return defaults."""
        data: dict[str, Any] = {}

        target_path = Path(config_path) if config_path is not None else None
        if target_path is None:
            if DEFAULT_CONFIG_JSON.exists():
                target_path = DEFAULT_CONFIG_JSON
            elif DEFAULT_CONFIG_YAML.exists():
                target_path = DEFAULT_CONFIG_YAML

        if target_path is not None and target_path.exists():
            if target_path.suffix.lower() == ".json":
                with open(target_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                try:
                    import yaml
                    with open(target_path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                except ImportError:
                    if DEFAULT_CONFIG_JSON.exists():
                        with open(DEFAULT_CONFIG_JSON, "r", encoding="utf-8") as f:
                            data = json.load(f)

        patches_data = data.get("patches", {})
        if "pad_color" in patches_data and isinstance(patches_data["pad_color"], list):
            patches_data["pad_color"] = tuple(patches_data["pad_color"])
        patches_cfg = SolDefPatchConfig(**patches_data)

        detector_data = data.get("detector", {})
        if "pad_color" in detector_data and isinstance(detector_data["pad_color"], list):
            detector_data["pad_color"] = tuple(detector_data["pad_color"])
        detector_cfg = SolDefDetectorConfig(**detector_data)

        qc_cfg = SolDefQCConfig(**data.get("qc", {}))

        paths_data = data.get("paths", {})
        paths_cfg = SolDefPathsConfig(
            split_manifest_dir=Path(paths_data.get("split_manifest_dir", "outputs/splits/soldef/phase1")),
            raw_labeled_dir=Path(paths_data.get("raw_labeled_dir", "Dateset/SolDef_AI/Labeled")),
            output_base_dir=Path(paths_data.get("output_base_dir", "outputs/preprocessing/soldef/phase_d")),
        )

        return cls(
            dataset_name=data.get("dataset_name", "soldef"),
            benchmark_name=data.get("benchmark_name", "primary_side_view"),
            views_included=data.get("views_included", ["V2", "V2.1"]),
            target_classes=data.get("target_classes", ["exc_solder", "spike", "poor_solder"]),
            class_to_id=data.get("class_to_id", {"exc_solder": 0, "spike": 1, "poor_solder": 2}),
            patches=patches_cfg,
            detector=detector_cfg,
            qc=qc_cfg,
            paths=paths_cfg,
        )

    @classmethod
    def from_yaml(cls, yaml_path: Path | str | None = None) -> SolDefPreprocessingConfig:
        return cls.from_file(yaml_path)
