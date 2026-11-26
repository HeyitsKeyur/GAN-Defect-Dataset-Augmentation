"""Configuration classes and loaders for real-only YOLOv8 detector baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


def _parse_yaml_value(val_str: str) -> Any:
    val_str = val_str.strip()
    if val_str.lower() in ("true", "yes", "on"):
        return True
    if val_str.lower() in ("false", "no", "off"):
        return False
    if val_str.lower() in ("null", "none", "~", ""):
        return None
    try:
        if "." in val_str or "e" in val_str.lower():
            return float(val_str)
        return int(val_str)
    except ValueError:
        return val_str.strip("'\"")


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Simple indent-aware YAML parser for standard key-value configs without PyYAML dependency."""
    lines = text.splitlines()
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        current_dict = stack[-1][1]

        if ":" in stripped:
            key, val = stripped.split(":", 1)
            key = key.strip()
            if key.isdigit():
                key = int(key)
            val = val.strip()
            if "#" in val:
                val = val.split("#", 1)[0].strip()

            if not val or val.startswith("#"):
                new_dict: dict[str, Any] = {}
                current_dict[key] = new_dict
                stack.append((indent, new_dict))
            else:
                current_dict[key] = _parse_yaml_value(val)

    return root


@dataclass
class ModelArchitectureConfig:
    model_family: str = "yolov8"
    variant: str = "s"  # "n" | "s" | "m"
    weights: str = "yolov8s.pt"
    pretrained: bool = True
    num_classes: int = 3
    names: dict[int, str] = field(
        default_factory=lambda: {0: "exc_solder", 1: "spike", 2: "poor_solder"}
    )


@dataclass
class BaselineAugmentationConfig:
    """Strict real-only augmentation policy. Disables synthetic composite operators."""
    fliplr: float = 0.5
    flipud: float = 0.0
    degrees: float = 0.0
    scale: float = 0.1
    hsv_h: float = 0.015
    hsv_s: float = 0.4
    hsv_v: float = 0.2
    translate: float = 0.05
    shear: float = 0.0
    perspective: float = 0.0
    mosaic: float = 0.0        # STRICTLY DISABLED: no artificial 4-image composites
    mixup: float = 0.0         # STRICTLY DISABLED: no phantom translucent joints
    copy_paste: float = 0.0    # STRICTLY DISABLED: no synthetic defect pasting
    erasing: float = 0.0


@dataclass
class TrainingScheduleConfig:
    image_size: int = 640
    batch_size: int = 16
    fallback_batch_size: int = 8
    epochs: int = 120
    patience: int = 25
    optimizer: str = "SGD"
    momentum: float = 0.937
    lr0: float = 0.01
    lrf: float = 0.01
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    cos_lr: bool = True
    seed: int = 42
    deterministic: bool = True
    device: str = "0"
    workers: int = 2
    save: bool = True
    save_period: int = -1
    val: bool = True


@dataclass
class BaselineExperimentConfig:
    experiment_name: str = "baseline_real_only"
    dataset_name: str = "soldef"
    benchmark_name: str = "primary_side_view"
    data_dir: str = "outputs/preprocessing/soldef/phase_d/detector"
    split_manifest_dir: str = "outputs/splits/soldef/phase1"
    output_dir: str = "outputs/experiments/baseline_yolov8s_real"
    model: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    schedule: TrainingScheduleConfig = field(default_factory=TrainingScheduleConfig)
    augmentation: BaselineAugmentationConfig = field(default_factory=BaselineAugmentationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_yaml(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            import yaml
            with open(p, "w", encoding="utf-8") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        except ImportError:
            self._save_simple_yaml(p)

    def _save_simple_yaml(self, p: Path) -> None:
        lines = []
        d = self.to_dict()
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f"{k}:")
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, dict):
                        lines.append(f"  {sub_k}:")
                        for s2_k, s2_v in sub_v.items():
                            lines.append(f"    {s2_k}: {s2_v}")
                    else:
                        lines.append(f"  {sub_k}: {sub_v}")
            else:
                lines.append(f"{k}: {v}")
        p.write_text("\n".join(lines), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselineExperimentConfig:
        model_data = data.get("model", {})
        schedule_data = data.get("schedule", {})
        aug_data = data.get("augmentation", {})

        return cls(
            experiment_name=data.get("experiment_name", "baseline_real_only"),
            dataset_name=data.get("dataset_name", "soldef"),
            benchmark_name=data.get("benchmark_name", "primary_side_view"),
            data_dir=data.get("data_dir", "outputs/preprocessing/soldef/phase_d/detector"),
            split_manifest_dir=data.get("split_manifest_dir", "outputs/splits/soldef/phase1"),
            output_dir=data.get("output_dir", "outputs/experiments/baseline_yolov8s_real"),
            model=ModelArchitectureConfig(**model_data) if model_data else ModelArchitectureConfig(),
            schedule=TrainingScheduleConfig(**schedule_data) if schedule_data else TrainingScheduleConfig(),
            augmentation=BaselineAugmentationConfig(**aug_data) if aug_data else BaselineAugmentationConfig(),
        )

    @classmethod
    def load(cls, path: Path | str) -> BaselineExperimentConfig:
        p = Path(path)
        if not p.is_file():
            alt = p.with_suffix(".json" if p.suffix in (".yaml", ".yml") else ".yaml")
            if alt.is_file():
                p = alt
            else:
                raise FileNotFoundError(f"Configuration file not found: {p}")

        text = p.read_text(encoding="utf-8")
        if p.suffix in (".yaml", ".yml"):
            try:
                import yaml
                data = yaml.safe_load(text)
            except ImportError:
                data = parse_simple_yaml(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    def generate_yolo_data_yaml(self, dest_yaml_path: Path | str, abs_data_root: Path | str | None = None) -> Path:
        """Generate the Ultralytics data.yaml file with strict test-set quarantine."""
        dest_p = Path(dest_yaml_path)
        dest_p.parent.mkdir(parents=True, exist_ok=True)

        root = Path(abs_data_root) if abs_data_root else Path(self.data_dir).resolve()

        train_imgs = (root / "train" / "images").as_posix()
        val_imgs = (root / "val" / "images").as_posix()

        lines = [
            "# SolDef_AI YOLOv8 Primary Side-View Dataset Specification",
            "# Generated automatically with strict test-set quarantine",
            f"path: {root.as_posix()}",
            f"train: {train_imgs}",
            f"val: {val_imgs}",
            "",
            "# TEST SET QUARANTINE NOTICE:",
            "# The test split is explicitly omitted from this training specification",
            "# to prevent accidental evaluation or hyperparameter tuning on held-out test data.",
            "",
            f"nc: {self.model.num_classes}",
            "names:",
        ]
        for cid, cname in self.model.names.items():
            lines.append(f"  {cid}: {cname}")

        dest_p.write_text("\n".join(lines), encoding="utf-8")
        return dest_p
