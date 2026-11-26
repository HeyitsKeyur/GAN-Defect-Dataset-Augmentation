"""Models package for defect detection baselines and evaluation."""

from src.models.baseline_config import (
    BaselineAugmentationConfig,
    BaselineExperimentConfig,
    ModelArchitectureConfig,
    TrainingScheduleConfig,
)

__all__ = [
    "BaselineAugmentationConfig",
    "BaselineExperimentConfig",
    "ModelArchitectureConfig",
    "TrainingScheduleConfig",
]
