"""Portable environment and project path configuration for local and Colab environments.

Supports:
- Local Windows Laptop 1 & Laptop 2
- Google Colab GPU Runtime
- Environment variable overrides (.env or os.environ)
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def is_colab() -> bool:
    """Return True if executing inside Google Colab."""
    return "google.colab" in sys.modules or os.environ.get("COLAB_GPU") is not None or Path("/content").is_dir()


def find_project_root() -> Path:
    """Detect the repository root based on markers like .git or AGENTS.md."""
    # Start looking from this file's location: src/utils/config.py -> project root is 2 levels up
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "AGENTS.md").exists() or (parent / "PROJECT_CONTEXT.md").exists():
            return parent
    # Fallback to current working directory
    return Path.cwd().resolve()


def load_dotenv(env_path: Path | None = None) -> dict[str, str]:
    """Simple parser for .env files without external dependencies."""
    loaded: dict[str, str] = {}
    if env_path is None:
        root = find_project_root()
        env_path = root / ".env"
    if not env_path.is_file():
        return loaded

    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return loaded

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip("'\"")
        if key and val:
            loaded[key] = val
            # Only set in os.environ if not already present
            if key not in os.environ:
                os.environ[key] = val
    return loaded


@dataclass(frozen=True)
class ProjectConfig:
    project_root: Path
    dataset_root: Path
    split_manifest_dir: Path
    google_drive_root: Path | None
    checkpoint_dir: Path
    experiment_output_dir: Path
    random_seed: int = 42
    environment_name: str = "local"
    extra_vars: dict[str, Any] = field(default_factory=dict)

    def validate_dataset(self) -> tuple[bool, list[str]]:
        """Verify that expected dataset directories exist."""
        errors: list[str] = []
        if not self.dataset_root.exists():
            errors.append(f"Dataset root does not exist: {self.dataset_root}")
            return False, errors

        required_subdirs = ["original image", "image", "lables"]
        for subdir in required_subdirs:
            p = self.dataset_root / subdir
            if not p.is_dir():
                errors.append(f"Required dataset subdirectory missing: {p}")

        return len(errors) == 0, errors

    def validate_splits(self) -> tuple[bool, list[str]]:
        """Verify that Phase 1 frozen manifests exist."""
        errors: list[str] = []
        if not self.split_manifest_dir.exists():
            errors.append(f"Split manifest directory missing: {self.split_manifest_dir}")
            return False, errors

        required_manifests = ["train.csv", "val.csv", "test.csv", "split_manifest.json"]
        for name in required_manifests:
            p = self.split_manifest_dir / name
            if not p.is_file():
                errors.append(f"Frozen split manifest missing: {p}")
            elif p.stat().st_size == 0:
                errors.append(f"Frozen split manifest is empty: {p}")

        return len(errors) == 0, errors


def get_config(
    dataset_root_override: Path | str | None = None,
    env_file: Path | None = None,
) -> ProjectConfig:
    """Build and return a ProjectConfig instance based on runtime and configuration."""
    # 1. Load .env if available
    load_dotenv(env_file)

    project_root = find_project_root()
    in_colab = is_colab()
    env_name = "colab" if in_colab else ("windows" if platform.system() == "Windows" else "unix")

    # 2. Determine Google Drive Root
    drive_root_str = os.environ.get("GOOGLE_DRIVE_ROOT")
    if drive_root_str:
        drive_root: Path | None = Path(drive_root_str).resolve()
    elif in_colab:
        drive_root = Path("/content/drive/MyDrive")
    else:
        drive_root = None

    # 3. Determine Dataset Root
    # Priority: override argument -> DATASET_ROOT env var -> Colab specific path -> Local discovery
    if dataset_root_override is not None:
        dataset_root = Path(dataset_root_override).resolve()
    elif "DATASET_ROOT" in os.environ and os.environ["DATASET_ROOT"].strip():
        dataset_root = Path(os.environ["DATASET_ROOT"].strip()).resolve()
    elif in_colab:
        # Verified Drive path as per user's screenshot: /content/drive/MyDrive/Ceramic-defect/data/data
        colab_default = Path("/content/drive/MyDrive/Ceramic-defect/data/data")
        dataset_root = colab_default
    else:
        # Default local repository discovery path
        dataset_root = project_root / "Dateset" / "Ceramic-defect" / "data" / "data"

    # 4. Determine Split Manifest Directory
    split_dir_str = os.environ.get("SPLIT_MANIFEST_DIR")
    if split_dir_str and split_dir_str.strip():
        split_dir = Path(split_dir_str.strip()).resolve()
    else:
        split_dir = project_root / "outputs" / "splits" / "phase1"

    # 5. Output and Checkpoint Directories
    out_dir_str = os.environ.get("EXPERIMENT_OUTPUT_DIR")
    if out_dir_str and out_dir_str.strip():
        out_dir = Path(out_dir_str.strip()).resolve()
    elif in_colab and drive_root and drive_root.is_dir():
        # Persist experiment outputs to Drive when on Colab
        out_dir = drive_root / "MajorProject_Outputs"
    else:
        out_dir = project_root / "outputs"

    ckpt_dir_str = os.environ.get("CHECKPOINT_DIR")
    if ckpt_dir_str and ckpt_dir_str.strip():
        ckpt_dir = Path(ckpt_dir_str.strip()).resolve()
    elif in_colab and drive_root and drive_root.is_dir():
        ckpt_dir = drive_root / "MajorProject_Outputs" / "checkpoints"
    else:
        ckpt_dir = out_dir / "checkpoints"

    seed_val = int(os.environ.get("RANDOM_SEED", 42))

    return ProjectConfig(
        project_root=project_root,
        dataset_root=dataset_root,
        split_manifest_dir=split_dir,
        google_drive_root=drive_root,
        checkpoint_dir=ckpt_dir,
        experiment_output_dir=out_dir,
        random_seed=seed_val,
        environment_name=env_name,
    )
