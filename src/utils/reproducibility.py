"""Reproducibility utilities and environment tracking for experiments.

Manages deterministic seed initialization across random, numpy, PyTorch, and CUDA,
and collects comprehensive execution metadata (commit SHA, hardware, dependencies).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from typing import Any


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds across Python, NumPy, PyTorch, and CUDA."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_git_info(repo_dir: Path | None = None) -> dict[str, str]:
    """Retrieve current Git commit SHA and branch name cleanly."""
    if repo_dir is None:
        repo_dir = Path(__file__).resolve().parent.parent.parent

    info = {"commit_sha": "unknown", "branch": "unknown", "is_dirty": "unknown"}
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "-C", str(repo_dir), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["commit_sha"] = sha
        info["branch"] = branch
        info["is_dirty"] = "true" if status else "false"
    except Exception:
        pass
    return info


def collect_environment_metadata(repo_dir: Path | None = None) -> dict[str, Any]:
    """Capture environment and dependency metadata for reproducibility."""
    meta: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "git": get_git_info(repo_dir),
        "packages": {},
        "gpu": {
            "cuda_available": False,
            "device_count": 0,
            "device_name": None,
            "total_memory_gb": None,
            "cuda_version": None,
        },
    }

    # Inspect PyTorch & CUDA
    try:
        import torch
        meta["packages"]["torch"] = torch.__version__
        meta["gpu"]["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            meta["gpu"]["device_count"] = torch.cuda.device_count()
            meta["gpu"]["device_name"] = torch.cuda.get_device_name(0)
            meta["gpu"]["total_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )
            meta["gpu"]["cuda_version"] = torch.version.cuda
    except ImportError:
        meta["packages"]["torch"] = "not_installed"

    # Inspect Ultralytics
    try:
        import ultralytics
        meta["packages"]["ultralytics"] = ultralytics.__version__
    except ImportError:
        meta["packages"]["ultralytics"] = "not_installed"

    # Inspect Pillow
    try:
        import PIL
        meta["packages"]["pillow"] = PIL.__version__
    except ImportError:
        meta["packages"]["pillow"] = "not_installed"

    # Inspect NumPy
    try:
        import numpy as np
        meta["packages"]["numpy"] = np.__version__
    except ImportError:
        meta["packages"]["numpy"] = "not_installed"

    return meta
