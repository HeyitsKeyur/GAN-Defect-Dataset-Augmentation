"""Phase 2 Smoke Test & Runtime Verification Module.

Performs non-training verification:
1. Environment & Python details
2. Installed dependencies
3. GPU detection, model name & VRAM memory
4. Google Drive mount status & dataset path integrity
5. Frozen Phase 1 split manifest integrity (disjointness, group leakage check)
6. Simple PyTorch CPU/GPU matrix multiplication
7. Generates machine-readable JSON & human-readable Markdown test reports.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.config import ProjectConfig, get_config, is_colab


@dataclass
class SmokeTestResult:
    timestamp_utc: str
    environment_name: str
    is_colab_runtime: bool
    python_version: str
    platform_details: str
    installed_packages: dict[str, str]
    gpu_available: bool
    gpu_name: str | None
    gpu_count: int
    vram_total_gb: float | None
    torch_version: str | None
    cuda_version: str | None
    dataset_verified: bool
    dataset_path: str
    dataset_errors: list[str]
    split_manifests_verified: bool
    split_manifest_dir: str
    split_counts: dict[str, int]
    split_leakage_detected: bool
    split_errors: list[str]
    tensor_op_success: bool
    tensor_device: str
    overall_status: str  # "PASS" | "FAIL" | "WARN"
    notes: list[str] = field(default_factory=list)


def inspect_packages() -> dict[str, str]:
    """Capture versions of relevant core packages without heavy dependencies."""
    packages = ["torch", "torchvision", "PIL", "numpy", "pytest"]
    results: dict[str, str] = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            results[pkg] = str(ver)
        except ImportError:
            results[pkg] = "not installed"
    return results


def check_gpu() -> tuple[bool, str | None, int, float | None, str | None, str | None]:
    """Detect GPU availability, device name, and memory via PyTorch."""
    try:
        import torch

        torch_ver = torch.__version__
        cuda_ver = torch.version.cuda
        available = torch.cuda.is_available()
        if available:
            count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(props.total_memory / (1024**3), 2)
            return True, name, count, vram_gb, torch_ver, cuda_ver
        else:
            return False, None, 0, None, torch_ver, cuda_ver
    except ImportError:
        return False, None, 0, None, None, None


def verify_frozen_splits(manifest_dir: Path) -> tuple[bool, dict[str, int], bool, list[str]]:
    """Verify that Phase 1 split manifests exist, are non-empty, disjoint, and group-leakage-free."""
    errors: list[str] = []
    split_counts: dict[str, int] = {}
    stems_seen: dict[str, str] = {}
    group_to_splits: dict[str, set[str]] = defaultdict(set)
    leakage_detected = False

    manifest_json_path = manifest_dir / "split_manifest.json"
    if not manifest_json_path.is_file():
        errors.append(f"split_manifest.json missing in {manifest_dir}")

    for split_name in ("train", "val", "test"):
        csv_path = manifest_dir / f"{split_name}.csv"
        if not csv_path.is_file():
            errors.append(f"Missing manifest file: {csv_path.name}")
            continue

        try:
            with csv_path.open(encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                count = 0
                for row in reader:
                    count += 1
                    stem = row.get("stem", "")
                    group_id = row.get("group_id", stem)

                    if stem in stems_seen:
                        errors.append(f"Duplicate stem across splits: {stem} (in {stems_seen[stem]} and {split_name})")
                    stems_seen[stem] = split_name
                    group_to_splits[group_id].add(split_name)

                split_counts[split_name] = count
        except Exception as exc:
            errors.append(f"Error reading {csv_path.name}: {exc}")

    # Check for group leakage
    leaking_groups = {gid: splits for gid, splits in group_to_splits.items() if len(splits) > 1}
    if leaking_groups:
        leakage_detected = True
        sample_leaks = list(leaking_groups.items())[:3]
        errors.append(f"Group leakage detected in {len(leaking_groups)} groups: {sample_leaks}")

    total_images = sum(split_counts.values())
    if total_images != 2892 and len(split_counts) == 3:
        errors.append(f"Expected 2892 total split images, got {total_images}")

    passed = len(errors) == 0 and not leakage_detected
    return passed, split_counts, leakage_detected, errors


def run_tensor_operation(gpu_available: bool = False) -> tuple[bool, str]:
    """Execute a lightweight non-training tensor operation on GPU (or CPU fallback, or NumPy if torch not installed)."""
    try:
        import torch

        device = torch.device("cuda:0" if gpu_available and torch.cuda.is_available() else "cpu")
        a = torch.ones((100, 100), device=device)
        b = torch.ones((100, 100), device=device)
        c = torch.matmul(a, b)
        val = float(c[0, 0].item())
        if abs(val - 100.0) < 1e-4:
            return True, f"torch ({device})"
        return False, f"torch ({device})"
    except ImportError:
        # Fallback to numpy or pure python if running on CPU-only local development without full PyTorch
        try:
            import numpy as np

            a = np.ones((100, 100), dtype=float)
            b = np.ones((100, 100), dtype=float)
            c = np.dot(a, b)
            if abs(float(c[0, 0]) - 100.0) < 1e-4:
                return True, "numpy (cpu-fallback)"
            return False, "numpy (cpu-fallback)"
        except ImportError:
            # Pure python fallback
            val = sum(1.0 * 1.0 for _ in range(100))
            if abs(val - 100.0) < 1e-4:
                return True, "python-builtin (cpu-fallback)"
            return False, "python-builtin (cpu-fallback)"
    except Exception as exc:
        return False, f"error: {exc}"


def run_smoke_test(config: ProjectConfig | None = None) -> SmokeTestResult:
    """Run all Phase 2 smoke tests and return structured result."""
    if config is None:
        config = get_config()

    timestamp = datetime.now(timezone.utc).isoformat()
    packages = inspect_packages()
    gpu_avail, gpu_name, gpu_cnt, vram, torch_ver, cuda_ver = check_gpu()

    dataset_ok, dataset_errors = config.validate_dataset()
    splits_ok, split_counts, leakage, split_errors = verify_frozen_splits(config.split_manifest_dir)

    tensor_ok, tensor_dev = run_tensor_operation(gpu_avail)

    notes: list[str] = []
    if config.environment_name == "colab":
        notes.append("Running inside Google Colab runtime.")
    else:
        notes.append(f"Running in local {platform.system()} environment.")

    if not gpu_avail:
        notes.append("No CUDA GPU detected; execution will fall back to CPU for local development.")
    else:
        notes.append(f"CUDA GPU detected: {gpu_name} with {vram} GB VRAM.")

    overall = "PASS"
    if not splits_ok or leakage or not tensor_ok:
        overall = "FAIL"
    elif not dataset_ok:
        # If dataset is missing in local environment (e.g. lightweight clone), mark WARN or FAIL depending on location
        overall = "WARN" if not config.environment_name == "colab" else "FAIL"

    return SmokeTestResult(
        timestamp_utc=timestamp,
        environment_name=config.environment_name,
        is_colab_runtime=is_colab(),
        python_version=platform.python_version(),
        platform_details=platform.platform(),
        installed_packages=packages,
        gpu_available=gpu_avail,
        gpu_name=gpu_name,
        gpu_count=gpu_cnt,
        vram_total_gb=vram,
        torch_version=torch_ver,
        cuda_version=cuda_ver,
        dataset_verified=dataset_ok,
        dataset_path=str(config.dataset_root),
        dataset_errors=dataset_errors,
        split_manifests_verified=splits_ok,
        split_manifest_dir=str(config.split_manifest_dir),
        split_counts=split_counts,
        split_leakage_detected=leakage,
        split_errors=split_errors,
        tensor_op_success=tensor_ok,
        tensor_device=tensor_dev,
        overall_status=overall,
        notes=notes,
    )


def save_smoke_test_report(result: SmokeTestResult, output_dir: Path) -> tuple[Path, Path]:
    """Save smoke test results as JSON and Markdown reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "smoke_test_report.json"
    md_path = output_dir / "smoke_test_report.md"

    # Save JSON
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(result), fh, indent=2)

    # Save Markdown
    md_content = f"""# Colab / Local Runtime Smoke Test Report

**Execution Timestamp (UTC):** `{result.timestamp_utc}`  
**Overall Status:** **{result.overall_status}**  
**Environment:** `{result.environment_name}` (Colab: {result.is_colab_runtime})  
**Python Version:** `{result.python_version}` on `{result.platform_details}`  

---

## 1. Hardware & Compute Acceleration
- **GPU Available:** {result.gpu_available}
- **GPU Model:** {result.gpu_name or "None (CPU runtime)"}
- **GPU Count:** {result.gpu_count}
- **VRAM Total:** {f"{result.vram_total_gb} GB" if result.vram_total_gb else "N/A"}
- **PyTorch Version:** `{result.torch_version}`
- **CUDA Runtime:** `{result.cuda_version or "None"}`
- **Tensor Operation Test:** {"PASSED" if result.tensor_op_success else "FAILED"} on `{result.tensor_device}`

---

## 2. Dataset & Storage Verification
- **Dataset Path:** `{result.dataset_path}`
- **Dataset Verified:** {"YES" if result.dataset_verified else "NO"}
{"" if not result.dataset_errors else "### Dataset Errors:\n" + "".join(f"- {e}\n" for e in result.dataset_errors)}

---

## 3. Phase 1 Frozen Split Manifest Verification
- **Manifest Directory:** `{result.split_manifest_dir}`
- **Manifests Verified:** {"YES" if result.split_manifests_verified else "NO"}
- **Split Counts:**
  - `train`: {result.split_counts.get("train", 0)} images
  - `val`: {result.split_counts.get("val", 0)} images
  - `test`: {result.split_counts.get("test", 0)} images
  - **Total Images:** {sum(result.split_counts.values())} (Expected: 2892)
- **Cross-Split Group Leakage Detected:** {"YES (CRITICAL VIOLATION)" if result.split_leakage_detected else "NO (100% LEAKAGE-FREE)"}
{"" if not result.split_errors else "### Split Manifest Errors:\n" + "".join(f"- {e}\n" for e in result.split_errors)}

---

## 4. Installed Packages Summary
| Package | Version |
|---|---|
{ "".join(f"| `{k}` | `{v}` |\n" for k, v in result.installed_packages.items()) }

---

## 5. Notes & Observations
{ "".join(f"- {n}\n" for n in result.notes) }
"""
    md_path.write_text(md_content, encoding="utf-8")
    return json_path, md_path
