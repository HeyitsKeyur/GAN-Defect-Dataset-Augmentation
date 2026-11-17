"""Unit tests for Phase 2: Configuration, Environment Resolver, and Smoke Test."""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from src.utils.config import ProjectConfig, find_project_root, get_config, is_colab, load_dotenv
from src.utils.smoke_test import check_gpu, inspect_packages, run_smoke_test, run_tensor_operation, verify_frozen_splits


def test_find_project_root():
    root = find_project_root()
    assert root.is_dir()
    assert (root / "AGENTS.md").is_file() or (root / "PROJECT_CONTEXT.md").is_file()


def test_get_config_defaults():
    config = get_config()
    assert config.project_root.is_dir()
    assert config.split_manifest_dir.name == "phase1"
    assert config.random_seed == 42
    assert config.environment_name in {"windows", "unix", "colab"}


def test_config_dataset_validation():
    config = get_config()
    ok, errors = config.validate_dataset()
    # In local repo, dataset is present under Dateset/...
    assert ok is True
    assert len(errors) == 0


def test_config_splits_validation():
    config = get_config()
    ok, errors = config.validate_splits()
    assert ok is True
    assert len(errors) == 0


def test_verify_frozen_splits_integrity():
    config = get_config()
    passed, counts, leakage, errors = verify_frozen_splits(config.split_manifest_dir)
    assert passed is True
    assert leakage is False
    assert len(errors) == 0
    assert counts.get("train") == 1967
    assert counts.get("val") == 479
    assert counts.get("test") == 446
    assert sum(counts.values()) == 2892


def test_tensor_operation_exec():
    success, device_name = run_tensor_operation(gpu_available=False)
    assert success is True
    assert "cpu" in device_name.lower() or "torch" in device_name.lower()


def test_run_smoke_test_pass():
    config = get_config()
    result = run_smoke_test(config)
    assert result.overall_status == "PASS"
    assert result.split_manifests_verified is True
    assert result.split_leakage_detected is False
    assert result.tensor_op_success is True
