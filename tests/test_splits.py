"""Unit tests for Phase 1 group-aware split generation and leakage prevention."""

from __future__ import annotations

from pathlib import Path
import pytest

from src.data.splits import (
    assign_splits,
    build_groups,
    merge_duplicate_groups,
    validate_manifest,
    SplitManifest,
    SplitRecord,
)


def test_build_groups_specimen_naming():
    stems = [
        "1_2_000001_0805100939749_NG_part0",
        "1_2_000001_0805100939749_NG_part1",
        "1_2_000001_0805100939749_NG_full",
        "defect001",
        "defect002",
        "uneven165",
    ]
    groups = build_groups(stems)
    assert "1_2_000001_0805100939749_NG" in groups
    assert set(groups["1_2_000001_0805100939749_NG"]) == {
        "1_2_000001_0805100939749_NG_part0",
        "1_2_000001_0805100939749_NG_part1",
        "1_2_000001_0805100939749_NG_full",
    }
    # Singletons
    assert groups["defect001"] == ["defect001"]
    assert groups["defect002"] == ["defect002"]
    assert groups["uneven165"] == ["uneven165"]


def test_merge_duplicate_groups():
    raw_groups = {
        "1_2_000244_1122094923254_NG": ["1_2_000244_1122094923254_NG_part0"],
        "defect533": ["defect533"],
        "uneven165": ["uneven165"],
        "uneven169": ["uneven169"],
        "other_group": ["other_stem"],
    }
    dup_pairs = [
        ("1_2_000244_1122094923254_NG_part0", "defect533"),
        ("uneven165", "uneven169"),
    ]
    merged = merge_duplicate_groups(raw_groups, dup_pairs)
    
    # Check that merged groups contain both stems together
    found_pair1 = False
    found_pair2 = False
    for gid, members in merged.items():
        if "1_2_000244_1122094923254_NG_part0" in members:
            assert "defect533" in members
            found_pair1 = True
        if "uneven165" in members:
            assert "uneven169" in members
            found_pair2 = True
    assert found_pair1 and found_pair2
    assert len(merged) == 3


def test_assign_splits_no_group_leakage():
    groups = {
        f"group_{i}": [f"group_{i}_part0", f"group_{i}_part1"]
        for i in range(100)
    }
    split_assignment = assign_splits(groups, train_ratio=0.7, val_ratio=0.15, seed=42)
    
    # Check that parts of the same group always share the split
    for gid, members in groups.items():
        splits_for_group = {split_assignment[m] for m in members}
        assert len(splits_for_group) == 1, f"Group {gid} leaked across splits: {splits_for_group}"

    # Check approximate distribution
    counts = {}
    for s in split_assignment.values():
        counts[s] = counts.get(s, 0) + 1
    assert counts["train"] == 140  # 70 groups * 2
    assert counts["val"] == 30    # 15 groups * 2
    assert counts["test"] == 30   # 15 groups * 2


def test_validate_manifest_detects_leakage():
    # Manifest with intentional leakage
    records = [
        SplitRecord("stem1", "group_A", "specimen", "train"),
        SplitRecord("stem2", "group_A", "specimen", "test"),  # LEAK!
        SplitRecord("stem3", "group_B", "singleton", "val"),
    ]
    manifest = SplitManifest(
        seed=42,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        total_images=3,
        total_groups=2,
        leakage_candidate_groups_in_vendor_split=0,
        records=records,
        split_counts={"train": 1, "val": 1, "test": 1},
    )
    violations = validate_manifest(manifest)
    assert any("GROUP LEAKAGE" in v for v in violations)


def test_validate_manifest_passes_valid():
    records = [
        SplitRecord("stem1", "group_A", "specimen", "train"),
        SplitRecord("stem2", "group_A", "specimen", "train"),
        SplitRecord("stem3", "group_B", "singleton", "val"),
        SplitRecord("stem4", "group_C", "singleton", "test"),
    ]
    manifest = SplitManifest(
        seed=42,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        total_images=4,
        total_groups=3,
        leakage_candidate_groups_in_vendor_split=0,
        records=records,
        split_counts={"train": 2, "val": 1, "test": 1},
    )
    violations = validate_manifest(manifest)
    assert len(violations) == 0
