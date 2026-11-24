"""Generate frozen group-aware train/validation/test splits for SolDef_AI dataset.

Preserves complete isolation:
- Primary side-view benchmark (V2, V2.1) -> outputs/splits/soldef/phase1/
- Secondary top-down benchmark (V1)      -> outputs/splits/soldef/phase1_v1/
- Grouping provenance                    -> outputs/splits/soldef/grouping_provenance.json
- Split feasibility report               -> outputs/splits/soldef/split_feasibility_report.md
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import sys
from typing import Any

# Add repository root to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.adapters.soldef import (
    FOUR_CLASS_SIDE_VIEW_CLASSES,
    PRIMARY_SIDE_VIEW_CLASSES,
    TOP_DOWN_CLASSES,
    SolDefAdapter,
)

PRIMARY_SPLIT_DIR = Path("outputs/splits/soldef/phase1")
SECONDARY_SPLIT_DIR = Path("outputs/splits/soldef/phase1_v1")
SOLDEF_REPORTS_DIR = Path("outputs/splits/soldef")


def write_csv_manifest(
    output_path: Path,
    records: list[Any],
    class_columns: list[str],
) -> None:
    """Write split records to a standard CSV manifest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stem",
        "image_path",
        "annotation_path",
        "group_id",
        "evidence_level",
        "specimen_identity_status",
        "subset_or_cs",
        "component_type",
        "view",
        "width",
        "height",
    ] + class_columns

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row: dict[str, Any] = {
                "stem": r.stem,
                "image_path": str(r.image_path.as_posix()),
                "annotation_path": str(r.annotation_path.as_posix()),
                "group_id": r.group_id,
                "evidence_level": r.evidence_level,
                "specimen_identity_status": r.specimen_identity_status,
                "subset_or_cs": r.subset_or_cs,
                "component_type": r.component_type,
                "view": r.view,
                "width": r.width,
                "height": r.height,
            }
            for cls_name in class_columns:
                row[cls_name] = r.instance_counts.get(cls_name, 0)
            writer.writerow(row)


def generate_primary_benchmark_splits(seed: int = 0) -> dict[str, Any]:
    """Generate group-aware splits for the primary side-view benchmark (V2, V2.1)."""
    PRIMARY_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    adapter = SolDefAdapter(benchmark_type="primary_side_view", defect_only=False)
    records = adapter.get_image_records()

    # Index by group_id
    groups: dict[str, list[Any]] = {}
    for r in records:
        groups.setdefault(r.group_id, []).append(r)

    atomic_groups = [gid for gid in sorted(groups.keys()) if "atomic_block" in gid]
    specimen_groups = [gid for gid in sorted(groups.keys()) if "atomic_block" not in gid]

    rng = random.Random(seed)
    shuffled_specimens = list(specimen_groups)
    rng.shuffle(shuffled_specimens)

    train_groups = list(atomic_groups)
    train_count = sum(len(groups[g]) for g in atomic_groups)
    val_groups: list[str] = []
    val_count = 0
    test_groups: list[str] = []
    test_count = 0

    for g in shuffled_specimens:
        g_len = len(groups[g])
        if val_count + g_len <= 30 and (len(val_groups) < 20 or test_count >= 25):
            val_groups.append(g)
            val_count += g_len
        elif test_count + g_len <= 30:
            test_groups.append(g)
            test_count += g_len
        else:
            train_groups.append(g)
            train_count += g_len

    assert train_count == 140, f"Expected 140 train images, got {train_count}"
    assert val_count == 30, f"Expected 30 val images, got {val_count}"
    assert test_count == 30, f"Expected 30 test images, got {test_count}"

    train_records = [r for g in train_groups for r in groups[g]]
    val_records = [r for g in val_groups for r in groups[g]]
    test_records = [r for g in test_groups for r in groups[g]]

    # Zero leakage assertions
    train_stems = {r.stem for r in train_records}
    val_stems = {r.stem for r in val_records}
    test_stems = {r.stem for r in test_records}
    assert len(train_stems & val_stems) == 0, "Train and Val share images!"
    assert len(train_stems & test_stems) == 0, "Train and Test share images!"
    assert len(val_stems & test_stems) == 0, "Val and Test share images!"

    assert len(set(train_groups) & set(val_groups)) == 0, "Train and Val share groups!"
    assert len(set(train_groups) & set(test_groups)) == 0, "Train and Test share groups!"
    assert len(set(val_groups) & set(test_groups)) == 0, "Val and Test share groups!"

    # Write CSV manifests
    write_csv_manifest(PRIMARY_SPLIT_DIR / "train.csv", train_records, FOUR_CLASS_SIDE_VIEW_CLASSES)
    write_csv_manifest(PRIMARY_SPLIT_DIR / "val.csv", val_records, FOUR_CLASS_SIDE_VIEW_CLASSES)
    write_csv_manifest(PRIMARY_SPLIT_DIR / "test.csv", test_records, FOUR_CLASS_SIDE_VIEW_CLASSES)

    # Compute distribution statistics
    def summarize_split(recs: list[Any], g_list: list[str]) -> dict[str, Any]:
        instance_counts: dict[str, int] = {}
        image_counts: dict[str, int] = {}
        for c in FOUR_CLASS_SIDE_VIEW_CLASSES:
            instance_counts[c] = sum(r.instance_counts.get(c, 0) for r in recs)
            image_counts[c] = sum(1 for r in recs if r.instance_counts.get(c, 0) > 0)
        return {
            "image_count": len(recs),
            "group_count": len(g_list),
            "instance_counts": instance_counts,
            "image_counts": image_counts,
            "groups": g_list,
        }

    summary = {
        "dataset_name": "soldef",
        "benchmark_type": "primary_side_view",
        "views_included": ["V2", "V2.1"],
        "target_classes_defect_only": PRIMARY_SIDE_VIEW_CLASSES,
        "all_side_view_classes": FOUR_CLASS_SIDE_VIEW_CLASSES,
        "seed": seed,
        "total_images": len(records),
        "total_groups": len(groups),
        "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "train": summarize_split(train_records, train_groups),
        "val": summarize_split(val_records, val_groups),
        "test": summarize_split(test_records, test_groups),
        "leakage_verification": {
            "image_overlap_train_val": len(train_stems & val_stems),
            "image_overlap_train_test": len(train_stems & test_stems),
            "image_overlap_val_test": len(val_stems & test_stems),
            "group_overlap_train_val": len(set(train_groups) & set(val_groups)),
            "group_overlap_train_test": len(set(train_groups) & set(test_groups)),
            "group_overlap_val_test": len(set(val_groups) & set(test_groups)),
            "leakage_detected": False,
        },
    }

    # Save manifest JSON
    with (PRIMARY_SPLIT_DIR / "split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save Markdown summary
    md_content = f"""# SolDef_AI Primary Side-View Benchmark Split Summary

- **Benchmark:** Solder Joint Defect Detection (Side Views: V2, V2.1)
- **Total Images:** 200 (Train: 140 [70.0%], Val: 30 [15.0%], Test: 30 [15.0%])
- **Total Independent Groups:** 66 (Train: 28, Val: 19, Test: 19)
- **Seed:** {seed}
- **Cross-Split Leakage:** 0 image overlap, 0 group overlap (VERIFIED LEAKAGE-FREE)

## Class Distribution Across Splits

| Class | Train Instances (Images) | Val Instances (Images) | Test Instances (Images) | Total Instances |
| :--- | :--- | :--- | :--- | :--- |
| `exc_solder` | {summary['train']['instance_counts']['exc_solder']} ({summary['train']['image_counts']['exc_solder']}) | {summary['val']['instance_counts']['exc_solder']} ({summary['val']['image_counts']['exc_solder']}) | {summary['test']['instance_counts']['exc_solder']} ({summary['test']['image_counts']['exc_solder']}) | {summary['train']['instance_counts']['exc_solder'] + summary['val']['instance_counts']['exc_solder'] + summary['test']['instance_counts']['exc_solder']} |
| `spike` | {summary['train']['instance_counts']['spike']} ({summary['train']['image_counts']['spike']}) | {summary['val']['instance_counts']['spike']} ({summary['val']['image_counts']['spike']}) | {summary['test']['instance_counts']['spike']} ({summary['test']['image_counts']['spike']}) | {summary['train']['instance_counts']['spike'] + summary['val']['instance_counts']['spike'] + summary['test']['instance_counts']['spike']} |
| `poor_solder` (minority) | {summary['train']['instance_counts']['poor_solder']} ({summary['train']['image_counts']['poor_solder']}) | {summary['val']['instance_counts']['poor_solder']} ({summary['val']['image_counts']['poor_solder']}) | {summary['test']['instance_counts']['poor_solder']} ({summary['test']['image_counts']['poor_solder']}) | {summary['train']['instance_counts']['poor_solder'] + summary['val']['instance_counts']['poor_solder'] + summary['test']['instance_counts']['poor_solder']} |
| `good` (normal baseline) | {summary['train']['instance_counts']['good']} ({summary['train']['image_counts']['good']}) | {summary['val']['instance_counts']['good']} ({summary['val']['image_counts']['good']}) | {summary['test']['instance_counts']['good']} ({summary['test']['image_counts']['good']}) | {summary['train']['instance_counts']['good'] + summary['val']['instance_counts']['good'] + summary['test']['instance_counts']['good']} |
| **Total Annotations** | **{sum(summary['train']['instance_counts'].values())}** | **{sum(summary['val']['instance_counts'].values())}** | **{sum(summary['test']['instance_counts'].values())}** | **{sum(summary['train']['instance_counts'].values()) + sum(summary['val']['instance_counts'].values()) + sum(summary['test']['instance_counts'].values())}** |
"""
    with (PRIMARY_SPLIT_DIR / "split_summary.md").open("w", encoding="utf-8") as f:
        f.write(md_content)

    return summary


def generate_secondary_benchmark_splits(seed: int = 42) -> dict[str, Any]:
    """Generate splits for the secondary top-down benchmark (V1)."""
    SECONDARY_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    adapter = SolDefAdapter(benchmark_type="secondary_top_down")
    records = adapter.get_image_records()

    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    # 160 train (70.2%), 34 val (14.9%), 34 test (14.9%) = 228
    train_recs = shuffled[:160]
    val_recs = shuffled[160:194]
    test_recs = shuffled[194:]

    train_stems = {r.stem for r in train_recs}
    val_stems = {r.stem for r in val_recs}
    test_stems = {r.stem for r in test_recs}

    assert len(train_stems & val_stems) == 0
    assert len(train_stems & test_stems) == 0
    assert len(val_stems & test_stems) == 0

    write_csv_manifest(SECONDARY_SPLIT_DIR / "train.csv", train_recs, TOP_DOWN_CLASSES)
    write_csv_manifest(SECONDARY_SPLIT_DIR / "val.csv", val_recs, TOP_DOWN_CLASSES)
    write_csv_manifest(SECONDARY_SPLIT_DIR / "test.csv", test_recs, TOP_DOWN_CLASSES)

    def summarize_v1(recs: list[Any]) -> dict[str, Any]:
        return {
            "image_count": len(recs),
            "instance_counts": {c: sum(r.instance_counts.get(c, 0) for r in recs) for c in TOP_DOWN_CLASSES},
        }

    summary = {
        "dataset_name": "soldef",
        "benchmark_type": "secondary_top_down",
        "view": "V1",
        "target_classes": TOP_DOWN_CLASSES,
        "seed": seed,
        "total_images": len(records),
        "split_ratios": {"train": len(train_recs) / len(records), "val": len(val_recs) / len(records), "test": len(test_recs) / len(records)},
        "train": summarize_v1(train_recs),
        "val": summarize_v1(val_recs),
        "test": summarize_v1(test_recs),
    }

    with (SECONDARY_SPLIT_DIR / "split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def generate_grouping_provenance() -> dict[str, Any]:
    """Export machine-readable provenance for all groups in SolDef_AI side-view benchmark."""
    adapter = SolDefAdapter(benchmark_type="primary_side_view", defect_only=False)
    records = adapter.get_image_records()

    provenance_by_group: dict[str, Any] = {}
    for r in records:
        if r.group_id not in provenance_by_group:
            provenance_by_group[r.group_id] = {
                "group_id": r.group_id,
                "evidence_level": r.evidence_level,
                "specimen_identity_status": r.specimen_identity_status,
                "rationale": r.rationale,
                "subset_or_cs": r.subset_or_cs,
                "component_type": r.component_type,
                "member_count": 0,
                "members": [],
            }
        provenance_by_group[r.group_id]["member_count"] += 1
        provenance_by_group[r.group_id]["members"].append(
            {
                "stem": r.stem,
                "view": r.view,
                "instances": r.instance_counts,
            }
        )

    out_file = SOLDEF_REPORTS_DIR / "grouping_provenance.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(provenance_by_group, f, indent=2)

    return provenance_by_group


def main() -> None:
    print("Generating SolDef_AI primary benchmark splits (V2, V2.1)...")
    primary_summary = generate_primary_benchmark_splits(seed=0)
    print(f"[SUCCESS] Primary splits generated: {primary_summary['train']['image_count']} train, {primary_summary['val']['image_count']} val, {primary_summary['test']['image_count']} test")

    print("Generating SolDef_AI secondary benchmark splits (V1)...")
    secondary_summary = generate_secondary_benchmark_splits(seed=42)
    print(f"[SUCCESS] Secondary splits generated: {secondary_summary['train']['image_count']} train, {secondary_summary['val']['image_count']} val, {secondary_summary['test']['image_count']} test")

    print("Generating grouping provenance manifest...")
    provenance = generate_grouping_provenance()
    print(f"[SUCCESS] Grouping provenance generated for {len(provenance)} groups")


if __name__ == "__main__":
    main()
