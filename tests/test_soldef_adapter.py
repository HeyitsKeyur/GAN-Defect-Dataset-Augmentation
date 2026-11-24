"""Unit tests for SolDef_AI dataset adapter and frozen group-aware splits."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest

from src.data.adapters import CeramicAdapter, SolDefAdapter, get_adapter
from src.data.adapters.base import DatasetImageRecord
from src.data.preprocessing.crop_geometry import BoundingBox

SOLDEF_ROOT = Path("Dateset/SolDef_AI")
PRIMARY_SPLIT_DIR = Path("outputs/splits/soldef/phase1")
SECONDARY_SPLIT_DIR = Path("outputs/splits/soldef/phase1_v1")


@pytest.mark.skipif(not SOLDEF_ROOT.exists(), reason="SolDef_AI dataset directory not found")
class TestSolDefAdapter:
    """Test suite for SolDefAdapter and benchmark modes."""

    def test_adapter_initialization(self) -> None:
        """Verify adapter initialization and property mappings."""
        adapter_pri = SolDefAdapter(benchmark_type="primary_side_view", defect_only=True)
        assert adapter_pri.dataset_name == "soldef"
        assert adapter_pri.target_classes == ["exc_solder", "spike", "poor_solder"]
        assert adapter_pri.class_to_id == {"exc_solder": 0, "spike": 1, "poor_solder": 2}
        assert adapter_pri.id_to_class == {0: "exc_solder", 1: "spike", 2: "poor_solder"}

        adapter_4c = SolDefAdapter(benchmark_type="primary_side_view", defect_only=False)
        assert "good" in adapter_4c.target_classes
        assert len(adapter_4c.target_classes) == 4

        adapter_v1 = SolDefAdapter(benchmark_type="secondary_top_down")
        assert adapter_v1.target_classes == ["good", "no_good"]

        adapter_all = SolDefAdapter(benchmark_type="all_inclusive")
        assert len(adapter_all.target_classes) == 5

        with pytest.raises(ValueError, match="Unknown benchmark_type"):
            SolDefAdapter(benchmark_type="invalid_type")

    def test_factory_function(self) -> None:
        """Verify get_adapter factory returns expected adapter types."""
        a_soldef = get_adapter("soldef")
        assert isinstance(a_soldef, SolDefAdapter)

        a_ceramic = get_adapter("ceramic")
        assert isinstance(a_ceramic, CeramicAdapter)

        with pytest.raises(ValueError, match="Unsupported dataset name"):
            get_adapter("unknown_dataset_xyz")

    def test_primary_side_view_records(self) -> None:
        """Verify primary side-view records count, resolution, and views."""
        adapter = SolDefAdapter(benchmark_type="primary_side_view")
        records = adapter.get_image_records()

        assert len(records) == 200, f"Expected 200 side-view images, got {len(records)}"
        for r in records:
            assert isinstance(r, DatasetImageRecord)
            assert r.dataset_name == "soldef"
            assert r.width == 2560
            assert r.height == 1440
            assert r.view in ("V2", "V2.1")
            assert r.evidence_level in ("verified", "inferred", "atomic_fallback")
            assert r.specimen_identity_status in ("verified", "inferred", "unknown")
            assert len(r.rationale) > 0

    def test_grouping_provenance_and_hierarchy(self) -> None:
        """Verify 66 independent groups with exact provenance tiers."""
        adapter = SolDefAdapter(benchmark_type="primary_side_view")
        records = adapter.get_image_records()

        groups = {r.group_id for r in records}
        assert len(groups) == 66, f"Expected 66 independent groups, got {len(groups)}"

        group_ev = {}
        group_counts = {}
        for r in records:
            group_ev[r.group_id] = r.evidence_level
            group_counts[r.group_id] = group_counts.get(r.group_id, 0) + 1

        ev_counts = {}
        for ev in group_ev.values():
            ev_counts[ev] = ev_counts.get(ev, 0) + 1

        assert ev_counts["verified"] == 14
        assert ev_counts["inferred"] == 50
        assert ev_counts["atomic_fallback"] == 2

        # Check atomic fallback blocks sizes
        assert group_counts["CS2_R0805_atomic_block"] == 36
        assert group_counts["CS6_R0805_atomic_block"] == 63

    def test_polygon_to_bounding_box_conversion(self) -> None:
        """Verify polygon-to-box conversion, coordinate clipping, and defect filtering."""
        adapter = SolDefAdapter(benchmark_type="primary_side_view", defect_only=True)
        records = adapter.get_image_records()

        # Find a record with annotations
        sample = next(r for r in records if len(r.classes_present) > 0)
        defect_boxes = adapter.load_annotations(sample.annotation_path, defect_only=True)

        for b in defect_boxes:
            assert isinstance(b, BoundingBox)
            assert 0.0 <= b.xmin < b.xmax <= 2560.0
            assert 0.0 <= b.ymin < b.ymax <= 1440.0
            assert b.class_name in ("exc_solder", "spike", "poor_solder")
            assert b.class_name != "good"

        # Check defect_only=False retains 'good' if present in raw annotations
        all_boxes = adapter.load_annotations(sample.annotation_path, defect_only=False)
        assert len(all_boxes) >= len(defect_boxes)


@pytest.mark.skipif(not PRIMARY_SPLIT_DIR.exists(), reason="SolDef split manifests not generated yet")
class TestSolDefSplitIntegrity:
    """Test suite for frozen SolDef split manifests and leakage prevention."""

    def test_primary_split_files_exist(self) -> None:
        """Verify all primary split manifest files are present."""
        assert (PRIMARY_SPLIT_DIR / "train.csv").exists()
        assert (PRIMARY_SPLIT_DIR / "val.csv").exists()
        assert (PRIMARY_SPLIT_DIR / "test.csv").exists()
        assert (PRIMARY_SPLIT_DIR / "split_manifest.json").exists()
        assert (PRIMARY_SPLIT_DIR / "split_summary.md").exists()

    def test_primary_split_zero_leakage_and_proportions(self) -> None:
        """Verify strict 70/15/15 ratio and 0 image/group leakage."""
        def load_manifest(csv_path: Path) -> tuple[set[str], set[str], dict[str, int]]:
            stems: set[str] = set()
            groups: set[str] = set()
            class_instances: dict[str, int] = {}
            with csv_path.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    stems.add(row["stem"])
                    groups.add(row["group_id"])
                    for c in ("exc_solder", "spike", "poor_solder", "good"):
                        class_instances[c] = class_instances.get(c, 0) + int(row.get(c, 0))
            return stems, groups, class_instances

        tr_stems, tr_groups, tr_classes = load_manifest(PRIMARY_SPLIT_DIR / "train.csv")
        va_stems, va_groups, va_classes = load_manifest(PRIMARY_SPLIT_DIR / "val.csv")
        te_stems, te_groups, te_classes = load_manifest(PRIMARY_SPLIT_DIR / "test.csv")

        # Counts: exactly 140 train, 30 val, 30 test
        assert len(tr_stems) == 140
        assert len(va_stems) == 30
        assert len(te_stems) == 30
        assert len(tr_stems | va_stems | te_stems) == 200

        # Zero image overlap
        assert len(tr_stems & va_stems) == 0
        assert len(tr_stems & te_stems) == 0
        assert len(va_stems & te_stems) == 0

        # Zero group overlap (zero view leakage)
        assert len(tr_groups & va_groups) == 0
        assert len(tr_groups & te_groups) == 0
        assert len(va_groups & te_groups) == 0
        assert len(tr_groups | va_groups | te_groups) == 66

        # Class presence: every defect class present in train, val, and test
        for c in ("exc_solder", "spike", "poor_solder", "good"):
            assert tr_classes[c] > 0, f"Train missing class {c}"
            assert va_classes[c] > 0, f"Val missing class {c}"
            assert te_classes[c] > 0, f"Test missing class {c}"

        # Minority class poor_solder representation
        assert tr_classes["poor_solder"] >= 30
        assert va_classes["poor_solder"] >= 8
        assert te_classes["poor_solder"] >= 10

    def test_secondary_v1_split_integrity(self) -> None:
        """Verify secondary V1 top-down split counts and leakage prevention."""
        assert (SECONDARY_SPLIT_DIR / "train.csv").exists()
        assert (SECONDARY_SPLIT_DIR / "val.csv").exists()
        assert (SECONDARY_SPLIT_DIR / "test.csv").exists()

        def load_v1_stems(csv_path: Path) -> set[str]:
            stems: set[str] = set()
            with csv_path.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    stems.add(row["stem"])
            return stems

        tr = load_v1_stems(SECONDARY_SPLIT_DIR / "train.csv")
        va = load_v1_stems(SECONDARY_SPLIT_DIR / "val.csv")
        te = load_v1_stems(SECONDARY_SPLIT_DIR / "test.csv")

        assert len(tr) == 160
        assert len(va) == 34
        assert len(te) == 34
        assert len(tr | va | te) == 228
        assert len(tr & va) == 0
        assert len(tr & te) == 0
        assert len(va & te) == 0
