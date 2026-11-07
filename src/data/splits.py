"""Phase 1: Group-aware, leakage-safe train/validation/test split.

Design decisions
----------------
- Groups are identified from filename stems using the existing ``infer_group_id``
  heuristic (``d_d_dddd_dddd_NG`` pattern and its ``_partN`` / ``_full`` variants).
- Singleton filenames (class-prefixed names like ``defect001``) form their own
  one-member groups.
- Confirmed exact MD5 duplicate pairs from the Phase 0 audit are merged into the
  same super-group so they always land in the same split.
- Groups are shuffled with a fixed seed and assigned to splits at the group level,
  so no substrate tile crosses the train/val/test boundary.
- Class distribution within each split is computed from JSON annotations (the
  primary format used throughout Phase 0 and 1).
- Split manifests are saved as CSV files and a JSON summary for reproducibility.
- This module does NOT modify any original dataset file.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data.dataset_paths import DatasetLayout, iter_files, resolve_layout
from src.data.image_utils import infer_group_id

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Observed YOLO ID → semantic class name mapping (from Phase 0 audit).
# This mapping is derived from annotation file inspection, not an official codebook.
YOLO_ID_TO_CLASS: dict[int, str] = {
    0: "adhesion",
    1: "scar",
    2: "spot",
    3: "uneven",
    4: "abnormal_thickness",
    5: "defect",
}

# Default split ratios (at group level, not image level).
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
# test ratio = 1.0 - train - val = 0.15

DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SplitRecord:
    """One row in a split manifest."""

    stem: str
    group_id: str
    group_evidence: str
    split: str
    classes_present: list[str] = field(default_factory=list)
    instance_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class SplitManifest:
    """Full split result with metadata."""

    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    total_images: int
    total_groups: int
    leakage_candidate_groups_in_vendor_split: int
    records: list[SplitRecord] = field(default_factory=list)
    split_counts: dict[str, int] = field(default_factory=dict)
    class_distribution: dict[str, dict[str, int]] = field(default_factory=dict)
    group_size_histogram: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Step 1 — Build groups from stems
# ---------------------------------------------------------------------------


def build_groups(stems: list[str]) -> dict[str, list[str]]:
    """Return a mapping group_id → list[stem].

    The ``infer_group_id`` function assigns the full stem as the group_id for
    singleton filenames, so every stem appears in exactly one group.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for stem in stems:
        group_id, _ = infer_group_id(stem)
        groups[group_id].append(stem)
    return dict(groups)


# ---------------------------------------------------------------------------
# Step 2 — Merge groups that contain confirmed exact duplicates
# ---------------------------------------------------------------------------


def load_md5_duplicate_pairs(audit_report_path: Path) -> list[tuple[str, str]]:
    """Load confirmed MD5 duplicate pairs from the Phase 0 audit JSON.

    Returns a list of (stem_a, stem_b) pairs.
    """
    pairs: list[tuple[str, str]] = []
    if not audit_report_path.exists():
        return pairs
    with audit_report_path.open(encoding="utf-8") as fh:
        report = json.load(fh)
    dup_groups: dict[str, list[str]] = (
        report.get("duplicates", {}).get("confirmed_exact_md5_duplicate_groups", {})
    )
    for stems in dup_groups.values():
        if len(stems) >= 2:
            # Pair each member with every other member.
            for i in range(len(stems)):
                for j in range(i + 1, len(stems)):
                    pairs.append((stems[i], stems[j]))
    return pairs


def merge_duplicate_groups(
    groups: dict[str, list[str]],
    duplicate_pairs: list[tuple[str, str]],
) -> dict[str, list[str]]:
    """Merge groups that share any confirmed duplicate pair.

    Uses a union-find approach so chains are merged correctly:
    if A↔B and B↔C then A, B, C all end up in the same group.

    Returns a new groups dict (original is not modified).
    """
    # Build stem → group_id mapping.
    stem_to_group: dict[str, str] = {}
    for group_id, members in groups.items():
        for stem in members:
            stem_to_group[stem] = group_id

    # Union-find parent dict.
    parent: dict[str, str] = {gid: gid for gid in groups}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for stem_a, stem_b in duplicate_pairs:
        gid_a = stem_to_group.get(stem_a)
        gid_b = stem_to_group.get(stem_b)
        if gid_a is not None and gid_b is not None and gid_a != gid_b:
            union(gid_a, gid_b)

    # Reconstruct merged groups.
    merged: dict[str, list[str]] = defaultdict(list)
    for group_id, members in groups.items():
        root = find(group_id)
        merged[root].extend(members)

    return dict(merged)


# ---------------------------------------------------------------------------
# Step 3 — Deterministic group-level split
# ---------------------------------------------------------------------------


def assign_splits(
    groups: dict[str, list[str]],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = DEFAULT_SEED,
) -> dict[str, str]:
    """Shuffle groups and assign each to train/val/test.

    All members of a group receive the same split label.
    Returns stem → split_name mapping.
    """
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1):
        raise ValueError(f"Invalid ratios: train={train_ratio}, val={val_ratio}")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"train_ratio + val_ratio must be < 1.0, got {train_ratio + val_ratio}"
        )

    group_ids = sorted(groups.keys())  # deterministic order before shuffle
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    n = len(group_ids)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)
    # Remaining groups go to test.

    split_assignment: dict[str, str] = {}
    for i, gid in enumerate(group_ids):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"
        for stem in groups[gid]:
            split_assignment[stem] = split

    return split_assignment


# ---------------------------------------------------------------------------
# Step 4 — Compute class distribution from YOLO TXT annotations
# ---------------------------------------------------------------------------


def compute_class_distribution_yolo(
    stems: list[str],
    txt_index: dict[str, list[Path]],
) -> tuple[dict[str, int], dict[str, int]]:
    """Return (image_counts_per_class, instance_counts_per_class) for a set of stems.

    Uses YOLO TXT annotations (normalized class IDs). Counts image-level class
    presence and total box instances separately, as required by AGENTS.md.
    """
    image_counts: dict[str, int] = defaultdict(int)
    instance_counts: dict[str, int] = defaultdict(int)

    for stem in stems:
        txt_paths = txt_index.get(stem, [])
        if not txt_paths:
            continue
        txt_path = txt_paths[0]
        try:
            text = txt_path.read_text(encoding="utf-8")
        except OSError:
            continue
        classes_in_image: set[str] = set()
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(parts[0])
            except ValueError:
                continue
            class_name = YOLO_ID_TO_CLASS.get(class_id, f"unknown_id_{class_id}")
            classes_in_image.add(class_name)
            instance_counts[class_name] += 1
        for cls in classes_in_image:
            image_counts[cls] += 1

    return dict(image_counts), dict(instance_counts)


# ---------------------------------------------------------------------------
# Step 5 — Build full manifest
# ---------------------------------------------------------------------------


def build_manifest(
    layout: DatasetLayout,
    audit_report_path: Path,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = DEFAULT_SEED,
) -> SplitManifest:
    """Run the full Phase 1 group-aware split pipeline and return a SplitManifest."""

    # --- Collect all stems ---
    orig_images = iter_files(layout.original_image_dir, {".jpg", ".jpeg", ".png"})
    all_stems = sorted({p.stem for p in orig_images})

    txt_files = iter_files(layout.txt_dir, {".txt"})
    txt_index: dict[str, list[Path]] = {}
    for p in txt_files:
        txt_index.setdefault(p.stem, []).append(p)

    # --- Build groups ---
    raw_groups = build_groups(all_stems)

    # --- Merge duplicate groups ---
    dup_pairs = load_md5_duplicate_pairs(audit_report_path)
    merged_groups = merge_duplicate_groups(raw_groups, dup_pairs)

    # Count groups with vendor-split leakage (informational).
    # We detect this by reading vendor split folder membership.
    vendor_split_stems: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        split_dir = layout.split_image_dir / split_name
        for p in iter_files(split_dir, {".jpg", ".jpeg", ".png"}):
            vendor_split_stems[p.stem] = split_name

    leakage_groups = 0
    for gid, members in merged_groups.items():
        vendor_splits_seen = {vendor_split_stems.get(m) for m in members} - {None}
        if len(vendor_splits_seen) > 1:
            leakage_groups += 1

    # --- Assign splits ---
    stem_to_split = assign_splits(merged_groups, train_ratio, val_ratio, seed)

    # --- Compute stem → group_id and evidence ---
    stem_to_group: dict[str, str] = {}
    stem_to_evidence: dict[str, str] = {}
    for stem in all_stems:
        _, evidence = infer_group_id(stem)
        # Find which merged group this stem now belongs to.
        raw_gid, _ = infer_group_id(stem)
        stem_to_group[stem] = raw_gid
        stem_to_evidence[stem] = evidence

    # For merged groups (duplicate pairs), use a "merged_duplicate" evidence tag.
    for gid, members in merged_groups.items():
        # If gid differs from infer_group_id for any member, it was merged.
        for stem in members:
            orig_gid, _ = infer_group_id(stem)
            if orig_gid != gid:
                stem_to_evidence[stem] = "merged_md5_duplicate"
                stem_to_group[stem] = gid

    # --- Build per-split class distributions ---
    split_stems: dict[str, list[str]] = defaultdict(list)
    for stem, split in stem_to_split.items():
        split_stems[split].append(stem)

    class_dist: dict[str, dict[str, int]] = {}
    for split_name, stems_in_split in split_stems.items():
        img_counts, inst_counts = compute_class_distribution_yolo(
            stems_in_split, txt_index
        )
        class_dist[split_name] = {
            "image_counts": img_counts,
            "instance_counts": inst_counts,
        }

    # --- Group size histogram ---
    size_hist: dict[str, int] = defaultdict(int)
    for gid, members in merged_groups.items():
        size_hist[str(len(members))] += 1

    # --- Build records ---
    records: list[SplitRecord] = []
    for stem in all_stems:
        split = stem_to_split.get(stem, "unknown")
        txt_paths = txt_index.get(stem, [])
        classes_present: list[str] = []
        inst_counts_per_stem: dict[str, int] = defaultdict(int)
        if txt_paths:
            try:
                text = txt_paths[0].read_text(encoding="utf-8")
                for line in text.splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cid = int(parts[0])
                            cname = YOLO_ID_TO_CLASS.get(cid, f"unknown_id_{cid}")
                            inst_counts_per_stem[cname] += 1
                        except ValueError:
                            pass
                classes_present = sorted(set(inst_counts_per_stem.keys()))
            except OSError:
                pass

        records.append(
            SplitRecord(
                stem=stem,
                group_id=stem_to_group.get(stem, stem),
                group_evidence=stem_to_evidence.get(stem, "unknown"),
                split=split,
                classes_present=classes_present,
                instance_counts=dict(inst_counts_per_stem),
            )
        )

    warnings: list[str] = []
    if leakage_groups > 0:
        warnings.append(
            f"Vendor split had {leakage_groups} groups crossing train/val/test boundaries. "
            "These are corrected in this group-aware split."
        )
    if dup_pairs:
        warnings.append(
            f"{len(dup_pairs)} confirmed MD5 duplicate pairs merged into shared groups."
        )

    return SplitManifest(
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=round(1.0 - train_ratio - val_ratio, 6),
        total_images=len(all_stems),
        total_groups=len(merged_groups),
        leakage_candidate_groups_in_vendor_split=leakage_groups,
        records=records,
        split_counts={k: len(v) for k, v in split_stems.items()},
        class_distribution=class_dist,
        group_size_histogram=dict(sorted(size_hist.items(), key=lambda kv: int(kv[0]))),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Step 6 — Save manifests
# ---------------------------------------------------------------------------


def save_manifests(manifest: SplitManifest, output_dir: Path) -> dict[str, Path]:
    """Save CSV and JSON manifests to output_dir. Returns dict of saved paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    # --- Per-split CSVs ---
    fieldnames = [
        "stem",
        "group_id",
        "group_evidence",
        "split",
        "classes_present",
        "instance_counts_json",
    ]
    split_records: dict[str, list[SplitRecord]] = defaultdict(list)
    for rec in manifest.records:
        split_records[rec.split].append(rec)

    for split_name in ("train", "val", "test"):
        csv_path = output_dir / f"{split_name}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for rec in sorted(split_records.get(split_name, []), key=lambda r: r.stem):
                writer.writerow(
                    {
                        "stem": rec.stem,
                        "group_id": rec.group_id,
                        "group_evidence": rec.group_evidence,
                        "split": rec.split,
                        "classes_present": "|".join(sorted(rec.classes_present)),
                        "instance_counts_json": json.dumps(rec.instance_counts),
                    }
                )
        saved[split_name] = csv_path

    # --- Full JSON manifest ---
    json_path = output_dir / "split_manifest.json"
    manifest_dict: dict[str, Any] = {
        "phase": 1,
        "description": (
            "Group-aware leakage-safe train/val/test split. "
            "All tiles of the same substrate are assigned to the same split. "
            "Confirmed MD5 duplicate pairs are merged into shared groups. "
            "The test split must remain real-only and must not be used "
            "for GAN training, synthetic sample selection, or hyperparameter tuning."
        ),
        "seed": manifest.seed,
        "ratios": {
            "train": manifest.train_ratio,
            "val": manifest.val_ratio,
            "test": manifest.test_ratio,
        },
        "counts": {
            "total_images": manifest.total_images,
            "total_groups": manifest.total_groups,
            "splits": manifest.split_counts,
            "leakage_groups_corrected": manifest.leakage_candidate_groups_in_vendor_split,
        },
        "class_distribution": manifest.class_distribution,
        "group_size_histogram": manifest.group_size_histogram,
        "yolo_id_to_class": {str(k): v for k, v in YOLO_ID_TO_CLASS.items()},
        "warnings": manifest.warnings,
        "records": [
            {
                "stem": r.stem,
                "group_id": r.group_id,
                "group_evidence": r.group_evidence,
                "split": r.split,
                "classes_present": r.classes_present,
                "instance_counts": r.instance_counts,
            }
            for r in manifest.records
        ],
    }
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest_dict, fh, indent=2)
    saved["manifest_json"] = json_path

    # --- Human-readable summary ---
    summary_path = output_dir / "split_summary.md"
    _write_split_summary(manifest, summary_path)
    saved["summary_md"] = summary_path

    return saved


def _write_split_summary(manifest: SplitManifest, path: Path) -> None:
    lines: list[str] = [
        "# Phase 1 — Group-Aware Split Summary",
        "",
        f"**Seed:** {manifest.seed}  ",
        f"**Ratios (group-level):** train={manifest.train_ratio:.0%} · "
        f"val={manifest.val_ratio:.0%} · test={manifest.test_ratio:.0%}  ",
        f"**Total images:** {manifest.total_images}  ",
        f"**Total groups:** {manifest.total_groups}  ",
        f"**Vendor-split leakage groups corrected:** "
        f"{manifest.leakage_candidate_groups_in_vendor_split}  ",
        "",
        "## Image counts per split",
        "",
        "| Split | Images |",
        "|---|---:|",
    ]
    for split_name in ("train", "val", "test"):
        count = manifest.split_counts.get(split_name, 0)
        lines.append(f"| {split_name} | {count} |")

    lines += [
        "",
        "## Instance counts per class per split",
        "",
        "| Class | train | val | test |",
        "|---|---:|---:|---:|",
    ]
    all_classes = sorted(YOLO_ID_TO_CLASS.values())
    for cls in all_classes:
        row = [cls]
        for split_name in ("train", "val", "test"):
            inst = (
                manifest.class_distribution
                .get(split_name, {})
                .get("instance_counts", {})
                .get(cls, 0)
            )
            row.append(str(inst))
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Image counts per class per split",
        "",
        "| Class | train | val | test |",
        "|---|---:|---:|---:|",
    ]
    for cls in all_classes:
        row = [cls]
        for split_name in ("train", "val", "test"):
            img = (
                manifest.class_distribution
                .get(split_name, {})
                .get("image_counts", {})
                .get(cls, 0)
            )
            row.append(str(img))
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Group size histogram",
        "",
        "| Members per group | Number of groups |",
        "|---:|---:|",
    ]
    for size, count in manifest.group_size_histogram.items():
        lines.append(f"| {size} | {count} |")

    if manifest.warnings:
        lines += ["", "## Warnings", ""]
        for warning in manifest.warnings:
            lines.append(f"- {warning}")

    lines += [
        "",
        "## Safety notes",
        "",
        "- The **test split is real-only**. Never use test images for GAN training, "
        "synthetic sample selection, or hyperparameter tuning.",
        "- The **GAN must be trained only on the train split**.",
        "- Split manifests are frozen. Do not regenerate without documenting the reason "
        "and changing the seed or output directory.",
        "- Original dataset files were not modified.",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_manifest(manifest: SplitManifest) -> list[str]:
    """Run integrity checks on the produced manifest. Return list of violations."""
    violations: list[str] = []

    # Build group → splits mapping to check for leakage.
    group_to_splits: dict[str, set[str]] = defaultdict(set)
    for rec in manifest.records:
        group_to_splits[rec.group_id].add(rec.split)

    leaking = {gid: splits for gid, splits in group_to_splits.items() if len(splits) > 1}
    if leaking:
        violations.append(
            f"GROUP LEAKAGE: {len(leaking)} groups appear in more than one split: "
            + str(list(leaking.keys())[:5])
        )

    # Every stem must have exactly one split.
    stems_per_split: dict[str, list[str]] = defaultdict(list)
    for rec in manifest.records:
        stems_per_split[rec.split].append(rec.stem)

    stem_set = {rec.stem for rec in manifest.records}
    if len(stem_set) != manifest.total_images:
        violations.append(
            f"Duplicate stems in manifest: "
            f"{manifest.total_images} expected, {len(stem_set)} unique"
        )

    for split_name in ("train", "val", "test"):
        if split_name not in stems_per_split:
            violations.append(f"Split '{split_name}' has zero images")

    return violations
