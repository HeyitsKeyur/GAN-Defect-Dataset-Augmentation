"""Audit script for Stage C artifacts, manifests, provenance, and class counts."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path

# Paths
primary_split_dir = Path("outputs/splits/soldef/phase1")
secondary_split_dir = Path("outputs/splits/soldef/phase1_v1")
prov_file = Path("outputs/splits/soldef/grouping_provenance.json")
labeled_dir = Path("Dateset/SolDef_AI/Labeled")


def audit_primary_splits() -> dict[str, object]:
    tr = list(csv.DictReader(open(primary_split_dir / "train.csv", encoding="utf-8")))
    va = list(csv.DictReader(open(primary_split_dir / "val.csv", encoding="utf-8")))
    te = list(csv.DictReader(open(primary_split_dir / "test.csv", encoding="utf-8")))

    all_rows = tr + va + te

    tr_groups = {r["group_id"] for r in tr}
    va_groups = {r["group_id"] for r in va}
    te_groups = {r["group_id"] for r in te}
    all_groups = tr_groups | va_groups | te_groups

    # Overlaps
    img_tv = set(r["stem"] for r in tr) & set(r["stem"] for r in va)
    img_tt = set(r["stem"] for r in tr) & set(r["stem"] for r in te)
    img_vt = set(r["stem"] for r in va) & set(r["stem"] for r in te)

    grp_tv = tr_groups & va_groups
    grp_tt = tr_groups & te_groups
    grp_vt = va_groups & te_groups

    # Classes
    classes = ["exc_solder", "spike", "poor_solder", "good"]
    counts = {}
    for name, rows in [("train", tr), ("val", va), ("test", te), ("total", all_rows)]:
        inst_c = Counter()
        img_c = Counter()
        for r in rows:
            for c in classes:
                cnt = int(r.get(c, 0))
                if cnt > 0:
                    inst_c[c] += cnt
                    img_c[c] += 1
        counts[name] = {
            "instances": dict(inst_c),
            "images": dict(img_c),
            "total_instances": sum(inst_c.values()),
            "total_images": len(rows),
        }

    # Verify against raw JSONs
    raw_classes = Counter()
    total_spikes = 0
    overlapping_spikes = 0

    for r in all_rows:
        jf = labeled_dir / f"{r['stem']}.json"
        with open(jf, encoding="utf-8") as f:
            d = json.load(f)
        shapes = d.get("shapes", [])
        spikes = []
        others = []
        for s in shapes:
            lbl = s.get("label", "").strip().lower().replace(" ", "_")
            raw_classes[lbl] += 1
            pts = s.get("points", [])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            box = (min(xs), min(ys), max(xs), max(ys))
            if lbl == "spike":
                spikes.append(box)
                total_spikes += 1
            else:
                others.append(box)

        for sb in spikes:
            for ob in others:
                ixmin = max(sb[0], ob[0])
                iymin = max(sb[1], ob[1])
                ixmax = min(sb[2], ob[2])
                iymax = min(sb[3], ob[3])
                if ixmax > ixmin and iymax > iymin:
                    inter_area = (ixmax - ixmin) * (iymax - iymin)
                    s_area = (sb[2] - sb[0]) * (sb[3] - sb[1])
                    o_area = (ob[2] - ob[0]) * (ob[3] - ob[1])
                    iou = inter_area / (s_area + o_area - inter_area)
                    if iou > 0.05:
                        overlapping_spikes += 1
                        break

    return {
        "total_images": len(all_rows),
        "split_image_counts": {"train": len(tr), "val": len(va), "test": len(te)},
        "total_groups": len(all_groups),
        "split_group_counts": {"train": len(tr_groups), "val": len(va_groups), "test": len(te_groups)},
        "image_overlaps": {"train_val": len(img_tv), "train_test": len(img_tt), "val_test": len(img_vt)},
        "group_overlaps": {"train_val": len(grp_tv), "train_test": len(grp_tt), "val_test": len(grp_vt)},
        "counts": counts,
        "raw_json_classes": dict(raw_classes),
        "total_spikes": total_spikes,
        "overlapping_spikes": overlapping_spikes,
    }


def main():
    res = audit_primary_splits()
    print("=== PRIMARY BENCHMARK AUDIT RESULTS ===")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
