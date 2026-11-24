# SolDef_AI Preprocessing & Visual QC Metrics Report (Stage D)

**Date:** 2026-11-22  
**Dataset:** SolDef_AI (European Space Agency ECSS-Q-ST-70-38C Standard)  
**Benchmark:** Side-View Solder Joint Defect Detection ($V_2, V_{2.1}$)  
**Status:** **Stage D COMPLETE — Verified Deterministic Preprocessing**

---

## 1. Dataset Accounting & Pipeline A Summary (Defect Patches for GANs)

- **Source Split:** Frozen Train Partition ONLY (`outputs/splits/soldef/phase1/train.csv`)
- **Total Source Train Images:** 140
- **Total Eligible Target Defects:** 256
- **Total Extracted Patches:** **256** (100% extraction rate, 0 rejected)
- **Unique Parent Images:** 135
- **Unique Grouping Units Represented:** 29
- **Boundary Crops Handled:** 0 (0.0%)
- **Patch Resolution:** 256x256 (100% uniform)

### Patch Counts by Target Defect Class:
| Target Class | Extracted Patches | % of Extracted Patches |
| :--- | :--- | :--- |
| `exc_solder` | 128 | 50.0% |
| `spike` | 96 | 37.5% |
| `poor_solder` (minority) | 32 | 12.5% |
| **Total** | **256** | **100.0%** |

---

## 2. Pipeline B Summary (Full-Image Detector Preprocessing)

- **Target Resolution:** 640x640
- **Aspect Ratio Policy:** Letterbox with padding (Original $2560 \times 1440 \rightarrow 640 \times 360$ scaled by $0.25$, top/bottom pad 140px)
- **Padding Color:** [114, 114, 114]

### Partition Breakdown:
| Split Partition | Images Preprocessed | Target Defect Boxes Transformed | `exc_solder` | `spike` | `poor_solder` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 140 | 256 | 128 | 96 | 32 |
| **Validation** | 30 | 51 | 26 | 15 | 10 |
| **Test** | 30 | 62 | 28 | 19 | 15 |
| **Total** | **200** | **369** | **182** | **130** | **57** |

---

## 3. Visual Quality Control Inspection

- **Visual QC Samples Generated:** 12 multi-panel comparisons saved to `outputs/preprocessing/soldef/phase_d/qc/visualizations/`.
- **Sample Coverage:**
  - Small bounding box defects (e.g. small `spike` < 150px).
  - Large bounding box defects (e.g. `exc_solder` > 400px).
  - Edge/boundary defects requiring context padding.
  - Spatial overlap cases: `spike` protrusion on underlying `exc_solder` fillet.
  - Multi-defect images with multiple solder joints.
  - Minority class `poor_solder` examples.

---

## 4. Integrity and Leakage Verification

- **Split Isolation:** GAN patches generated **strictly from Train (140 images)**. Zero patches extracted from Validation or Test.
- **Annotation Preservation:** Original raw JSONs are untouched; `good` annotations preserved in source records; overlapping spike annotations preserved as distinct bounding boxes.
- **Mathematical Invertibility:** Letterbox scaling and translation are analytically invertible with zero coordinate drift.
