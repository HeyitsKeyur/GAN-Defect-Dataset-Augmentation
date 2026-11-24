# PHASE3_QC_AUDIT.md — Preprocessing Quality Control Review & Acceptance Audit

**Project:** Defect Dataset Augmentation using Generative Adversarial Networks (GAN)  
**Academic Context:** M.Tech CSE, Nirma University, 2025–2027 | Guide: Prof. Sonia Mittal  
**Repository:** https://github.com/HeyitsKeyur/GAN-Defect-Dataset-Augmentation.git  
**Current Phase:** Phase 3 — Preprocessing Pipeline and Defect-Centred Crops (Stage C Audit)  
**Branch:** `phase-3-preprocessing`  
**Date:** 2026-11-21  
**Audit Status:** **Complete — Accept Phase 3 with Documented Architectural Recommendations**

---

## 1. Audit Methodology & Scope

The purpose of this Stage C audit is to independently verify that the preprocessing pipeline implemented in Stage B produces defect crops and full-image representations that are geometrically, statistically, and visually sound for downstream defect augmentation and detector training.

### Audit Protocols:
1. **Source Data & Manifest Immutability:** Verify that the original raw images, LabelMe JSON files, and Phase 1 frozen manifests (`train.csv`, `val.csv`, `test.csv`, `split_manifest.json`) were untouched.
2. **Provenance & Exclusion Tracing:** Verify that every single bounding box in a source image is accounted for in `crop_provenance_mapping.json` (no silent dropping of annotations).
3. **Geometry & Coordinate Math:** Verify that all translated bounding boxes, crop window dimensions, padding offsets, letterbox transformations, and normalized YOLO coordinates are mathematically exact within $[0.0, 1.0]$.
4. **Visual Quality Inspection:** Inspect actual generated composite quality-control images (`outputs/qc_phase3/`) covering all 6 defect classes (`spot`, `scar`, `adhesion`, `defect`, `uneven`, `abnormal_thickness`), edge/corner conditions, large bounding boxes, and multi-defect/multi-class clusters.
5. **Architectural Suitability Analysis:** Examine the implications of defect spatial scales on patch-based generative models versus full-image detectors.

---

## 2. Actual Inspected Artifacts & Image Paths

All audits were conducted directly against real generated files on disk:

| Artifact Category | File Path / Directory | Verified Content |
| :--- | :--- | :--- |
| **Sample Crop Patches** | [`data/processed/phase3_sample/patches_256/`](file:///e:/MTech[CSE]%20Nirma_University/S3/Major%20Project/Defect%20Dataset%20Augmentation%20using%20Generative%20Adversarial%20Networks%20(GAN)/data/processed/phase3_sample/patches_256) | 64 crop images (`.jpg`) and 64 YOLO label files (`.txt`) |
| **Sample Full Images** | [`data/processed/phase3_sample/full_640/`](file:///e:/MTech[CSE]%20Nirma_University/S3/Major%20Project/Defect%20Dataset%20Augmentation%20using%20Generative%20Adversarial%20Networks%20(GAN)/data/processed/phase3_sample/full_640) | 15 letterboxed images (640×640) and 15 YOLO label files |
| **Crop Manifests** | [`data/processed/phase3_sample/manifests/`](file:///e:/MTech[CSE]%20Nirma_University/S3/Major%20Project/Defect%20Dataset%20Augmentation%20using%20Generative%20Adversarial%20Networks%20(GAN)/data/processed/phase3_sample/manifests) | `train_crops.csv`, `val_crops.csv`, `test_crops.csv` |
| **Provenance Mapping** | [`crop_provenance_mapping.json`](file:///e:/MTech[CSE]%20Nirma_University/S3/Major%20Project/Defect%20Dataset%20Augmentation%20using%20Generative%20Adversarial%20Networks%20(GAN)/data/processed/phase3_sample/manifests/crop_provenance_mapping.json) | 64 comprehensive JSON records tracking all box transformations |
| **Visual QC Panels** | [`outputs/qc_phase3/`](file:///e:/MTech[CSE]%20Nirma_University/S3/Major%20Project/Defect%20Dataset%20Augmentation%20using%20Generative%20Adversarial%20Networks%20(GAN)/outputs/qc_phase3) | 11 multi-panel inspection figures ($1280 \times 512\text{ px}$, RGB) |

---

## 3. Visual Audit Findings by Defect Category

The visual audit inspected 11 composite QC images in `outputs/qc_phase3/` generated directly from source images and annotations:

### A. Small `spot` Defects
- **Sample File:** [`outputs/qc_phase3/qc_1_2_000002_0810104212116_NG_part7.jpg`](file:///e:/MTech[CSE]%20Nirma_University/S3/Major%20Project/Defect%20Dataset%20Augmentation%20using%20Generative%20Adversarial%20Networks%20(GAN)/outputs/qc_phase3/qc_1_2_000002_0810104212116_NG_part7.jpg)
- **Observations:** 4 small `spot` instances (mean size: $30.8 \times 37.2\text{ px}$). The $256 \times 256$ crop provides generous surrounding substrate context ($\approx 85\%$ background area). The defect maintains native camera pixel resolution (zero downsampling distortion), making it optimal for generative modeling.

### B. Elongated and Large `scar` Defects
- **Sample Files:** [`qc_1_2_000004_0807141312951_NG_part2.jpg`](file:///e:/MTech[CSE]%20Nirma_University/S3/Major%20Project/Defect%20Dataset%20Augmentation%20using%20Generative%20Adversarial%20Networks%20(GAN)/outputs/qc_phase3/qc_1_2_000004_0807141312951_NG_part2.jpg) and [`qc_1_2_000008_0807142032039_NG_part10.jpg`](file:///e:/MTech[CSE]%20Nirma_University/S3/Major%20Project/Defect%20Dataset%20Augmentation%20using%20Generative%20Adversarial%20Networks%20(GAN)/outputs/qc_phase3/qc_1_2_000008_0807142032039_NG_part10.jpg)
- **Observations:** Contains a large continuous linear scratch ($101.1 \times 613.3\text{ px}$). The adaptive context window appropriately expanded the crop dimension to encompass the entire continuous defect without premature cropping.

### C. Surface Contamination `adhesion`
- **Sample Files:** [`qc_1_2_000008_0807142032039_NG_part13.jpg`](file:///e:/MTech[CSE]%20Nirma_University/S3/Major%20Project/Defect%20Dataset%20Augmentation%20using%20Generative%20Adversarial%20Networks%20(GAN)/outputs/qc_phase3/qc_1_2_000008_0807142032039_NG_part13.jpg) and [`qc_1_2_000002_0810104212116_NG_part7.jpg`](file:///e:/MTech[CSE]%20Nirma_University/S3/Major%20Project/Defect%20Dataset%20Augmentation%20using%20Generative%20Adversarial%20Networks%20(GAN)/outputs/qc_phase3/qc_1_2_000002_0810104212116_NG_part7.jpg)
- **Observations:** Shows discrete particulate clusters adhering to the substrate surface. The bounding box tightly bounds the defect particle boundaries with no label drift.

### D. Generic Structural `defect`
- **Sample Files:** [`qc_defect007.jpg`](file:///e:/MTech[CSE]%20Nirma_University/S3/Major%20Project/Defect%20Dataset%20Augmentation%20using%20Generative%20Adversarial%20Networks%20(GAN)/outputs/qc_phase3/qc_defect007.jpg) and [`qc_defect028.jpg`](file:///e:/MTech[CSE]%20Nirma_University/S3/Major%20Project/Defect%20Dataset%20Augmentation%20using%20Generative%20Adversarial%20Networks%20(GAN)/outputs/qc_phase3/qc_defect028.jpg)
- **Observations:** `defect007` ($95.5 \times 231.8\text{ px}$) is centered smoothly. `defect028` ($xmin=6.4\text{ px}$) demonstrates exact border boundary clamping at the substrate margin.

### E. Surface Topography `uneven`
- **Sample File:** [`qc_uneven004.jpg`](file:///e:/MTech[CSE]%20Nirma_University/S3/Major%20Project/Defect%20Dataset%20Augmentation%20using%20Generative%20Adversarial%20Networks%20(GAN)/outputs/qc_phase3/qc_uneven004.jpg)
- **Observations:** Box size $132.2 \times 135.6\text{ px}$. Shows low-contrast diffuse shading gradients indicating surface unevenness. Centered neatly in a $256 \times 256$ crop.

### F. Massive Dimensionality `abnormal_thickness`
- **Sample File:** [`qc_Abnormal thickness007.jpg`](file:///e:/MTech[CSE]%20Nirma_University/S3/Major%20Project/Defect%20Dataset%20Augmentation%20using%20Generative%20Adversarial%20Networks%20(GAN)/outputs/qc_phase3/qc_Abnormal thickness007.jpg)
- **Observations:** Massive defect ($3,718.2 \times 827.3\text{ px}$) on a $4,000 \times 4,096\text{ px}$ image. Triggers full-image enclosure with symmetric padding. (See Section 6 for in-depth analysis).

### G. Boundary and Corner Defects
- **Sample Files:** `qc_1_2_000008_0807142032039_NG_part8.jpg` ($ymin=5.33\text{ px}$, $xmin=1,007.5\text{ px}$ on 1024-wide frame) and `qc_1_2_000002_0810104212116_NG_part7.jpg` ($xmin=0.0\text{ px}$).
- **Observations:** When defects touch the image border, `calculate_crop_window` prevents window overflow by clamping `src_x1 = 0` or `src_x2 = image_width`. The defect remains fully enclosed within the patch without negative array indexing.

### H. Multi-Defect & Multi-Class Crops
- **Sample Files:** [`qc_1_1_000004_0121102105699_NG.jpg`](file:///e:/MTech[CSE]%20Nirma_University/S3/Major%20Project/Defect%20Dataset%20Augmentation%20using%20Generative%20Adversarial%20Networks%20(GAN)/outputs/qc_phase3/qc_1_1_000004_0121102105699_NG.jpg) and [`qc_1_2_000002_0121101850711_NG.jpg`](file:///e:/MTech[CSE]%20Nirma_University/S3/Major%20Project/Defect%20Dataset%20Augmentation%20using%20Generative%20Adversarial%20Networks%20(GAN)/outputs/qc_phase3/qc_1_2_000002_0121101850711_NG.jpg)
- **Observations:**
  - Crop `1_1_000004_0121102105699_NG_c10_288`: Dimension $288 \times 288\text{ px}$. Correctly encloses 3 distinct bounding boxes spanning 2 classes (`scar` and `spot`).
  - Crop `1_2_000002_0121101850711_NG_c04_512`: Dimension $512 \times 512\text{ px}$. Retains 2 boxes across `adhesion` and `scar`.
  - All neighbouring defects are assigned separate, verified lines in the corresponding YOLO `.txt` file.

---

## 4. Exclusion & Retention Analysis

An audit of all 64 sample crop provenance records in `crop_provenance_mapping.json` revealed:

| Metric | Count | Percentage | Method / Significance |
| :--- | :--- | :--- | :--- |
| **Total Crops Audited** | **64** | 100.0% | Extracted across 15 sample images |
| **Primary Defect Loss Count** | **0** | **0.0%** | **100% of crops retain their primary target defect** |
| **Total Retained Defect Instances** | **80** | — | Mean: 1.25 defects per crop |
| **Total Excluded Defect Records** | **526** | — | Other defects in source image evaluated against this crop |
| **Reason: `outside_crop_window`** | **524** | **99.62%** | Coordinate lies entirely outside the crop window |
| **Reason: `low_retention_ratio`** | **2** | **0.38%** | Clipped sliver below 30% retention threshold |
| **Reason: `clipped_area_too_small`** | **0** | **0.0%** | No micro-artifacts ($< 9.0\text{ px}^2$) occurred |

### Retention Ratio Analysis for Clipped Annotations:
Only 2 bounding boxes across the entire sample set were clipped by crop boundaries:
1. Crop `1_2_000002_0121101850711_NG_c03_352` (class `adhesion`):
   - Original box area: $192.5\text{ px}^2$. Retained clipped area: $22.1\text{ px}^2$.
   - Computed Retention Ratio: **0.1150 (11.5%)**. Excluded because $0.115 < 0.30$.
2. Crop `1_2_000001_0810104210546_NG_part0_c02_256` (class `adhesion`):
   - Original box area: $246.8\text{ px}^2$. Retained clipped area: $35.0\text{ px}^2$.
   - Computed Retention Ratio: **0.1418 (14.2%)**. Excluded because $0.142 < 0.30$.

### Disproportionality Assessment:
- Both excluded boxes were peripheral slivers of distant neighbour defects whose centroids were outside the window.
- Their primary defects were fully preserved in their own respective dedicated crops.
- **Zero minority-class defects were lost or disproportionately excluded.**

---

## 5. Crop Geometry and Clustering Analysis

| Dimension Policy | Crop Count | Share | Typical Target Defect |
| :--- | :--- | :--- | :--- |
| **$256 \times 256\text{ px}$** | **57** | **89.06%** | Standard isolated `spot`, `adhesion`, `scar` |
| **$288 \times 288\text{ px}$** | **1** | **1.56%** | Close paired defects |
| **$320 \times 320\text{ px}$** | **1** | **1.56%** | Moderately elongated defect cluster |
| **$352 \times 352\text{ px}$** | **3** | **4.69%** | Multi-defect cluster ($\le 5$ boxes) |
| **$512 \times 512\text{ px}$** | **1** | **1.56%** | Spatially dispersed defect pair |
| **$1,312 \times 1,312\text{ px}$** | **1** | **1.56%** | Broad composite scratch group |

### Defect Multiplicity per Crop:
- **1 defect:** 52 crops (81.25%)
- **2 defects:** 10 crops (15.63%)
- **3 defects:** 1 crop (1.56%)
- **5 defects:** 1 crop (1.56%)

### Redundancy & Overlap Verification:
- Computed pairwise Intersection-over-Union (IoU) among all crops extracted from the same source image.
- **Pairs with $\text{IoU} > 0.50$:** **0 pairs**.
- **Conclusion:** The connected-component clustering algorithm ($D_{\text{cluster}} = 128\text{ px}$) completely prevents redundant, duplicate crops by merging nearby defects rather than extracting multiple overlapping windows.

---

## 6. Architectural Analysis of `abnormal_thickness`

The audit revealed an important structural property of the `abnormal_thickness` class:
- **Instance Count:** 156 total annotations across 156 images.
- **Spatial Dimensions:** Often spans almost the entire field of view (e.g. $3,718.2 \times 827.3\text{ px}$, with maximum dimensions reaching $3,845 \times 3,795\text{ px}$ on $4,000 \times 4,096\text{ px}$ substrates).

### Impact on Preprocessing Streams:
1. **Patch Extraction (Stream 1):**
   - With the adaptive context margin ($20\%$), extracting a patch around a $3,718\text{ px}$ defect produces a window of $5,216 \times 5,216\text{ px}$.
   - Because $5,216\text{ px} > 4,096\text{ px}$ (image height), the window is clamped and padded symmetrically with neutral gray (`pad_left=608, pad_top=560, pad_right=608, pad_bottom=560`).
   - *Implication for GANs:* Generative patch models (such as DCGAN, WGAN-GP, or StyleGAN) operate on fixed square patches ($128 \times 128$ or $256 \times 256$). Downsampling a $5,216\text{ px}$ patch to $256 \times 256$ results in a severe $20\times$ resolution reduction, losing fine ceramic texture.
2. **Full-Image Letterboxing (Stream 2):**
   - In Stream 2, the original image is letterboxed to $640 \times 640$ with scale factor $s = 640 / 4096 \approx 0.156$.
   - The bounding box rescales cleanly to $[23.4, 353.9, 604.4, 483.2]$ on the $640 \times 640$ canvas.
   - *Implication for Detectors:* Standard detectors (YOLOv8, Faster R-CNN) process full-frame letterboxed images seamlessly, allowing `abnormal_thickness` to be detected in its full macroscopic context.

---

## 7. Statistical & Coordinate Geometry Verification

Executed an exhaustive verification across all 64 crop outputs and 15 full-image outputs:

1. **Output File Existence:** 100% of images and YOLO `.txt` files exist on disk.
2. **Dimension Match:** Every generated image file precisely matches its target width and height ($256 \times 256$, $640 \times 640$, etc.).
3. **Box Count Concordance:** The number of lines in each YOLO `.txt` file exactly equals `retained_boxes_count` in the provenance metadata.
4. **Normalized Coordinate Boundaries:**
   $$\forall \text{ box } b: \quad 0.0 \le c_x \le 1.0, \quad 0.0 \le c_y \le 1.0, \quad 0.0 < w \le 1.0, \quad 0.0 < h \le 1.0$$
   $$\left(c_x \pm \frac{w}{2}\right) \in [0.0, 1.0], \quad \left(c_y \pm \frac{h}{2}\right) \in [0.0, 1.0]$$
   - Out of bounds YOLO coordinates: **0 errors detected**.
5. **Coordinate Translation Accuracy:**
   - Un-normalizing YOLO coordinates back to absolute crop pixels reproduces the clipped bounding box coordinates within float precision ($< 10^{-5}\text{ px}$).

---

## 8. Split Integrity & Leakage Verification

Audited split allocation across all frozen Phase 1 manifests and sample crop outputs:

| Split Partition | Source Stems | Sample Crops Generated | Group Leakage Rate |
| :--- | :--- | :--- | :--- |
| **`train`** | 1,967 | 25 | 0.0% |
| **`val`** | 479 | 11 | 0.0% |
| **`test`** | 446 | 28 | 0.0% |
| **Total** | **2,892** | **64** | **0.0% (Zero cross-partition leakage)** |

- **Frozen Manifest Hash Integrity:** `outputs/splits/phase1/{train.csv, val.csv, test.csv, split_manifest.json}` were unchanged.
- **Split Inheritance:** Every single crop manifest entry strictly inherited the `split` of its source image.
- **Test Set Isolation:** No test set images or annotations were used to select, tune, or optimize crop dimensions or retention thresholds.

---

## 9. Identified Issues Classification

| Issue Description | Severity | Classification | Mitigation / Recommendation |
| :--- | :--- | :--- | :--- |
| **Single-channel Grayscale Images:** Some images (e.g. `defect007`, `Abnormal thickness007`) are stored as mode `"L"` in raw data. | **Minor** | Operational | Always apply `.convert("RGB")` during image loading (already applied in `visualize_crops.py`). |
| **Macro-scale Defect Discrepancy:** `abnormal_thickness` instances span up to $3,800\text{ px}$, causing excessive padding in patch crops. | **Major** | Architectural Limitation | Use Stream 2 (640×640 full-image letterboxing) for baseline detection of `abnormal_thickness`; restrict patch-based GAN augmentation (Stream 1) to localized defect classes (`spot`, `scar`, `adhesion`, `defect`, `uneven`). |
| **Peripheral Defect Clipping:** Neighbor defects with $< 30\%$ area retained inside crop window are excluded. | **No Issue** | Functioning as Designed | Fully tracked in `crop_provenance_mapping.json`. Zero primary defects lost. |
| **Duplicate Crop Avoidance:** Defect clustering eliminates redundant overlapping crops. | **No Issue** | Quality Improvement | IoU $> 0.50$ rate is 0.0%. Multi-defect clusters cleanly captured in enlarged crops. |
| **Zero Split Leakage:** Frozen Phase 1 splits strictly preserved across all manifests. | **No Issue** | Verification Success | Split integrity 100% verified. |

---

## 10. Final Acceptance Recommendation

### Verdict: **ACCEPT PHASE 3 WITH DOCUMENTED ARCHITECTURAL RECOMMENDATIONS**

**Justification:**
1. All Phase 3 deliverables (`crop_geometry.py`, `defect_cluster.py`, `crop_extractor.py`, `full_image_pipeline.py`, `pipeline.py`, `preprocess_dataset.py`, `visualize_crops.py`) are fully implemented and verified.
2. Complete test suite passes cleanly: **46 of 46 tests passing**.
3. Bounding box mathematics, coordinate translation, boundary clamping, and YOLO format normalization are 100% error-free.
4. Frozen Phase 1 split manifests remain strictly preserved with zero group leakage.
5. The dual-stream architecture successfully reconciles the differing requirements of patch-based GAN generation (Stream 1) and full-image detector baselines (Stream 2).

### Guidelines for Phase 4 (Detector Baselines) & Phase 7 (GAN Augmentation):
- **For Phase 4 (Real-Only Detector Baseline):** Train and evaluate the baseline detector primarily on Stream 2 ($640 \times 640$ letterboxed images) so all 6 defect classes, including macro-scale `abnormal_thickness`, are evaluated in natural full-substrate context.
- **For Phase 7 (GAN Augmentation):** Target generative patch augmentation on the localized defect classes (`spot`, `scar`, `adhesion`, `defect`, `uneven`) where $256 \times 256$ crops provide ideal texture resolution.
