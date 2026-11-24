# SolDef_AI Split Feasibility & Acceptance Report

**Date:** 2026-11-22  
**Dataset:** SolDef_AI (European Space Agency ECSS-Q-ST-70-38C Standard)  
**Primary Benchmark:** Side-View Solder Joint Defect Detection (Views $ and {2.1}$)  
**Feasibility Verdict:** **STRONGLY FEASIBLE (100% Group-Disjoint, Zero View Leakage)**

---

## 1. Group Inventory by Provenance Confidence

| Evidence Tier | Criteria | Group Count | Image Count | % of Images |
| :--- | :--- | :--- | :--- | :--- |
| **Verified Specimen Groups** | Explicit component serial suffix _C<num> (CS6 C0805) | **11** | **15** | 10.5% |
| **Inferred Specimen Groups** | Chronological capture matching with equal view counts | **53** | **86** | 40.0% |
| **Atomic Fallback Blocks** | Asymmetric runs grouped to prevent multi-view cross-leakage | **2** | **99** | 49.5% |
| **Total Independent Units** | — | **66** | **200** | **100.0%** |

---

## 2. 70 / 15 / 15 Split Feasibility Analysis

- **Target Split:** 70% Train (140 images), 15% Validation (30 images), 15% Test (30 images).
- **Atomic Block Placement:** Both atomic fallback blocks (CS2_R0805_atomic_block [36 images] and CS6_R0805_atomic_block [63 images]) reside in the **Train** partition (\text{ images}$, .5\%$). Placing either in Validation or Test would exceed the 15% (30-image) capacity.
- **Specimen Stratification:** The remaining 64 specimen groups (101 images) are stratified:
  - **Train:** 26 specimen groups (41 images) $\rightarrow$ Total Train = **140 images (70.0%)**
  - **Validation:** 19 specimen groups (30 images) $\rightarrow$ Total Val = **30 images (15.0%)**
  - **Test (Real-Only):** 19 specimen groups (30 images) $\rightarrow$ Total Test = **30 images (15.0%)**

---

## 3. Class Instance Distribution Across Partitions

| Class | Train Instances (Images) | Val Instances (Images) | Test Instances (Images) | Total Instances | Class Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| exc_solder | 128 (101) | 26 (21) | 28 (19) | 182 | Target Defect |
| spike | 96 (81) | 15 (14) | 19 (19) | 130 | Target Defect |
| poor_solder | 32 (27) | 10 (10) | 15 (14) | 57 | Minority Target Defect |
| good | 50 (43) | 14 (14) | 10 (10) | 74 | Normal Baseline |
| **Total** | **306** | **65** | **72** | **443** | — |

---

## 4. Leakage Verification Gate

- **Image Overlap:** Exactly 0 shared image stems between any two partitions.
- **Group Overlap:** Exactly 0 shared groups between any two partitions.
- **Multi-View Integrity:** All paired captures ($ and {2.1}$) for any specimen or capture run remain strictly confined to a single partition.
- **Real-Only Test Discipline:** Test set is 100% real images and instances, ready for frozen downstream evaluation.
