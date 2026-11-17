# Colab / Local Runtime Smoke Test Report

**Execution Timestamp (UTC):** `2026-11-17T18:35:27.500005+00:00`  
**Overall Status:** **PASS**  
**Environment:** `windows` (Colab: False)  
**Python Version:** `3.12.10` on `Windows-11-10.0.26300-SP0`  

---

## 1. Hardware & Compute Acceleration
- **GPU Available:** False
- **GPU Model:** None (CPU runtime)
- **GPU Count:** 0
- **VRAM Total:** N/A
- **PyTorch Version:** `None`
- **CUDA Runtime:** `None`
- **Tensor Operation Test:** PASSED on `python-builtin (cpu-fallback)`

---

## 2. Dataset & Storage Verification
- **Dataset Path:** `E:\MTech[CSE] Nirma_University\S3\Major Project\Defect Dataset Augmentation using Generative Adversarial Networks (GAN)\Dateset\Ceramic-defect\data\data`
- **Dataset Verified:** YES


---

## 3. Phase 1 Frozen Split Manifest Verification
- **Manifest Directory:** `E:\MTech[CSE] Nirma_University\S3\Major Project\Defect Dataset Augmentation using Generative Adversarial Networks (GAN)\outputs\splits\phase1`
- **Manifests Verified:** YES
- **Split Counts:**
  - `train`: 1967 images
  - `val`: 479 images
  - `test`: 446 images
  - **Total Images:** 2892 (Expected: 2892)
- **Cross-Split Group Leakage Detected:** NO (100% LEAKAGE-FREE)


---

## 4. Installed Packages Summary
| Package | Version |
|---|---|
| `torch` | `not installed` |
| `torchvision` | `not installed` |
| `PIL` | `11.3.0` |
| `numpy` | `not installed` |
| `pytest` | `8.4.2` |


---

## 5. Notes & Observations
- Running in local Windows environment.
- No CUDA GPU detected; execution will fall back to CPU for local development.

