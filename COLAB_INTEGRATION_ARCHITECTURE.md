# COLAB_INTEGRATION_ARCHITECTURE.md

**Major Project:** Defect Dataset Augmentation using Generative Adversarial Networks (GAN)  
**Document Purpose:** Blueprint for Google Colab, Google Drive, GitHub, and Two-Laptop Integration  
**Current Phase:** Phase 2 (Environment, Configuration, and Reproducibility)  
**Target Repository:** `https://github.com/HeyitsKeyur/GAN-Defect-Dataset-Augmentation.git`  
**Colab Notebook:** `DataSet_Augmentation_GAN.ipynb` (URL: `https://colab.research.google.com/drive/1o3o9C6HJTphazQJR39xmHCfbynIzm4b7`)

---

## 1. System Architecture Overview

The project relies on a decoupled, synchronized architecture separating development and execution:

```
+-------------------------------------------------------------------------------+
|                        CANONICAL GITHUB REPOSITORY                            |
|             (Source code, tests, configs, split manifests, docs)              |
|              https://github.com/HeyitsKeyur/GAN-Defect-Dataset-Augmentation    |
+----------------------+---------------------------------+----------------------+
                       ^                                 ^
                       | push / pull                     | clone / pull
                       v                                 v
+--------------------------------------+   +------------------------------------+
|     LOCAL DEVELOPMENT ENVIRONMENTS   |   |        GOOGLE COLAB GPU RUNTIME    |
| (VS Code / Antigravity / Windows OS) |   |    (Thin orchestration notebook)   |
|                                      |   |                                    |
| - Laptop 1 (Drive E: Workstation)    |   | - GPU discovery & verification     |
| - Laptop 2 (Portable Development)    |   | - High-throughput preprocessing    |
| - Preprocessing logic & Unit tests   |   | - Baseline detector training       |
| - Configuration & Git management     |   | - GAN training & evaluation        |
+--------------------------------------+   +------------------+-----------------+
                                                              | mount Drive
                                                              v
                                           +------------------------------------+
                                           |        GOOGLE DRIVE / MYDRIVE      |
                                           |                                    |
                                           | - Original Ceramic-defect Dataset  |
                                           |   (/MyDrive/Ceramic-defect/...)    |
                                           | - Resumable Model Checkpoints      |
                                           |   (/MyDrive/MajorProject_Outputs/) |
                                           | - Experiment Logs & Metrics        |
                                           +------------------------------------+
```

---

## 2. Environment Roles and Responsibilities

| Component | Primary Environment | Storage Location | Key Responsibilities |
|---|---|---|---|
| **Source Code & Unit Tests** | Local Laptops (VS Code / Antigravity) | Git & GitHub | Algorithm development, test-driven validation, code quality. |
| **Colab Orchestration** | Google Colab | `notebooks/colab/` (Git) & Drive | Runtime setup, Drive mounting, pulling Git commits, executing scripts. |
| **Original Dataset** | Read-only in Colab & Local | Google Drive & Local Disk | Unmodified raw Ceramic-Defect images and annotations. |
| **Frozen Split Manifests** | Git & GitHub | `outputs/splits/phase1/` | Deterministic train/val/test assignment; group leakage prevention. |
| **Model Weights & Logs** | Google Colab GPU | Google Drive (`MajorProject_Outputs/`) | Resumable training checkpoints, metrics, evaluation reports. |

---

## 3. Directory and Path Mapping

All reusable code uses `src.utils.config.get_config()`, ensuring zero hardcoded drive letters or absolute paths across machines:

| Setting Key | Local Laptop 1 (Workstation) | Local Laptop 2 (Portable) | Google Colab Runtime |
|---|---|---|---|
| `DATASET_ROOT` | `E:/.../Dateset/Ceramic-defect/data/data` | `D:/Datasets/Ceramic-defect/data/data` | `/content/drive/MyDrive/Ceramic-defect/data/data` |
| `SPLIT_MANIFEST_DIR` | `<repo>/outputs/splits/phase1` | `<repo>/outputs/splits/phase1` | `/content/GAN-Defect-Dataset-Augmentation/outputs/splits/phase1` |
| `GOOGLE_DRIVE_ROOT` | `None` | `None` | `/content/drive/MyDrive` |
| `CHECKPOINT_DIR` | `<repo>/outputs/checkpoints` | `<repo>/outputs/checkpoints` | `/content/drive/MyDrive/MajorProject_Outputs/checkpoints` |
| `EXPERIMENT_OUTPUT_DIR` | `<repo>/outputs` | `<repo>/outputs` | `/content/drive/MyDrive/MajorProject_Outputs` |

---

## 4. Colab Notebook Synchronization Workflow

To ensure `DataSet_Augmentation_GAN.ipynb` never diverges between GitHub and Google Drive:

1. **Canonical Source of Truth:**
   The version-controlled notebook in Git is:
   `notebooks/colab/DataSet_Augmentation_GAN.ipynb`
2. **Opening in Google Colab:**
   - Option A: In Colab, choose **File -> Open notebook -> GitHub -> select repository and branch** `phase-2-colab-integration`.
   - Option B: If opening directly from Google Drive (`https://colab.research.google.com/drive/1o3o9C6HJTphazQJR39xmHCfbynIzm4b7`), select **File -> Save a copy in GitHub** whenever notebook orchestration cells are updated.
3. **Execution Separation:**
   Core logic resides in `.py` modules under `src/`. The notebook only issues calls like:
   ```python
   !python scripts/run_colab_smoke_test.py --dataset-root "$DATASET_ROOT"
   ```
   This prevents code drift between notebook cell contents and repository files.

---

## 5. Google Drive Structure

Based on live verification, the Google Drive directory structure is organized as:

```
/content/drive/MyDrive/
├── Ceramic-defect/
│   └── data/
│       └── data/
│           ├── image/           (Vendor split folders: train, val, test)
│           ├── lables/          (json, xml, txt annotations)
│           └── original image/  (2,892 full-resolution JPEG images)
└── MajorProject_Outputs/         (Created automatically by Colab notebook)
    ├── checkpoints/             (Resumable GAN & detector model weights)
    ├── logs/                    (Tensorboard / training execution logs)
    └── smoke_test/              (Smoke test JSON & Markdown reports)
```

> **Strict Protection:** Under no circumstances should original dataset images in `/content/drive/MyDrive/Ceramic-defect/` be moved, renamed, deleted, or overwritten.

---

## 6. Two-Laptop Workflow Guide

When alternating between your two laptops:

1. **Before Starting Work:**
   ```bash
   git status
   git pull --ff-only origin <current-branch>
   ```
2. **Local Machine Settings (`.env`):**
   - Copy `.env.example` to `.env` on each machine.
   - Configure machine-specific `DATASET_ROOT` in `.env`.
   - `.env` is ignored by Git, preventing path conflicts across laptops.
3. **After Completing Work:**
   ```bash
   git status
   pytest
   git add <modified-files>
   git commit -m "Descriptive commit message"
   git push origin <current-branch>
   ```

---

## 7. Resumable Checkpoint Protocol

To guard against Google Colab runtime disconnections during GPU training (Phase 4 and Phase 7):

1. **Periodic Checkpoint Saving:**
   Models save checkpoints every $N$ epochs directly to `/content/drive/MyDrive/MajorProject_Outputs/checkpoints/`.
2. **State Dictionary Contents:**
   Checkpoints store:
   - `epoch`: Current epoch index
   - `model_state_dict`: Generator / Detector network parameters
   - `optimizer_state_dict`: Optimizer weights and momentum buffers
   - `loss`: Current training and validation loss metrics
   - `random_seed`: Seed for deterministic resumption
   - `commit_sha`: Git commit hash used to train the checkpoint
3. **Resumption Routine:**
   Before initiating a training loop, scripts check for existing checkpoint files, verify parameter compatibility, and resume without loss of training progress.
