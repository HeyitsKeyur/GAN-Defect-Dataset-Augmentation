# PHASE5_STAGE_F_BASELINE_PLAN.md
# Stage F: Real-Only Detector Baseline Planning & Experimental Design

**Project Title:** Defect Dataset Augmentation using Generative Adversarial Networks (GAN)  
**Academic Context:** M.Tech Major Project (Computer Science & Engineering), Nirma University (2025–2027)  
**Author / Student:** Keyur Surti (Roll No. 23MTCSE018)  
**Project Supervisor:** Prof. Sonia Mittal  
**Current Git Branch:** `soldef-dataset-migration`  
**Verified Published Commit:** `5cb89f6f7acfa6f24c1630c50eb8b87190a8cc63`  
**Remote Repository:** `https://github.com/HeyitsKeyur/GAN-Defect-Dataset-Augmentation.git`  
**Document Status:** Complete Planning Document (Read-Only / No Code Execution)  
**Date:** 2026-11-25  

---

## 1. Executive Summary & Objective

The objective of Stage F is to establish a rigorous, technically justified, and scientifically sound experimental plan for training the first **Real-Only Object Detection Baseline** for the SolDef_AI PCB solder defect dataset.

This baseline serves as the fundamental scientific control and reference benchmark for the entire M.Tech research project. All subsequent experimental interventions—including conventional image data augmentation and conditional Generative Adversarial Network (GAN) synthetic defect augmentation—will be evaluated against this exact baseline under strictly controlled conditions.

### Strict Stage F Boundaries
- **Scope:** Planning, architectural design, hyperparameter selection, evaluation protocol specification, and Colab execution blueprint.
- **Prohibitions:** Strictly zero model training, zero training code implementation, zero notebook execution, zero synthetic generation, zero modification of dataset splits or annotations, and zero unauthorized Git commits or pushes.

---

## 2. Mandatory Repository & Environment Verification

An empirical, read-only audit of the repository was conducted to establish ground truth prior to experimental planning:

| Inspection Item | Observed Value / Path | Verification Status |
|---|---|---|
| **Active Git Branch** | `soldef-dataset-migration` | Verified (`git branch -vv`) |
| **Current HEAD Commit** | `5cb89f6f7acfa6f24c1630c50eb8b87190a8cc63` | Verified (Clean working tree) |
| **Remote Tracking** | `origin/soldef-dataset-migration` | Verified (Published & Up to date) |
| **`main` Branch Status** | `fe4cf2f9852291872d6ada8f73d0b7659c873d0a` | Verified (Completely untouched) |
| **Dataset Adapter** | `src/data/adapters/soldef.py` | Verified (Multi-view grouping implemented) |
| **Preprocessing Pipeline** | `src/data/preprocessing/soldef_detector_pipeline.py` | Verified ($640 \times 640$ letterbox pipeline) |
| **Configuration Files** | `configs/soldef_preprocessing.json` & `.yaml` | Verified (Portable path mapping) |
| **Split Manifests** | `outputs/splits/soldef/phase1/split_manifest.json` | Verified (200 images, 66 groups, 0 leakage) |
| **Split CSV Records** | `outputs/splits/soldef/phase1/{train,val,test}.csv` | Verified (140 train, 30 val, 30 test) |
| **Detector Manifests** | `outputs/preprocessing/soldef/phase_d/metadata/` | Verified (369 target defect boxes) |
| **Existing Colab Notebook** | `notebooks/colab/DataSet_Augmentation_GAN.ipynb` | Verified (Phase 2 skeleton; cells 17-22 placeholder) |
| **Existing Model Code** | `src/models/` | **Verified Absent** (No detector code exists) |
| **Full Unit Test Suite** | 61 / 61 tests passing (100%) | Verified via pytest |

---

## 3. Dataset & Benchmark Scope (Primary Side-View V2 + V2.1)

The baseline model is planned exclusively for the primary side-view solder joint benchmark ($V_2$ and $V_{2.1}$). Top-down images ($V_1$) represent component-level placement qualification (`good` / `no_good`) and are strictly isolated as a secondary exploratory benchmark.

### 3.1 Input Geometry & Preprocessing
- **Source Resolution:** $2560 \times 1440$ pixels, 24-bit sRGB color.
- **Detector Input Resolution:** $640 \times 640$ pixels ($3 \times 640 \times 640$).
- **Transformation:** Isotropic letterboxing with fixed scale factor $s = 0.25$, vertical padding $pad_{top} = 140\text{ px}$, $pad_{bottom} = 140\text{ px}$, and $pad_{left} = pad_{right} = 0\text{ px}$.
- **Coordinate System:** YOLO normalized format $[c_x, c_y, w, h] \in [0, 1]^4$, strictly bounded with zero coordinate distortion.

### 3.2 Target Defect Classes
The baseline defect detector operates on exactly **three defect classes**:
1. **Class ID 0 — `exc_solder` (Excess Solder):** Solder volume exceeding wetting specifications, convex bulges obscuring component leads.
2. **Class ID 1 — `spike` (Solder Spike / Protrusion):** Sharp, pointed metallic protrusions extending outward from the fillet.
3. **Class ID 2 — `poor_solder` (Insufficient Solder / Poor Wetting):** Deficient solder volume, concave wetting angle failure, exposed pad metallization.

> [!IMPORTANT]
> **Negative Image Policy:** The compliant `good` joint class (74 annotations in source records) is excluded from the defect detection target set. Images containing only `good` joints (5 images in Train) are retained in the training set with empty label files (`0` bounding boxes) to serve as legitimate background negative samples, training the detector to suppress false positives on defect-free joints.

### 3.3 Partition Distribution & Defect Counts

```
+---------------------------------------------------------------------------------------------------+
|                                SOLDEF_AI SIDE-VIEW BENCHMARK (V2 + V2.1)                          |
+-------------------+---------------+------------------+--------------------------------------------+
| Partition         | Images Count  | Grouping Units   | Target Defect Bounding Boxes               |
|                   |               |                  | exc_solder  | spike   | poor_solder | Total |
+-------------------+---------------+------------------+-------------+---------+-------------+-------+
| Train Split       | 140 (70.0%)   | 30 groups        | 128         | 96      | 32          | 256   |
| Validation Split  |  30 (15.0%)   | 19 groups        |  26         | 15      | 10          |  51   |
| Test Split (Held) |  30 (15.0%)   | 17 groups        |  28         | 19      | 15          |  62   |
+-------------------+---------------+------------------+-------------+---------+-------------+-------+
| Total Benchmark   | 200 (100.0%)  | 66 groups        | 182 (49.3%) | 130     | 57 (15.4%)  | 369   |
+-------------------+---------------+------------------+-------------+---------+-------------+-------+
```

---

## 4. Dataset Limitations & Scientific Constraints

A rigorous scientific baseline must acknowledge empirical dataset constraints:

1. **Very Small Sample Size:** With only 140 training images and 256 defect instances, deep neural networks are subject to rapid memorization and overfitting.
2. **Severe Class Imbalance:** `poor_solder` is heavily minority-class (32 instances in Train, 10 in Val, 15 in Test).
3. **Exploratory Power of Test Metrics on `poor_solder`:** The held-out test set contains only 15 instances of `poor_solder`. Consequently, an error of 1 or 2 missed detections shifts the per-class recall by $6.7\%$ to $13.3\%$. These metrics must be reported as indicative/exploratory rather than high-powered statistical proof.
4. **Group Clustering & Specimen Fallback Blocks:** 74.2% of training defect annotations ($190 / 256$) originate from two large atomic fallback blocks (`CS6_R0805`: 117 boxes, `CS2_R0805`: 73 boxes). Zero leakage across grouping units prevents training on identical image pairs, but background circuit board textures share visual similarity.
5. **Overlapping Defect Instances:** 37 of 130 `spike` defects ($28.5\%$) spatially overlap an underlying `exc_solder` or solder fillet box. The detector must localize both the broad fillet and the narrow spike simultaneously without Non-Maximum Suppression (NMS) erroneously suppressing the spike.

---

## 5. Model Selection Analysis & Architecture Comparison

To determine the most appropriate object detector architecture, four representative candidate families were evaluated against the project's specific constraints:

### 5.1 Architecture Comparison Matrix

| Evaluation Criteria | YOLOv8s (Ultralytics) | YOLOv8n (Nano) | Faster R-CNN (ResNet-50) | RT-DETR (Transformer) |
|---|---|---|---|---|
| **Parameter Count** | **11.2 M** | 3.2 M | 41.8 M | 32.9 M |
| **Computational Cost (FLOPs)** | **28.6 GFLOPs** | 8.7 GFLOPs | ~180 GFLOPs | 110 GFLOPs |
| **Detection Paradigm** | Anchor-free, Decoupled Head | Anchor-free, Decoupled Head | Two-stage, Anchor-based RPN | Transformer encoder-decoder |
| **Small Object Sensitivity** | **High (P3/8, P4/16, P5/32 FPN)**| Moderate | Moderate-High (RPN anchors) | High (Multi-scale attention) |
| **Overfitting Risk on 140 Images** | **Low-Moderate (Pretrained COCO)**| **Lowest** | **High** (High capacity) | **Severe** (Attention data hungry) |
| **Colab T4 Memory (Batch 16)** | **~3.5 GB VRAM (Fits easily)** | ~2.0 GB VRAM | ~9.8 GB VRAM | ~8.5 GB VRAM |
| **Training Time (120 epochs)** | **~15–20 min on T4** | ~10 min on T4 | ~60 min on T4 | ~45 min on T4 |
| **Overlapping Defect Handling** | **Excellent (Decoupled head)** | Good | Needs anchor/IoU tuning | Good |
| **Literature Comparability** | **De facto standard (2024-2026)**| Common baseline | Historical baseline | Emerging |

### 5.2 Model Selection Verdict

1. **PRIMARY BASELINE MODEL: YOLOv8s (`yolov8s.pt`)**
   - **Rationale:** YOLOv8s represents the optimal balance of representational capacity and regularization for a 140-image transfer learning task. Its anchor-free decoupled head eliminates the need for manual anchor box clustering, which is especially critical for SolDef where defect aspect ratios range from tall, needle-like spikes ($w/h < 0.3$) to wide solder fillets ($w/h > 3.0$). Pretrained COCO weights provide strong edge, texture, and metallic surface priors, preventing overfitting when trained with standard regularization.
2. **OPTIONAL SECONDARY / ABLATION MODEL: YOLOv8n (`yolov8n.pt`)**
   - **Rationale:** With only 3.2M parameters, YOLOv8n provides an ultra-lightweight ablation to verify whether detector capacity impacts minority-class feature representation. It serves as a rapid local smoke test candidate.
3. **MODELS EXCLUDED AS INAPPROPRIATE:**
   - **Faster R-CNN:** Excluded due to high parameter count (41.8M) leading to rapid overfitting on 140 images, rigid anchor aspect-ratio sensitivity, and sluggish Colab execution.
   - **RT-DETR / Deformable DETR:** Excluded because Transformer architectures lack inductive biases (such as translational equivariance) and notoriously require tens of thousands of samples to avoid attention collapse on small datasets.

---

## 6. Baseline Training Strategy & Hyperparameter Planning

To ensure scientific fairness and prevent data snooping, all hyperparameters are established *a priori* based on small-sample computer vision principles:

### 6.1 Proposed Training Configuration Table

| Hyperparameter | Recommended Baseline Value | Candidate Range (Validation Tuning) | Technical Justification |
|---|---|---|---|
| **Model Checkpoint** | `yolov8s.pt` (COCO pretrained) | `yolov8s.pt` / `yolov8n.pt` | Transfer learning initialization; random weights fail on 140 samples. |
| **Input Image Size** | $640 \times 640$ | Fixed at 640 | Matches Stage D letterbox preprocessing resolution natively. |
| **Batch Size** | `16` | `8`, `16` | Optimal gradient stability on Colab T4 without memory thrashing. |
| **Optimizer** | `SGD` (momentum = 0.937) | `SGD` vs `AdamW` (lr0 = 0.001) | SGD with momentum exhibits superior generalization on small datasets compared to AdamW. |
| **Initial Learning Rate ($lr_0$)**| `0.01` | `0.005` – `0.01` | Standard Ultralytics SGD base rate with linear warmup. |
| **Final Learning Rate ($lrf$)** | `0.01` (final lr = $10^{-4}$) | Fixed | Cosine annealing schedule gradually decays learning rate. |
| **Warmup Epochs** | `3.0` | `3.0` – `5.0` | Stabilizes pretrained weights during initial batch exposure. |
| **Weight Decay** | `0.0005` | `0.0005` – `0.001` | $L_2$ penalty to constrain weights and curb overfitting on small sample. |
| **Maximum Epochs** | `120` | `100` – `150` | Sufficient for convergence on 140 images with early stopping guard. |
| **Early Stopping Patience** | `25 epochs` | `20` – `30` | Halts training if validation `mAP@0.50` fails to improve for 25 consecutive epochs. |
| **Checkpoint Criterion** | Max Validation `mAP@0.50` | `fitness` ($0.1 \text{mAP}_{50} + 0.9 \text{mAP}_{50-95}$)| Best checkpoint selected strictly on Validation split metrics. |
| **Random Seed** | `42` | Fixed | Universal project seed ensuring deterministic mini-batch ordering. |
| **Device** | `0` (CUDA / GPU) | `0` or `cpu` (smoke tests) | Targets Colab NVIDIA Tesla T4 runtime. |

### 6.2 Strict Real-Only Augmentation Policy

In standard YOLO training, aggressive synthetic transformations (Mosaic, MixUp, Copy-Paste) are enabled by default. In our research, these transformations create synthetic composite artifacts that would confound our scientific comparison against GAN-generated samples.

**Planned Real-Only Augmentation Configuration:**
- `mosaic: 0.0` (DISABLED) — Prevents artificial 4-image stitching that alters joint context.
- `mixup: 0.0` (DISABLED) — Prevents blending solder joints into translucent phantoms.
- `copy_paste: 0.0` (DISABLED) — Prevents pasting defect patches onto non-defect joints.
- `fliplr: 0.5` (ENABLED) — Horizontal flip simulates viewing the joint from the opposite direction (physically valid).
- `flipud: 0.0` (DISABLED) — Vertical flip produces inverted gravity and upside-down PCB pads (physically invalid).
- `scale: 0.1` (MINIMAL) — Minor scale jitter ($\pm 10\%$) simulates slight camera distance variation.
- `hsv_v: 0.2` (MINIMAL) — Minor brightness variation simulates microscopic lighting fluctuations.
- `degrees: 0.0` (DISABLED) — Arbitrary rotation produces unphysical tilted PCB orientations.

This configuration defines a clean, authentic **Real-Only baseline** that reflects genuine microscopic inspection conditions.

---

## 7. Comprehensive Evaluation Framework

### 7.1 Metric Hierarchy

```
+---------------------------------------------------------------------------------------------------+
|                                      METRIC EVALUATION HIERARCHY                                  |
+-------------------+----------------------+--------------------------------------------------------+
| Level             | Metric Name          | Scientific Role & Interpretation                       |
+-------------------+----------------------+--------------------------------------------------------+
| **PRIMARY**       | **mAP@0.50**         | Overall benchmark accuracy across all 3 defect classes |
|                   |                      | at IoU threshold 0.50 (Standard PASCAL VOC criterion). |
+-------------------+----------------------+--------------------------------------------------------+
| **SECONDARY**     | **mAP@0.50:0.95**    | Strict bounding-box localization precision across      |
|                   |                      | 10 IoU thresholds (0.50, 0.55, ..., 0.95).             |
+-------------------+----------------------+--------------------------------------------------------+
| **CLASS-SPECIFIC**| **AP@0.50 [class]**  | Individual Average Precision per defect class:         |
|                   |                      | - AP_exc_solder (Majority class)                       |
|                   |                      | - AP_spike (Overlapping defect class)                  |
|                   |                      | - AP_poor_solder (Severe minority class)               |
+-------------------+----------------------+--------------------------------------------------------+
| **OPERATIONAL**   | **Precision & Recall**| Operational trade-off curves at optimal F1 threshold:   |
|                   | **F1-Score**         | $F1 = 2 \cdot (P \cdot R) / (P + R)$.                  |
+-------------------+----------------------+--------------------------------------------------------+
```

### 7.2 Diagnostic Error Analysis Strategy
Quantitative metrics alone cannot capture industrial failure modes. The baseline evaluation will produce:
1. **Confusion Matrix (Normalized & Raw):** Identifies inter-class confusion (e.g. `poor_solder` misclassified as `exc_solder`) and background false positives.
2. **False Negative (Missed Defect) Audit:** Visual inspection of false negatives on the test set, specifically examining whether missed spikes were obscured by overlapping fillets.
3. **False Positive (Ghost Defect) Audit:** Inspection of negative image predictions to verify that compliant solder fillets are not falsely flagged as defects.
4. **Precision-Recall (PR) Curves:** Generated for each class to inspect recall behavior under strict confidence thresholds.

---

## 8. Validation and Test Protocol (Strict Leakage Prevention)

To preserve scientific validity, strict partition discipline is enforced:

```
+-----------------------------------------------------------------------------------------------+
|                                    PARTITION ROLE & ISOLATION                                 |
+-----------------------------------------------------------------------------------------------+
| 1. TRAIN SPLIT (140 images, 30 groups)                                                        |
|    - Exclusively used for gradient updates during model optimization.                         |
|    - No validation or test images are ever exposed to the training loop.                      |
+-----------------------------------------------------------------------------------------------+
| 2. VALIDATION SPLIT (30 images, 19 groups)                                                    |
|    - Evaluated at the end of every epoch.                                                     |
|    - Controls early stopping (patience = 25 epochs).                                          |
|    - Determines the `best.pt` model weights.                                                  |
|    - Candidate hyperparameter adjustments (batch size, lr) are compared ONLY on this split.   |
+-----------------------------------------------------------------------------------------------+
| 3. TEST SPLIT (30 images, 17 groups) — STRICTLY HELD-OUT REAL-ONLY BENCHMARK                  |
|    - Completely quarantined during training and hyperparameter selection.                     |
|    - Evaluated exactly ONCE per model using the final `best.pt` checkpoint.                   |
|    - Zero threshold tuning or checkpoint cherry-picking is permitted on the Test split.       |
|    - Remains real-only for ALL future experiment arms (Real-Only, Real+Aug, Real+GAN).        |
+-----------------------------------------------------------------------------------------------+
```

---

## 9. Google Colab Implementation Blueprint

The baseline training will be orchestrated on Google Colab GPU runtimes following the architecture defined in `COLAB_INTEGRATION_ARCHITECTURE.md`:

```
+-------------------------------------------------------------------------------+
|                             COLAB EXECUTION PIPELINE                          |
+-------------------------------------------------------------------------------+
| Step 1: Runtime Setup & Verification                                          |
|         - Verify Python >= 3.10, PyTorch >= 2.0, CUDA available.               |
|         - Inspect GPU model (NVIDIA Tesla T4 / V100 / A100) and VRAM.         |
|                                                                               |
| Step 2: Google Drive Mounting                                                 |
|         - Mount `/content/drive/MyDrive/` to access raw SolDef data and       |
|           persistent output destination.                                      |
|                                                                               |
| Step 3: Canonical Repository Integration                                      |
|         - Clone or pull `https://github.com/HeyitsKeyur/GAN-Defect-Dataset...`|
|         - Checkout branch `soldef-dataset-migration`.                         |
|         - Verify HEAD commit matches `5cb89f6`.                               |
|                                                                               |
| Step 4: Environment Installation                                              |
|         - Install dependencies: `pip install -q ultralytics pyyaml`.          |
|                                                                               |
| Step 5: Data Integrity & Manifest Verification                                |
|         - Verify existence of `outputs/splits/soldef/phase1/{train,val,test}` |
|         - Run lightweight dataset integrity verification script.              |
|                                                                               |
| Step 6: Dataset Configuration Setup                                           |
|         - Generate dynamic YOLO dataset YAML pointing to preprocessed images. |
|                                                                               |
| Step 7: Baseline Training Execution (Modular Script)                          |
|         - Execute `python scripts/train_detector_baseline.py --config ...`    |
|         - Train YOLOv8s with seed=42, 120 epochs, patience=25, mosaic=0.0.    |
|                                                                               |
| Step 8: Validation & Checkpoint Selection                                     |
|         - Identify `best.pt` based on validation mAP@0.50.                    |
|                                                                               |
| Step 9: Final Held-Out Test Evaluation                                        |
|         - Execute single-pass evaluation on Test split:                       |
|           `python scripts/evaluate_detector.py --weights best.pt --split test`|
|                                                                               |
| Step 10: Artifact Persistence & Export                                        |
|         - Copy checkpoints, training curves, logs, and evaluation JSONs to    |
|           `/content/drive/MyDrive/MajorProject_Outputs/experiments/...`        |
|         - Save environment metadata and execution duration log.               |
+-------------------------------------------------------------------------------+
```

---

## 10. Experiment Artifact Directory Structure

All generated outputs from the baseline training run will be organized deterministically:

```
outputs/experiments/baseline_yolov8s_real/
├── config/
│   ├── experiment_config.json              # Complete training hyperparameters and paths
│   └── dataset_soldef.yaml                 # YOLO dataset configuration file
├── metadata/
│   ├── run_metadata.json                   # Git commit SHA, timestamp, seed, duration
│   └── environment.json                    # Python, PyTorch, CUDA, GPU model, OS details
├── weights/
│   ├── best.pt                             # Best checkpoint selected via validation mAP
│   └── last.pt                             # Final epoch checkpoint for resumability
├── logs/
│   ├── train.log                           # Full console stdout and training logs
│   ├── results.csv                         # Per-epoch loss and validation metrics
│   └── args.yaml                           # Ultralytics internal runtime arguments
├── metrics/
│   ├── val_summary_metrics.json            # Final validation mAP50, mAP50-95, P, R
│   ├── test_summary_metrics.json           # Held-out test mAP50, mAP50-95, P, R
│   └── per_class_test_metrics.csv          # Per-class AP, P, R, F1 for exc, spike, poor
├── plots/
│   ├── results.png                         # Loss curves and mAP trajectories
│   ├── confusion_matrix.png                # Normalized confusion matrix on test split
│   ├── precision_recall_curve.png          # Per-class and aggregate PR curves
│   └── f1_confidence_curve.png             # F1 score vs confidence threshold curve
└── visualizations/
    ├── batch_0_labels.jpg                  # Sample ground-truth training batch
    └── test_predictions/                   # Visual predictions on 30 held-out test images
```

---

## 11. Scientific Alignment with Future Research Paper

The experimental protocol directly supports the eventual M.Tech thesis and peer-reviewed research publication:

### 11.1 Controlled Experimental Matrix

```
+-------------------------------------------------------------------------------------------------------+
|                                  PLANNED PROJECT EXPERIMENTAL MATRIX                                  |
+-----+-----------------------+---------------------+-------------------+-------------------------------+
| Arm | Experiment Name       | Training Set        | Test Set          | Scientific Objective          |
+-----+-----------------------+---------------------+-------------------+-------------------------------+
| E0  | **Real-Only Baseline**| 140 Real Images     | 30 Real Images    | **Reference Control (Stage F/G)**|
|     | (This Plan)           | (256 Real Defect Bxs| (62 Real Defect   | Establishes raw benchmark     |
|     |                       |                     |  Boxes)           | performance without synthesis.|
+-----+-----------------------+---------------------+-------------------+-------------------------------+
| E1  | Conventional Aug      | 140 Real + Classic  | 30 Real Images    | Quantifies impact of standard |
|     | Comparison            | Augmentations       | (Identical Test)  | geometric/color distortions.  |
+-----+-----------------------+---------------------+-------------------+-------------------------------+
| E2  | Real + Patch GAN      | 140 Real + Synthetic| 30 Real Images    | Evaluates whether GAN crops   |
|     | Augmentation          | Patches (Phase 7)   | (Identical Test)  | improve rare defect detection.|
+-----+-----------------------+---------------------+-------------------+-------------------------------+
| E3  | Real + Class-Balanced | 140 Real + Targeted | 30 Real Images    | Evaluates focused balancing of|
|     | GAN Augmentation      | `poor_solder` GAN   | (Identical Test)  | the 32-sample minority class. |
+-----+-----------------------+---------------------+-------------------+-------------------------------+
```

### 11.2 Experimental Control Invariants
Across all experimental arms:
- The **Detector Architecture** is held constant (`YOLOv8s`).
- The **Test Set** is held constant (identical 30 real images, 62 defect instances).
- The **Validation Set** is held constant (identical 30 real images).
- The **Random Seed** is held constant (`42`).
- The **Evaluation Metric Code** is identical.
- The **Only Independent Variable** is the composition of the training data.

---

## 12. Comprehensive Risk Assessment & Mitigation Register

| Risk Factor | Severity | Probability | Impact on Experiment | Planned Mitigation Strategy |
|---|---|---|---|---|
| **Small Sample Overfitting** | High | High | Detector memorizes training backgrounds and fails to generalize on test set. | Pretrained COCO initialization, strong $L_2$ weight decay ($0.0005$), early stopping (patience 25), and minimal photometric jitter. |
| **Minority Class Starvation (`poor_solder`)** | High | High | Detector predicts zero `poor_solder` instances (AP = 0.0) due to extreme imbalance. | Monitor per-class recall closely. Document raw baseline baseline behavior without artificial class weighting so future GAN impact can be measured cleanly. |
| **Overlapping Defect Suppression** | Moderate | Moderate | YOLO NMS suppresses `spike` detections that overlap `exc_solder` fillets. | Utilize decoupled head architecture; inspect class-agnostic NMS settings (`agnostic_nms = False`) so different classes do not suppress each other. |
| **Colab Runtime Disconnection** | Moderate | High | Long training run interrupted before completion. | 120 epochs on YOLOv8s executes in ~15–20 minutes, well within standard Colab timeout windows. Regular checkpointing (`last.pt`) to Google Drive. |
| **Test Set Snooping / Leakage**| Critical | Low | Evaluating test set iteratively during tuning invalidates scientific baseline. | Enforce strict quarantine protocol: test evaluation occurs exactly ONCE using `best.pt` after all training decisions are frozen. |

---

## 13. Stage G Implementation & Execution Handoff

### 13.1 Stage G Scope of Work (Future Authorized Tasks)
When Stage G is approved by the user, the following tasks will be implemented:
1. **Module Implementation:** Author modular CLI script `scripts/train_detector_baseline.py` and evaluation script `scripts/evaluate_detector.py`.
2. **Config Generation:** Create `configs/soldef_baseline_yolov8s.yaml` defining YOLO dataset mappings and hyperparameters.
3. **Colab Notebook Update:** Update `notebooks/colab/DataSet_Augmentation_GAN.ipynb` cells 17–22 to orchestrate execution on Colab GPU.
4. **Local Smoke Testing:** Run 1-epoch CPU smoke test locally with synthetic/dummy flags to verify script syntax and artifact generation before Colab upload.
5. **Colab Execution:** Execute full 120-epoch training on Colab T4 GPU.
6. **Artifact Collection:** Retrieve checkpoints, logs, and evaluation reports.

### 13.2 Explicitly Excluded from Stage G
- GAN training or synthetic generation.
- Top-down V1 dataset training.
- Conventional augmentation experimentation (reserved for subsequent phase).
- Modification of frozen splits or source annotations.

---

## 14. Document History & Revision Tracking

| Version | Date | Author | Description |
|---|---|---|---|
| **1.0** | 2026-11-25 | Keyur Surti | Initial comprehensive baseline planning document for SolDef primary benchmark. |
