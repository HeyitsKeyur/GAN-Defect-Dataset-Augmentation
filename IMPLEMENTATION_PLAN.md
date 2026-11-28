# Major Project — Implementation Plan and Environment Allocation

**Project:** Defect Dataset Augmentation using Generative Adversarial Networks (GAN)

This plan translates `PROJECT_CONTEXT.md` into implementation phases. The canonical codebase is developed in VS Code and computationally intensive experiments are executed in Google Colab GPU. The two environments use one Git repository; they are not separate projects.

## 1. Environment allocation at a glance

| Phase | Main work | VS Code / local | Google Colab |
|---|---|---|---|
| 0 | Repository and dataset audit | Main execution; inspect files, annotations, dimensions, integrity | Not normally needed |
| 1 | Validation and splitting | Implement validation and group-aware split scripts; create manifests | Optional large-scale validation |
| 2 | Environment and reproducibility | Set up repository, dependencies, configs, logging, seeds, portable paths | Verify GPU/runtime compatibility |
| 3 | Preprocessing pipeline | Implement loaders, crops, resizing and label transforms; CPU tests | Optional performance tests on large data |
| 4 | Real-only detector baseline | Prepare config/scripts; inspect results | Main detector training and evaluation |
| 5 | Conventional augmentation | Implement/test augmentation code | Train/evaluate detector with conventional augmentation |
| 6 | GAN data preparation | Analyze boxes; create training-only crop/patch pipeline | Optional bulk processing if local resources insufficient |
| 7 | GAN implementation and training | Implement model, loss, configs and tests | Main GAN training on GPU |
| 8 | Synthetic generation and validation | Build validation/analysis scripts and inspect outputs | Generate batches and run GPU-heavy scoring where appropriate |
| 9 | Comparative experiments | Maintain configs and experiment records | Run controlled training/evaluation arms |
| 10 | Results analysis | Metrics analysis, plots, tables, error analysis | Only if GPU-specific evaluation is needed |
| 11 | Final report and presentation | Write methodology, results, limitations and documentation | Not normally needed |

**Simple rule:** write and maintain code in VS Code; use Colab to execute GPU-heavy training and generation. Dataset audit, preprocessing, testing, reporting and Git management primarily happen locally.

## 2. Detailed phases

### Phase 0 — Repository and dataset audit
**Primary environment:** VS Code / local CPU  
**Goal:** establish exactly what data and code are available before implementation.

Tasks:
1. Inspect the existing repository and do not overwrite existing work.
2. Locate the downloaded dataset archive and extracted folders.
3. Inspect annotation formats, class names/IDs, image paths and category counts.
4. Count images and bounding-box instances per class separately.
5. Inspect dimensions, aspect ratios, box sizes and annotation validity.
6. Identify missing/corrupt images, invalid boxes and unknown class IDs.
7. Check duplicate/near-duplicate images and possible paired/group relationships.
8. Produce a machine-readable audit report and a human-readable summary.

**Deliverables:** dataset audit script, audit JSON/CSV/Markdown report, confirmed data layout, unresolved dataset issues.

**Acceptance gate:** no training until dataset paths, labels and annotation integrity are understood. Do not assume reported web/article counts match the local archive.

### Phase 1 — Data validation and leakage-safe splits
**Primary environment:** VS Code / local CPU

Tasks:
1. Implement deterministic annotation validation.
2. Determine whether images can be grouped by substrate/specimen/batch or related views.
3. Split at group level where possible; otherwise document the limitation.
4. Create train, validation and test manifests.
5. Ensure the test set is real-only.
6. Freeze split definitions before GAN training.
7. Save split seed and class distributions.

**Deliverables:** split manifests, validation report, reproducible split script.

**Acceptance gate:** no duplicate or related samples should cross splits where identifiable; test set is excluded from GAN training and synthetic-data tuning.

### Phase 2 — Repository, environment and reproducibility
**Primary environment:** VS Code; verify in Colab

Tasks:
1. Inspect current project structure before proposing changes.
2. Create/adjust a Python environment and dependency file.
3. Add configuration files for dataset paths, model settings and experiments.
4. Use relative/configurable paths and environment variables; avoid hardcoded Windows/Colab paths.
5. Add logging, random seed handling and run metadata.
6. Add `.gitignore` for raw data, environments, caches, outputs and large checkpoints.
7. Add CPU-safe smoke tests.
8. Verify scripts can be imported/run in Colab.

**Deliverables:** reproducible project setup, configs, README instructions, `.gitignore`, smoke tests.

**Acceptance gate:** core scripts run without depending on a specific machine path; no secrets or large raw data are committed.

### Phase 3 — Preprocessing and data pipeline (Completed & Verified)
**Primary environment:** implement/test in VS Code; optional Colab scale checks

Tasks:
1. Implement dataset readers and annotation conversion if required. (Done)
2. Inspect defect bounding-box size distribution and context. (Done: 7,689 bboxes analyzed)
3. Choose a justified crop/patch size based on Phase 0/3 analysis. (Done: 256x256 adaptive patches + 640x640 letterbox)
4. Implement resizing/cropping while transforming labels correctly. (Done: crop_geometry, crop_extractor, full_image_pipeline)
5. Keep train-only transformations separate from validation/test processing. (Done: frozen Phase 1 split routing, zero leakage)
6. Add tests for coordinate transformations and boundary conditions. (Done: 11 tests in test_preprocessing.py, 46 total pass)
7. Visualize sample images with boxes before and after transformations. (Done: visualize_crops.py, 4-panel figures)

**Deliverables:** reusable data loader/preprocessing modules (`src/data/preprocessing/`), visual sanity-check samples (`outputs/qc_phase3/`), tests (`tests/test_preprocessing.py`), progress documentation (`PHASE3_HANDOFF.md`, `PHASE3_QC_AUDIT.md`).

**Acceptance gate:** boxes remain correct after transformations; validation/test data are not augmented or used to train the GAN. (Met: tested and verified for Ceramic-Defect reference pipeline)

---

### Dataset Migration Milestone — Controlled Migration to SolDef_AI (Aerospace PCB Solder Defects)
**Context:** Per academic guide recommendation, transitioning primary benchmark from Ceramic-Defect (proxy) to SolDef_AI (authentic aerospace PCB solder defect dataset, ECSS-Q-ST-70-38C standard). Ceramic-Defect implementation is preserved as reference.

**Stages:**
- **Stage A — Dataset Audit & Verification:** Completed. 428 labeled images ($2560 \times 1440$ RGB), 671 polygons, 5 classes (`good`, `exc_solder`, `spike`, `no_good`, `poor_solder`), multi-view structure identified ($V_1$ component-level vs $V_2/V_{2.1}$ fillet-level), 1,208 raw images documented.
- **Stage B — Migration Architecture & Plan Revision:** Completed. Dataset adapter pattern formulated, strict dataset isolation (`outputs/.../soldef/` vs `outputs/.../ceramic/`), unlabeled data exclusion policy established.
- **Stage C — Dataset Adapter & Verified Group-Aware Split:** Completed (Audited & Verified). Implemented `BaseDatasetAdapter`, `CeramicAdapter`, and `SolDefAdapter` in `src/data/adapters/`. Generated frozen 70/15/15 splits in `outputs/splits/soldef/phase1/` (200 images, 66 groups [30/19/17], 0 leakage) and `outputs/splits/soldef/phase1_v1/` (228 images). All 54 tests pass.
- **Stage D — SolDef Preprocessing & Visual QC:** Completed. Implemented dual preprocessing pipelines in `src/data/preprocessing/`. Extracted 256 patches ($256 \times 256$) strictly from Train split; preprocessed 200 letterbox detector images ($640 \times 640$); generated 12 visual QC panels and exhaustive metrics report. All 61 tests pass.
- **Stage E — Preprocessing Acceptance Audit:** Completed (Audited & Accepted). Read-only empirical audit completed; verified 256 patches ($256 \times 256$), 200 letterbox detector images ($640 \times 640$), 369 target boxes, and 0 leakage across splits.
- **Stage F — Real-Only Detector Baseline Planning:** Completed. Formulated comprehensive experimental plan in `PHASE5_STAGE_F_BASELINE_PLAN.md`, specifying YOLOv8s baseline architecture, hyperparameter schedule, evaluation hierarchy, test-set quarantine, and Colab execution blueprint.
- **Stage G1 — Real-Only Baseline Infrastructure Implementation:** Completed (No Training). Implemented preflight dataset validation engine (`src/data/validation/`), configuration architecture and YAML generation (`src/models/baseline_config.py`), reproducibility utilities (`src/utils/reproducibility.py`), metric reporting and threshold selection (`src/models/metrics_collector.py`), training CLI (`scripts/train_detector_baseline.py`), quarantined test evaluation CLI (`scripts/evaluate_detector_test.py`), and Colab orchestration notebook (`notebooks/colab/DataSet_Augmentation_GAN.ipynb`). All 74 unit tests passing; dry run verified; zero model training executed.
- **Stage G2 — Colab Baseline Dry Run & GPU Environment Validation:** Completed (Audited & Fully Accepted in Colab). Audited all 23 Colab notebook cells, added Google Drive write/read verification, verified preflight dataset validation (140/30/30 images, 369 boxes, 0 leakage), verified training script `--dry-run` safeguard in live Google Colab environment (zero model training, zero weight downloads), and validated persistent Drive logging.
- **Stage G3-A — Real-Only Baseline YOLOv8s Training Execution:** Completed (Accepted). Executed real-only YOLOv8s baseline training on Colab Tesla T4 GPU (57 epochs completed, early stopping at patience=25, best checkpoint at Epoch 32). Reached validation mAP@0.50 of 82.4% (peak 84.7%) and mAP@0.50:0.95 of 58.8% (`exc_solder`: 92.6%, `poor_solder`: 94.2%, `spike`: 60.4%). All checkpoints (`best.pt`, `last.pt`), curves, and logs persisted to Google Drive. Test set 100% quarantined.
- **Stage G3-B — Quarantined Test Set Baseline Evaluation:** Completed (Accepted). Executed official single-pass evaluation of frozen G3-A `best.pt` on the quarantined test split (30 images, 62 defect boxes). Established official E0 baseline numbers: Overall mAP@0.50 of 57.23%, mAP@0.50:0.95 of 39.12%, Precision of 65.43%, Recall of 62.69%, F1-Score of 64.03%. Per-class breakdown: `exc_solder` (mAP@0.50: 83.7%), `poor_solder` (mAP@0.50: 62.6%), `spike` (mAP@0.50: 25.3%). Revealed severe spike defect generalization gap (25.3% test mAP vs 60.4% val mAP), providing empirical justification for defect-focused generative augmentation (GAN). All test artifacts persisted to Google Drive.
- **Phase 5 (Stage H) — Conventional Augmentation Baseline (E1):** (Next Milestone — Planning & Implementation). Design, implement, and evaluate conventional geometric and color augmentations to establish the non-generative augmentation benchmark before introducing GANs.

---

### Phase 4 (Legacy Plan) / Stage G (SolDef) — Real-only detector baseline
**Code development:** VS Code  
**Training:** Google Colab GPU

Tasks:
1. Select and document a suitable detector, subject to project/guide requirements.
2. Prepare detector configuration using real training data only.
3. Run a small smoke test, then full baseline training in Colab.
4. Save runtime/hardware details, config, logs and checkpoint.
5. Evaluate on validation data for selection and on the held-out real-only test set under the evaluation protocol.
6. Record overall and per-class precision, recall, F1 and mAP.

**Deliverables:** baseline checkpoint, config, logs, evaluation metrics and sample predictions.

**Acceptance gate:** reproducible real-only baseline with saved evidence. Do not assume a baseline result.

### Phase 5 — Conventional augmentation baseline
**Code development/testing:** VS Code  
**Training/evaluation:** Google Colab GPU

Tasks:
1. Select conservative, domain-valid conventional transformations.
2. Avoid transformations that change defect semantics or invalidate boxes.
3. Implement and test transformations.
4. Train detector with real data plus conventional augmentation.
5. Evaluate using the same held-out real-only test set and comparable training protocol.

**Deliverables:** augmentation config, training logs, metrics and comparison with Phase 4.

**Acceptance gate:** conventional augmentation arm is documented and comparable to the real-only baseline.

### Phase 6 — GAN training data preparation
**Primary environment:** VS Code / local CPU

Tasks:
1. Use only the training split for GAN training data.
2. Create defect-centred crops/patches based on Phase 3 analysis.
3. Retain enough context for the defect to remain meaningful.
4. Store class labels and provenance metadata for each crop.
5. Check class counts and possible leakage/memorization risks.
6. Visualize crop samples and confirm preprocessing.
7. Decide how synthetic outputs can be assigned valid detector labels.

**Deliverables:** training-only crop dataset, metadata/manifests, visual review samples, documented label strategy.

**Acceptance gate:** crop source split and class are traceable; no validation/test images are used. A generated image without valid detector annotation cannot be treated as a labelled detection sample.

### Phase 7 — GAN implementation and training
**Code development/debugging:** VS Code  
**Full training:** Google Colab GPU

Tasks:
1. Implement a DCGAN baseline first.
2. Add conditional generation only if class data volume and stability justify it.
3. Make model, loss, optimizer, image size, batch size and duration configurable.
4. Add checkpointing, sample generation and loss logging.
5. Run a short smoke test before a longer Colab run.
6. Save checkpoints and generated preview grids regularly to persistent storage.
7. Monitor mode collapse, memorization, unstable training and class-conditioning failures.
8. Record GPU/runtime details and hyperparameters.

**Deliverables:** generator/discriminator code, configs, training logs, checkpoints, sample grids and run metadata.

**Acceptance gate:** training run is reproducible enough to inspect; outputs are not accepted merely because training completed.

### Phase 8 — Synthetic generation and validation
**Generation:** Google Colab GPU  
**Validation/analysis:** VS Code and/or Colab

Tasks:
1. Generate samples by class using documented settings.
2. Preserve provenance: model checkpoint, seed, class condition and generation run.
3. Check image integrity, class consistency, duplicates and near duplicates.
4. Review samples for artifacts, memorization and unrealistic defects.
5. Calculate appropriate distribution/diversity metrics (for example FID/KID/LPIPS where justified).
6. Use SSIM/PSNR only for meaningful paired comparisons.
7. Define acceptance/rejection rules before selecting samples.
8. If detection training is intended, ensure each accepted sample has a valid label/box.

**Deliverables:** synthetic sample set, provenance manifest, quality/diversity report, accepted/rejected counts and reasons.

**Acceptance gate:** synthetic samples are validated and correctly labelled for the intended task. Visual plausibility alone is insufficient.

### Phase 9 — Controlled comparative experiments
**Experiment configuration:** VS Code  
**Training/evaluation:** Google Colab GPU

Run these arms under a controlled protocol:
- A: real-only;
- B: real + conventional augmentation;
- C: real + validated GAN samples;
- D: real + conventional + validated GAN samples.

Tasks:
1. Keep the real-only test set fixed.
2. Use documented training configurations and comparable detector settings.
3. Track generated sample counts and class balance.
4. Save all logs, checkpoints and metrics.
5. Repeat runs or quantify variability if resources permit.
6. Compare overall and per-class performance; inspect failure cases.

**Deliverables:** experiment matrix, run manifests, logs, metrics table and model predictions.

**Acceptance gate:** every result is traceable to a config, split, checkpoint and run log. Report neutral or negative results honestly.

### Phase 10 — Results analysis
**Primary environment:** VS Code

Tasks:
1. Consolidate experiment metrics.
2. Compare real-only, conventional, GAN and combined approaches.
3. Analyze per-class gains/regressions and minority-class behavior.
4. Plot training curves and performance comparisons.
5. Review false positives/false negatives and representative predictions.
6. Discuss generator quality alongside downstream detector utility.
7. Document limitations, threats to validity and future work.

**Deliverables:** clean tables, figures, error analysis and evidence-backed findings.

### Phase 11 — Final report and presentation
**Primary environment:** VS Code

Tasks:
1. Document problem statement, motivation and research gap.
2. Present literature review with verified references.
3. Describe dataset source, license, class counts and limitations accurately.
4. Explain split strategy, preprocessing, GAN architecture and detector.
5. Document experimental arms and evaluation metrics.
6. Report results with tables/figures and reproducibility details.
7. Discuss limitations and future work.
8. Prepare guide-review material and final presentation.

**Deliverables:** final report, presentation, code/reproducibility instructions and experiment evidence.

## 3. VS Code ↔ Colab working procedure

1. Keep one canonical Git repository.
2. Develop code and configuration in VS Code.
3. Run formatting, unit tests and CPU-safe smoke tests locally.
4. Commit and push changes to Git.
5. In Colab, clone or pull the same repository.
6. Connect the dataset and persistent artifact storage without putting large files into Git.
7. Check actual runtime GPU, CUDA/framework versions and available memory.
8. Run repository scripts from the notebook or command line; keep notebook code to setup/orchestration.
9. Save checkpoints, logs and sample outputs frequently to persistent storage.
10. Record the Git commit hash and config for each experiment.
11. Sync metrics and selected outputs back to the project workspace.
12. Analyze and document results in VS Code.

### Colab disconnection precautions
- Save checkpoints periodically, not only at the end.
- Save optimizer state when resuming training matters.
- Save logs and generated previews to persistent storage.
- Make training scripts resume from a checkpoint where practical.
- Do not assume runtime filesystem survives a disconnect.
- Keep data and model artifacts out of Git unless small and intentionally versioned.

## 4. First task for the coding agent

Start with **Phase 0 only**.

The coding agent should:
1. inspect the current repository and identify its structure;
2. locate or ask for the dataset path if it is not present;
3. inspect the archive and annotation files without modifying originals;
4. implement a reproducible audit script;
5. produce counts, dimensions, class mapping, annotation validation and data-integrity reports;
6. identify unknowns and discrepancies;
7. run tests/sanity checks and report exact commands and outcomes.

Do not begin GAN or detector training until Phase 0 and Phase 1 split requirements have been reviewed and accepted.

## 5. General completion rules

A phase is not complete merely because code was written. Each phase must have code/configuration or documented manual steps, test or execution evidence, output files/logs where applicable, acceptance criteria checked, and known limitations recorded.

The agent must not fabricate data counts, model results, citations, licenses, approvals or completed work. It must ask before destructive changes, large training runs, or changes to the research scope.
