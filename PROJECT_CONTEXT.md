# Major Project — Complete Project Context

**Project title:** Defect Dataset Augmentation using Generative Adversarial Networks (GAN)  
**Academic context:** M.Tech Computer Science & Engineering, 2025–2027 batch, Nirma University  
**Project guide:** Prof. Sonia Mittal

> This is the central knowledge base for the project. It records the project context, prior exploration, decisions discussed, proposed research direction, constraints, and unresolved questions. It distinguishes prior exploration from implementation that is still planned or must be verified. Do not treat a proposed task as completed unless repository files, data, logs, or user confirmation establish that it is complete.

## 1. Project overview

The project investigates whether Generative Adversarial Networks can help address limited availability and class imbalance in industrial defect datasets, particularly for defects associated with ceramic/electronic packaging and components relevant to MIC/MMIC/LTCC applications.

The proposed approach is to train a generative model using real defect examples, produce additional synthetic defect images, assess their quality and diversity, and evaluate whether adding suitable synthetic samples improves a downstream defect detector on a held-out set of real images.

**Proposed pipeline:**

Real defect dataset → dataset audit and annotation validation → leakage-safe train/validation/test split → preprocessing and defect-centred crop preparation → GAN training → synthetic image generation → synthetic image and label validation → quality/diversity evaluation → real + synthetic training set → object detector training → evaluation on an unchanged real-only test set → comparative analysis and reporting.

**Central research question:** Can carefully generated and validated synthetic defect samples improve downstream industrial defect detection, particularly for underrepresented defect classes, compared with training on real data alone and with conventional augmentation?

This question must be answered experimentally. Do not assume GAN augmentation improves detection.

## 2. Domain and scope

The intended application domain is inspection of mission-critical electronic/ceramic components and packaging, with interest in MIC (microwave integrated circuit), MMIC (monolithic microwave integrated circuit), and LTCC (low-temperature co-fired ceramic) contexts.

A scope limitation has already been identified: a publicly available ceramic-substrate defect dataset may be relevant to electronic/ceramic manufacturing, but it is not automatically a dataset of actual MIC/MMIC/LTCC assembly defects. Unless the data source proves otherwise and the guide approves the framing, describe it as a **related ceramic-substrate proxy dataset**, not as a direct MIC/MMIC/LTCC dataset.

Make clear what component/material the dataset depicts, what defect categories it represents, how they relate to the target domain, which target-domain defects are not represented, and what conclusions can and cannot be generalized. Do not overstate domain relevance.

## 3. Work and exploration to date

The following items have been discussed or explored in project history. They are not all equivalent to completed implementation; verify actual files and evidence before marking them complete.

### 3.1 Literature and GAN fundamentals
- Studied GAN fundamentals, including generator/discriminator training.
- Studied DCGAN and Conditional GAN as candidate approaches.
- Collected approximately 40–45 research papers for the literature review.
- Literature review themes: survey papers; industrial defect detection; GANs for data augmentation; GANs for industrial defect generation; recent 2025–2026 state-of-the-art work.
- Scopus search planning covered approximately 2021–2026.
- The guide requested a clear explanation of research gaps, proposed model, key considerations, evaluation measures, and a benchmark paper.
- Verify the precise bibliography, publication details, claims and benchmark choice against the user's actual literature-review records. Do not fabricate citations or paper findings.

### 3.2 Initial dataset exploration: MVTec AD
- Downloaded or explored MVTec AD, including `hazelnut` and `screw` categories.
- MVTec AD is useful as a general industrial anomaly-detection benchmark/prototype, but it is not specific to ceramic substrates or MIC/MMIC/LTCC defects.
- Its common structure includes normal training images and anomalous test images with masks; do not assume it contains YOLO bounding-box annotations.
- Treat MVTec as optional/general experimentation unless project scope is explicitly expanded to include it.

### 3.3 Candidate ceramic defect dataset
A candidate dataset was identified:
- **Name/description used during exploration:** Ceramic-defect / ceramic substrate defect dataset.
- **Mendeley URL:** https://data.mendeley.com/datasets/55yb7hh55r/1
- **DOI previously noted:** `10.17632/55yb7hh55r.1`
- **Reported size:** approximately 2,892 images.
- **Reported categories:** six types, described in prior exploration as adhesion, scar, spot, uneven, abnormal thickness, and edge defect.
- **Reported annotations:** COCO JSON, with YOLO and VOC conversions also reported.
- **Reported dimensions:** approximately 1024×1024 up to 4096×4000.
- **Reported instances:** dataset page previously noted as reporting 7,360 boxed instances, while an associated article was noted as reporting 7,689. The article was also noted as reporting 3,165 spot instances and 156 abnormal-thickness instances.

These are previously observed/reported figures, not independently re-verified facts. Recheck the current dataset page, archive, paper and annotation files before using them in a report. Do not silently reconcile the 7,360 versus 7,689 discrepancy. The reported instance imbalance (3,165 spot versus 156 abnormal-thickness boxes) is not image-count imbalance; calculate both from actual files.

**License caution:** prior exploration noted a possible discrepancy between CC BY 4.0 on the dataset page and CC BY-NC in the associated article. Verify the license attached to the exact downloadable version and its terms before redistribution or publication. Do not assume the more permissive license applies.

### 3.4 Other candidate datasets mentioned
- CPS2D-AD was identified as another ceramic-package-substrate-related dataset, with approximately 20,781 samples reported in prior exploration. Verify source, labels, license and relevance before use.
- A second ceramic-substrate dataset was mentioned as published in May 2026; its identity and suitability require verification.
- A solder-joint dataset was mentioned as a possible supplementary source. Do not combine it with the primary dataset without a documented rationale, compatible labels and guide approval.

### 3.5 Model and evaluation concepts discussed
Candidate generative models:
- DCGAN as a baseline.
- Conditional GAN if class data volume and stability justify it.

Potential generative-image evaluation:
- FID and KID for distribution-level similarity, with careful interpretation on small datasets.
- LPIPS for perceptual diversity/similarity where the comparison design supports it.
- SSIM and PSNR only where a meaningful paired/reference relationship exists; do not use them as universal measures for unpaired generated samples.

Downstream detector evaluation:
- precision, recall, F1;
- mAP@0.5 and mAP@0.5:0.95;
- per-class metrics and confusion analysis.

Downstream detector performance is the key test of augmentation usefulness. Visually plausible images alone do not prove that augmentation improves detection.

## 4. Proposed experimental design

Use the same real-only held-out test set for every experiment. Keep test images out of GAN training, synthetic-data selection/tuning, and augmentation decisions.

Recommended comparison arms:

| Experiment | Training data |
|---|---|
| A — Real-only baseline | Real training images only |
| B — Conventional augmentation | Real training images + conventional augmentations |
| C — GAN augmentation | Real training images + validated GAN-generated samples |
| D — Combined augmentation | Real training images + conventional + validated GAN-generated samples |

Keep detector architecture, training schedule, evaluation split and relevant hyperparameters as controlled as possible. Record deviations.

Report overall and per-class precision, recall, F1 and mAP; class-wise sample counts before/after augmentation; generated images accepted/rejected and reasons; generator configuration, duration and seeds where practical; detector configuration and checkpoint selection rule; variability if feasible; limitations and failure cases.

Do not claim a positive result unless measured results support it. Neutral or negative results are valid findings if methodology is sound.

## 5. Data engineering and validation requirements

Before model training, perform a reproducible audit of the exact downloaded dataset.

### 5.1 Dataset audit
Inspect and report:
- archive and directory structure;
- image count and supported formats;
- annotation formats and paths;
- class names and class-ID mapping;
- image counts per class, where image-level labels exist;
- bounding-box instance counts per class;
- dimensions and aspect ratios;
- missing files, unreadable/corrupt images and malformed annotations;
- boxes outside image bounds, zero/negative-area boxes and invalid class IDs;
- duplicate and near-duplicate images;
- paired views, repeated substrates, acquisition batches or other group relationships;
- small-object prevalence and bounding-box size distribution.

Do not infer class image counts from box counts. Preserve original files and create derived outputs separately.

### 5.2 Splitting and leakage prevention
- Split by substrate/specimen/group where identifiers or relationships are available, rather than randomly splitting related individual images.
- Keep near duplicates, alternate views and related samples in the same split where identifiable.
- Establish train/validation/test splits before GAN training.
- Train the GAN only on the training partition.
- Use validation data only for model/parameter selection under a documented protocol.
- Keep the test set real-only and untouched until final evaluation.
- Save split manifests and random seeds for reproducibility.

### 5.3 Crops and image resolution
Full-resolution images may be too large for practical GAN training, especially on constrained hardware. First inspect defect bounding-box dimensions and surrounding context. Then choose and justify a crop/patch size (for example, 256 or 512 pixels) based on actual data.

For object-detection training, a generated image must have valid corresponding annotations. A generated image-only patch with no trustworthy bounding box/class label must not be inserted into a YOLO training set as a fully labelled sample.

If using defect-centred crops, document crop sampling, retained context, label transformations, and how synthetic patches are placed or used. Validate boxes and realism after compositing.

## 6. Hardware and development environment

Two devices were previously described:

1. **Device 1:** Intel Core i5-10210U, 8 GB RAM (about 7.83 GB usable), NVIDIA GeForce MX250 with 2 GB VRAM, storage reported as 553 GB used / 1.14 TB total.
2. **Device 2:** Intel Core i5-1135G7, 8 GB RAM (about 7.71 GB usable), integrated graphics.

These specifications are constrained for high-resolution GAN and detector training. Use the local machine primarily for development, data inspection and CPU-safe validation. Use Google Colab GPU runtime (or another approved cloud GPU) for computationally intensive training. Do not assume a particular Colab GPU is always available; inspect runtime hardware at execution time.

### VS Code role
- Maintain the single source-code repository.
- Write/edit Python modules, configs and scripts.
- Implement dataset auditing, annotation conversion/validation and split generation.
- Implement preprocessing and data loaders.
- Implement GAN and detector training scripts portably.
- Write tests and run CPU-safe checks.
- Inspect logs, analyze results and maintain Git/documentation.

### Google Colab role
- Run GPU-intensive detector baseline training.
- Run GAN training.
- Generate larger batches of synthetic samples.
- Run GPU-heavy experiments and evaluation.
- Save checkpoints, logs, generated samples and metrics regularly.

Colab notebooks should be thin orchestration layers that clone/pull the same repository, configure paths/runtime and call scripts/modules from the repository. Avoid duplicating core implementation inside notebooks.

### One repository and artifact handling
Maintain one canonical Git repository. Do not maintain separate independent VS Code and Colab codebases.

Workflow:
1. Develop and test in VS Code.
2. Commit and push code/configuration to Git.
3. Clone or pull the repository in Colab.
4. Connect approved storage for dataset and experiment outputs.
5. Run the same repository scripts on GPU.
6. Save checkpoints/logs frequently because hosted runtimes can disconnect.
7. Sync relevant metrics, plots and selected samples back.
8. Analyze and document results in VS Code.

Do not commit large raw datasets, virtual environments, credentials, temporary caches or large checkpoints to Git. Use `.gitignore` and documented external storage. Never hardcode a local Windows path or Colab-only path into reusable code; use configuration/environment variables and path validation.

## 7. Suggested project structure

Adapt to the existing repository rather than overwriting it blindly. Example:

```text
gan-defect-augmentation/
├── AGENTS.md
├── PROJECT_CONTEXT.md
├── IMPLEMENTATION_PLAN.md
├── README.md
├── configs/
├── data/
│   ├── raw/                 # ignored by Git
│   ├── interim/             # ignored by Git
│   └── processed/           # ignored by Git
├── manifests/
├── notebooks/
│   └── colab/               # orchestration only
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── generation/
│   ├── evaluation/
│   └── utils/
├── tests/
├── scripts/
├── outputs/                 # ignored or selectively versioned
│   ├── checkpoints/
│   ├── generated/
│   ├── logs/
│   └── metrics/
└── reports/
```

This is a proposal, not a claim that these folders already exist. Inspect the repository first and preserve useful existing structure.

## 8. Implementation roadmap at a glance

1. Phase 0 — Repository and dataset audit: inspect repository, archive, annotations, labels, dimensions and integrity. Primarily VS Code/local CPU.
2. Phase 1 — Data validation and split manifests: validate labels, establish group-aware train/validation/test splits, save manifests. VS Code/local CPU.
3. Phase 2 — Environment and reproducibility: dependencies, configuration, logging, seeds and portable paths. VS Code; verify Colab compatibility.
4. Phase 3 — Preprocessing pipeline: reproducible loading, resizing/cropping and annotation transformations. Develop/test in VS Code; scale checks in Colab if needed.
5. Phase 4 — Real-only detector baseline: code in VS Code; GPU training in Colab.
6. Phase 5 — Conventional augmentation: code in VS Code; detector training/evaluation in Colab.
7. Phase 6 — GAN data preparation: VS Code/local CPU.
8. Phase 7 — GAN implementation/training: implement/debug in VS Code; train in Colab GPU.
9. Phase 8 — Synthetic generation/quality evaluation: generate in Colab; validate/analyze with repository tools.
10. Phase 9 — Controlled comparative experiments: Colab GPU.
11. Phase 10 — Results analysis/graphs: VS Code.
12. Phase 11 — Final report/presentation: VS Code.

See `IMPLEMENTATION_PLAN.md` for detailed tasks and gates.

## 9. Risks and decisions that must remain explicit

- **Domain mismatch:** ceramic substrate dataset is a proxy unless direct MIC/MMIC/LTCC relevance is verified.
- **Small dataset:** GANs can memorize examples; evaluate diversity and nearest-neighbour similarity.
- **Class imbalance:** calculate image counts and instance counts separately.
- **Label validity:** image-only generation is not sufficient for object detection without valid annotations.
- **Data leakage:** no test-set images in GAN training or synthetic tuning.
- **Resolution/compute:** use evidence-based crops and cloud GPU training.
- **License uncertainty:** verify the exact dataset version and terms.
- **Metric limitations:** use FID/KID/LPIPS thoughtfully; do not overinterpret SSIM/PSNR for unpaired images.
- **Reproducibility:** record seeds, configs, versions, split manifests, logs and checkpoints.
- **Research integrity:** report negative or inconclusive results honestly.

## 10. Open questions / verification checklist

- [ ] Confirm exact primary dataset version and archive contents.
- [ ] Confirm dataset license and redistribution conditions.
- [ ] Confirm authoritative class names and ID mapping.
- [ ] Reconcile (or transparently report) the 7,360 versus 7,689 reported instance counts.
- [ ] Calculate actual image counts and box counts per class from annotations.
- [ ] Determine whether images have specimen/group identifiers or paired views.
- [ ] Check duplicates and near duplicates before splitting.
- [ ] Confirm annotations are suitable for the chosen detector.
- [ ] Determine crop size from actual box/context analysis.
- [ ] Decide how synthetic images will receive valid labels.
- [ ] Confirm benchmark paper and research gap from the literature review.
- [ ] Confirm supplementary datasets are permitted and how they will be combined.
- [ ] Confirm available Colab GPU/runtime constraints at each run.

## 11. Rules for project reporting

Use these distinctions:
- **Explored/discussed:** considered or researched, but implementation may not exist.
- **Implemented:** code exists and has been inspected or tested.
- **Executed:** a run was performed and logs/results exist.
- **Validated:** output passed stated checks.
- **Completed:** acceptance criteria and evidence are satisfied.

Never invent dataset statistics, experiment results, paper citations, training outcomes, or guide approvals. If information is unknown, label it unknown and propose a verification step.
