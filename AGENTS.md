# AGENTS.md — Instructions for AI Coding Agents

## Project identity
This repository supports the M.Tech Major Project **“Defect Dataset Augmentation using Generative Adversarial Networks (GAN)”**. The project background, prior exploration, research direction, constraints and open questions are documented in `PROJECT_CONTEXT.md`. The phased execution plan and VS Code/Colab allocation are documented in `IMPLEMENTATION_PLAN.md`.

Read both documents before making substantive changes. Treat them as project context, not proof that every proposed task has already been implemented.

## 1. Core operating principles
1. Inspect the existing repository before changing it. Preserve useful code, data organization and decisions.
2. Work on one clearly defined milestone at a time.
3. Do not fabricate dataset counts, annotations, paper citations, licenses, model results, training outcomes or guide approvals.
4. Distinguish explored, implemented, executed, validated and completed work.
5. Explain assumptions and mark unknowns explicitly.
6. Do not change research scope, primary dataset or experimental design without explaining implications and asking for approval.
7. Never modify or delete original raw data. Write derived data to separate, documented locations.
8. Do not launch long GPU training jobs or destructive operations without explicit approval.
9. Provide exact commands run, test outcomes, generated files and remaining issues at the end of each milestone.

## 2. Development environment policy: VS Code and Colab
- **VS Code/local machine is the primary development environment.** Write and maintain canonical source code, configuration, tests, data-audit tools, preprocessing, model modules, evaluation tools, documentation and Git history here.
- **Google Colab is primarily the GPU execution environment.** Use it for computationally intensive detector training, GAN training, bulk synthetic generation and GPU-heavy experiments.
- Maintain **one canonical Git repository**. Do not create a separate, independently maintained Colab implementation.
- Colab notebooks should be thin orchestration notebooks: runtime setup, repository clone/pull, storage mounting, configuration selection and calls to reusable repository scripts/modules. Do not duplicate core model/data logic in notebook cells.
- Make reusable code portable between local and Colab environments. Use configuration files, command-line arguments and environment variables rather than hardcoded machine-specific paths.
- Inspect actual Colab GPU and available memory at runtime; never assume a particular GPU is available.
- Save checkpoints, logs and generated artifacts to persistent storage regularly because hosted runtimes can disconnect.
- Keep raw datasets, virtual environments, credentials, caches and large checkpoints out of Git by default. Maintain an appropriate `.gitignore`.

## 3. First milestone: Phase 0 dataset/repository audit only
Unless the user explicitly directs otherwise, the first implementation task is Phase 0 from `IMPLEMENTATION_PLAN.md`.

Required behavior:
1. Inspect repository and report its current structure before proposing modifications.
2. Locate dataset archive/extracted folder. If absent or ambiguous, ask user for its location; do not guess.
3. Inspect dataset structure and annotations without altering source files.
4. Calculate image counts and bounding-box instance counts per class separately.
5. Report class names and class-ID mapping from actual annotations.
6. Inspect image dimensions, aspect ratios and bounding-box sizes.
7. Detect/report missing files, unreadable images, malformed annotations, invalid class IDs and invalid/out-of-bounds boxes.
8. Check duplicate/near-duplicate images and possible specimen/group relationships where feasible.
9. Produce a reproducible script and machine-readable plus human-readable audit outputs.
10. State exact commands, environment and limitations.

Do not begin GAN training or detector training during this first milestone.

## 4. Dataset and research integrity
- Treat the ceramic-substrate dataset as a related proxy dataset for MIC/MMIC/LTCC applications unless direct domain relevance is verified and approved.
- Do not describe MVTec AD as a MIC/MMIC/LTCC-specific dataset. It may be used as a general industrial benchmark/prototype if justified.
- Verify exact dataset version, class mapping, annotation format and license from source files/current authoritative material before reporting.
- Preserve discrepancies in source-reported counts; do not silently choose or reconcile numbers.
- Compute actual counts from the downloaded archive and report the counting method.
- Do not confuse image counts with object-instance/bounding-box counts.
- Keep citations traceable to real papers and sources. Never invent titles, authors, DOIs or findings.

## 5. Leakage prevention and split discipline
- Create train/validation/test manifests reproducibly and record the seed.
- Group related specimens, paired views, batches and duplicates into the same split where identifiers or evidence permit.
- Train GANs only on the training split.
- Do not use held-out test images for GAN training, synthetic sample selection, hyperparameter tuning or augmentation decisions.
- Keep the final test set real-only.
- Save split manifests and do not silently regenerate or change them between experiment arms.
- If grouping is impossible, state the limitation and use the most defensible split method available.

## 6. Synthetic data and detection labels
- Do not assume generated images improve detector performance.
- Evaluate image quality/diversity and downstream detector utility separately.
- Use FID/KID/LPIPS only with appropriate interpretation and documented settings. Do not apply SSIM/PSNR indiscriminately to unpaired images.
- A generated image-only crop is not a valid object-detection training example unless its class and bounding-box annotation are known and verified.
- Any crop, resize, paste or compositing operation must transform boxes correctly and be covered by tests.
- Check for memorization, duplicates, artifacts, class mismatch and implausible defect morphology.
- Record synthetic sample provenance: generator checkpoint, class condition, seed/run and validation status.

## 7. Code quality and reproducibility
- Prefer modular, readable Python code with clear function boundaries and type hints where useful.
- Keep hyperparameters and paths in configuration rather than scattering constants.
- Add unit tests for annotation parsing, box transformations, split logic and other critical data operations.
- Include CPU-safe smoke tests so basic correctness can be checked locally without a GPU.
- Handle missing files and invalid annotations with informative errors/reports.
- Set and record random seeds where practical.
- Log package versions, hardware/runtime details, config, Git commit hash and run identifiers for experiments.
- Make scripts resumable where practical, especially long-running Colab training.
- Avoid unnecessary dependencies and explain why new packages are needed.

## 8. Git and file handling
- Do not commit raw datasets, `.env` files, credentials, virtual environments, cache folders or large generated artifacts by default.
- Do not overwrite user files or make broad refactors unrelated to the current milestone.
- Before changing existing files, summarize the planned change if it is substantial.
- Use meaningful file names and update README/documentation when commands or workflows change.
- Do not claim an output file exists unless it was actually created.

## 9. Reporting after each task
At the end of each milestone, report:
1. What was inspected or changed.
2. Files created/modified.
3. Commands executed.
4. Tests and their actual results.
5. Data counts/results with source and method.
6. What remains incomplete or uncertain.
7. The next recommended task.

Do not mark a phase complete until its acceptance criteria in `IMPLEMENTATION_PLAN.md` are met.

## Mandatory Milestone Handover and Progress Tracking

Maintain a continuously updated (Ex.`PHASE0_HANDOFF.md`) throughout Every Phase.

This is a mandatory requirement, not an optional task.

### When to update it

Update (Ex. `PHASE0_HANDOFF.md`) whenever:
- A meaningful implementation milestone is completed.
- A dataset inspection or audit stage is completed.
- A new parser or validation module is implemented and tested.
- A test suite is executed and results are available.
- An audit report or visualization is generated.
- A significant issue or blocker is discovered.
- The implementation approach or an important decision changes.
- Work is paused, interrupted, or transferred to another model.

Do not wait until the usage limit is reached.

### Required contents

Keep the file concise but sufficiently detailed. Include:

1. Project title and current phase.
2. Current overall progress.
3. Latest completed milestone.
4. Completed tasks and verification evidence.
5. Tasks implemented but not yet executed.
6. Files created and modified.
7. Commands executed and actual results.
8. Tests executed and their outcomes.
9. Dataset findings discovered so far.
10. Known issues, blockers and unresolved questions.
11. Important implementation decisions.
12. Exact next task to continue.

### Accuracy requirements

- Update the file immediately after each meaningful milestone.
- Never mark a task complete without evidence.
- Preserve previous useful progress information.
- Do not invent results or completion percentages.
- Clearly distinguish completed, partially completed and pending work.
- Keep the handover consistent with the actual repository state.

### Model handover

Whenever another AI model takes over, it must read:
- `AGENTS.md`
- `PROJECT_CONTEXT.md`
- `IMPLEMENTATION_PLAN.md`
- Ex.`PHASE0_HANDOFF.md`, if it exists.

The new model must inspect the actual workspace and verify the handover against the files and execution evidence before continuing.

Do not restart completed work unnecessarily.

At the end of every meaningful milestone, update (Ex.`PHASE0_HANDOFF.md`) before reporting completion to the user.
