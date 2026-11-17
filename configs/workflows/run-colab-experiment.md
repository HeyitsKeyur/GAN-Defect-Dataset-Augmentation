# Antigravity Reusable Workflow: run-colab-experiment

## Purpose
Orchestrates experiment verification, Git branch synchronization, Colab runtime inspection, Drive dataset validation, and structured logging between the local Antigravity environment and Google Colab.

---

## Workflow Steps and Human-in-the-Loop Gates

```
+-------------------------------------------------------------+
| Step 1: Pre-flight Local Verification (Automated)          |
| - Inspect working tree (must be clean)                     |
| - Run test suite (pytest)                                  |
| - Verify Phase 1 frozen split manifests                     |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| Step 2: Push Approved Code to GitHub (Automated)            |
| - Commit staged modifications                              |
| - Push to designated development branch (never 'main')      |
| - Record commit SHA                                         |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| Step 3: Colab Session Setup (Browser / Manual Assistance)  |
| - Open notebook URL in browser:                             |
|   https://colab.research.google.com/drive/1o3o9C6H...       |
| - Verify runtime accelerator is GPU                         |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| Step 4: Non-Training Smoke Test in Colab (Automated/Semi)   |
| - Mount Google Drive                                        |
| - Clone/pull latest commit SHA from GitHub                  |
| - Run scripts/run_colab_smoke_test.py                       |
| - Verify GPU, Drive access, and 0 split leakage             |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| [GATE] Step 5: User Explicit Approval Required             |
| *** STOP & ASK USER BEFORE EXPENSIVE GPU TRAINING ***       |
| Present: Commit SHA, GPU model, VRAM, and planned training  |
| Await explicit user authorization                           |
+------------------------------+------------------------------+
                               | (Upon User Approval)
                               v
+-------------------------------------------------------------+
| Step 6: Execute Experiment & Persist Artifacts (Colab)      |
| - Run training script (detector / GAN)                      |
| - Stream epoch progress to logs                             |
| - Save checkpoints periodically to Drive                    |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| Step 7: Handoff & Experiment Logging (Automated)            |
| - Retrieve evaluation metrics from Drive                    |
| - Update PHASE*_HANDOFF.md with results and commit SHA      |
+-------------------------------------------------------------+
```

---

## Detailed Step Specification

### Step 1: Pre-flight Local Verification
Execute in the local development environment:
```powershell
git status
pytest
python scripts/run_colab_smoke_test.py
```
*Gate:* All tests must pass, and frozen split manifests must report 0 group leakage.

### Step 2: Push Approved Code to GitHub
```powershell
git add <approved_files>
git commit -m "Phase X: <descriptive message>"
git push -u origin <current_branch>
git rev-parse HEAD
```

### Step 3: Colab Notebook Synchronization
- Open `DataSet_Augmentation_GAN.ipynb` in Colab.
- Verify that `REPO_BRANCH` in Section 1 matches your current Git branch (e.g., `phase-2-colab-integration`).
- Ensure GPU runtime is connected (`Runtime -> Change runtime type -> T4 or better GPU`).

### Step 4: Run Smoke Test in Colab
Execute Section 4 (Mount Drive), Section 5 (Git pull), Section 6 (Verify Splits), and Section 7 (Smoke test).
*Verification:* Inspect generated `smoke_test_report.md` on Google Drive (`/content/drive/MyDrive/MajorProject_Outputs/smoke_test/`).

### Step 5: Explicit User Approval Gate
Before any GPU detector or GAN training cell is launched, the following information must be presented to the user:
- Current Commit SHA
- GPU Accelerator Model and VRAM Total
- Target Experiment Identifier (e.g., Baseline Real-only Detector)
- Estimated Duration and Checkpoint Save Interval

> **Rule:** Antigravity must never automatically start training runs without user sign-off.

### Step 6: Post-Experiment Artifact Synchronization
After training completes:
- Checkpoints reside in `/content/drive/MyDrive/MajorProject_Outputs/checkpoints/`
- Logs reside in `/content/drive/MyDrive/MajorProject_Outputs/logs/`
- Sync evaluation tables and metrics into repository documentation.
