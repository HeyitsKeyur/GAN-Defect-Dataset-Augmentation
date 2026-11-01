# Defect Dataset Augmentation using Generative Adversarial Networks (GAN)

M.Tech major project, Nirma University. See `PROJECT_CONTEXT.md` and `IMPLEMENTATION_PLAN.md` for research context and phases.

## Phase 0 — dataset audit

The Ceramic-Defect archive in this workspace lives at:

`Dateset/Ceramic-defect/data/data`

(The folder names `Dateset` and `lables` are the on-disk spellings.)

Original dataset files are not modified. Audit outputs go to `outputs/audit/phase0/`.

### Environment (local CPU)

Phase 0 needs only Python 3, Pillow (image inspection and overlay visualizations) and pytest (unit tests). No GPU packages are required.

```powershell
Set-Location -LiteralPath "<project-root>"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-phase0.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\audit_ceramic_defect.py
```

If the dataset is not at the default path:

```powershell
.\.venv\Scripts\python.exe scripts\audit_ceramic_defect.py --dataset-root "<path-to-data\data>"
```
