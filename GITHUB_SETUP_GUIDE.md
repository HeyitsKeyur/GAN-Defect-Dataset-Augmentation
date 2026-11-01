# GitHub Setup and Two-Laptop Workflow — Major Project

**Project:** Defect Dataset Augmentation using Generative Adversarial Networks (GAN)  
**GitHub username:** `HeyitsKeyur`  
**Repository:** `GAN-Defect-Dataset-Augmentation`  
**Remote:** `https://github.com/HeyitsKeyur/GAN-Defect-Dataset-Augmentation.git`  
**Default branch:** `main`

## 1. Purpose

Connect the existing Major Project folder to the existing GitHub repository, safely publish source code and documentation, and continue development from another laptop.

Use one canonical Git repository. GitHub stores source code, documentation, tests and configuration. Large datasets, generated image collections, training checkpoints and local environments should normally remain outside Git.

## 2. Exact local project location

```text
E:\MTech[CSE] Nirma_University\S3\Major Project\Defect Dataset Augmentation using Generative Adversarial Networks (GAN)
```

Because this path contains spaces and square brackets, quote it in PowerShell:

```powershell
Set-Location -LiteralPath 'E:\MTech[CSE] Nirma_University\S3\Major Project\Defect Dataset Augmentation using Generative Adversarial Networks (GAN)'
```

Do not create a new project folder or copy the project elsewhere unless the owner explicitly asks.

## 3. Remote repository

Repository page: https://github.com/HeyitsKeyur/GAN-Defect-Dataset-Augmentation

Remote URL:

```text
https://github.com/HeyitsKeyur/GAN-Defect-Dataset-Augmentation.git
```

Expected default branch: `main`.

Inspect whether the remote is empty or already contains commits/files before pushing. Do not assume it is empty.

## 4. Safety rules

1. Inspect the existing project folder and Git status first.
2. Check whether `.git` already exists.
3. If Git is already initialized, preserve its history and inspect remotes and branches. Do not run `git init` blindly.
4. Inspect the GitHub remote before pushing. If it contains commits/files, fetch and reconcile safely; do not force-push or overwrite remote content.
5. Do not add raw Ceramic-Defect images, extracted dataset folders, archives, or other large research data to Git by default.
6. Do not commit `.env`, credentials, API keys, tokens, passwords, virtual environments, caches, or large model checkpoints.
7. Review the staged file list before committing.
8. Never use `git push --force`, `git reset --hard`, destructive cleanup, or history rewriting as a routine fix.
9. Do not install Git LFS or upload large files through LFS unless the owner approves a storage plan.
10. Never expose credentials in prompts, source files, logs, or commit messages.

## 5. Recommended `.gitignore` rules

Inspect the existing `.gitignore` first. Merge necessary rules instead of overwriting existing project-specific rules.

```gitignore
# Python environments
venv/
.venv/
env/
ENV/

# Python cache and test cache
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/

# Secrets and local environment configuration
.env
.env.*
!.env.example

# Jupyter temporary files
.ipynb_checkpoints/

# Dataset data (keep original and derived data outside Git by default)
data/raw/
data/interim/
data/processed/

# Common experiment output directories
outputs/checkpoints/
outputs/generated/
outputs/logs/
outputs/metrics/

# Model artifacts
*.pt
*.pth
*.ckpt
*.h5
*.keras
*.onnx

# Temporary and OS files
*.tmp
*.log
.DS_Store
Thumbs.db

# IDE settings
.idea/
```

Notes:
- Do not blindly ignore all reports or all images. Small audit reports and selected visualizations may be versioned if useful, non-sensitive and reasonably sized.
- `.gitignore` does not automatically untrack files already tracked. Check this before committing.
- Do not remove tracked files without approval.

## 6. Suitable files for GitHub

Usually appropriate:
- `AGENTS.md`
- `PROJECT_CONTEXT.md`
- `IMPLEMENTATION_PLAN.md`
- `PHASE0_HANDOFF.md`
- `GITHUB_SETUP_GUIDE.md`
- `README.md`
- Python source code, tests and configuration templates
- Small notebooks
- Small audit reports and selected figures after review

Usually keep outside Git:
- Original Ceramic-Defect dataset and archives
- Large processed datasets and crop collections
- Large synthetic image collections
- GAN and detector checkpoints
- Virtual environments
- Caches and temporary outputs
- Credentials and private configuration

Do not redistribute dataset images unless the license and redistribution conditions have been verified.

## 7. First-laptop setup

### Step 1 — Inspect

- Verify the exact project folder exists.
- List its top-level files and directories.
- Locate the dataset and determine whether it is inside or outside the project folder.
- Check for `.git`, `.gitignore`, README and existing Git metadata.
- If already a repository, inspect `git status`, branches, recent commits and remotes.
- Check for large files and likely secrets before staging.

### Step 2 — Preserve existing Git history

If `.git` already exists:
- Do not reinitialize.
- Inspect `git status`, `git log`, `git branch -vv` and `git remote -v`.
- Preserve existing commits and branches.
- If `origin` points elsewhere, report it and do not silently replace it.

If `.git` does not exist:
- Initialize Git in the existing project root only after inspection.
- Use `main` as the default branch if there is no existing branch/history to preserve.

### Step 3 — Inspect GitHub remote

- If the remote is empty, connect the local repository and push the initial commit.
- If the remote has commits/files, fetch it and inspect the relationship between local and remote histories.
- Reconcile safely. Do not force-push, overwrite, or discard either history.
- If reconciliation is ambiguous, stop and ask the owner.

### Step 4 — Review ignore rules and staged files

- Create or merge `.gitignore`.
- Ensure raw dataset and large artifacts are excluded.
- Search for secrets and credentials.
- Stage only intended files.
- Review `git status` and `git diff --cached --name-only`.
- If a dataset, archive, checkpoint, environment folder or secret is staged, stop and correct staging before committing.

### Step 5 — Commit and push

Only after the checks above pass:
- Create a clear initial commit if there are uncommitted project files.
- Connect `origin` to the exact URL above if appropriate.
- Push to `main` or the already-established intended branch.
- Set upstream tracking where needed.
- Verify the push and working tree.

Never force-push.

## 8. Authentication

Use Git Credential Manager / browser-based GitHub authentication if available on Windows.

The owner should authenticate directly in the browser or approved credential manager when prompted.

Never ask the owner to paste a password, personal access token, or SSH private key into Cursor chat. Never put tokens into a remote URL, source file, committed `.env`, or command history.

If authentication is not configured, explain the prompt and guide browser authentication or GitHub CLI setup. Resume only after authentication succeeds.

## 9. Second-laptop setup

On the second laptop:

1. Install Git for Windows: https://git-scm.com/downloads/win
2. Install VS Code: https://code.visualstudio.com/
3. Optionally install Cursor: https://cursor.com/
4. Sign in to the same GitHub account.
5. Configure Git name and email.
6. Choose a destination folder, for example `D:\Projects`.
7. Clone the repository:

```powershell
Set-Location -LiteralPath 'D:\Projects'
git clone https://github.com/HeyitsKeyur/GAN-Defect-Dataset-Augmentation.git
Set-Location -LiteralPath '.\GAN-Defect-Dataset-Augmentation'
code .
```

8. Create a new local Python virtual environment; do not copy the first laptop's `venv`.
9. Install dependencies using the project's documented dependency file.
10. Configure local dataset paths through an ignored local configuration or environment variable.
11. Make sure the same licensed dataset is available on the second laptop. `git clone` does not download ignored datasets.
12. Run documented tests/smoke checks before continuing development.

## 10. Two-laptop daily workflow

Before starting work:

```bash
git status
git pull --ff-only
```

`git pull --ff-only` is a safe default when the local branch can simply advance. If it refuses because histories diverged, stop and inspect instead of forcing a merge.

After changes:

```bash
git status
git diff
git add <specific-files-or-folders>
git diff --cached --name-only
git diff --cached
git commit -m "Describe the actual change"
git push
```

Use specific paths with `git add` when practical. `git add .` is acceptable only after checking `.gitignore` and reviewing staged files.

Before switching laptops:
1. Save files.
2. Run relevant tests.
3. Review staged changes.
4. Commit.
5. Push.
6. Confirm push succeeded.

On the other laptop, check status and pull before starting.

Avoid editing the same branch independently on both laptops before synchronizing.

## 11. Branch strategy

Keep `main` as the stable branch.

For Phase 0, a working branch such as `phase0-dataset-audit` may be created after the initial repository connection is verified:

```bash
git switch -c phase0-dataset-audit
git push -u origin phase0-dataset-audit
```

Work and commit on that branch. When Phase 0 is complete and reviewed, create a GitHub Pull Request into `main`.

Do not create branches unnecessarily if the existing repository already has an agreed strategy. Do not change or delete existing branches without approval.

## 12. Dataset and large artifact strategy

Keep the dataset in a stable external location available to each laptop, such as an external drive or approved cloud storage. Configure the dataset root separately on each machine.

Example local configuration values (do not commit the actual `.env` file):

Laptop 1:
```env
DATASET_ROOT=E:/Datasets/Ceramic-Defect
```

Laptop 2:
```env
DATASET_ROOT=D:/Datasets/Ceramic-Defect
```

Use `.env.example` with placeholder values if needed.

Store large checkpoints, generated samples and experiment outputs in approved persistent storage (for example, Google Drive). Record their locations/metadata in project documentation where useful.

## 13. Common Git commands

| Command | Purpose |
|---|---|
| `git status` | Show current changes and branch state |
| `git remote -v` | Show remote repository URLs |
| `git branch -vv` | Show branches and upstream tracking |
| `git log --oneline --decorate -10` | Show recent commit history |
| `git diff` | Show unstaged changes |
| `git diff --cached` | Show staged changes |
| `git add <path>` | Stage selected changes |
| `git commit -m "message"` | Create a local commit |
| `git push` | Upload local commits |
| `git pull --ff-only` | Safely advance from remote when possible |
| `git clone <url>` | Copy repository to another laptop |
| `git switch <branch>` | Switch branches |
| `git switch -c <branch>` | Create and switch to a branch |

## 14. Recovery and conflict rules

If a push is rejected:
- Do not force-push.
- Run `git status`, `git log --oneline --decorate -10` and inspect branch tracking.
- Fetch remote updates and inspect differences.
- Reconcile using a safe merge or rebase only when the situation is understood.
- If unsure, stop and ask the owner.

If a merge conflict occurs:
- Preserve both sides until the correct content is understood.
- Resolve conflicts deliberately.
- Run tests after resolution.
- Do not use destructive reset commands to make the conflict disappear.

Never use these as routine fixes:

```bash
git push --force
git reset --hard
git clean -fd
```

## 15. Ready-to-use Cursor Agent Prompt

Open the exact project folder in Cursor and submit the following prompt.

---

# CURSOR TASK — Safely Connect Existing Major Project to GitHub

You are working in the existing Major Project folder:

```text
E:\MTech[CSE] Nirma_University\S3\Major Project\Defect Dataset Augmentation using Generative Adversarial Networks (GAN)
```

Connect this existing project to the owner's GitHub repository:

```text
https://github.com/HeyitsKeyur/GAN-Defect-Dataset-Augmentation.git
```

Expected default branch: `main`.

### Context files

Read these project files before making changes, if they exist:
- `AGENTS.md`
- `PROJECT_CONTEXT.md`
- `IMPLEMENTATION_PLAN.md`
- `PHASE0_HANDOFF.md`

This is a Git/GitHub setup task only. Do not implement Phase 0 audit functionality, train models, train GANs, download datasets, or reorganize the project.

### Mandatory safe procedure

**A. Inspect first**
1. Verify the current workspace is the exact project folder above.
2. Inspect top-level files and folders.
3. Locate the dataset and large output folders.
4. Check whether `.git` already exists.
5. If Git is initialized, inspect `git status`, `git log`, branches and remotes.
6. Check whether `origin` already exists and where it points.
7. Inspect the GitHub remote's existing state before pushing.
8. Identify large files, dataset archives, raw images, checkpoints, virtual environments and secrets that must not be committed.

**B. Preserve existing work**
1. Never delete or overwrite project files.
2. Never run `git init` if a Git repository already exists.
3. Never discard uncommitted changes.
4. Never rewrite existing commit history.
5. Never use `git push --force`.
6. If local and remote histories conflict or the remote contains unexpected files/commits, stop and report before risky reconciliation.

**C. Configure `.gitignore`**
1. Inspect the existing `.gitignore`.
2. Add or merge appropriate rules for Python environments, caches, `.env` secrets, raw/interim/processed datasets, large checkpoints, generated image collections and temporary outputs.
3. Preserve useful existing rules.
4. Do not blindly ignore all reports or useful small visualizations.
5. Remember that `.gitignore` does not untrack files already tracked.

**D. Protect dataset and secrets**
1. Do not upload Ceramic-Defect or any other raw dataset by default.
2. Do not upload large archives, generated image datasets, model checkpoints or virtual environments.
3. Do not upload credentials, tokens, passwords, API keys or `.env` files.
4. Review staged files carefully before committing.
5. If a large file or secret is staged, stop and correct staging before committing.

**E. Connect the remote**

Use the exact remote URL:

```text
https://github.com/HeyitsKeyur/GAN-Defect-Dataset-Augmentation.git
```

If the local repository has no remote, add `origin`.

If `origin` exists and points elsewhere, do not silently replace it. Report the existing remote and ask before changing it.

If GitHub authentication is required, ask the owner to authenticate through Git Credential Manager/browser. Never ask for a password or token in chat.

**F. Initial commit and push**
1. Determine whether there are changes needing an initial commit.
2. Review all staged file names and staged diffs.
3. Commit only appropriate source code and documentation.
4. If GitHub is empty and local history is ready, push the intended branch and set upstream tracking.
5. If GitHub already contains commits, inspect and safely reconcile; never overwrite remote history.
6. Never force-push.

**G. Verify**

Report:
- Exact local project path.
- Git installation/version.
- Whether the local repository existed previously.
- Current branch.
- Current commit.
- Remote URL.
- Whether upstream tracking is configured.
- Whether push succeeded.
- Current `git status`.
- Files included in the initial commit.
- Files intentionally excluded.
- Authentication or remote-history blockers.

### Two-laptop documentation

Update `GITHUB_SETUP_GUIDE.md` only if necessary to reflect the actual setup. Preserve its existing instructions.

Ensure the guide explains:
- How to clone on another laptop.
- How to configure Git identity and authentication.
- How to create a separate local Python environment.
- How to configure dataset paths separately on each laptop.
- How to pull before work and commit/push after work.
- How to avoid simultaneous edits and resolve conflicts safely.

### Completion conditions

Do not claim setup is complete unless:
- The correct local folder is connected to the intended remote.
- Sensitive/large data are excluded from the commit.
- The correct branch is established.
- The intended commit has been pushed successfully, or a clearly explained blocker remains.
- Git status and remote tracking have been checked.

At the end, provide exact commands executed and actual outputs/results. If an action is unsafe or ambiguous, stop and ask the owner instead of guessing.

---

## 16. Official references

- GitHub: Adding locally hosted code — https://docs.github.com/en/repositories/creating-and-managing-repositories/adding-locally-hosted-code-to-github
- GitHub: Cloning a repository — https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository
- GitHub: Ignoring files — https://docs.github.com/en/get-started/git-basics/ignoring-files
- GitHub: About large files — https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files
- GitHub: Authentication — https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github
- Git documentation — https://git-scm.com/doc

## 17. Final checklist

- [ ] Existing project folder inspected.
- [ ] Existing Git history preserved.
- [ ] Existing GitHub remote inspected.
- [ ] `.gitignore` reviewed and updated safely.
- [ ] Raw dataset excluded from Git.
- [ ] Large checkpoints/generated outputs excluded by default.
- [ ] Credentials and `.env` excluded.
- [ ] Staged files reviewed.
- [ ] Initial commit created if needed.
- [ ] Correct GitHub remote configured.
- [ ] Push completed without force-push.
- [ ] Remote tracking verified.
- [ ] Second-laptop clone instructions reviewed.
- [ ] Local dataset path strategy documented.
