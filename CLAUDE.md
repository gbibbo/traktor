# TRAKTOR ML (Surrey HPC) — Project Context & Instructions

## 1. Project Overview
**Goal:** Unified ML pipeline to analyze, classify, and organize a local music collection (Techno/TechHouse).
**Root Directory:** `/mnt/fast/nobackup/users/gb0048/traktor`
**Philosophy:** Hybrid workflow. Heavy training on HPC (Slurm) -> Lightweight inference on Local Windows.

## 2. Environment & Infrastructure
- **Host:** VS Code SSH on `datamove1`.
- **Slurm Commands:** NOT in PATH. Mandatory wrapper:
  `./slurm/tools/on_submit.sh <squeue|sbatch|scancel|sacct|scontrol> <args...>`
- **Job Templates:** New jobs MUST derive from `slurm/templates/generic_job.job` and include:
  `#SBATCH --time=48:00:00` (default)
- **Python Env:** Conda environment `traktor_ml`.

## 3. Core Process (Mandatory)
**"Plan-First" Protocol:**
- Non-trivial changes (e.g., adding a new model, changing data flow) require a written plan in `./plans/YYYYMMDD_feature_name.md` BEFORE coding.
- **Stop & Check:** If asked to add a feature, first audit existing scripts. Do NOT create new scripts (`train_v2.py`) if existing ones can be refactored or extended.

**Scripting Standards:**
- New/edited `.py` files MUST start with a docstring containing `PURPOSE` and `CHANGELOG`.
- Use `pathlib` for all file paths (OS-agnostic).
- **Maintenance:** Update `docs/PROJECT_MAP.md` when adding files.

## 4. Architecture Strategy
Do not pollute the root. Stick to this structure:

| Directory | Purpose |
| :--- | :--- |
| `scripts/hpc/ingest/` | Data download & validation (Slurm). |
| `scripts/hpc/process/` | Feature extraction (Essentia/Librosa). |
| `scripts/hpc/train/` | Model training loops (PyTorch). |
| `scripts/local/` | **Windows Inference.** The final product. |
| `scripts/common/` | Shared Logic (DataLoaders, Model Definitions). |
| `data/raw_audio/` | Read-only input (Symlinks preferred). |

## 5. Quality Gate & Completion Protocol
1.  **Validation:** Ensure `scripts/quality_gate.sh` passes (Linting + Unit Tests).
2.  **Approval:** Ask user to approve/disapprove the outcome.
3.  **Testing:** If approved, propose 3 candidate tests for the new feature; implement the user's choice.

## 6. Lessons Learned (Knowledge Base)
- **Source of Truth:** `docs/LESSONS_LEARNED.md`.
- **No Spam:** Before adding a lesson, search for similar entries. Merge/expand existing ones instead of creating duplicates.

## 7. Storage Policy
- **Heavy Caches:** Use `scripts/migrate_caches_to_scratch4weeks.sh` if storage gets tight.
- **Final Results:** Never leave critical results *only* on scratch. Copy final artifacts to `results/`.