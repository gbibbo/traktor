# TRAKTOR ML

Audio ML pipeline for organizing a DJ music library by rhythmic, timbral and transition similarity.

This repository is shared as portfolio evidence. The private music collection and large generated artifacts are not included, because the source audio is personal and copyright protected. The repo includes the V4 pipeline code, Slurm jobs, tests, documentation, UI screenshot and exported playlist examples.

![Streamlit clustering explorer](interface.png)

## Problem

Large DJ libraries are hard to organize with filenames, folders or broad genre tags alone. For Techno and Tech House, tracks that look similar in metadata can behave very differently in a set because of groove, percussion, timbre, BPM and harmonic compatibility.

The practical problem was to reduce manual sorting time and generate Traktor-ready playlists that group musically similar tracks while preserving useful transition order inside each group.

## What I built

I built an end-to-end audio ML system that turns a folder of tracks into versioned M3U playlists for Traktor DJ.

The V4 pipeline has five stages:

| Stage | What it does | Main output |
|---|---|---|
| Phase 0: Ingest | Scans audio files, validates paths, computes stable track IDs and builds a canonical catalog | `catalog.parquet`, `ingest_report.json` |
| Phase 1: Feature extraction | Runs GPU feature extraction with Demucs, MERT and Essentia | `mert_perc.npy`, `mert_full.npy`, `bpm_key.parquet` |
| Phase 2: Clustering | Builds two-level clusters with PCA, HDBSCAN and UMAP | `results_<hash>.parquet`, `config_<hash>.json` |
| Phase 3: Naming | Assigns readable cluster names using metadata when available and safe fallbacks otherwise | `names_<hash>.json` |
| Phase 4: Ordering | Orders tracks inside clusters for smoother transitions using embeddings, BPM and Camelot key compatibility | `ordered_<hash>.parquet` |
| Phase 5: Export | Writes versioned M3U playlists with Windows paths ready for Traktor DJ | `playlists/V4_<N>/` |

The system also includes a Streamlit dashboard for reviewing clustering results, changing clustering parameters locally and re-exporting playlists.

## Key design decisions

### Separate groove from full-track similarity

The pipeline extracts two MERT embedding views:

- `mert_perc.npy`: embeddings from the percussive/drums stem, used for first-level clustering by groove.
- `mert_full.npy`: embeddings from the full mix, used for second-level clustering and transition ordering.

This keeps the first grouping close to DJ-relevant rhythm and groove, while the second grouping can still capture broader timbral and musical context.

### Use PCA before HDBSCAN

MERT embeddings are 1024-dimensional. Direct density clustering was unstable on a small, homogeneous Techno/Tech House collection. V4 applies PCA before HDBSCAN. On the validated `test_20` run, `pca_dim=50` retained 93.7% of variance and produced useful L1 structure.

### Keep a canonical N across artifacts

The code treats `track_uids.json` as the source of truth after GPU extraction. This prevents misalignment between catalog rows, embedding matrices, BPM/key features, clustering outputs and exported playlists when one track fails during processing.

### Export for a real downstream tool

The final artifact is not only a plot or notebook. The pipeline writes UTF-8 M3U playlists with Windows paths and `#EXTM3U` metadata, so they can be imported directly into Traktor DJ.

## Tech stack

| Area | Tools |
|---|---|
| Core language | Python 3.11 |
| Deep learning | PyTorch, torchaudio, Hugging Face Transformers |
| Audio representation | `m-a-p/MERT-v1-330M` |
| Source separation | Demucs `htdemucs` |
| Music features | Essentia BPM, beat confidence and key extraction |
| Clustering and reduction | scikit-learn PCA, scikit-learn HDBSCAN, UMAP |
| Data artifacts | pandas, Parquet, NumPy arrays, JSONL logs |
| UI | Streamlit, Plotly |
| HPC execution | Slurm, Apptainer, A100 GPU jobs |
| Validation | Block-level Python and shell tests |

## Result and evidence

The V4 implementation was run end to end on a private 243-track Techno/Tech House collection.

| Evidence | Result |
|---|---|
| Catalog size | 243 audio files scanned |
| Canonical processed set | 239 tracks with successful embeddings |
| GPU smoke test | 3-track smoke test passed in 46.9 s |
| Full GPU extraction | 239 successful tracks in 1 h 03 m |
| Embedding artifacts | `mert_perc.npy` and `mert_full.npy`, each with shape `(239, 1024)` |
| BPM/key artifact | 239 rows, BPM range 86 to 167, median BPM 123.9 |
| Clustering run | 8 first-level clusters with UMAP coordinates |
| Noise handling | Raw HDBSCAN noise retained for diagnosis, with optional 1-NN reassignment for playlist export |
| Playlist export | 239 of 239 canonical tracks exported |
| Latest playlist set | `playlists/V4_5/`, 14 M3U files across 8 L1 groups |
| Ordering check | `transition_score=0.797` reported in the V4 validation tracker |

The repository includes several concrete evidence points:

- `src/v4/`: current V4 implementation.
- `slurm/jobs/v4/`: reproducible HPC jobs for GPU extraction and CPU phases.
- `tests/v4/`: block-level tests for setup, common utilities, pipeline, clustering, export and system integration.
- `docs/v4/JOBS_STATUS.md`: job history and validated artifact shapes.
- `docs/v4/TODO.md`: completed implementation tracker.
- `playlists/V4_1/` to `playlists/V4_5/`: exported playlist examples.
- `interface.png`: Streamlit UI screenshot from the clustering explorer.

## Repository structure

```text
config/v4.yaml              Main configuration for paths, datasets, clustering and ordering
src/v4/common/              Shared utilities for config, paths, catalog, audio, Demucs, MERT and logs
src/v4/pipeline/            Phase 0 to Phase 5 pipeline scripts
src/v4/evaluation/          Metrics for clustering, retrieval, ordering and noise analysis
src/v4/ui/app.py            Streamlit clustering explorer
src/v4/adaptation/          Projection-head and contrastive-training stubs for future adaptation
slurm/jobs/v4/              Slurm jobs for GPU and CPU execution
tests/v4/                   Block-level validation tests
docs/                       Usage guide, project map, job status and lessons learned
playlists/                  Exported M3U playlist examples
legacy/                     Older V1 to V3 experiments retained for traceability
```

## How to run

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_v4.txt
```

On Surrey HPC, Phase 1 uses Slurm and Apptainer because the A100 nodes do not provide the required Python environment natively. See `docs/V4_USAGE.md` and `slurm/jobs/v4/` for the validated HPC workflow.

### 2. Configure the dataset

Edit `config/v4.yaml`:

```yaml
datasets:
  test_20:
    audio_root: "/path/to/audio"
    expected_n: null

paths:
  local_windows_audio_dir: "C:\\Music\\My DJ Library"
```

The repository does not include the private audio dataset. To reproduce the pipeline, point `audio_root` to your own local collection.

### 3. Build the catalog

```bash
python src/v4/pipeline/phase0_ingest.py --dataset-name test_20
```

### 4. Extract features on GPU

Small dataset, single job:

```bash
./slurm/tools/on_submit.sh sbatch slurm/jobs/v4/phase1_extract.job test_20
./slurm/tools/on_submit.sh sbatch slurm/jobs/v4/phase1_merge.job test_20
```

Larger dataset, sharded extraction:

```bash
./slurm/tools/on_submit.sh sbatch slurm/jobs/v4/phase1_extract_array.job test_20
./slurm/tools/on_submit.sh sbatch slurm/jobs/v4/phase1_merge.job test_20
```

### 5. Run clustering, naming, ordering and export

```bash
python src/v4/pipeline/phase2_cluster.py --dataset-name test_20 --config-tag baseline
python src/v4/pipeline/phase3_name.py --dataset-name test_20
python src/v4/pipeline/phase4_order.py --dataset-name test_20
python src/v4/pipeline/phase5_export.py \
  --dataset-name test_20 \
  --windows-audio-dir "C:\\Music\\My DJ Library"
```

Or run the combined Slurm job for Phases 2 to 5:

```bash
./slurm/tools/on_submit.sh sbatch slurm/jobs/v4/phase2_to_5.job test_20
```

### 6. Open the Streamlit demo

From the machine running Streamlit:

```bash
streamlit run src/v4/ui/app.py --server.port 8501
```

From Windows, using an SSH tunnel to the HPC login node:

```powershell
ssh -L 8501:localhost:8501 datamove1
```

Then open:

```text
http://localhost:8501
```

The UI lets you inspect the UMAP embedding map, filter L1/L2 clusters, view BPM/key metadata, test local clustering parameters and export playlist versions.

## Validation

Run the available checks from the repository root:

```bash
python tests/v4/test_block1_common.py
python tests/v4/test_block2_pipeline.py
python tests/v4/test_block3_clustering.py
python tests/v4/test_block4_export.py
python tests/v4/test_block5_system.py
```

Some tests require generated artifacts from Phase 1 onward. If the private dataset or embeddings are not present, those checks will skip or report missing artifacts rather than reproducing the original private run.

## Limitations

- The private music collection is not included.
- Large generated artifacts under `artifacts/v4/` are not included.
- Cluster names are conservative when external metadata is missing.
- `src/v4/adaptation/` contains future-facing stubs for contrastive adaptation, not a completed fine-tuning pipeline.
- Musical quality still requires human review in Traktor, because clustering metrics cannot fully judge DJ transition quality.

## Why this repo matters

This project shows that I can build beyond a notebook: GPU audio feature extraction, HPC execution, artifact alignment, clustering, evaluation, UI inspection and final export into a real user workflow.

The core engineering pattern is transferable to audio ML, recommendation, retrieval, music intelligence and applied machine learning systems where model outputs must become usable artifacts for a downstream user.
