# TRAKTOR ML

Audio ML pipeline for organizing a Techno / Tech House DJ library into Traktor-ready playlists.

This project started from a practical DJ workflow: a large folder of tracks does not show which songs share groove, percussion, timbre, tempo range or harmonic movement. Metadata helps, but it is too coarse for building playable groups. TRAKTOR ML turns a local music collection into versioned M3U playlists by extracting audio representations, clustering similar tracks, ordering each group for smoother transitions and exposing the result in a Streamlit dashboard.

The source audio and large generated artifacts are not included because the collection is private and copyright protected. The repository includes the V4 pipeline code, configuration, Slurm jobs, tests, documentation, UI screenshot and exported playlist examples.

![Streamlit clustering explorer](interface.png)

## From audio folder to playlists

```text
Audio files
  -> catalog with stable track IDs
  -> Demucs stems, MERT embeddings, Essentia BPM/key features
  -> PCA, HDBSCAN and UMAP clustering
  -> cluster naming and transition-aware ordering
  -> Traktor-ready M3U playlists
  -> Streamlit review and re-export
```

The pipeline is split into phases so that GPU-heavy feature extraction can run on an HPC node, while clustering, ordering, export and inspection can run on CPU.

| Phase | Main task | Output |
|---|---|---|
| 0 | Scan the collection, validate paths and build the catalog | `catalog.parquet`, `ingest_report.json` |
| 1 | Extract Demucs stems, MERT embeddings and Essentia BPM/key features | `mert_perc.npy`, `mert_full.npy`, `bpm_key.parquet` |
| 2 | Cluster tracks with PCA, HDBSCAN and UMAP | `results_<hash>.parquet` |
| 3 | Assign readable names to clusters | `names_<hash>.json` |
| 4 | Order tracks inside each cluster using embedding, BPM and key compatibility | `ordered_<hash>.parquet` |
| 5 | Export UTF-8 M3U playlists with Windows paths for Traktor | `playlists/V4_<N>/` |

## Implementation

The current V4 implementation is a Python 3.11 pipeline under `src/v4/`.

- `m-a-p/MERT-v1-330M` provides 1024-dimensional audio embeddings.
- Demucs `htdemucs` separates stems so the system can compare both percussion-focused and full-mix representations.
- Essentia extracts BPM, beat confidence and musical key metadata.
- PCA reduces the MERT space before HDBSCAN, which is more stable on a small and stylistically homogeneous Techno / Tech House collection.
- UMAP gives the dashboard a 2D view of the clustering result.
- Track ordering combines cosine distance, BPM distance and Camelot key compatibility.
- Artifacts are stored as Parquet, NumPy arrays, JSON and JSONL logs.
- GPU work is submitted through Slurm jobs using Apptainer on A100 nodes.
- The Streamlit and Plotly dashboard supports visual inspection, cluster filtering and playlist re-export.

A key detail in V4 is the canonical track set. After feature extraction, `track_uids.json` becomes the source of truth for row alignment across catalog rows, embeddings, BPM/key features, clustering outputs and exported playlists. This avoids silent mismatches when an audio file fails during processing.

## Current V4 run

The validated run used a private Techno / Tech House collection.

| Item | Result |
|---|---:|
| Files found in catalog | 243 |
| Canonical tracks after feature extraction | 239 |
| GPU smoke test | 3 tracks in 46.9 s |
| Full GPU extraction | 239 tracks in 1 h 03 m |
| MERT percussion embeddings | `(239, 1024)` |
| MERT full-mix embeddings | `(239, 1024)` |
| BPM/key rows | 239 |
| BPM range | 86 to 167 |
| Median BPM | 123.9 |
| First-level clusters | 8 |
| Latest exported set | `playlists/V4_5/` |
| Exported tracks | 239 / 239 |
| Exported playlists | 14 M3U files |
| Transition score reported by the validation tracker | 0.797 |

The latest included export is `playlists/V4_5/`:

```text
playlists/V4_5/
├── L1_A_Group A/L2_A1.m3u
├── L1_B_Group B/L2_B1.m3u
├── L1_C_Group C/L2_C1.m3u
├── L1_D_Group D/L2_D1.m3u
├── L1_D_Group D/L2_D2.m3u
├── L1_E_Group E/L2_E1.m3u
├── L1_F_Group F/L2_F1.m3u
├── L1_F_Group F/L2_F2.m3u
├── L1_F_Group F/L2_F3.m3u
├── L1_F_Group F/L2_F4.m3u
├── L1_G_Group G/L2_G1.m3u
├── L1_G_Group G/L2_G2.m3u
├── L1_G_Group G/L2_G3.m3u
├── L1_H_Group H/L2_H1.m3u
└── _summary.txt
```

Raw HDBSCAN noise is preserved for diagnosis. For playlist export, V4 can reassign noise points to the nearest cluster with 1-NN so that every processed track can be placed in a usable playlist.

## Repository layout

```text
config/v4.yaml              Main configuration for paths, datasets, clustering and ordering
src/v4/common/              Config, path, catalog, audio, Demucs, embedding and logging utilities
src/v4/pipeline/            Phase 0 to Phase 5 pipeline scripts
src/v4/evaluation/          Clustering, retrieval, ordering and noise metrics
src/v4/ui/app.py            Streamlit clustering explorer
src/v4/adaptation/          Projection-head and contrastive-training scaffolding
slurm/jobs/v4/              GPU and CPU jobs for the HPC workflow
tests/v4/                   Block-level validation tests
docs/                       Usage guide, project map, job status and implementation notes
playlists/                  Exported M3U playlist examples
legacy/                     Earlier V2 and V3 experiments retained for traceability
```

## Run it on your own collection

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_v4.txt
```

Configure a dataset in `config/v4.yaml`:

```yaml
datasets:
  my_collection:
    audio_root: "/path/to/audio"
    metadata_csv: null
    manifest_csv: null
    expected_n: null

paths:
  local_windows_audio_dir: "C:\\Music\\My DJ Library"
```

Build the catalog:

```bash
python src/v4/pipeline/phase0_ingest.py --dataset-name my_collection
```

Run feature extraction on GPU:

```bash
./slurm/tools/on_submit.sh sbatch slurm/jobs/v4/phase1_extract.job my_collection
./slurm/tools/on_submit.sh sbatch slurm/jobs/v4/phase1_merge.job my_collection
```

For larger collections, use the array job:

```bash
./slurm/tools/on_submit.sh sbatch slurm/jobs/v4/phase1_extract_array.job my_collection
./slurm/tools/on_submit.sh sbatch slurm/jobs/v4/phase1_merge.job my_collection
```

Run clustering, naming, ordering and export:

```bash
python src/v4/pipeline/phase2_cluster.py --dataset-name my_collection --config-tag baseline
python src/v4/pipeline/phase3_name.py --dataset-name my_collection
python src/v4/pipeline/phase4_order.py --dataset-name my_collection
python src/v4/pipeline/phase5_export.py \
  --dataset-name my_collection \
  --windows-audio-dir "C:\\Music\\My DJ Library"
```

Or submit the combined CPU job:

```bash
./slurm/tools/on_submit.sh sbatch slurm/jobs/v4/phase2_to_5.job my_collection
```

## Open the dashboard

Start Streamlit from the repository root:

```bash
streamlit run src/v4/ui/app.py --server.port 8501
```

When running on the HPC login node from Windows, open an SSH tunnel first:

```powershell
ssh -L 8501:localhost:8501 datamove1
```

Then open:

```text
http://localhost:8501
```

The dashboard loads the clustering artifacts, shows the UMAP map, filters by L1 and L2 clusters, displays BPM/key metadata and supports local re-export after parameter changes.

## Validation commands

```bash
python tests/v4/test_block1_common.py
python tests/v4/test_block2_pipeline.py
python tests/v4/test_block3_clustering.py
python tests/v4/test_block4_export.py
python tests/v4/test_block5_system.py
```

Some checks require generated artifacts from Phase 1 onward. Without the private audio collection or embeddings, those checks will report missing artifacts or skip the artifact-dependent parts.

## Reproducibility notes

This repository is meant to show the complete workflow and implementation, but it is not a drop-in reproduction package. The private audio files are not distributed, and the exported M3U examples point to local Windows paths from the original collection.

The completed V4 path covers ingestion, feature extraction, clustering, naming, ordering, export, UI inspection, Slurm execution and tests. The files under `src/v4/adaptation/` define scaffolding for future contrastive adaptation, but they are not part of the completed playlist-generation path.

The clustering result is used as a decision aid, not as an automatic replacement for DJ judgment. The dashboard and playlist versions are part of the workflow because musical grouping still benefits from listening, inspection and iteration.
