# TRAKTOR ML V4 — Guía de Uso

Pipeline completo para clustering y organización de colección de música electrónica.

---

## 1. Requisitos

**HPC (Surrey):**
- Slurm con particiones `a100` (GPU) y `debug` (CPU)
- Apptainer SIF: `pytorch_2.7.0_cu128.sif` en scratch4weeks
- Wrapper de Slurm: `./slurm/tools/on_submit.sh`

**Local/login node:**
- Python 3.11 en `/usr/bin/python3.11`
- Dependencias CPU: `pip install pandas pyarrow scikit-learn hdbscan umap-learn streamlit plotly`

**Config mínima** (adaptar `config/v4.yaml`):
```yaml
paths:
  local_windows_audio_dir: "C:\\Música\\2020 new - copia"
datasets:
  test_20:
    audio_root: "/mnt/fast/nobackup/users/gb0048/traktor/data/raw_audio/test_20"
    expected_n: 243
```

---

## 2. Pipeline completo (Phase 0 → 5)

### Phase 0 — Ingesta y catálogo (login node, ~1 min)

```bash
cd /mnt/fast/nobackup/users/gb0048/traktor
python src/v4/pipeline/phase0_ingest.py --dataset-name test_20
```

Output: `artifacts/v4/datasets/test_20/catalog.parquet`, `ingest_report.json`

### Phase 1 — Extracción de features (GPU, ~1h para ~250 tracks)

```bash
# Single job (recomendado para datasets pequeños ≤500 tracks)
./slurm/tools/on_submit.sh sbatch slurm/jobs/v4/phase1_extract.job test_20

# Array job (para datasets grandes ≥1000 tracks, sharding paralelo en 4 GPUs)
./slurm/tools/on_submit.sh sbatch slurm/jobs/v4/phase1_extract_array.job test_20

# Merge shards (tras cualquier variante)
./slurm/tools/on_submit.sh sbatch slurm/jobs/v4/phase1_merge.job test_20
```

Output:
- `embeddings/mert_perc.npy` (N, 1024) — embeddings percusivos
- `embeddings/mert_full.npy` (N, 1024) — embeddings full mix
- `embeddings/track_uids.json` — N canónico (N ≤ catalog)
- `catalog_success.parquet` — catálogo filtrado a N exitosos
- `features/bpm_key.parquet` — BPM, key, beat_confidence

**Nota sobre N canónico:** `N = len(track_uids.json)` ≤ N del catálogo. Los tracks fallidos
(archivos corruptos) se excluyen. `catalog_success.parquet` es la fuente de verdad para Phase 2-5.

### Phase 2 — Clustering (login node o debug, ~2-5 min)

```bash
# Sin UMAP (más rápido, recomendado primero)
python src/v4/pipeline/phase2_cluster.py \
    --dataset-name test_20 --skip-umap --config-tag baseline

# Con UMAP (necesario para visualización en UI)
python src/v4/pipeline/phase2_cluster.py --dataset-name test_20 --config-tag v1

# Ajustar parámetros si el clustering no satisface:
python src/v4/pipeline/phase2_cluster.py \
    --dataset-name test_20 \
    --l1-min-cluster-size 7 --l1-min-samples 2 \
    --l2-min-cluster-size 3 --l2-min-samples 2 \
    --config-tag v2
```

Output: `clustering/results_<hash>.parquet`, `clustering/config_<hash>.json`

**PAUSA HUMANA:** Revisar reporte de clustering antes de continuar:
```bash
python tests/v4/test_block3_clustering.py
```

### Phases 3-5 — Naming, Ordering, Export (todo en CPU)

```bash
# Opción 1: job Slurm (recomendado)
./slurm/tools/on_submit.sh sbatch slurm/jobs/v4/phase2_to_5.job test_20

# Opción 2: ejecutar localmente (login node)
python src/v4/pipeline/phase3_name.py --dataset-name test_20
python src/v4/pipeline/phase4_order.py --dataset-name test_20
python src/v4/pipeline/phase5_export.py \
    --dataset-name test_20 \
    --windows-audio-dir "C:\\Música\\2020 new - copia"
```

Output: `playlists/V4_<N>/` con M3U por subcluster L2.

---

## 3. UI Streamlit

```bash
# Desde repo root (en login node o local con SSH tunnel)
streamlit run src/v4/ui/app.py --server.port 8501

# Acceder desde navegador: http://localhost:8501
# (si es SSH, hacer tunnel: ssh -L 8501:localhost:8501 datamove1)
```

**Funcionalidades:**
- Scatter UMAP interactivo (o BPM vs label si no hay UMAP)
- Filtrar por cluster L1 → ver subclusters L2
- Re-clustering desde la UI (local, con sliders)
- Export de playlists (Phase 3+4+5) desde la UI

---

## 4. Tests de validación

```bash
# Block 1: common utilities (sin GPU)
python tests/v4/test_block1_common.py

# Block 2: pipeline scripts + Phase 0
python tests/v4/test_block2_pipeline.py

# Block 3: clustering (requiere embeddings de Phase 1)
python tests/v4/test_block3_clustering.py

# Block 4: export pipeline (requiere clustering)
python tests/v4/test_block4_export.py

# Block 5: sistema completo
python tests/v4/test_block5_system.py
```

---

## 5. Escalar a 2000 tracks

```yaml
# Añadir en config/v4.yaml:
datasets:
  full_2000:
    audio_root: "/path/to/full_2000/audio"
    expected_n: null  # No verificar count exacto
```

```bash
# Phase 0
python src/v4/pipeline/phase0_ingest.py --dataset-name full_2000

# Phase 1 con sharding en array (4 GPUs paralelas)
./slurm/tools/on_submit.sh sbatch slurm/jobs/v4/phase1_extract_array.job full_2000

# Merge + resto del pipeline
./slurm/tools/on_submit.sh sbatch slurm/jobs/v4/phase1_merge.job full_2000
./slurm/tools/on_submit.sh sbatch slurm/jobs/v4/phase2_to_5.job full_2000
```

**Nota:** Para >1000 tracks, aumentar `l1-min-cluster-size` a 15-20 para evitar demasiados clusters pequeños.

---

## 6. Troubleshooting

| Problema | Solución |
|---------|---------|
| `ModuleNotFoundError` | `export PYTHONPATH=/mnt/fast/nobackup/users/gb0048/traktor:$PYTHONPATH` |
| Phase 1 salta tracks | Revisar `run_manifest.json` → `failed_uids` para debug |
| Smoke test vs full run conflict | Ya corregido con `run_id` en progress files (v2026-03-01) |
| Noise rate > 50% | Bajar `l1-min-cluster-size` a 5-7, `l1-min-samples` a 2 |
| Solo 1 cluster L1 | Subir `l1-min-cluster-size` — dataset demasiado homogéneo |
| M3U no carga en Traktor | Verificar `local_windows_audio_dir` en config/v4.yaml |
| UMAP tarda mucho | Usar `--skip-umap` para exploración inicial |

---

## 7. Estructura de artifacts

```
artifacts/v4/datasets/<dataset>/
├── catalog.parquet              # Todos los tracks escaneados (Phase 0)
├── catalog_success.parquet      # N canónico: tracks con embeddings (Phase 1 merge)
├── ingest_report.json
├── run_manifest.json            # Metadatos del run + processed/failed/skipped UIDs
├── embeddings/
│   ├── mert_perc.npy (N,1024)
│   ├── mert_full.npy (N,1024)
│   └── track_uids.json [N UIDs]  ← N canónico
├── features/
│   └── bpm_key.parquet
└── clustering/
    ├── results_<hash>.parquet
    ├── config_<hash>.json
    ├── names_<hash>.json
    └── ordered_<hash>.parquet

playlists/V4_<N>/
├── L1_A_<nombre>/
│   └── L2_A1_<nombre>.m3u
├── All_Noise.m3u
└── _summary.txt
```
