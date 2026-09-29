# TRAKTOR ML V4 — Guía de Uso

Pipeline completo para clustering y organización de colección de música electrónica.

---

## 0. Workflow actual: local, CPU-first (por defecto)

El desarrollo por defecto es local y en CPU. Surrey HPC/Slurm y Lightning AI Studio
son infraestructura opcional/legacy y nunca se arrancan ni se facturan automáticamente
(ver `AGENTS.md`). Las secciones con `sbatch`/`on_submit.sh` de abajo son de referencia
histórica para HPC; no son el flujo actual.

**Requisitos locales (CPU):**
- Python 3.11.
- Dependencias: `pip install -r requirements_v4.txt`.
- Chequeo pre-commit (una vez por clon): `git config core.hooksPath tools/git_hooks`. Bloquea audio,
  arrays, pesos, credenciales y archivos de más de 5 MB.
- Nota Windows: Essentia no tiene build nativo para Windows. En Windows nativo corren
  Phase 0, Phases 2-5, tests y la UI (torch/torchaudio/sklearn/umap/hdbscan ya sirven en CPU);
  la extracción CPU de Phase 1 (`--essentia-only`) se ejecuta bajo WSL/Linux con Python 3.11.

**Dataset local:** `data/raw_audio/test_20/` (243 tracks). `data/` está ignorado por Git;
en local esta ruta suele ser un enlace de filesystem (symlink en Linux/WSL, junction en Windows)
a la carpeta de música real. Nunca se commitea.

```bash
# Phase 0 — ingesta y catálogo (CPU, ~30 s)
python src/v4/pipeline/phase0_ingest.py --dataset-name test_20

# Phase 1 — extracción CPU sin GPU:
#   solo BPM/key (Essentia, WSL/Linux):
python src/v4/pipeline/phase1_extract.py --dataset-name test_20 --device cpu --essentia-only
#   percusión sin Demucs (HPSS, CPU):
python src/v4/pipeline/phase1_extract.py --dataset-name test_20 --device cpu --percussion hpss
python src/v4/pipeline/phase1_merge_shards.py --dataset-name test_20

# Phases 2-5 — clustering, naming, ordering, export (CPU)
python src/v4/pipeline/phase2_cluster.py --dataset-name test_20 --skip-umap --config-tag baseline
python src/v4/pipeline/phase3_name.py --dataset-name test_20
python src/v4/pipeline/phase4_order.py --dataset-name test_20
python src/v4/pipeline/phase5_export.py --dataset-name test_20 --windows-audio-dir "C:\\Música\\2020 new - copia"
```

Demucs + MERT en GPU siguen siendo opcionales y solo con aprobación explícita (ver `AGENTS.md`).

---

## 0b. MVP: biblioteca completa → Rekordbox + Traktor (Windows, sin Essentia) — 2026-09-27

Dataset `musica`: toda la biblioteca copiada en `Música/` (raíz del repo, git-ignorada), con
subcarpetas. BPM, tonalidad y energía salen de los tags (Beatport / Mixed In Key; el BPM que falta
se estima del audio); el parecido sale de CLAP (determinista, 3 segmentos x 3 ventanas de 10 s).
Los grupos se arman con Ward (número de grupos fijado): en CLAP, HDBSCAN deja casi todo como ruido.
Todo corre en CPU en Windows; ffmpeg en PATH cubre los MP3 que libsndfile no abre.

```bash
# 0. Catálogo recursivo con tags; duplicados exactos → duplicates.csv (gana la copia organizada)
python src/v4/pipeline/phase0_ingest.py --dataset-name musica
# 1. BPM / tonalidad / energía desde tags (+ BPM estimado del audio donde falte)
python src/v4/pipeline/phase1_tags.py --dataset-name musica --estimate-missing
# 1b. CLAP. Primero una carpeta chica para verificar (misma caché), luego 2 procesos con prioridad
#     baja (fuera < 90 s y > 15 min: muestras, FX, mixes enteros) y ensamblado
python src/v4/pipeline/extract_representations.py --dataset-name musica --models clap --min-duration 90 --max-duration 900 --folder "2022 sin clasificar"
python src/v4/pipeline/extract_representations.py --dataset-name musica --models clap --threads 8 --min-duration 90 --max-duration 900 --shard 2:0 --low-priority
python src/v4/pipeline/extract_representations.py --dataset-name musica --models clap --threads 8 --min-duration 90 --max-duration 900 --shard 2:1 --low-priority
python src/v4/pipeline/extract_representations.py --dataset-name musica --models clap --min-duration 90 --max-duration 900 --assemble-only
# 1c. Etiqueta Vocal (CLAP zero-shot, umbral 0.145). Primero sin escribir; --write-tags agrega
#     " - Vocal" al comentario (respaldo CSV en features/; --revert <csv> lo deshace)
python src/v4/pipeline/tag_vocals.py --dataset-name musica --method clap --check-list 12
python src/v4/pipeline/tag_vocals.py --dataset-name musica --method clap --write-tags
# 2-5. Grupos (Ward sobre CLAP + BPM), nombres (género de los tags), orden (embedding + BPM +
#      tonalidad + energía), export
python src/v4/pipeline/phase2_cluster.py --dataset-name musica --rep clap_full --bpm-weight 0.3 --method ward --n-l1 15 --l2-target-size 35 --pca-dim 50
python src/v4/pipeline/phase3_name.py --dataset-name musica
python src/v4/pipeline/phase4_order.py --dataset-name musica --rep clap_full
python src/v4/pipeline/phase5_export.py --dataset-name musica --rep clap_full --formats m3u8,rekordbox,traktor --out-root artifacts/v4/datasets/musica/exports
```

**Organizaciones estables (2026-09-29).** Las playlists para tocar salen de una organización con
nombre y versiones (`src/v4/pipeline/organize.py`, plan `docs/plans/20260929_organizaciones_estables.md`):

```bash
# convertir la organización elegida en 'biblioteca' v1 (sin cambios; reproduce el mismo mapa)
python src/v4/pipeline/organize.py import --name biblioteca --from-hash fb78f2f6
# música nueva en Música/<carpeta>: catálogo, BPM, CLAP y Vocal solo de esa carpeta
python src/v4/pipeline/organize.py ingest --scope "2026 Octubre"
# a) agregarla sin tocar nada de lo existente (congelado)
python src/v4/pipeline/organize.py add --name biblioteca --scope "2026 Octubre"
# b) o rehacer todo desde cero (nueva versión)
python src/v4/pipeline/organize.py build --name biblioteca
# organizar solo esa carpeta, con su propio mapa
python src/v4/pipeline/organize.py build --name octubre --scope "2026 Octubre"
# semilla: estos temas van juntos (congelado: solo se mueven ellos; --rebuild: todo desde cero)
python src/v4/pipeline/organize.py link --name biblioteca --track "Artista - Título" --track "Otro tema"
python src/v4/pipeline/organize.py show --name biblioteca
# export y página de la versión actual
python src/v4/pipeline/phase5_export.py --dataset-name musica --org-name biblioteca --formats m3u8,rekordbox,traktor --out-root artifacts/v4/datasets/musica/exports
python tools/playlist_review/build_review_page.py --dataset-name musica --org-name biblioteca
```

**Antes de importar, revisar escuchando:** `python tools/playlist_review/build_review_page.py
--dataset-name musica` escribe `Música/_revision_playlists.html` (necesita Phase 2 sin `--skip-umap`).
Se abre con doble clic; los veredictos se exportan a CSV. Para comparar dos organizaciones a ciegas:
`--org <hash1> --org <hash2> --blind` (la correspondencia queda en `evaluation/review_key_*.json`).

**Rekordbox (pendrive):** Preferencias > Avanzado > Base de datos > rekordbox xml: elegir
`rekordbox.xml`. Preferencias > Vista > Diseño: activar "rekordbox xml". En el árbol,
rekordbox xml > Playlists > clic derecho sobre la carpeta "TRAKTOR ML ..." > Importar playlist.
Analizar los temas y arrastrar la carpeta al pendrive (modo Export).
**Traktor:** Browser > clic derecho en Playlists > Import Playlist > `traktor.nml`. Alternativa
universal: las `.m3u8` de `m3u8/` (una por playlist).

---

## 1. Requisitos (referencia histórica HPC / entornos remotos)

> Legacy/opcional. No es el flujo actual (ver sección 0).

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

## Modo Essentia-only (CPU)

Para obtener solo BPM y tonalidad sin GPU (por ejemplo para los baselines de tripletas):

```bash
python src/v4/pipeline/phase1_extract.py --dataset-name test_20 --device cpu --essentia-only
```

Usa un checkpoint propio en `features/shards/progress_essentia_shard_XX.json` (no toca el de
embeddings) y escribe `features/bpm_key.parquet` al terminar. Unos 11 s por tema en CPU.
