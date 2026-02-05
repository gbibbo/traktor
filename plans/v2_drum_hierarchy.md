# Plan V2: "Drum-First Hierarchy" Pipeline

**Fecha:** 2025-02-04
**Estado:** Aprobado - Pendiente Implementación
**Dataset:** `data/raw_audio/test_20/` (245 tracks, 4.2GB)

---

## 1. Objetivo

Organizar una colección musical en una estructura de carpetas de **dos niveles** basada en criterios semánticos:

1. **Nivel 1 (Percusión/Groove):** Agrupa por patrones rítmicos similares (drums)
2. **Nivel 2 (Melódica/Vibe):** Sub-agrupa por características melódicas/texturales
3. **Etiquetado Semántico:** Renombra carpetas según género dominante

---

## 2. Arquitectura Optimizada

### 2.1 Principios de Diseño

| Principio | Descripción |
|-----------|-------------|
| **Pipeline On-the-Fly** | Sin archivos WAV intermedios. Stems procesados en memoria GPU. |
| **Consolidación Slurm** | Solo 2 scripts principales → 2 jobs. |
| **Eficiencia de Disco** | Solo persistir embeddings (.npy) y resultados (.csv). |

### 2.2 Estructura de Scripts

```
scripts/
├── common/
│   ├── __init__.py
│   ├── audio_utils.py        # Carga audio, conversión formatos
│   ├── embedding_utils.py    # Funciones Essentia (effnet, genre)
│   ├── clustering_utils.py   # UMAP/HDBSCAN reutilizables
│   └── demucs_utils.py       # Wrapper Demucs en memoria
└── hpc/
    └── process/
        └── v2/
            ├── __init__.py
            ├── download_models.py    # Descarga genre_discogs400
            ├── phase1_extraction.py  # GPU: Demucs + embeddings
            └── phase2_analysis.py    # CPU: Clustering + reporte
```

---

## 3. Phase 1: Extracción (GPU)

### 3.1 Flujo de Datos

```
Para cada track en raw_audio/:
┌─────────────────────────────────────────────────────────────┐
│  1. torchaudio.load(track.mp3) → wav tensor                 │
│                         │                                   │
│  2. demucs.apply_model(wav) → stems dict (en memoria)       │
│     [drums, bass, vocals, other]                            │
│                         │                                   │
│  3a. essentia.effnet(stems["drums"]) → drum_embedding       │
│  3b. essentia.effnet(wav_full) → full_embedding             │
│                         │                                   │
│  4. del stems  # Liberar memoria GPU                        │
│                         │                                   │
│  5. Acumular embeddings en arrays numpy                     │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Pseudocódigo Principal

```python
def process_track(audio_path: Path, demucs_model, effnet_model) -> tuple:
    """
    Procesa un track completo sin escribir WAV intermedios.

    Args:
        audio_path: Ruta al archivo MP3/WAV
        demucs_model: Modelo Demucs cargado (htdemucs)
        effnet_model: Path al modelo effnet .pb

    Returns:
        tuple: (drum_embedding, full_embedding) cada uno shape (1280,)
    """
    import torch
    import torchaudio
    from demucs.apply import apply_model

    # 1. Cargar audio
    wav, sr = torchaudio.load(audio_path)
    wav = torchaudio.functional.resample(wav, sr, demucs_model.samplerate)
    wav = wav.mean(0, keepdim=True)  # Mono si es stereo

    # 2. Separar stems (en memoria GPU)
    with torch.no_grad():
        wav_gpu = wav.cuda()[None]  # (1, 1, samples)
        sources = apply_model(demucs_model, wav_gpu)[0]  # (4, 1, samples)

    drums_idx = demucs_model.sources.index("drums")
    drums_wav = sources[drums_idx].cpu()

    # 3a. Embedding del drum stem
    drums_16k = convert_to_essentia(drums_wav, target_sr=16000)
    drum_embedding = extract_effnet_embedding(drums_16k, effnet_model)

    # 3b. Embedding del full track
    wav_16k = convert_to_essentia(wav, target_sr=16000)
    full_embedding = extract_effnet_embedding(wav_16k, effnet_model)

    # 4. Liberar memoria
    del sources, drums_wav, wav_gpu
    torch.cuda.empty_cache()

    return drum_embedding, full_embedding
```

### 3.3 Output

```
data/embeddings/v2/
├── drum_embeddings.npy       # shape: (245, 1280)
├── fulltrack_embeddings.npy  # shape: (245, 1280)
└── manifest_v2.json          # metadata: tracks, dims, timestamp
```

**manifest_v2.json:**
```json
{
  "version": "2.0",
  "pipeline": "drum_first_hierarchy",
  "created": "2025-02-04T12:00:00Z",
  "tracks": ["track1.mp3", "track2.mp3", ...],
  "n_tracks": 245,
  "embeddings": {
    "drum": {"file": "drum_embeddings.npy", "dim": 1280, "model": "discogs-effnet-bs64-1"},
    "fulltrack": {"file": "fulltrack_embeddings.npy", "dim": 1280, "model": "discogs-effnet-bs64-1"}
  },
  "demucs_model": "htdemucs"
}
```

---

## 4. Phase 2: Análisis (CPU)

### 4.1 Flujo de Datos

```
┌──────────────────────────────────────────────────────────────┐
│  1. Cargar embeddings (.npy) + manifest                      │
│                                                              │
│  2. CLUSTERING NIVEL 1 (Drums)                               │
│     L2_normalize(drum_embeddings)                            │
│     UMAP(n_neighbors=15, min_dist=0.1) → coords_2d           │
│     HDBSCAN(min_cluster_size=5) → cluster_labels             │
│     Asignar letras: 0→A, 1→B, 2→C, -1→Noise                  │
│                                                              │
│  3. CLUSTERING NIVEL 2 (Por cada cluster L1)                 │
│     Para cluster 'A' con N tracks:                           │
│       if N >= 10:  # Solo si hay suficientes tracks          │
│         fulltrack_A = fulltrack_embeddings[cluster==A]       │
│         UMAP(fulltrack_A) → sub_coords                       │
│         HDBSCAN(min_cluster_size=3) → sub_labels             │
│         Asignar: A1, A2, A3...                               │
│       else:                                                  │
│         Asignar A1 a todos                                   │
│                                                              │
│  4. CLASIFICACIÓN DE GÉNEROS                                 │
│     genre_discogs400(audio) → top-3 predicciones por track   │
│                                                              │
│  5. VOTACIÓN Y RENOMBRADO                                    │
│     Por carpeta L1: contar géneros → majority vote           │
│     Ej: A(100 Techno, 20 House) → "Group_A_[Techno_House]"   │
│     Por subcarpeta L2: mismo proceso                         │
│                                                              │
│  6. GENERAR OUTPUTS                                          │
│     - level1_clusters.csv                                    │
│     - level2_clusters.csv                                    │
│     - genre_predictions.csv                                  │
│     - final_organization.csv                                 │
│     - visualization.html (Plotly interactivo)                │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 Formatos de Salida

**level1_clusters.csv:**
```csv
track,cluster_l1,umap_x,umap_y
&ME - Garden.mp3,A,1.23,4.56
0010X0010 - Escape.mp3,A,1.45,4.78
Adana Twins - Origo.mp3,B,5.67,2.34
```

**level2_clusters.csv:**
```csv
track,cluster_l1,cluster_l2,sub_umap_x,sub_umap_y
&ME - Garden.mp3,A,A1,0.12,0.34
0010X0010 - Escape.mp3,A,A2,0.56,0.78
```

**genre_predictions.csv:**
```csv
track,genre_1,genre_2,genre_3,conf_1,conf_2,conf_3
&ME - Garden.mp3,Techno,Minimal Techno,Industrial Techno,0.82,0.11,0.04
```

**final_organization.csv:**
```csv
track,folder_l1,folder_l2,final_path
&ME - Garden.mp3,Group_A_[Techno_Minimal],A1_[Industrial],results/v2_hierarchy/Group_A_[Techno_Minimal]/A1_[Industrial]/&ME - Garden.mp3
```

---

## 5. Jobs Slurm

### 5.1 Configuración

| Job | Script | Recursos | Tiempo |
|-----|--------|----------|--------|
| `v2_phase1.job` | `phase1_extraction.py` | 1x A100 GPU, 8 CPU, 32GB RAM | ~5h |
| `v2_phase2.job` | `phase2_analysis.py` | CPU only, 8 CPU, 16GB RAM | ~1h |

### 5.2 Template v2_phase1.job

```bash
#!/bin/bash
#SBATCH --job-name=v2_phase1_extraction
#SBATCH --partition=a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Activar entorno
source ~/.bashrc
export PYTHONPATH="/mnt/fast/nobackup/users/gb0048/traktor:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Ejecutar
python3 scripts/hpc/process/v2/phase1_extraction.py \
    --audio-dir data/raw_audio/test_20 \
    --output-dir data/embeddings/v2 \
    --demucs-model htdemucs \
    --effnet-model models/essentia/discogs-effnet-bs64-1.pb \
    --device cuda
```

### 5.3 Template v2_phase2.job

```bash
#!/bin/bash
#SBATCH --job-name=v2_phase2_analysis
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Activar entorno
source ~/.bashrc
export PYTHONPATH="/mnt/fast/nobackup/users/gb0048/traktor:$PYTHONPATH"

# Ejecutar
python3 scripts/hpc/process/v2/phase2_analysis.py \
    --embeddings-dir data/embeddings/v2 \
    --audio-dir data/raw_audio/test_20 \
    --output-dir results/v2_hierarchy \
    --genre-model models/essentia/genre_discogs400-discogs-effnet-1.pb \
    --min-cluster-size 5 \
    --min-subcluster-size 3
```

---

## 6. Dependencias

### 6.1 A Instalar

```bash
# Demucs para separación de stems
pip install --user demucs

# Descargar modelo genre_discogs400
python scripts/hpc/process/v2/download_models.py
```

### 6.2 Ya Disponibles

| Paquete | Ubicación/Versión |
|---------|-------------------|
| `essentia-tensorflow` | Sistema |
| `torch`, `torchaudio` | Sistema |
| `umap-learn` | `~/.local/` |
| `scikit-learn` (HDBSCAN) | `~/.local/` |
| `plotly` | `~/.local/` |
| `discogs-effnet-bs64-1.pb` | `models/essentia/` |

### 6.3 Modelo a Descargar

```
URL: https://essentia.upf.edu/models/classification-heads/genre_discogs400/
Archivos:
  - genre_discogs400-discogs-effnet-1.pb (~5MB)
  - genre_discogs400-discogs-effnet-1.json
Destino: models/essentia/
```

---

## 7. Gestión de Almacenamiento

| Tipo | Ubicación | Tamaño Est. | Persistencia |
|------|-----------|-------------|--------------|
| Embeddings | `data/embeddings/v2/` | ~3 MB | Permanente |
| Resultados | `results/v2_hierarchy/` | ~2 MB | Permanente |
| Stems WAV | **NINGUNO** | 0 | On-the-fly |
| Symlinks/Org | `results/v2_hierarchy/*/` | ~0 | Permanente |

**Beneficio total:** ~5 MB vs ~36 GB si guardáramos stems.

---

## 8. Código Reutilizable de V1

| Archivo Legacy | Función | Destino V2 |
|----------------|---------|------------|
| `legacy/v1/scripts/hpc/process/extract_embeddings.py` | `extract_effnet_embedding()` | `common/embedding_utils.py` |
| `legacy/v1/scripts/hpc/process/reduce_and_cluster.py` | `apply_umap()`, `apply_hdbscan()` | `common/clustering_utils.py` |
| `legacy/v1/scripts/hpc/process/download_essentia_models.py` | `download_file()` | `v2/download_models.py` |
| `legacy/v1/scripts/hpc/process/generate_visualization.py` | `create_scatter_plot()` | `phase2_analysis.py` |

---

## 9. Verificación

### 9.1 Tests Propuestos

1. **Test unitario (1 track):**
   - Verificar shapes: `drum_embedding.shape == (1280,)`
   - Verificar que no se escriben archivos WAV

2. **Test integración (10 tracks):**
   - Verificar clustering produce al menos 2 clusters
   - Verificar CSV tiene columnas correctas

3. **Test completo (245 tracks):**
   - Ejecutar pipeline completo
   - Validar auditivamente: tracks en mismo cluster suenan similares

### 9.2 Métricas de Éxito

- **Nivel 1:** Clusters con coherencia percusiva (bombos similares)
- **Nivel 2:** Sub-clusters con vibe/melodía coherente
- **Etiquetado:** >70% tracks con género asignado (no noise)
- **Tiempo total:** <6 horas en A100

---

## 10. Secuencia de Ejecución

```bash
# 1. Descargar modelo de géneros (manual, una vez)
python scripts/hpc/process/v2/download_models.py

# 2. Phase 1: Extracción (GPU)
./slurm/tools/on_submit.sh sbatch slurm/jobs/v2/v2_phase1.job

# 3. Esperar a que termine (~5h)
./slurm/tools/on_submit.sh squeue -u $USER

# 4. Phase 2: Análisis (CPU)
./slurm/tools/on_submit.sh sbatch slurm/jobs/v2/v2_phase2.job

# 5. Revisar resultados
ls results/v2_hierarchy/
cat results/v2_hierarchy/final_organization.csv
```

---

## Changelog

| Fecha | Cambio |
|-------|--------|
| 2025-02-04 | Creación inicial del plan V2 |
| 2025-02-04 | Optimización: Pipeline on-the-fly (sin WAV intermedios) |
| 2025-02-04 | Optimización: Consolidación a 2 fases (phase1 + phase2) |
