# TRAKTOR-ML Project Map

**Última actualización:** 2025-02-05
**Versión actual:** V2 (Drum-First Hierarchy)

## Estructura del Proyecto

```
traktor/
├── CLAUDE.md                           # Instrucciones del proyecto
├── data/
│   ├── raw_audio/
│   │   └── test_20/                    # 245 tracks MP3 de prueba
│   ├── embeddings/
│   │   └── v2/                         # [NUEVO] Embeddings V2
│   └── outputs/
│       └── fase1_validation/           # Outputs de Fase 1 (legacy)
├── docs/
│   └── PROJECT_MAP.md                  # Este archivo
├── legacy/                             # [NUEVO] Código V1 archivado
│   └── v1/
│       ├── scripts/hpc/process/        # Scripts V1 originales
│       └── slurm/jobs/                 # Jobs V1 originales
├── logs/                               # Logs de jobs Slurm
├── models/
│   └── essentia/                       # Modelos Essentia TensorFlow
│       ├── discogs-effnet-bs64-1.pb
│       ├── discogs-maest-30s-pw-1.pb
│       └── genre_discogs400-*.pb       # [PENDIENTE] Clasificador géneros
├── plans/
│   ├── 20250203_fase1_validacion_embeddings.md
│   └── v2_drum_hierarchy.md            # [NUEVO] Plan V2
├── results/
│   ├── fase1_validation/               # Resultados V1
│   └── v2_hierarchy/                   # [NUEVO] Resultados V2
├── scripts/
│   ├── common/                         # [NUEVO] Utilidades compartidas V2
│   │   ├── audio_utils.py              # Carga audio, conversión
│   │   ├── embedding_utils.py          # Funciones Essentia
│   │   ├── clustering_utils.py         # UMAP/HDBSCAN
│   │   └── demucs_utils.py             # Wrapper Demucs
│   ├── local/                          # [NUEVO] Scripts para ejecución local
│   │   └── generate_visualization.py   # Genera visualization.html con toggle L1/L2
│   └── hpc/
│       └── process/
│           └── v2/                     # [NUEVO] Pipeline V2
│               ├── download_models.py
│               ├── phase1_extraction.py  # GPU: Demucs + embeddings
│               └── phase2_analysis.py    # CPU: Clustering + reporte
└── slurm/
    ├── jobs/
    │   └── v2/                         # [NUEVO] Jobs V2
    │       ├── v2_phase1.job
    │       └── v2_phase2.job
    ├── templates/
    │   └── generic_job.job             # Template genérico
    └── tools/
        └── on_submit.sh                # Wrapper para Slurm
```

## Pipeline V2: Drum-First Hierarchy

### Arquitectura

| Fase | Script | Recurso | Descripción |
|------|--------|---------|-------------|
| Setup | `download_models.py` | CPU | Descarga genre_discogs400 |
| Phase 1 | `phase1_extraction.py` | GPU A100 | Demucs → embeddings (on-the-fly) |
| Phase 2 | `phase2_analysis.py` | CPU | Clustering L1/L2 + géneros + reporte |

### Flujo de Datos

```
raw_audio/*.mp3
      │
      ▼
┌─────────────────────────────┐
│  Phase 1 (GPU, ~5h)         │
│  - Demucs: separar drums    │
│  - Essentia: embeddings     │
│  - TODO en memoria (0 WAV)  │
└─────────────────────────────┘
      │
      ▼
embeddings/v2/
├── drum_embeddings.npy       (245, 1280)
├── fulltrack_embeddings.npy  (245, 1280)
└── manifest_v2.json
      │
      ▼
┌─────────────────────────────┐
│  Phase 2 (CPU, ~1h)         │
│  - Clustering Nivel 1       │
│  - Clustering Nivel 2       │
│  - Clasificación géneros    │
│  - Votación + renombrado    │
└─────────────────────────────┘
      │
      ▼
results/v2_hierarchy/
├── level1_clusters.csv
├── level2_clusters.csv
├── genre_predictions.csv
├── final_organization.csv
└── visualization.html         # Generado por generate_visualization.py
      │
      ▼
┌─────────────────────────────┐
│  Visualización (Local)      │
│  generate_visualization.py  │
│  - Toggle L1 (Drums) / L2   │
│  - Colores únicos por L2    │
│  - Double-click playback    │
└─────────────────────────────┘
```

## Visualización Interactiva

### Script: `scripts/local/generate_visualization.py`

Genera `visualization.html` desde los CSVs de resultados, independiente del pipeline de procesamiento.

**Características:**
- Toggle para alternar entre vista L1 (Según Drums) y L2 (Según Todo)
- Colores únicos para cada subcluster L2 derivados del color del padre L1
- Double-click en puntos para reproducir audio (Windows)
- Sin dependencias externas (solo stdlib de Python)

**Uso:**
```bash
# Regenerar visualización
python scripts/local/generate_visualization.py results/v2_hierarchy/

# Con directorio de audio personalizado
python scripts/local/generate_visualization.py results/v2_hierarchy/ --audio-dir "D:/Mi Musica/"

# Guardar en ubicación diferente
python scripts/local/generate_visualization.py results/v2_hierarchy/ --output mi_viz.html
```

**Entrada (CSVs):**
- `level1_clusters.csv` → track, cluster_l1, umap_x, umap_y
- `level2_clusters.csv` → track, cluster_l1, cluster_l2
- `final_organization.csv` → track, folder_l1, folder_l2

**Salida:**
- `visualization.html` con toggle L1/L2

## Código Legacy (V1)

Los scripts originales de Fase 1 están archivados en `legacy/v1/`:

| Archivo | Propósito |
|---------|-----------|
| `download_essentia_models.py` | Descarga modelos Essentia |
| `extract_embeddings.py` | Extrae embeddings (effnet/maest) |
| `reduce_and_cluster.py` | UMAP + HDBSCAN |
| `generate_visualization.py` | Plotly HTML |

**Reutilización:** Funciones core extraídas a `scripts/common/`.

## Jobs Slurm V2

| Job | Partición | GPUs | RAM | Tiempo |
|-----|-----------|------|-----|--------|
| `v2_phase1.job` | a100 | 1 | 32GB | ~5h |
| `v2_phase2.job` | cpu | 0 | 16GB | ~1h |

**Ejecución:**
```bash
./slurm/tools/on_submit.sh sbatch slurm/jobs/v2/v2_phase1.job
./slurm/tools/on_submit.sh sbatch slurm/jobs/v2/v2_phase2.job
```

## Dependencias V2

| Paquete | Estado | Propósito |
|---------|--------|-----------|
| `demucs` | **Pendiente** | Separación de stems |
| `essentia-tensorflow` | Instalado | Embeddings |
| `torch`, `torchaudio` | Instalado | Backend GPU |
| `umap-learn` | Instalado | Reducción dimensional |
| `scikit-learn` | Instalado | HDBSCAN clustering |
| `plotly` | Instalado | Visualización |

## Convenciones

- **Paths:** Usar `pathlib` (OS-agnostic)
- **Docstrings:** Incluir `PURPOSE` y `CHANGELOG`
- **Slurm:** Siempre usar `./slurm/tools/on_submit.sh`
- **Planes:** Crear en `plans/*.md` antes de implementar
- **Legacy:** Código obsoleto va a `legacy/vN/`
