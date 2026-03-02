# TRAKTOR ML

Pipeline de ML para analizar, clasificar y organizar una colección de música electrónica (Techno / Tech House).

Extrae embeddings de audio con MERT y Demucs, agrupa los tracks por similitud rítmica y tímbrica con HDBSCAN jerárquico, y exporta playlists M3U listas para importar en Traktor DJ.

---

## Arquitectura

| Dónde | Qué hace |
|---|---|
| **HPC Surrey (GPU)** | Extracción de features (Phases 0–1) — tarda ~1h para ~250 tracks |
| **Login node / debug** | Clustering y export (Phases 2–5) — ~5 min, solo CPU |
| **Local Windows** | Visualización interactiva y uso de los playlists en Traktor DJ |

---

## Ver la UI desde tu ordenador Windows

La UI es un dashboard Streamlit que muestra el UMAP interactivo, los clusters y permite re-exportar playlists.

**1. Abre PowerShell y conéctate con túnel SSH:**

```powershell
ssh -L 8501:localhost:8501 datamove1
```

**2. En la sesión SSH, activa el entorno y arranca Streamlit:**

```bash
conda activate traktor_ml
cd /mnt/fast/nobackup/users/gb0048/traktor
streamlit run src/v4/ui/app.py --server.port 8501
```

**3. Abre en tu navegador Windows:**

```
http://localhost:8501
```

Mantén el PowerShell abierto mientras uses la UI.

---

## Pipeline completo (HPC)

### Phase 0 — Ingesta (login node, ~1 min)

```bash
python src/v4/pipeline/phase0_ingest.py --dataset-name test_20
```

### Phase 1 — Extracción de features (GPU, ~1h)

```bash
./slurm/tools/on_submit.sh sbatch slurm/jobs/v4/phase1_extract.job test_20
./slurm/tools/on_submit.sh sbatch slurm/jobs/v4/phase1_merge.job test_20
```

### Phases 2–5 — Clustering, naming, ordering, export (~5 min)

```bash
./slurm/tools/on_submit.sh sbatch slurm/jobs/v4/phase2_to_5.job test_20
```

Los playlists se generan en `playlists/V4_<N>/` con rutas Windows listas para Traktor.

---

## Estructura del repo

```
src/v4/pipeline/     # Scripts de cada phase (0–5)
src/v4/common/       # Utilidades compartidas
src/v4/ui/           # Dashboard Streamlit
slurm/jobs/v4/       # Jobs de Slurm
artifacts/v4/        # Artefactos generados (embeddings, clustering)
playlists/           # M3U exportados
config/v4.yaml       # Configuración principal
tests/v4/            # Tests de integración por bloques
docs/                # Documentación detallada
```

---

## Documentación

- [`docs/V4_USAGE.md`](docs/V4_USAGE.md) — Guía completa de uso
- [`docs/PROJECT_MAP.md`](docs/PROJECT_MAP.md) — Mapa de archivos
- [`docs/v4/JOBS_STATUS.md`](docs/v4/JOBS_STATUS.md) — Estado de los jobs y artefactos validados
- [`docs/LESSONS_LEARNED.md`](docs/LESSONS_LEARNED.md) — Lecciones aprendidas (HPC, modelos, bugs)
