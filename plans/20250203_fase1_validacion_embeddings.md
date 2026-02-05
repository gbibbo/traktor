# Plan: Fase 1 - Validación del Pipeline de Embeddings

**Fecha:** 2025-02-03
**Estado:** Implementado
**Autor:** Claude Code

## Objetivo

Validar el pipeline de extracción de embeddings con 20 tracks de prueba usando dos modelos Essentia:
1. **discogs-effnet-bs64** (EfficientNet, 18MB, 1280-dim)
2. **discogs-maest-30s-pw** (Transformer, 300MB, 768-dim)

## Componentes Implementados

### Scripts Python (`scripts/hpc/process/`)

| Script | Propósito |
|--------|-----------|
| `download_essentia_models.py` | Descarga modelos de essentia.upf.edu |
| `extract_embeddings.py` | Extrae embeddings con EffNet y MAEST |
| `reduce_and_cluster.py` | UMAP + HDBSCAN clustering |
| `generate_visualization.py` | Scatter plot interactivo con Plotly |

### Jobs Slurm (`slurm/jobs/`)

| Job | Partición | Recursos | Tiempo |
|-----|-----------|----------|--------|
| `fase1_download_models.job` | CPU | 2 CPU, 4GB | 1h |
| `fase1_extract_embeddings.job` | A100 | 1 GPU, 8 CPU, 32GB | 4h |
| `fase1_cluster_visualize.job` | CPU | 8 CPU, 16GB | 1h |

## Ejecución

```bash
# 1. Descargar modelos
./slurm/tools/on_submit.sh sbatch slurm/jobs/fase1_download_models.job

# 2. Extraer embeddings (esperar a que termine paso 1)
./slurm/tools/on_submit.sh sbatch slurm/jobs/fase1_extract_embeddings.job

# 3. Clustering y visualización (esperar a que termine paso 2)
./slurm/tools/on_submit.sh sbatch slurm/jobs/fase1_cluster_visualize.job
```

## Monitoreo

```bash
# Ver cola de trabajos
./slurm/tools/on_submit.sh squeue -u $USER

# Ver logs en tiempo real
tail -f logs/fase1_*.out
```

## Outputs Esperados

| Archivo | Ubicación |
|---------|-----------|
| `embeddings_effnet.npy` | `data/outputs/fase1_validation/` |
| `embeddings_maest.npy` | `data/outputs/fase1_validation/` |
| `results.csv` | `data/outputs/fase1_validation/` |
| `cluster_visualization.html` | `data/outputs/fase1_validation/` |

## Parámetros

- **Tracks:** 20 aleatorios (seed=42)
- **UMAP:** n_neighbors=15, min_dist=0.1, metric=cosine
- **HDBSCAN:** min_cluster_size=3, min_samples=2

## Dependencias

```
essentia-tensorflow
umap-learn
hdbscan
plotly
pandas
tqdm
numpy
```
