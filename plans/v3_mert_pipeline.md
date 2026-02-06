# V3: MERT Embeddings & Modular Clustering Pipeline

## 0. Stack Tecnológico
* **Embeddings:** MERT-v1-330M (24kHz, layer combination)
* **Dimensionality:** PCA a 128D
* **Clustering:** Híbrido (Plano K-elegible / HDBSCAN y Jerárquico de 2 niveles)
* **UI:** Streamlit para exploración interactiva

## 1. Diseño de Arquitectura (Pipeline Desacoplado)

### A) Preprocesamiento (Costoso, Run-Once)
**Inputs:** Audio files
**Outputs (Artifacts en `artifacts/dataset/<id>/`):**
1. `manifest.parquet`: Metadata, checksums, paths.
2. `embeddings_mert.npy`: Matriz [N, 1024].
3. `X_pca128.npy`: Matriz [N, 128] (Features finales para clustering).
4. `pca_128.joblib`: Objeto PCA entrenado.

### B) Clustering (Barato, Interactivo)
**Inputs:** `X_pca128.npy` + `manifest.parquet`
**Outputs:** CSVs con labels, configs y métricas.
**Constraint:** El clustering NUNCA recalcula embeddings.

## 2. Estructura de Directorios (Obligatoria)
```text
traktor_ml_v3/
  src/
    preprocess/
      build_manifest.py       # Scan, checksums, audio validation
      extract_embeddings.py   # MERT inference (GPU)
      fit_pca.py             # Normalization + PCA
    clustering/
      interface.py           # Base abstract class
      flat.py                # KMeans, Agglomerative, HDBSCAN
      hierarchical.py        # Two-stage logic
    ui/
      app.py                 # Streamlit dashboard
  artifacts/                 # Storage for generated .npy/.parquet
  configs/                   # YAMLs for defaults
  slurm/
    jobs/v3/                 # New job templates
```

## 3. Detalles de Implementación Críticos

**MERT Config:** Resample a 24kHz. Pooling de segmentos de 5s (padding start/end 45s).

**PCA:** Entrenar sobre todo el dataset. Output fijo a 128 dimensiones.

**Jerarquía:**
- Nivel 1: Clusters macro (Groove/Energy).
- Nivel 2: Sub-clusters (Timbre/Elements).
