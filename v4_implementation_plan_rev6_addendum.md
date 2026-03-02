# TRAKTOR ML V4: Plan de Implementación (rev.6 — Addendum)
# Actualización post-implementación: PCA, sklearn HDBSCAN, noise reassignment, bug fixes

Este documento actualiza secciones específicas de `v4_implementation_plan.md` (rev.5).
Leer rev.5 como base; este addendum SUPERSEDE las secciones indicadas.

Fecha: 2026-03-02
Contexto: Bloques 0-5 ya implementados. Este addendum documenta los cambios realizados
durante la implementación y añade la feature de noise reassignment (1-NN).


---
---

# CAMBIOS TRANSVERSALES

## Dependencia hdbscan eliminada

El paquete pip `hdbscan` (que requiere compilación C y falla en nodos sin Python.h)
fue reemplazado por `sklearn.cluster.HDBSCAN` (disponible en scikit-learn >= 1.3).
La API es equivalente; el único cambio es `core_dist_n_jobs` → `n_jobs`.

Esto afecta:
  - `requirements_v4.txt`: línea `hdbscan` eliminada.
  - `src/v4/pipeline/phase2_cluster.py`: `import hdbscan` → `from sklearn.cluster import HDBSCAN`.
  - Todos los Slurm jobs que hacían `pip install hdbscan`: eliminado de la línea de pip.

## Curse of dimensionality: PCA antes de HDBSCAN

HDBSCAN no estima densidad correctamente en 1024 dimensiones. Se añade PCA como paso
intermedio entre L2-normalize y HDBSCAN. Retiene ~93.7% de varianza con 50 dims.

Pipeline de clustering actualizado:
```
embeddings → L2 normalize → PCA(n_components=pca_dim) → HDBSCAN → noise reassignment
```

## Noise reassignment (1-NN)

HDBSCAN es conservador: con N=239 tracks techno, ~51% queda como noise. Para DJing es
preferible asignar todos los tracks a un cluster. Se añade post-procesamiento 1-NN:
cada punto noise se asigna al cluster del punto no-noise más cercano (distancia euclidiana
en el espacio PCA-reducido). Esto es más robusto que asignación a centroide porque
respeta clusters no convexos.


---
---

# SECCIONES SUPERSEDIDAS


---

## TAREA 0.3 (parcial): config.py — Clustering defaults

SUPERSEDE la sección "Clustering defaults" de `src/v4/config.py` (rev.5 líneas 319-323).

```python
# === Clustering defaults ===
PCA_DIM = 50               # Dims PCA antes de HDBSCAN (0 = desactivado)
ASSIGN_NOISE = True         # Reasignar puntos noise al cluster vecino más cercano (1-NN)
L1_MIN_CLUSTER_SIZE = 6     # Tuned para test_20 (N=239): 8 clusters L1 con PCA=50
L1_MIN_SAMPLES = 1          # ms=1 necesario para densidad suficiente en N pequeño
L2_MIN_CLUSTER_SIZE = 4
L2_MIN_SAMPLES = 2
```

Nota: Los valores originales del plan (mcs=10, ms=3) producían 80%+ noise sin PCA
y 79% noise con PCA. Los valores tuneados (mcs=6, ms=1, pca=50) dan 8 clusters
con ~51% noise pre-reassignment, 0% post-reassignment.


---

## TAREA 0.3 (parcial): config/v4.yaml — Sección clustering

SUPERSEDE la sección `clustering:` de `config/v4.yaml` (rev.5 líneas 393-397).

```yaml
clustering:
  pca_dim: 50              # dims PCA antes de HDBSCAN (0 = sin PCA); ~93.7% var retained en 1024→50
  assign_noise: true       # reasignar puntos noise al cluster vecino más cercano (1-NN)
  l1_min_cluster_size: 6   # tuned para test_20 (N=239): 8 clusters L1, 0% noise con reassignment
  l1_min_samples: 1
  l2_min_cluster_size: 4
  l2_min_samples: 2
```


---

## TAREA 0.3 (parcial): requirements_v4.txt

SUPERSEDE `requirements_v4.txt` (rev.5 líneas 408-423).

```
transformers
torchaudio
scikit-learn
pandas
pyarrow
soundfile
tqdm
plotly
joblib
demucs
streamlit
umap-learn
pyyaml
```

Nota: `hdbscan` eliminado. Se usa `sklearn.cluster.HDBSCAN` (incluido en scikit-learn >= 1.3).


---

## TAREA 3.1: Phase 2 — Clustering jerárquico (REESCRITA)

SUPERSEDE TAREA 3.1 completa (rev.5 líneas 1068-1093).

Implementar `src/v4/pipeline/phase2_cluster.py`.

Entrada: `mert_perc.npy`, `mert_full.npy`, `catalog.parquet` (o `track_uids.json`).

Pipeline:
  Paso 1: L2-normalize ambas matrices.
  Paso 2: PCA sobre mert_perc normalizados (default 50 dims). Imprimir varianza retenida.
  Paso 3: L1 clustering con sklearn.cluster.HDBSCAN sobre embeddings PCA-reducidos.
  Paso 4: Noise reassignment L1 — 1-NN: cada punto noise se asigna al cluster del punto
           no-noise más cercano en el espacio PCA. Guardar labels originales como label_l1_raw.
  Paso 5: L2 clustering con HDBSCAN sobre mert_full (normalizado, opcionalmente PCA-reducido
           si el cluster L1 tiene >= 2*pca_dim tracks), dentro de cada cluster L1.
  Paso 6: Noise reassignment L2 — misma lógica 1-NN dentro de cada cluster L1.
           Guardar labels originales como label_l2_raw.
  Paso 7: UMAP 2D para visualización (sobre mert_perc PCA-reducido).
  Paso 8: Guardar `clustering/results_<hash>.parquet` y `clustering/config_<hash>.json`.

Funciones internas clave:

```python
def _apply_pca(X: np.ndarray, pca_dim: int, label: str = "") -> np.ndarray:
    """PCA con sklearn. Si pca_dim<=0 o pca_dim>=X.shape[1], retorna X sin cambios.
    Imprime: [INFO] PCA [label]: 1024→50 dims, 93.7% variance retained."""

def _hdbscan_cluster(X, min_cluster_size, min_samples) -> np.ndarray:
    """Wrapper de sklearn.cluster.HDBSCAN. Retorna labels (int array, -1=noise)."""

def _reassign_noise(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """1-NN reassignment. Para cada punto con label=-1, asigna el label del punto
    no-noise más cercano (distancia euclidiana). Usa sklearn.neighbors.NearestNeighbors.
    Si no hay noise o todo es noise, retorna sin cambios.
    Imprime: [INFO] Noise reassignment: N noise points → assigned to nearest cluster."""

def run_clustering(dataset_name, config, l1_min_cluster_size, l1_min_samples,
                   l2_min_cluster_size, l2_min_samples, config_tag,
                   skip_umap=False, pca_dim=0, assign_noise=True) -> Path:
    """Función principal. Retorna path al parquet de resultados."""
```

Columnas del parquet de salida:
  - track_uid, filename (identificación)
  - label_l1 (int, >= 0 si assign_noise=True, puede ser -1 si assign_noise=False)
  - label_l2 (int, >= 0 si assign_noise=True)
  - label_l1_raw (int, labels originales HDBSCAN, puede ser -1)
  - label_l2_raw (int, labels originales HDBSCAN L2, puede ser -1)
  - umap_x, umap_y (float, 0.0 si skip_umap=True)

Config hash: incluye pca_dim y assign_noise para que diferentes configuraciones
generen parquets distintos.

CLI:
```
python src/v4/pipeline/phase2_cluster.py \
    --dataset-name <nombre> \
    --l1-min-cluster-size 6 --l1-min-samples 1 \
    --l2-min-cluster-size 4 --l2-min-samples 2 \
    --pca-dim 50 \
    --assign-noise \
    --config-tag "baseline"
```

Flags:
  --pca-dim N         Dimensiones PCA antes de HDBSCAN (default: 50, 0 = desactivar)
  --assign-noise      Reasignar noise al cluster vecino más cercano (default)
  --no-assign-noise   Mantener noise como -1 (comportamiento original rev.5)

Verificación:
  Parquet tiene tantas filas como tracks.
  Cada track tiene label_l1 y label_l2 (>= 0 si assign_noise=True).
  Columnas label_l1_raw y label_l2_raw existen.
  UMAP coords finitas (si no skip_umap).
  Imprime estadísticas: N clusters L1, tamaños, noise original, tracks reasignados.


---

## TEST-3: Verificación del Bloque 3 (ACTUALIZADO)

SUPERSEDE TEST-3 Parte B (rev.5 líneas 1143-1167).

### Parte B: Clustering real sobre test_20 (automático, CPU)

4. Ejecutar Phase 2 sobre test_20 con parámetros de config (pca_dim=50, assign_noise=true,
   mcs=6, ms=1).
5. Verificar:
   - Parquet tiene N filas (N = len(track_uids.json)).
   - Con assign_noise=True: noise rate final == 0% (todos asignados).
   - Columna label_l1_raw existe y su noise rate es el valor pre-reassignment.
   - Clusters L1 >= 1 (assert duro).
   - Clusters L1 entre 3-15 y noise_raw < 60% (guía humana, no assert).
6. Ejecutar eval_runner (noise_rate + cluster stats).
7. Imprimir resumen de clustering para revisión humana, incluyendo:
   - N clusters L1 y tamaños
   - Noise original (de label_l1_raw) vs noise final (de label_l1)
   - N tracks reasignados por cluster

### Parte C: Revisión humana del clustering

### !! INTERVENCIÓN HUMANA REQUERIDA !!

El test imprime un reporte con:
```
=== CLUSTERING REPORT (para revisión humana) ===
Clusters L1: N, tamaños: [X, Y, Z, ...]
Noise rate original (HDBSCAN): X%
Noise rate final (post-reassignment): 0%
Tracks reasignados: M (por cluster: A+N1, B+N2, ...)
Sub-clusters L2 por grupo: A=[A1(n), A2(n)], B=[B1(n), B2(n)], ...
```

El humano debe:
  1. Revisar si el número de clusters L1 es razonable (3-10 para ~250 tracks).
  2. Revisar si los tamaños son balanceados (el mayor no sea >10x el menor).
  3. Revisar cuántos tracks fueron reasignados a cada cluster: si un cluster absorbió
     demasiado noise relativo a su tamaño original, puede ser señal de mala calidad.
  4. Si no es satisfactorio: re-ejecutar Phase 2 con otros parámetros.


---

## TAREA 4.3: Phase 5 — Export (NOTA DE BUG FIX)

NOTA sobre rev.5 TAREA 4.3 (líneas 1217-1237).

Bug encontrado y corregido: Cuando un cluster L1 tenía todos sus tracks con label_l2=-1
(todo noise en L2), el código los exportaba dos veces: una en All_Noise.m3u y otra
en una playlist L2 trivial. Fix: filtrar `df[mask_l1 & (df["label_l2"] >= 0)]` en el
path trivial y hacer `continue` si está vacío.

Con assign_noise=True en L2, este bug ya no se manifiesta (no hay label_l2=-1),
pero el fix permanece como protección defensiva.

Estructura de export actualizada (con assign_noise=True):
```
playlists/V4_<N>/
  L1_A_[nombre]/
    L2_A1_[nombre].m3u
    L2_A2_[nombre].m3u
  L1_B_[nombre]/
    L2_B1_[nombre].m3u
  ...
  _summary.txt
```

Nota: Con assign_noise=True, `All_Noise.m3u` no se genera (o queda vacío) porque
no hay tracks con label_l1=-1. Esto es correcto.


---

## TAREA 5.1: UI Streamlit (NOTAS DE ACTUALIZACIÓN)

NOTAS sobre rev.5 TAREA 5.1 (líneas 1289-1303).

Ajustes realizados al plan original:
  - load_data() debe seleccionar el parquet más reciente que tenga coordenadas UMAP
    reales (umap_x.abs().sum() > 0), no simplemente el más reciente por mtime.
  - El botón Re-cluster debe pasar pca_dim (leído de config) y assign_noise=True
    a la invocación de phase2_cluster.
  - Si existe la columna label_l1_raw, mostrar métrica adicional "Noise original: X%"
    junto al noise rate actual.
  - El filtro "Mostrar noise" queda deshabilitado/oculto cuando assign_noise=True
    (no hay noise que filtrar).


---

## Slurm jobs (NOTAS)

NOTA sobre TAREA 2.4 (rev.5 líneas 974-1006).

Cambio en `slurm/jobs/v4/phase2_to_5.job`:
  - Eliminado `hdbscan` de la línea de pip install.
  - Los scripts Python usan `sklearn.cluster.HDBSCAN` que viene con scikit-learn.

Cambio en todos los jobs que referencian pip install:
  - Si existía `hdbscan` en la línea de pip, eliminarlo.


---

## Bug fix: phase1_merge_shards.py

NOTA sobre TAREA 2.3 (rev.5 líneas 952-969).

Bug encontrado: `pd.DataFrame(success_rows).reset_index()` producía una columna
llamada `index` en vez de `track_uid`. Fix: añadir `.rename(columns={"index": "track_uid"})`.

Esto afectaba a `catalog_success.parquet` y causaba fallo en test_block5.


---
---

# LECCIONES APRENDIDAS (para futura referencia)

1. HDBSCAN en alta dimensionalidad: No funciona bien en 1024D. PCA a 50D es esencial.
   La varianza retenida (~93.7%) confirma que las dimensiones eliminadas eran ruido.

2. sklearn vs paquete hdbscan: sklearn.cluster.HDBSCAN (desde v1.3) es drop-in replacement
   y no requiere compilación C. Preferir siempre el de sklearn en entornos HPC donde
   las herramientas de build pueden no estar disponibles.

3. Noise en HDBSCAN: Para datasets pequeños (~250 tracks) de música similar (todo techno),
   HDBSCAN es muy conservador. La reasignación 1-NN post-hoc es una solución práctica
   que preserva la estructura de clusters descubierta por HDBSCAN sin descartar tracks.

4. UMAP solo para visualización: No usar UMAP como paso de pre-clustering (V2 lo hacía
   y distorsionaba distancias globales). UMAP 2D es solo para el scatter plot de Streamlit.

5. Config hashing: Incluir TODOS los parámetros que afectan el output en el hash
   (pca_dim, assign_noise, etc.) para evitar colisiones de parquet entre configs diferentes.

6. Parquet stale en UI: La UI debe verificar que las columnas UMAP contienen valores
   reales (no todos 0.0) antes de usarlas para el scatter plot.
