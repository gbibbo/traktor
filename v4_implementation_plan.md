# TRAKTOR ML V4: Plan de Implementación (rev.5)
# Prompt autocontenido para Claude Code — Ejecución incremental con trazabilidad (robustez HPC + ingest por manifests)

## Contexto del proyecto

Este es un sistema para organizar una librería de música electrónica (Techno/Tech House, ~250 tracks expandible a ~2000) en una jerarquía de dos niveles optimizada para DJing. El sistema agrupa tracks primero por groove/percusión (L1) y luego por vibe/timbre/armonía (L2) dentro de cada grupo, genera nombres legibles para cada grupo, ordena los tracks dentro de cada grupo para facilitar transiciones suaves, y exporta playlists compatibles con Traktor DJ.

El proyecto está en su cuarta iteración. Las versiones anteriores (V1, V2, V3) están en el repositorio y deben moverse a `legacy/`. V4 se construye desde cero en `src/v4/`, reutilizando funciones probadas de versiones anteriores cuando sea apropiado.

## Entorno de ejecución

El proyecto se ejecuta en el HPC de la University of Surrey.

Acceso a Slurm: los comandos de Slurm NO están en PATH. Se debe usar el wrapper `./slurm/tools/on_submit.sh <squeue|sbatch|scancel|...> <args>`, que hace SSH a `aisurrey-submit01.surrey.ac.uk`. Consultar `CLAUDE.md` en la raíz del repositorio y el template en `slurm/jobs/` para detalles sobre cómo crear y enviar jobs.

Python: Sistema Python 3.11 en `/usr/bin/python3.11`. NO usar conda. Las dependencias se instalan con `pip install --user` redirigiendo PYTHONUSERBASE a scratch4weeks para evitar llenar HOME.

GPU: Partición `a100` para jobs con GPU. Partición `debug` o `cpu` para jobs sin GPU.

Cache de modelos (recomendado):
  - export `TRAKTOR_CACHE_ROOT=/mnt/fast/nobackup/scratch4weeks/$USER/cache`
  - export `TRAKTOR_HF_CACHE=$TRAKTOR_CACHE_ROOT/hf` (o `HF_HOME`/`TRANSFORMERS_CACHE`)
  - export `TRAKTOR_TORCH_CACHE=$TRAKTOR_CACHE_ROOT/torch` (o `TORCH_HOME`)
  Esto evita saturar HOME y acelera re-runs.

El repositorio raíz es `/mnt/fast/nobackup/users/gb0048/traktor`.

IMPORTANTE: Toda inferencia de modelos (MERT, Demucs, CLAP) y cualquier operación que requiera GPU debe ejecutarse mediante Slurm jobs, nunca directamente en el nodo de login. Las verificaciones locales de estos módulos se limitan a tests de importación y validación de sintaxis, no a inferencia real.

## Contrato de sample rates (CRITICO, no modificar)

Cada modelo y herramienta requiere un sample rate específico. Violar esto produce resultados silenciosamente incorrectos.

MERT-v1-330M: 24000 Hz (documentado en la model card de HuggingFace).
Essentia RhythmExtractor2013: 44100 Hz (documentado en la referencia de Essentia).
Essentia KeyExtractor: 44100 Hz (documentado en la referencia de Essentia).
Demucs (htdemucs): 44100 Hz (sample rate nativo del modelo).

El pipeline debe cargar el audio UNA sola vez a 44100 Hz y derivar dos versiones en memoria:
  audio_44k: para Essentia (BPM, key) y Demucs (stem separation).
  audio_24k: resampleado desde audio_44k, para MERT.

NUNCA resamplear a 16000 Hz en V4. Esa constante era de V1/V2 para Essentia EffNet y no aplica aquí.

## Contrato de track_id (CRITICO, no modificar)

El identificador de cada track debe ser estable ante cambios de ruta (portabilidad HPC a laptop, reorganización de carpetas, merges entre datasets).

track_uid = SHA256 de los primeros 1MB del archivo + filesize_bytes.

Esto es rápido (no lee el archivo completo) y suficientemente robusto para detección de duplicados y portabilidad. Guardar siempre junto al track_uid: source_path (ruta absoluta actual), filename, filesize_bytes.

## Código existente reutilizable

Estas funciones de versiones anteriores están probadas y pueden adaptarse para V4. Las rutas son relativas al repositorio raíz. Al reutilizarlas, copiar y adaptar (no importar de legacy).

Audio loading y conversión (de `legacy/v2/scripts/common/audio_utils.py`):
  `get_audio_files(audio_dir)` para escanear directorios de audio.
  `load_audio_torch(audio_path)` para cargar con torchaudio.
  `torch_to_essentia(waveform, source_sr, target_sr)` para convertir torch tensor a numpy.
  `validate_audio_file(audio_path)` para validar archivos.

Demucs stem separation (de `legacy/v2/scripts/common/demucs_utils.py`):
  `load_demucs_model(model_name, device)` para cargar htdemucs.
  `load_audio_for_demucs(audio_path, target_sr)` para preparar audio para Demucs.
  `separate_stems(model, waveform, device)` para separar en drums/bass/vocals/other.
  `stem_to_mono_numpy(stem, target_sr, source_sr)` para convertir stem a mono numpy.
  `process_track_stems(audio_path, model, model_sr, device)` pipeline completo. NOTA: V4 debe cambiar target_sr de 16000 a 24000 para MERT.

Clustering utilities (de `legacy/v2/scripts/common/clustering_utils.py`):
  `l2_normalize(embeddings)` para normalización L2.
  `apply_umap(embeddings, ...)` para reducción dimensional (solo visualización en V4).
  `apply_hdbscan(embeddings, ...)` para clustering. NOTA: V4 debe modificar esta función para clusterizar en espacio de embeddings completo (no sobre UMAP 2D como hacía V2).
  `cluster_to_letter(cluster_id)`, `subcluster_label(parent, sub_id)` para naming.
  `get_cluster_stats(labels)` para estadísticas.
  `simplify_genre_name(genre)` para limpiar nombres de género.

Playlist generation (de `generate_playlists.py` en raíz):
  `generate_m3u_playlist(tracks, playlist_name, local_audio_dir, output_path)` para generar M3U.

Slurm job templates: Usar el patrón de `slurm/jobs/v3/extract_embeddings.job` como base para nuevos jobs (setup de scratch4weeks, bootstrap de pip, verificación de imports, etc.). Consultar `CLAUDE.md` para instrucciones completas sobre el wrapper de Slurm.

## Estructura de directorios de V4

```
traktor/
  config/
    v4.yaml                      # Config de rutas, datasets, hiperparámetros
  src/v4/
    __init__.py
    config.py                    # Constantes hardcoded (SRs, dims, defaults)
    common/
      __init__.py
      audio_utils.py             # Carga, validación, segmentación de audio
      demucs_utils.py            # Separación de stems (adaptado de V2)
      embedding_utils.py         # Extracción MERT
      catalog.py                 # Catálogo Parquet + merge metadata externa
      config_loader.py           # Carga config YAML + env overrides
      path_resolver.py           # Resolución de rutas HPC/local
    pipeline/
      __init__.py
      phase0_ingest.py           # Escaneo + validación + catálogo + metadata merge
      phase1_extract.py          # GPU: Demucs + MERT + Essentia (con sharding)
      phase1_merge_shards.py     # CPU: Consolida shards de phase1
      phase2_cluster.py          # CPU: Clustering jerárquico L1/L2
      phase3_name.py             # CPU: Naming semántico
      phase4_order.py            # CPU: Ordenamiento intra-cluster
      phase5_export.py           # CPU: Generación de playlists M3U
    evaluation/
      __init__.py
      metrics.py                 # ARI, NMI, Recall@k, MRR, NDCG, transition score
      eval_runner.py             # Loop de evaluación automatizada
    adaptation/
      __init__.py
      projection_head.py         # MLP projection head (frozen backbone + head)
      contrastive_trainer.py     # Entrenamiento contrastivo (hooks, no impl completa)
    ui/
      app.py                     # Streamlit dashboard
  tests/v4/
    test_block0_setup.sh         # Verificación estructural
    test_block1_common.py        # Integration test de common utilities
    test_block2_pipeline.py      # Verificación de pipeline scripts
    test_block3_clustering.py    # Verificación post-clustering
    test_block4_export.py        # Verificación de export pipeline
    test_block5_system.py        # Verificación end-to-end
  slurm/jobs/v4/
    phase1_extract.job           # GPU job (single o con --shard-id)
    phase1_extract_array.job     # GPU array job para sharding
    phase1_merge.job             # CPU job para merge shards
    phase2_to_5.job              # CPU job (fases 2-5)
    eval_sweep.job               # CPU job para ablation sweeps
    smoke_test_gpu.job           # GPU job: test rápido de MERT+Demucs con 3 tracks
  artifacts/v4/
    datasets/<dataset_name>/
      run_manifest.json          # metadata de corrida (config hash, git, host, slurm, versiones)
      ingest_report.json         # stats Phase 0 (ok/fail, fuentes, etc.)
      catalog.parquet            # single source of truth del dataset
      logs/
        phase0_*.jsonl
        phase1_*.jsonl
        phase2_*.jsonl
        phase3_*.jsonl
        phase4_*.jsonl
        phase5_*.jsonl
      download_cache/            # opcional, si se usa manifest http (puede estar fuera via config)
      embeddings/
        mert_perc.npy            # (N, 1024) final consolidado
        mert_full.npy            # (N, 1024) final consolidado
        shards/                  # temporales durante extracción
      features/
        bpm_key.parquet          # incluye bpm_confidence, beat_confidence, key_confidence
      clustering/
        results_<config_hash>.parquet
      evaluation/
        scores_<config_hash>.json
  evaluation_data/
    dev_set.csv                  # [HUMANO CREA] Labels manuales L1/L2
    dj_pairs_<n>.csv             # Pares adyacentes de DJ sets
  playlists/V4_<N>/
  docs/v4/
    TODO.md                      # Tracker de progreso (creado por Tarea 0.1)
```


---
---

# SISTEMA DE TRAZABILIDAD

Al completar cada tarea, Claude Code debe:
  1. Editar `docs/v4/TODO.md`: marcar la tarea con `[x]` y agregar la fecha.
  2. Hacer commit con mensaje descriptivo: `v4: complete task X.Y - <descripción breve>`.

El archivo `TODO.md` es la fuente de verdad sobre el progreso del proyecto.


---
---

# TAREAS DE IMPLEMENTACIÓN (ejecutar en orden)

Las tareas están organizadas en BLOQUES. Cada bloque agrupa tareas relacionadas.
Al final de cada bloque hay una tarea de TEST que valida el bloque completo.
NO avanzar al siguiente bloque sin pasar el test del bloque actual.

Cada tarea individual tiene su propia verificación rápida.
Los tests de bloque son más exhaustivos e integran todo lo construido en ese bloque.

---
---

# BLOQUE 0: Setup y organización del repositorio

---

## TAREA 0.1: Lectura del plan y creación del tracker de progreso

Esta es la primera tarea. Antes de escribir cualquier código:

1. Leer este archivo completo (`v4_implementation_plan_rev4.md`) de principio a fin.
2. Leer `CLAUDE.md` en la raíz del repositorio para entender el entorno de ejecución y las instrucciones de Slurm.
3. Leer el template de Slurm job existente en `slurm/jobs/` para entender el patrón.
4. Crear el directorio `docs/v4/` si no existe.
5. Crear el archivo `docs/v4/TODO.md` con el contenido definido abajo.

### Contenido de `docs/v4/TODO.md`:

```markdown
# TRAKTOR ML V4 — Progress Tracker

Claude Code: al completar cada tarea, marcar [x] y agregar fecha de finalización.
Formato: - [x] Tarea X.Y — Descripción | Completado: YYYY-MM-DD

---

## BLOQUE 0: Setup y organización
- [ ] 0.1 Leer plan completo + crear este archivo TODO.md          | Completado: ____
- [ ] 0.2 Mover V3 a legacy                                        | Completado: ____
- [ ] 0.3 Crear estructura V4 + config.py + v4.yaml + requirements | Completado: ____
- [ ] TEST-0 Verificación de bloque 0                              | Completado: ____

## BLOQUE 1: Common utilities
- [ ] 1.1 config_loader.py + path_resolver.py                      | Completado: ____
- [ ] 1.2 catalog.py                                               | Completado: ____
- [ ] 1.3 audio_utils.py (carga + segmentación DJ)                 | Completado: ____
- [ ] 1.4 demucs_utils.py                                          | Completado: ____
- [ ] 1.5 embedding_utils.py (MERTEmbedder)                        | Completado: ____
- [ ] 1.6 logging_utils.py (JSONL + run manifests)                 | Completado: ____
- [ ] TEST-1 Verificación de bloque 1 (integration test)           | Completado: ____

## BLOQUE 2: Pipeline scripts + Slurm
- [ ] 2.1 phase0_ingest.py                                         | Completado: ____
- [ ] 2.2 phase1_extract.py                                        | Completado: ____
- [ ] 2.3 phase1_merge_shards.py                                   | Completado: ____
- [ ] 2.4 Slurm jobs V4 (todos)                                    | Completado: ____
- [ ] TEST-2 Verificación de bloque 2 (Phase 0 run + validaciones) | Completado: ____

## >>> PAUSA HUMANA: ejecutar Phase 0, submit Phase 1 GPU, revisar embeddings <<<

## BLOQUE 3: Clustering + evaluación
- [ ] 3.1 phase2_cluster.py                                        | Completado: ____
- [ ] 3.2 metrics.py + eval_runner.py                              | Completado: ____
- [ ] TEST-3 Verificación de bloque 3 (clustering + eval stats)    | Completado: ____

## >>> PAUSA HUMANA: revisar clustering, ajustar hiperparámetros <<<

## BLOQUE 4: Export pipeline
- [ ] 4.1 phase3_name.py                                           | Completado: ____
- [ ] 4.2 phase4_order.py                                          | Completado: ____
- [ ] 4.3 phase5_export.py                                         | Completado: ____
- [ ] TEST-4 Verificación de bloque 4 (playlists + human review)   | Completado: ____

## BLOQUE 5: UI + finalización
- [ ] 5.1 UI Streamlit                                             | Completado: ____
- [ ] 5.2 Adaptation stubs (projection_head + contrastive_trainer) | Completado: ____
- [ ] 5.3 Integración end-to-end + documentación                   | Completado: ____
- [ ] TEST-5 Verificación final del sistema                        | Completado: ____
```

Verificación de Tarea 0.1:
  `docs/v4/TODO.md` existe y es legible.
  Actualizar la tarea 0.1 en el TODO como completada con la fecha de hoy.


---

## TAREA 0.2: Mover V3 a legacy

Crear directorio `legacy/v3/` y mover allí:
  `src/` (el directorio src actual completo) a `legacy/v3/src/`
  `slurm/jobs/v3/` a `legacy/v3/slurm/jobs/v3/`
  `plans/v3_mert_pipeline.md` a `legacy/v3/plans/`
  `requirements_v3.txt` a `legacy/v3/`
  `generate_playlists.py` a `legacy/v3/`

Verificación:
  `legacy/v1/`, `legacy/v2/`, `legacy/v3/` existen con el código correspondiente.
  `src/` ya no contiene código V3.


---

## TAREA 0.3: Crear estructura de directorios V4 y archivos de configuración

Crear toda la estructura de directorios de V4 según el árbol de arriba. Crear `__init__.py` vacíos donde corresponda.

Crear `src/v4/config.py`:

```python
"""
PURPOSE: Constantes centrales de TRAKTOR ML V4.
         Los sample rates son contratos de los modelos upstream. NO modificar.
CHANGELOG:
  - 2026-XX-XX: Creación inicial V4.
"""
from pathlib import Path

# === Rutas base ===
REPO_ROOT = Path(__file__).resolve().parents[2]  # traktor/
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "v4.yaml"

# === Sample rates (contratos de modelos, NO cambiar) ===
MERT_SAMPLE_RATE = 24000       # MERT-v1-330M espera 24kHz
ESSENTIA_SAMPLE_RATE = 44100   # RhythmExtractor2013 y KeyExtractor requieren 44.1kHz
DEMUCS_SAMPLE_RATE = 44100     # htdemucs opera a 44.1kHz

# === Segmentación (defaults; overrideable via config.v4.yaml > segmentation.*) ===
SEGMENT_DURATION_S = 5.0       # Duración de cada ventana en segundos
SEGMENT_DURATION_BARS = 16     # Alternativa: duración en barras musicales (si hay beat tracking)
N_INTRO_SEGMENTS = 1
N_MID_SEGMENTS = 2
N_OUTRO_SEGMENTS = 1

# === Modelos ===
MERT_MODEL_NAME = "m-a-p/MERT-v1-330M"
MERT_EMBEDDING_DIM = 1024
DEMUCS_MODEL_NAME = "htdemucs"

# === Clustering defaults ===
L1_MIN_CLUSTER_SIZE = 10
L1_MIN_SAMPLES = 3
L2_MIN_CLUSTER_SIZE = 4
L2_MIN_SAMPLES = 2

# === Evaluación ===
RETRIEVAL_K_VALUES = [5, 10, 20]

# === Track ID ===
TRACK_UID_BYTES_TO_READ = 1_048_576  # 1MB para hash parcial

# === Ordering weights (defaults, override desde v4.yaml) ===
ORDERING_WEIGHTS = {"embedding": 0.5, "bpm": 0.3, "key": 0.2}

# === CLAP naming vocabulary (techno/tech house oriented) ===
CLAP_DESCRIPTORS = [
    "dark rolling techno", "melodic progressive techno", "acid techno",
    "minimal deep techno", "hard industrial techno", "tribal percussive techno",
    "atmospheric ambient techno", "peak time techno", "deep hypnotic techno",
    "raw warehouse techno", "dub techno", "Detroit techno",
    "groovy tech house", "funky tech house", "deep tech house",
    "minimal tech house", "vocal tech house", "jackin tech house",
    "afro tech house", "organic house", "progressive house",
    "breaks and electro", "downtempo electronica",
]
```

Crear `config/v4.yaml`:

```yaml
# TRAKTOR ML V4 - Configuración de rutas y datasets
# Valores por defecto para HPC (y laptop). Override con --config o env vars.
# Precedencia de rutas: CLI override > env var > YAML > defaults.

paths:
  artifacts_root: null        # env: TRAKTOR_ARTIFACTS_ROOT
  audio_roots:                # env: TRAKTOR_AUDIO_ROOTS (separado por ":")
    - null                    # null => REPO_ROOT/data/raw_audio/
  hf_cache: null              # env: TRAKTOR_HF_CACHE o TRAKTOR_CACHE_ROOT/hf
  torch_cache: null           # env: TRAKTOR_TORCH_CACHE o TRAKTOR_CACHE_ROOT/torch
  download_cache: null        # (opcional) donde guardar descargas HTTP antes de mover a audio_root
  local_windows_audio_dir: "C:\Música\2020 new - copia"  # para export m3u en Windows

hashing:
  mode: "full"                # "full" (default robusto) o "fast"
  fast_bytes_to_read: 1048576 # 1MB (solo aplica si mode="fast")

segmentation:
  mode: "auto"                # "auto" | "bars" | "seconds"
  beat_conf_threshold: 0.5    # si beat_confidence < threshold => fallback a seconds
  segment_duration_s: 5.0
  segment_duration_bars: 16
  n_intro_segments: 1
  n_mid_segments: 2
  n_outro_segments: 1

datasets:
  # Cada dataset puede tener config específica.
  # Recomendado: definir expected_n para tests reproducibles en test_20.
  #
  # ejemplo:
  #   test_20:
  #     audio_root: "/path/to/test_20/audio"
  #     metadata_csv: "/path/to/beatport_export.csv"
  #     manifest_csv: null
  #     expected_n: 242
  #
  #   full_2000:
  #     audio_root: "/path/to/full_2000/audio"
  #     metadata_csv: "/path/to/beatport_export.csv"
  #     manifest_csv: "/path/to/manifest_full_2000.csv"
  #     expected_n: null

clustering:
  l1_min_cluster_size: 10
  l1_min_samples: 3
  l2_min_cluster_size: 4
  l2_min_samples: 2

ordering:
  weights:
    embedding: 0.5
    bpm: 0.3
    key: 0.2
```

Crear `requirements_v4.txt`:

```
transformers
torchaudio
scikit-learn
hdbscan
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

Actualizar `CLAUDE.md` en la raíz para reflejar que V4 es la versión activa y que el código vive en `src/v4/`.

Verificación:
  `python -c "from src.v4.config import *; print('OK')"` (desde repo root).
  `python -c "import yaml; yaml.safe_load(open('config/v4.yaml')); print('OK')"`.
  El directorio `src/v4/` tiene toda la estructura de subdirectorios con `__init__.py`.


---

## TEST-0: Verificación del Bloque 0

Crear y ejecutar `tests/v4/test_block0_setup.sh`:

```bash
#!/bin/bash
# Test de bloque 0: verificación estructural del repositorio V4
set -e
ERRORS=0

echo "=== TEST BLOCK 0: Setup y organización ==="

# 1. Legacy dirs exist
for dir in legacy/v1 legacy/v2 legacy/v3; do
    if [ -d "$dir" ]; then echo "  OK: $dir exists"; else echo "  FAIL: $dir missing"; ERRORS=$((ERRORS+1)); fi
done

# 2. V4 structure exists
for dir in src/v4/common src/v4/pipeline src/v4/evaluation src/v4/adaptation src/v4/ui tests/v4 slurm/jobs/v4 config; do
    if [ -d "$dir" ]; then echo "  OK: $dir exists"; else echo "  FAIL: $dir missing"; ERRORS=$((ERRORS+1)); fi
done

# 3. Config imports
python3 -c "from src.v4.config import MERT_SAMPLE_RATE, REPO_ROOT; assert MERT_SAMPLE_RATE == 24000; print('  OK: config.py imports')" || { echo "  FAIL: config.py"; ERRORS=$((ERRORS+1)); }

# 4. YAML parseable
python3 -c "import yaml; c=yaml.safe_load(open('config/v4.yaml')); assert 'paths' in c; assert 'clustering' in c; print('  OK: v4.yaml parseable')" || { echo "  FAIL: v4.yaml"; ERRORS=$((ERRORS+1)); }

# 5. No V3 code in src/ (only V4)
if [ -d "src/v4" ] && [ ! -f "src/preprocess" ] 2>/dev/null; then echo "  OK: src/ is clean (V4 only)"; else echo "  WARN: src/ may still have V3 code"; fi

# 6. TODO.md exists
if [ -f "docs/v4/TODO.md" ]; then echo "  OK: TODO.md exists"; else echo "  FAIL: TODO.md missing"; ERRORS=$((ERRORS+1)); fi

echo ""
if [ $ERRORS -eq 0 ]; then echo "BLOCK 0: ALL TESTS PASSED"; else echo "BLOCK 0: $ERRORS TESTS FAILED"; exit 1; fi
```

Ejecutar: `bash tests/v4/test_block0_setup.sh`

Todos los checks deben pasar. Si alguno falla, corregir antes de continuar.


---
---

# BLOQUE 1: Common utilities

---

## TAREA 1.1: Config loader + path resolver

Crear `src/v4/common/config_loader.py`:

```python
"""
PURPOSE: Cargar configuración V4 con cascada: CLI --config > env TRAKTOR_CONFIG > config/v4.yaml.
         Rutas pueden overridearse también con env vars (ver path_resolver).
"""

def load_config(config_path: Optional[Path] = None) -> dict:
    """
    Orden de precedencia:
      1. config_path (argumento explícito)
      2. Variable de entorno TRAKTOR_CONFIG
      3. REPO_ROOT/config/v4.yaml

    Importante: además del YAML, las rutas pueden overridearse con env vars:
      - TRAKTOR_ARTIFACTS_ROOT
      - TRAKTOR_AUDIO_ROOTS (lista separada por ":")
      - TRAKTOR_CACHE_ROOT (root para caches) o TRAKTOR_HF_CACHE / TRAKTOR_TORCH_CACHE
    La precedencia de rutas es: CLI override > env var > YAML > defaults en config.py.

    Dentro del YAML, strings con ${VAR} se expanden con os.environ.
    Valores null se resuelven a sus defaults en config.py.
    """
```

Crear `src/v4/common/path_resolver.py`:

```python
"""
PURPOSE: Resolver rutas de audio, artifacts, caches y logs de forma portátil (laptop + HPC).
"""

def resolve_artifacts_root(config: dict) -> Path:
    """Precedencia: env TRAKTOR_ARTIFACTS_ROOT > config.paths.artifacts_root > REPO_ROOT/artifacts/v4/datasets/."""

def resolve_dataset_audio_root(dataset_name: str, config: dict, cli_override: Optional[str] = None) -> Path:
    """
    Busca en orden:
      1. cli_override
      2. config.datasets[dataset_name].audio_root
      3. Cada directorio en env TRAKTOR_AUDIO_ROOTS (separado por ":") + /<dataset_name>/
      4. Cada directorio en config.paths.audio_roots + /<dataset_name>/
      5. REPO_ROOT/data/raw_audio/<dataset_name>/
    Si no lo encuentra, lanza FileNotFoundError con mensaje que lista rutas probadas.
    """

def resolve_dataset_metadata(dataset_name: str, config: dict) -> Optional[Path]:
    """Buscar metadata CSV externa. Retorna None si no hay."""


def resolve_dataset_manifest(dataset_name: str, config: dict) -> Optional[Path]:
    """Si existe, retorna ruta a manifest CSV (local/http)."""

def resolve_hf_cache(config: dict) -> Optional[Path]:
    """Precedencia: env TRAKTOR_HF_CACHE > env TRAKTOR_CACHE_ROOT/hf > config.paths.hf_cache > None."""

def resolve_torch_cache(config: dict) -> Optional[Path]:
    """Precedencia: env TRAKTOR_TORCH_CACHE > env TRAKTOR_CACHE_ROOT/torch > config.paths.torch_cache > None."""

def resolve_logs_root(config: dict) -> Path:
    """Directorio para logs JSONL. Por defecto: <artifacts_root>/<dataset>/logs/"""
```

Verificación:
  Importar ambos módulos sin errores.
  `load_config()` carga v4.yaml correctamente.
  `resolve_dataset_audio_root("test_20", config)` encuentra `data/raw_audio/test_20/` (si existe) o lanza error informativo.


---

## TAREA 1.2: Catálogo

Crear `src/v4/common/catalog.py`:

```python
"""
PURPOSE: Catálogo central de dataset. Single source of truth para metadata.
"""

def compute_track_uid(filepath: Path, bytes_to_read: int = 1_048_576) -> str:
    """Hash estable de contenido.
    Default (robusto): SHA256 streaming de TODO el archivo (hex de 64 chars).
    Modo rápido (opcional): SHA256 de (primeros bytes_to_read bytes + filesize_bytes) si config.hashing.mode == "fast".
    NO truncar el hash: guardar 64 chars para evitar colisiones.
    """

def build_catalog(audio_dir: Path, dataset_name: str, config: dict) -> pd.DataFrame:
    """
    Escanear directorio y construir catálogo.
    Columnas mínimas: track_uid, filename, source_path, duration_s, filesize_bytes.
    Metadata: artist, title, beatport_genre_raw/norm (si existe), label, year.
    Proveniencia (si manifest): source_type, source_uri, sha256 (si existe).
    Si hay metadata CSV: merge por filename normalizado.
    Guarda en artifacts/v4/datasets/<dataset_name>/catalog.parquet.
    """

def load_catalog(dataset_name: str, config: dict) -> pd.DataFrame:
    """Cargar catálogo existente."""

def update_catalog_columns(dataset_name: str, config: dict, updates: pd.DataFrame):
    """Agregar/actualizar columnas. updates debe tener track_uid para join."""
```

Para el merge de metadata externa:
  Intentar join por `track_uid` primero, luego por `filename` normalizado, luego por heurística `artist + title`.
  Loggear cuántos matchearon.

Verificación:
  `compute_track_uid()` da mismo hash para mismo archivo, distinto para distintos. Testar con 3 archivos de test_20.
  `build_catalog()` sobre test_20 genera DataFrame con N>0 filas y track_uids únicos.
  Si config.datasets.test_20.expected_n está definido, verificar que N == expected_n.
  `load_catalog()` recupera el mismo DataFrame.


---

## TAREA 1.3: Audio utils (carga + segmentación)

Crear `src/v4/common/audio_utils.py`. Adaptar de `legacy/v2/scripts/common/audio_utils.py`.

Cambios respecto a V2: Eliminar constante ESSENTIA_SAMPLE_RATE = 16000. Importar SRs de config.py.

Mantener de V2: `get_audio_files()`, `load_audio_torch()`, `torch_to_essentia()`, `validate_audio_file()`.

Agregar:

```python
def get_dj_segments(
    audio: np.ndarray, sr: int,
    segment_duration_s: float = 5.0,
    n_intro: int = 1, n_mid: int = 2, n_outro: int = 1,
    beat_ticks: Optional[np.ndarray] = None,
    bpm: Optional[float] = None,
    bars_per_segment: int = 16,
) -> List[np.ndarray]:
    """
    Modo 1 (beat-aware, si beat_ticks y bpm proporcionados con confianza):
      Barra = 4 beats. Segmento = bars_per_segment barras.
      Intro: primeros bars_per_segment barras. Mid: barras centrales. Outro: últimas barras.
      Truncar/pad a segment_duration_s * sr muestras.
      Fallback a Modo 2 si confianza < 0.5 o beat_ticks < 32.

    Modo 2 (fallback porcentaje):
      Intro: 0%-15%. Mid: 35%-65%. Outro: 85%-100%.
      Segmentos equiespaciados de segment_duration_s dentro de cada zona.

    Returns: Lista de arrays, cada uno exactamente int(segment_duration_s * sr) muestras.
    """
```

Verificación:
  `get_audio_files()` encuentra N archivos en test_20 (N>0). Si config.datasets.test_20.expected_n está definido, verificar N == expected_n.
  `load_audio_torch()` carga un archivo, resamplea a 44.1kHz mono y retorna (waveform, sr=44100).
  `get_dj_segments()` fallback (sin beat info) retorna 4 segmentos (1+2+1), cada uno de exactamente `5.0 * sr` muestras.
  `get_dj_segments()` con beat_ticks sintéticos (BPM=128, ticks regulares a 44100Hz) retorna 4 segmentos del largo correcto.


---

## TAREA 1.4: Demucs utils

Crear `src/v4/common/demucs_utils.py`. Copiar y adaptar de `legacy/v2/scripts/common/demucs_utils.py`.

Cambios respecto a V2: `process_track_stems()` acepta parámetro `target_sr` con default `MERT_SAMPLE_RATE` (24000). `stem_to_mono_numpy()` también acepta `target_sr`.

IMPORTANTE sobre sample rates: Demucs opera internamente a 44.1kHz. El `target_sr` se aplica DESPUÉS de la separación, al convertir stem a numpy: audio 44.1kHz → Demucs separa a 44.1kHz → stem drums 44.1kHz → resample a target_sr (24kHz) → numpy mono.

Verificación:
  El módulo se importa sin errores: `python -c "import src.v4.common.demucs_utils"`.
  `grep -n "16000" src/v4/common/demucs_utils.py` no retorna resultados (ningún 16000 hardcoded).
  Las funciones tienen signatures correctas (inspección visual del código).
  NOTA: la inferencia real de Demucs requiere GPU y se testea vía Slurm en TEST-2.


---

## TAREA 1.5: Embedding utils (MERTEmbedder)

Crear `src/v4/common/embedding_utils.py`:

```python
"""
PURPOSE: Extracción de embeddings con MERT-v1-330M.
"""

class MERTEmbedder:
    def __init__(self, model_name: str = MERT_MODEL_NAME, device: str = "cuda"):
        """Cargar Wav2Vec2FeatureExtractor y AutoModel. Audio DEBE ser 24kHz."""

    def embed_audio(self, audio_24k: np.ndarray) -> np.ndarray:
        """Feature extractor -> model forward -> last hidden state -> mean pool. Returns (1024,)."""

    def embed_segments(self, segments: List[np.ndarray]) -> np.ndarray:
        """Múltiples segmentos. Returns (n_segments, 1024)."""

    def aggregate_segments(self, segment_embeddings: np.ndarray, method: str = "mean") -> np.ndarray:
        """method="mean": (1024,). method="mean_std": (2048,)."""
```

Verificación:
  El módulo se importa sin errores: `python -c "import src.v4.common.embedding_utils"`.
  La clase MERTEmbedder existe con los métodos `embed_audio`, `embed_segments`, `aggregate_segments`.
  NOTA: la inferencia real de MERT requiere GPU y se testea vía Slurm en TEST-2. No intentar cargar el modelo en el nodo de login.


---

## TEST-1: Verificación del Bloque 1 (integration test)

Crear y ejecutar `tests/v4/test_block1_common.py`. Este script valida que todas las common utilities funcionan juntas, SIN usar GPU.

```python
"""
TEST BLOCK 1: Integration test de common utilities.
Ejecutar desde repo root: python tests/v4/test_block1_common.py

Este test NO carga modelos (MERT, Demucs). Solo valida:
  1. Config carga correctamente
  2. Path resolver encuentra rutas
  3. Catálogo se construye correctamente
  4. Audio se carga y segmenta correctamente
  5. Módulos de Demucs y MERT importan sin errores
"""

import sys, os
# Agregar repo root al path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

def test_config_and_paths():
    """Config carga -> paths resuelven -> artifacts root existe."""
    from src.v4.common.config_loader import load_config
    from src.v4.common.path_resolver import resolve_artifacts_root, resolve_dataset_audio_root
    config = load_config()
    artifacts = resolve_artifacts_root(config)
    audio_root = resolve_dataset_audio_root("test_20", config)
    assert audio_root.exists(), f"Audio root not found: {audio_root}"
    print(f"  OK: config loads, artifacts={artifacts}, audio={audio_root}")

def test_catalog():
    """Catálogo se construye con N tracks (N>0), UIDs únicos. Si expected_n está definido, validar N == expected_n."""
    from src.v4.common.config_loader import load_config
    from src.v4.common.path_resolver import resolve_dataset_audio_root
    from src.v4.common.catalog import build_catalog, load_catalog
    config = load_config()
    audio_root = resolve_dataset_audio_root("test_20", config)
    catalog = build_catalog(audio_root, "test_20", config)
    expected_n = config.get("datasets", {}).get("test_20", {}).get("expected_n")
    assert len(catalog) > 0
    if expected_n is not None:
        assert len(catalog) == expected_n, f"Expected {expected_n} tracks, got {len(catalog)}"
    assert catalog["track_uid"].nunique() == len(catalog), "Duplicate track_uids!"
    # Reload test
    catalog2 = load_catalog("test_20", config)
    assert len(catalog2) == len(catalog), "Reload mismatch"
    print(f"  OK: catalog has {len(catalog)} tracks, all UIDs unique")

def test_audio_loading_and_segmentation():
    """Cargar 1 track, segmentar en modo fallback."""
    from src.v4.common.config_loader import load_config
    from src.v4.common.path_resolver import resolve_dataset_audio_root
    from src.v4.common.audio_utils import get_audio_files, load_audio_torch, get_dj_segments
    from src.v4.config import ESSENTIA_SAMPLE_RATE
    config = load_config()
    audio_root = resolve_dataset_audio_root("test_20", config)
    files = get_audio_files(audio_root)
    expected_n = config.get("datasets", {}).get("test_20", {}).get("expected_n")
    assert len(files) > 0
    if expected_n is not None:
        assert len(files) == expected_n, f"Expected {expected_n} files, got {len(files)}"
    # Load first file
    waveform, sr = load_audio_torch(files[0])
    assert sr == ESSENTIA_SAMPLE_RATE, f"Unexpected SR (must be {ESSENTIA_SAMPLE_RATE}): {sr}"
    # Segment in fallback mode (no beat info)
    import numpy as np
    audio_np = waveform.squeeze().numpy() if hasattr(waveform, 'numpy') else waveform
    segments = get_dj_segments(audio_np, sr)
    expected_n = 4  # 1 intro + 2 mid + 1 outro
    expected_len = int(5.0 * sr)
    assert len(segments) == expected_n, f"Expected {expected_n} segments, got {len(segments)}"
    for i, seg in enumerate(segments):
        assert len(seg) == expected_len, f"Segment {i}: expected {expected_len} samples, got {len(seg)}"
    print(f"  OK: loaded {files[0].name}, {len(segments)} segments of {expected_len} samples")

def test_demucs_import():
    """Demucs module imports, no 16000 hardcoded."""
    import src.v4.common.demucs_utils  # Should not fail
    import inspect
    source = inspect.getsource(src.v4.common.demucs_utils)
    assert "16000" not in source, "Found hardcoded 16000 in demucs_utils!"
    print("  OK: demucs_utils imports, no hardcoded 16000")

def test_mert_import():
    """MERT embedder module imports (no model loading)."""
    from src.v4.common.embedding_utils import MERTEmbedder
    assert hasattr(MERTEmbedder, 'embed_audio')
    assert hasattr(MERTEmbedder, 'embed_segments')
    assert hasattr(MERTEmbedder, 'aggregate_segments')
    print("  OK: MERTEmbedder class exists with expected methods")

if __name__ == "__main__":
    print("=== TEST BLOCK 1: Common utilities integration ===\n")
    tests = [test_config_and_paths, test_catalog, test_audio_loading_and_segmentation,
             test_demucs_import, test_mert_import]
    passed, failed = 0, 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1
    print(f"\n{'='*50}")
    print(f"BLOCK 1: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    print("BLOCK 1: ALL TESTS PASSED")
```

Ejecutar: `python tests/v4/test_block1_common.py`

Todos los tests deben pasar. Si `test_catalog` falla porque `data/raw_audio/test_20/` no existe en este entorno, ese test puede marcarse como skip, pero los demás deben pasar.



---

## TAREA 1.6: Logging + manifests (JSONL)

Crear `src/v4/common/logging_utils.py`:

```python
"""
PURPOSE: Logging consistente (JSONL) + helpers para manifests reproducibles.
"""

def compute_config_hash(obj: dict) -> str:
    """Hash corto (p.ej. sha1) de un dict canonizado (json.dumps(sort_keys=True))."""

def get_git_commit(repo_root: Path) -> str:
    """Retorna commit actual o 'unknown' si no hay git."""

def get_slurm_job_id() -> Optional[str]:
    """Lee SLURM_JOB_ID / SLURM_ARRAY_JOB_ID si existen."""

def open_phase_log(logs_root: Path, phase_name: str) -> TextIO:
    """Crea `logs/<phase>_<timestamp>.jsonl` y retorna file handle."""

def log_event(fh: TextIO, event: dict) -> None:
    """Escribe 1 línea JSON con timestamp_utc agregado."""
```

Estandarizar schema de eventos (mínimo):
- `timestamp_utc`, `phase`, `dataset_name`, `event_type`
- opcionales: `track_uid`, `filepath`, `status`, `duration_ms`, `error`

Requisito de implementación:
- Phase 0..5 deben escribir logs JSONL en `resolve_logs_root(config)/<dataset_name>/logs/`.
- Si un track falla en Phase 1, se loguea `event_type='track_failed'` con error y se continúa.

Verificación:
- Importar el módulo sin errores.
- Crear un log JSONL, escribir 2 eventos, y verificar que el archivo contiene 2 líneas JSON válidas.

---
---

# BLOQUE 2: Pipeline scripts + Slurm

---

## TAREA 2.1: Phase 0 — Ingesta y catálogo

Implementar `src/v4/pipeline/phase0_ingest.py`.

CLI:
```
python src/v4/pipeline/phase0_ingest.py \
    --dataset-name <nombre> \
    [--audio-root /ruta/override] \
    [--manifest-csv /ruta/manifest.csv] \
    [--metadata-csv /ruta/beatport_export.csv] \
    [--download-dir /ruta/cache_descargas] \  # default: config.paths.download_cache o <artifacts_root>/<dataset>/download_cache
    [--download-workers 8] \
    [--verify-sha256] \
    [--config /ruta/v4.yaml]
```

### Formato de `manifest.csv` (opcional, recomendado)

Usar un CSV (o Parquet) con estas columnas mínimas:

- `filename` (string): nombre final deseado del archivo dentro de `audio_root/` (sin subdirectorios).
- `source_type` (string): `local` o `http`
- `source_uri` (string):
  - si `local`: ruta absoluta a un archivo existente
  - si `http`: URL directa descargable (p.ej. presigned URL / servidor propio). **No scraping / no DRM**
- `sha256` (opcional): hash de contenido para verificación
- `bytes_expected` (opcional): tamaño esperado en bytes

Columnas de metadata opcionales (si están disponibles): `artist`, `title`, `beatport_genre`, `label`, `year`, `bpm`, `key`.

Reglas:
- Si `filename` ya existe en `audio_root`, no re-descargar (idempotente).
- Si `--verify-sha256` y `sha256` existe: validar; si falla → marcar como error y no incluir en catálogo.


Flujo:
  1. Cargar configuración.
  2. Resolver rutas (artifacts_root, audio_root, caches).
  3. Determinar fuente de ingesta:
     a) Si --manifest-csv está presente (o config.datasets[dataset].manifest_csv): leer manifest.
        - `source_type=local`: copiar (o linkear) a audio_root de dataset.
        - `source_type=http`: descargar SOLO URLs directas autorizadas (sin scraping/DRM) a download_dir y luego mover a audio_root.
        - Si --verify-sha256: validar hash si la columna `sha256` existe.
     b) Si no hay manifest: escanear audio_root: .mp3, .wav, .flac.
  4. Por archivo final en audio_root: validar, track_uid (hash estable), extraer artist/title del filename, duración con soundfile, filesize.
  5. Merge metadata CSV si existe (Beatport u otro), por filename normalizado y/o por track_uid si está disponible.
  6. Guardar catalog.parquet + `ingest_report.json` con stats (n_files_ok, n_failed, sources).
  7. Imprimir resumen (N tracks, errores, dónde quedaron los artifacts).

Verificación:
  `python -c "import src.v4.pipeline.phase0_ingest"` sin errores.
  El script tiene argparse con los flags indicados.
  En una corrida real, genera `catalog.parquet`, `ingest_report.json` y un log JSONL de Phase 0.


---

## TAREA 2.2: Phase 1 — Extracción (script principal)

Implementar `src/v4/pipeline/phase1_extract.py`.

CLI:
```
python src/v4/pipeline/phase1_extract.py \
    --dataset-name <nombre> --device cuda \
    --shard-id 0 --num-shards 1 --checkpoint-every 25
```

Flujo por track:
  1. Cargar audio a 44.1kHz mono.
  2. Essentia: BPM + bpm_confidence + beat_ticks + beat_confidence + key + key_confidence (44.1kHz).
  3. Demucs drums stem (44.1kHz → resample a 24kHz post-separación).
  4. Resample full mix a 24kHz.
  5. Segmentos DJ de drums_24k y full_24k.
     - Si beat_confidence >= config.segmentation.beat_conf_threshold: segmentación por BARRAS (SEGMENT_DURATION_BARS).
     - Si no: fallback a segmentación por duración fija en segundos (SEGMENT_DURATION_S) y posiciones intro/mid/outro.
  6. MERT embeddings de cada segmento. Agregar a vector por track.
  7. Checkpoint cada N tracks.

Sharding: dividir catálogo en N shards, procesar solo el indicado.
Output: `embeddings/shards/mert_perc_shard_XX.npy`, `mert_full_shard_XX.npy`, `track_uids_shard_XX.json`, `progress_shard_XX.json`. Features: `features/shards/bpm_key_shard_XX.parquet`.

IMPORTANTE: Este script se ejecuta vía Slurm (partición a100, GPU). NUNCA ejecutar directamente en el nodo de login.

Verificación:
  `python -c "import src.v4.pipeline.phase1_extract"` sin errores de sintaxis.
  La lógica de sharding es correcta: con N tracks y K shards, cada shard tiene ~N/K tracks.
  En una corrida real, genera log JSONL de Phase 1 + progress_shard_XX.json reentrante.


---

## TAREA 2.3: Phase 1 — Merge de shards

Implementar `src/v4/pipeline/phase1_merge_shards.py`.

CLI: `python src/v4/pipeline/phase1_merge_shards.py --dataset-name <nombre>`

Lee shards, verifica cobertura completa del catálogo, consolida en:
  `embeddings/mert_perc.npy` (N, 1024), `embeddings/mert_full.npy` (N, 1024), `features/bpm_key.parquet`.
Actualiza catálogo con bpm, bpm_confidence, beat_confidence, key, key_confidence.
Genera/actualiza:
  - `run_manifest.json`: {dataset_name, N_tracks, config_hash, git_commit, timestamp_utc, hostname, user, slurm_job_id (si aplica),
     model_versions (MERT, Demucs, Essentia), python/torch/transformers versions, paths resueltas, shards usados}.
  - `dataset_manifest.json`: lista de artifacts producidos + checksums básicos (tamaños, shapes, min/max finitos).

Verificación:
  `python -c "import src.v4.pipeline.phase1_merge_shards"` sin errores.
  En una corrida real, produce `mert_perc.npy`, `mert_full.npy`, `bpm_key.parquet`, `run_manifest.json`.


---

## TAREA 2.4: Slurm jobs V4

Crear todos los Slurm jobs en `slurm/jobs/v4/`. Basados en el patrón de `slurm/jobs/v3/extract_embeddings.job` y las instrucciones de `CLAUDE.md`.

### 2.4.1 `phase1_extract.job`
  Partición a100, 1 GPU, 8 CPUs, 32GB RAM, 12 horas.
  Setup scratch4weeks, bootstrap pip.
  Exportar caches:
    - HF_HOME/TRANSFORMERS_CACHE desde `resolve_hf_cache()` (o $TRAKTOR_HF_CACHE)
    - TORCH_HOME desde `resolve_torch_cache()` (o $TRAKTOR_TORCH_CACHE)
  Pre-descargar MERT + Demucs al cache antes de procesar (primer run).
  Ejecuta: `$PY src/v4/pipeline/phase1_extract.py --dataset-name $DATASET_NAME --device cuda --checkpoint-every 25`

### 2.4.2 `phase1_extract_array.job`
  `#SBATCH --array=0-3`. Ejecuta con `--shard-id $SLURM_ARRAY_TASK_ID --num-shards 4`.

### 2.4.3 `phase1_merge.job`
  Partición debug, 4 CPUs, 4GB, 30 min.

### 2.4.4 `phase2_to_5.job`
  Partición debug, 4 CPUs, 8GB, 1 hora. Ejecuta secuencialmente Phases 2, 3, 4, 5.

### 2.4.5 `eval_sweep.job`
  Partición debug, 4 CPUs, 8GB, 2 horas.

### 2.4.6 `smoke_test_gpu.job`
  Partición a100, 1 GPU, 4 CPUs, 16GB, 30 minutos.
  Este job es para TEST-2: ejecuta phase1_extract con solo 3 tracks para validar que el pipeline GPU funciona.
  Ejecuta: `$PY src/v4/pipeline/phase1_extract.py --dataset-name test_20 --device cuda --num-shards 1 --checkpoint-every 1 --max-tracks 3`
  (Requiere que phase1_extract.py acepte `--max-tracks` como flag opcional para limitar procesamiento.)

Verificación:
  `bash -n slurm/jobs/v4/*.job` sin errores de sintaxis.
  Los jobs referencian scripts Python que existen.


---

## TEST-2: Verificación del Bloque 2

Crear `tests/v4/test_block2_pipeline.py` que valida:

### Parte A: Verificaciones locales (automáticas, sin GPU)

1. Todos los scripts de pipeline importan sin errores de sintaxis.
2. Todos los Slurm jobs pasan `bash -n`.
3. Phase 0 se ejecuta correctamente sobre test_20 y genera catálogo con N tracks (N>0; si expected_n está definido, N == expected_n).
4. phase1_merge_shards funciona con datos dummy (crear 2 shards fake con np.random, verificar merge).

### Parte B: Smoke test GPU (vía Slurm)

5. Enviar `smoke_test_gpu.job` al cluster:
   `./slurm/tools/on_submit.sh sbatch slurm/jobs/v4/smoke_test_gpu.job`
6. Esperar a que termine.
7. Verificar que se generaron embeddings para 3 tracks: shapes (3, 1024), valores finitos, BPMs en rango.

### !! INTERVENCIÓN HUMANA REQUERIDA !!
La Parte B requiere acceso al cluster y esperar ~15-30 minutos. El humano debe:
  1. Confirmar que se envió el smoke test.
  2. Verificar el output del job (`slurm-XXXXX.out`).
  3. Reportar si los embeddings se generaron correctamente.

Si la Parte A pasa y la Parte B se confirma, el bloque 2 está validado.


---
---

# >>> PAUSA HUMANA MAYOR <<<

Antes de continuar con Bloque 3:
  0. Asegurar rutas:
     - Si el audio NO está en `REPO_ROOT/data/raw_audio/<dataset>/`, definir `--audio-root` o `config.datasets.<dataset>.audio_root`, o exportar `TRAKTOR_AUDIO_ROOTS`.
     - Si ingestas vía URLs directas/autorizadas, preparar `manifest.csv` y correr Phase 0 con `--manifest-csv`.
  1. Ejecutar Phase 0 completo: `python src/v4/pipeline/phase0_ingest.py --dataset-name test_20`
  2. Revisar catálogo: N tracks (N>0; si expected_n está definido, N == expected_n), UIDs únicos, artist/title parseados.
  3. Enviar Phase 1 completo: `./slurm/tools/on_submit.sh sbatch slurm/jobs/v4/phase1_extract.job test_20`
  4. Esperar 2-4 horas. Verificar embeddings:
     - `mert_perc.npy` shape (N, 1024) donde N = #tracks del catálogo, valores finitos.
     - `mert_full.npy` shape (N, 1024) donde N = #tracks del catálogo, valores finitos.
     - BPMs en `bpm_key.parquet` en rango 110-155.
     - Embeddings percusivos y full son distintos (distancia coseno media > 0.05).
  5. Si usó sharding: ejecutar merge `./slurm/tools/on_submit.sh sbatch slurm/jobs/v4/phase1_merge.job test_20`

Solo continuar cuando los embeddings estén validados.


---
---

# BLOQUE 3: Clustering + evaluación

---

## TAREA 3.1: Phase 2 — Clustering jerárquico

Implementar `src/v4/pipeline/phase2_cluster.py`.

Entrada: `mert_perc.npy`, `mert_full.npy`, `catalog.parquet`.

Paso 1: L2-normalize ambas matrices.
Paso 2: L1 clustering con HDBSCAN sobre mert_perc. Parámetros desde v4.yaml (con override CLI).
Paso 3: L2 clustering con HDBSCAN sobre mert_full, dentro de cada cluster L1 (min 8 tracks).
Paso 4: UMAP 2D para visualización.
Paso 5: Guardar `clustering/results_<hash>.parquet` y `clustering/config_<hash>.json`.

CLI:
```
python src/v4/pipeline/phase2_cluster.py \
    --dataset-name <nombre> \
    --l1-min-cluster-size 10 --l1-min-samples 3 \
    --l2-min-cluster-size 4 --l2-min-samples 2 \
    --config-tag "baseline"
```

Verificación:
  Parquet tiene tantas filas como tracks.
  Cada track tiene label_l1 y label_l2.
  UMAP coords finitas.
  Imprime estadísticas: N clusters L1, tamaños, noise rate.


---

## TAREA 3.2: Evaluation framework

Implementar `src/v4/evaluation/metrics.py` y `src/v4/evaluation/eval_runner.py`.

### metrics.py

```python
def clustering_ari(labels_pred, labels_true) -> float: ...
def clustering_nmi(labels_pred, labels_true) -> float: ...
def retrieval_recall_at_k(embeddings, positive_pairs, k=10) -> float: ...
def retrieval_mrr(embeddings, positive_pairs) -> float: ...
def retrieval_ndcg_at_k(embeddings, positive_pairs, k=10) -> float: ...
def pairwise_auc(embeddings, positive_pairs, n_negatives=1000) -> float: ...
def transition_score(ordering, embeddings, bpm, keys, weights=None) -> float:
    """weights default from config.ORDERING_WEIGHTS."""
def noise_rate(labels) -> float: ...
def composite_score(metrics_dict, weights=None) -> float: ...
```

### eval_runner.py

```python
def run_evaluation(dataset_name, clustering_config_hash, config,
                   dev_set_path=None, dj_pairs_path=None) -> dict:
    """Carga todo, calcula métricas aplicables, guarda scores JSON."""
```

Verificación:
  `clustering_ari(labels, labels)` == 1.0.
  `noise_rate(np.array([0,0,1,-1,-1]))` == 0.4.
  Las funciones retornan valores numéricos en rangos esperados.


---

## TEST-3: Verificación del Bloque 3

Crear y ejecutar `tests/v4/test_block3_clustering.py`.

### Parte A: Unit tests de métricas (automático)

1. `clustering_ari(x, x) == 1.0`
2. `noise_rate` con inputs conocidos.
3. `transition_score` con datos sintéticos retorna valor en [0, 1].

### Parte B: Clustering real sobre test_20 (automático, CPU)

4. Ejecutar Phase 2 sobre test_20 con parámetros default.
5. Verificar: clusters L1 entre 3-15, noise rate < 30%, parquet tiene N filas (N = #tracks del catálogo).
6. Ejecutar eval_runner (solo noise_rate + cluster stats, sin evaluation data externa).
7. Imprimir resumen de clustering para revisión humana.

### Parte C: Revisión humana del clustering

### !! INTERVENCIÓN HUMANA REQUERIDA !!

El test imprime un reporte con:
```
=== CLUSTERING REPORT (para revisión humana) ===
Clusters L1: N, tamaños: [X, Y, Z, ...]
Noise rate: X%
Sub-clusters L2 por grupo: A=[A1(n), A2(n)], B=[B1(n), B2(n)], ...
Tracks en noise: [lista de filenames]
```

El humano debe:
  1. Revisar si el número de clusters L1 es razonable (3-10 para un dataset pequeño (~250 tracks); ajustar a escala si N crece).
  2. Revisar si los tamaños de clusters son balanceados (no hay 1 cluster con 200 tracks y 5 con 2).
  3. Revisar si los tracks en noise son realmente atípicos o si el clustering está descartando demasiado.
  4. Si no es satisfactorio: re-ejecutar Phase 2 con otros parámetros.


---
---

# BLOQUE 4: Export pipeline

---

## TAREA 4.1: Phase 3 — Naming semántico

Implementar `src/v4/pipeline/phase3_name.py`.

Estrategia 1 (genre voting): Si hay `beatport_genre_norm` en catálogo, nombre = top-2 géneros.
Si no hay metadata: nombre genérico "Group A", "Group B".

Estrategia 2 (CLAP, opcional): Si CLAP instalable, matching con CLAP_DESCRIPTORS.
Si no: documentar como mejora futura.

Verificación:
  Cada cluster tiene nombre no vacío.
  Nombres de clusters distintos no son todos iguales.


---

## TAREA 4.2: Phase 4 — Ordenamiento intra-cluster

Implementar `src/v4/pipeline/phase4_order.py`.

```python
def key_compatibility(key_a: str, key_b: str) -> float:
    """Camelot wheel: 1.0=compatible, 0.5=semitono, 0.0=incompatible."""

def order_cluster_tracks(track_indices, embeddings, bpm, keys, weights=None) -> List[int]:
    """Nearest-neighbour greedy. weights default from config.ORDERING_WEIGHTS."""
```

Tabla Camelot completa (1A-12A minor, 1B-12B major).

Verificación:
  `key_compatibility("Cm", "Cm")` == 1.0.
  `key_compatibility("Cm", "Gm")` == 1.0 (5A y 6A).
  Ordering es permutación válida.
  Transition score > aleatorio (test con 100 permutaciones random).


---

## TAREA 4.3: Phase 5 — Export

Implementar `src/v4/pipeline/phase5_export.py`.

Estructura:
```
playlists/V4_<N>/
  L1_A_[nombre]/
    L2_A1_[nombre].m3u
  All_Noise.m3u
  _summary.txt
```

Rutas Windows desde `config.paths.local_windows_audio_dir`.

### !! INTERVENCIÓN HUMANA REQUERIDA !!
Confirmar ruta Windows (default: `C:\Música\2020 new - copia`).

Verificación:
  M3U con header `#EXTM3U`.
  Total tracks en todos los M3U == N (#tracks del catálogo).


---

## TEST-4: Verificación del Bloque 4

Crear y ejecutar `tests/v4/test_block4_export.py`.

### Parte A: Verificaciones automáticas

1. Ejecutar Phases 3, 4, 5 secuencialmente sobre test_20.
2. Verificar: todos los M3U existen, total tracks == N (#tracks del catálogo), rutas Windows en M3U.
3. Calcular transition_score promedio por playlist y comparar con baseline aleatorio.
4. Verificar que `_summary.txt` existe y lista todos los clusters.

### Parte B: Revisión humana de calidad

### !! INTERVENCIÓN HUMANA REQUERIDA !!

El test genera un reporte de calidad para el humano:

```
=== EXPORT QUALITY REPORT (para revisión humana) ===

Playlist: L1_A_[Group A] / L2_A1
  Tracks: 12
  BPM range: 125-130
  Keys: Cm(4), Gm(3), Dm(2), Fm(2), Am(1)
  Transition score: 0.78 (vs random baseline: 0.42)
  Primeros 5 tracks en orden:
    1. Artist1 - Title1 (127 BPM, Cm)
    2. Artist2 - Title2 (128 BPM, Gm)
    3. Artist3 - Title3 (126 BPM, Cm)
    ...

[repite para cada playlist L2]
```

El humano debe:
  1. Revisar si los BPM ranges dentro de cada playlist son coherentes (no mezclar 120 con 150).
  2. Revisar si el ordering tiene sentido (BPMs cercanos consecutivos, keys compatibles).
  3. Opcionalmente: abrir 2-3 M3U en un reproductor y escuchar las transiciones.


---
---

# BLOQUE 5: UI + finalización

---

## TAREA 5.1: UI Streamlit

Implementar `src/v4/ui/app.py`.

Funcionalidades:
  1. Selector de dataset.
  2. Scatter plot UMAP interactivo (plotly). Color por L1. Hover: filename, artist, BPM, key, L1, L2.
  3. Filtro por L1 para ver sub-clusters L2.
  4. Panel de estadísticas: N clusters, tamaños, noise rate, distribución BPM/key.
  5. Panel de evaluación (si hay scores).
  6. Controles de re-clustering: sliders para min_cluster_size/min_samples + botón "Re-cluster".
  7. Botón "Export Playlists" que ejecuta Phases 3+4+5.

Verificación:
  `streamlit run src/v4/ui/app.py` arranca sin errores.
  Scatter plot se renderiza con datos de test_20.


---

## TAREA 5.2: Adaptation stubs

Implementar `src/v4/adaptation/projection_head.py` y `contrastive_trainer.py`.

`ProjectionHead(nn.Module)`: MLP 1024 → 512 → 256.
`ContrastiveTrainer`: interfaz definida, `train_epoch` levanta `NotImplementedError`.

Verificación:
  `ProjectionHead` forward con tensor (16, 1024) sin errores.
  `ContrastiveTrainer` se instancia.


---

## TAREA 5.3: Integración end-to-end + documentación

### 5.3.1 Smoke test con 5 tracks

Ejecutar pipeline completo: Phase 0 → 1 → 2 → 3 → 4 → 5. Verificar todos los outputs.
NOTA: Phase 1 con 5 tracks se puede ejecutar vía Slurm (smoke_test_gpu.job) o en CPU si el tiempo es aceptable.

### 5.3.2 Documentación

Actualizar `docs/PROJECT_MAP.md`.
Crear `docs/V4_USAGE.md` con instrucciones de uso completas.

Verificación:
  Documentación existe y es coherente con el código.


---

## TEST-5: Verificación final del sistema

### Parte A: Verificación automática

1. Todos los módulos importan sin errores.
2. Todos los tests de bloques anteriores siguen pasando.
3. `docs/v4/TODO.md` tiene todas las tareas marcadas como completadas.

### Parte B: Revisión humana del sistema completo

### !! INTERVENCIÓN HUMANA REQUERIDA !!

El humano debe:
  1. Lanzar `streamlit run src/v4/ui/app.py` y navegar la UI.
  2. Verificar que el scatter UMAP muestra clusters visualmente separados.
  3. Probar re-clustering con distintos parámetros desde la UI.
  4. Exportar playlists desde la UI.
  5. Importar 2-3 playlists M3U en Traktor DJ y verificar que las rutas son correctas.
  6. Escuchar transiciones entre tracks consecutivos en al menos 2 playlists.
  7. Evaluar subjetivamente: ¿los clusters agrupan tracks que tienen sentido juntos?

### Criterio de aceptación:
  El pipeline completo corre sin errores.
  La UI es funcional.
  Las playlists se importan correctamente en Traktor.
  El humano considera que al menos el 70% de los clusters tienen coherencia musical.


---
---

# Resumen de intervenciones humanas requeridas

| Momento | Qué se necesita | Por qué |
|---------|----------------|---------|
| TEST-2 Parte B | Confirmar smoke test GPU | Validar pipeline de inferencia |
| Pausa mayor (pre-Bloque 3) | Ejecutar Phase 0+1, revisar embeddings | Phase 1 es costosa (2-4h GPU) |
| TEST-3 Parte C | Revisar clustering stats | El clustering óptimo depende del dataset |
| Antes de Tarea 4.3 | Confirmar ruta Windows para M3U | M3U necesita rutas absolutas |
| TEST-4 Parte B | Revisar calidad de playlists | Validación subjetiva de ordering |
| TEST-5 Parte B | Probar sistema completo + Traktor | Validación final de usabilidad |


---

## Orden de ejecución y sesiones de Claude Code

```
BLOQUE 0: Setup (3 tareas + 1 test)
  SESIÓN 1:  Tarea 0.1  Leer plan + crear TODO.md
  SESIÓN 2:  Tarea 0.2  Mover V3 a legacy
  SESIÓN 3:  Tarea 0.3  Estructura V4 + config
  SESIÓN 4:  TEST-0     Verificación estructural

BLOQUE 1: Common utilities (5 tareas + 1 test)
  SESIÓN 5:  Tarea 1.1  config_loader + path_resolver
  SESIÓN 6:  Tarea 1.2  catalog.py
  SESIÓN 7:  Tarea 1.3  audio_utils.py + segmentación
  SESIÓN 8:  Tarea 1.4  demucs_utils.py
  SESIÓN 9:  Tarea 1.5  embedding_utils.py
  SESIÓN 10: TEST-1     Integration test common

BLOQUE 2: Pipeline + Slurm (4 tareas + 1 test)
  SESIÓN 11: Tarea 2.1  phase0_ingest.py
  SESIÓN 12: Tarea 2.2  phase1_extract.py
  SESIÓN 13: Tarea 2.3  phase1_merge_shards.py
  SESIÓN 14: Tarea 2.4  Slurm jobs
  SESIÓN 15: TEST-2     Pipeline validation + smoke test GPU

  >>> PAUSA HUMANA: Phase 0 + Phase 1 completo + revisar embeddings <<<

BLOQUE 3: Clustering + evaluación (2 tareas + 1 test)
  SESIÓN 16: Tarea 3.1  phase2_cluster.py
  SESIÓN 17: Tarea 3.2  evaluation framework
  SESIÓN 18: TEST-3     Clustering + eval + human review

  >>> PAUSA HUMANA: revisar clustering, ajustar params <<<

BLOQUE 4: Export pipeline (3 tareas + 1 test)
  SESIÓN 19: Tarea 4.1  phase3_name.py
  SESIÓN 20: Tarea 4.2  phase4_order.py
  SESIÓN 21: Tarea 4.3  phase5_export.py
  SESIÓN 22: TEST-4     Export validation + human review

BLOQUE 5: UI + finalización (3 tareas + 1 test)
  SESIÓN 23: Tarea 5.1  UI Streamlit
  SESIÓN 24: Tarea 5.2  Adaptation stubs
  SESIÓN 25: Tarea 5.3  Integración + docs
  SESIÓN 26: TEST-5     Verificación final del sistema
```

Total: 26 sesiones de Claude Code.
  20 sesiones de implementación.
  6 sesiones de test de bloque.
  2 pausas humanas mayores (post-Bloque 2 y post-Bloque 3).
  6 intervenciones humanas para revisión (marcadas con !! INTERVENCIÓN HUMANA REQUERIDA !!).
