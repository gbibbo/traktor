# TRAKTOR ML V4 — Project Map

Inventario de archivos del proyecto. Actualizar al añadir ficheros nuevos.

## Configuración

| Archivo | Descripción |
| :--- | :--- |
| `config/v4.yaml` | Config principal: rutas, datasets, clustering, ordering |
| `src/v4/config.py` | Constantes centrales (sample rates, model names, defaults) |

## Common Utilities (`src/v4/common/`)

| Archivo | Descripción |
| :--- | :--- |
| `config_loader.py` | Cargar config con cascada: CLI > env > YAML |
| `path_resolver.py` | Resolver rutas de artifacts, audio, caches (laptop + HPC) |
| `catalog.py` | Construir y cargar catálogo de tracks (catalog.parquet) |
| `audio_utils.py` | Carga de audio, segmentación DJ, get_audio_files |
| `demucs_utils.py` | Separación de stems (Demucs htdemucs) |
| `embedding_utils.py` | MERTEmbedder: embeddings MERT-v1-330M |
| `logging_utils.py` | Logger JSONL + run manifests |

## Pipeline (`src/v4/pipeline/`)

| Archivo | Descripción |
| :--- | :--- |
| `phase0_ingest.py` | Escaneo + validación + catálogo + metadata merge |
| `phase1_extract.py` | GPU: Demucs + MERT + Essentia (con sharding) |
| `phase1_merge_shards.py` | CPU: Consolida shards de Phase 1 → mert_perc.npy, mert_full.npy, track_uids.json |
| `phase2_cluster.py` | CPU: HDBSCAN L1/L2 + UMAP 2D → clustering/results_<hash>.parquet |
| `phase3_name.py` | CPU: Naming semántico de clusters (genre voting + fallback genérico) |
| `phase4_order.py` | CPU: Ordering greedy NN (cosine + BPM + Camelot key) → ordered_<hash>.parquet |
| `phase5_export.py` | CPU: Export M3U Traktor (UTF-8, rutas Windows) → playlists/V4_<N>/ |

## Evaluation (`src/v4/evaluation/`)

| Archivo | Descripción |
| :--- | :--- |
| `metrics.py` | ARI, NMI, Recall@k, MRR, NDCG, pairwise_auc, transition_score, noise_rate |
| `eval_runner.py` | Loop de evaluación: carga artifacts, calcula métricas, guarda JSON |

## Slurm Jobs (`slurm/jobs/v4/`)

| Archivo | Descripción |
| :--- | :--- |
| `smoke_test_gpu.job` | Smoke test GPU: 3 tracks, 30min, a100 |
| `phase0_ingest.job` | Phase 0 en CPU (debug partition) |
| `phase1_extract.job` | Phase 1 completo GPU: Demucs + MERT + Essentia |
| `phase1_extract_array.job` | Phase 1 en array de Slurm (sharding paralelo) |
| `phase1_merge.job` | Phase 1 merge shards en CPU |

## UI y Adaptation (`src/v4/`)

| Archivo | Descripción |
| :--- | :--- |
| `ui/app.py` | Streamlit dashboard: scatter UMAP, filtros L1/L2, re-clustering local, export |
| `adaptation/projection_head.py` | MLP projection head: 1024→512→256 L2-normalizado (stub fine-tuning) |
| `adaptation/contrastive_trainer.py` | Entrenador contrastivo (stub — interfaz definida, NotImplementedError) |

## Tests (`tests/v4/`)

| Archivo | Descripción |
| :--- | :--- |
| `test_block0_setup.sh` | Smoke test de estructura de directorios |
| `test_block1_common.py` | Integration test: config, catalog, audio, utils |
| `test_block2_pipeline.py` | Integration test: Phase 0 + validaciones |
| `test_block3_clustering.py` | Unit tests métricas (Parte A) + clustering real (Parte B, skip-safe) |
| `test_block4_export.py` | Tests export pipeline: Phase 3+4+5, N canónico, transition score, quality report |
| `test_block5_system.py` | Verificación final: todos los módulos importan, catalog_success, ProjectionHead |

## Artifacts (generados, no en git)

```
artifacts/v4/datasets/<dataset_name>/
├── catalog.parquet                # Phase 0: catálogo de tracks
├── ingest_report.json             # Phase 0: estadísticas de ingesta
├── embeddings/
│   ├── mert_perc.npy              # Phase 1: embeddings percusivos (N, 1024)
│   ├── mert_full.npy              # Phase 1: embeddings full mix (N, 1024)
│   ├── track_uids.json            # Phase 1: UIDs en orden (fuente de verdad de alineación)
│   └── shards/                    # Shards temporales Phase 1
├── features/
│   └── bpm_key.parquet            # Phase 1: BPM, key, beat_confidence
├── catalog_success.parquet        # Phase 1 merge: catalog filtrado a N canónico (track_uids.json)
├── clustering/
│   ├── results_<hash>.parquet     # Phase 2: label_l1, label_l2, umap_x, umap_y
│   ├── config_<hash>.json         # Phase 2: parámetros + label_semantics
│   ├── names_<hash>.json          # Phase 3: nombres de clusters
│   └── ordered_<hash>.parquet     # Phase 4: results + columna 'position'
├── evaluation/
│   └── <hash>_scores.json         # eval_runner: métricas calculadas
└── logs/
    └── *.jsonl                    # Logs estructurados por phase

playlists/V4_<N>/
├── L1_A_[nombre]/
│   └── L2_A1_[nombre].m3u        # Phase 5: playlists ordenadas por subcluster
├── All_Noise.m3u                  # Phase 5: tracks en ruido L1
└── _summary.txt                   # Phase 5: tabla resumen de clusters y tracks
```

## Documentación (`docs/`)

| Archivo | Descripción |
| :--- | :--- |
| `docs/PROJECT_MAP.md` | Este archivo |
| `docs/V4_USAGE.md` | Instrucciones de uso end-to-end (Phase 0→5, UI, escalar a 2000 tracks) |
| `docs/v4/TODO.md` | Progress tracker con fechas de completado |
| `docs/v4/JOBS_STATUS.md` | Estado de jobs Slurm + comandos de monitoreo |
| `docs/LESSONS_LEARNED.md` | Base de conocimiento de lecciones aprendidas |
| `v4_implementation_plan.md` | Plan de implementación completo (rev.5) |
