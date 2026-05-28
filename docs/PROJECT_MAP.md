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

---

## DJ Clustering Pipeline (feature/dj-clustering-v1)

### Operational plan and trackers

| File | Description |
| :--- | :--- |
| `docs/plans/dj_clustering_plan.md` | Sole operational source of truth for DJ clustering work |
| `docs/progress/dj_clustering_progress.md` | Markdown task tracker |
| `docs/progress/dj_clustering_progress.yaml` | YAML state tracker |
| `reports/dj_clustering/v4_reuse_map.md` | D0.5 V4 component audit and classification |
| `reports/dj_clustering/library_inventory_summary.md` | D1.2 inventory summary |
| `reports/dj_clustering/metadata_quality_report.md` | D1.3 metadata quality and sanity field validation |
| `reports/dj_clustering/feature_quality_report.md` | D2.2 frozen feature quality |
| `reports/dj_clustering/reference_availability_report.md` | D2.3 V2/V4 reference availability |
| `reports/dj_clustering/pair_features_summary.md` | D3.1 pair feature aggregate summary |
| `reports/dj_clustering/triplet_queue_summary.md` | D3.2 triplet question queue generation summary |
| `reports/dj_clustering/v4_playlist_mapping_report.md` | D3.2 V4 playlist mapping coverage and decision |

### Source modules (`src/dj_clustering/`)

| File | Description |
| :--- | :--- |
| `inventory.py`, `identity.py` | D1.1/D1.2 inventory and track identity |
| `segments.py`, `mert.py`, `hpss.py`, `features.py` | D2.1/D2.2 segment selection, MERT extraction, HPSS, orchestration |
| `pair_features.py` | D3.1 pair enumeration, cosine pair features, raw-BPM tempo equivalence, metadata similarity |
| `similarity_profiles.py` | D3.1 committed similarity profile registry (audio_only, audio_plus_metadata_light) |
| `triplets.py` | D3.2 KNN index, V4 playlist cluster-map parsing, NN and boundary triplet sampling, de-duplication, question DataFrame assembly; D4.2 active selection — top-3 leaderboard config tie-break, per-config rank differences, disagreement detection / margin / variance ranking, existing-queue de-dup, active-set assembly |
| `triplet_ingest.py` | D3.3 answer template creation, validation (B/C/skip, unknown/duplicate qids), manual triplet and skip log assembly, summary report generation; D4.3 active-answer support — `collect_answered_ids`, `filter_unanswered`, `merge_answer_rows` |
| `match_1001.py` | D3.4 metadata-only 1001Tracklists matching: name/marker/duration normalization, confidence policy, accepted/rejected/ambiguous categorization, no-usable-source path, aggregate-only report |
| `sweep.py` | D4.1 Regime 1 sweep planning: deterministic vector-grid expansion, invalid-combination pruning, seed-13 capped sampling, fixed-baseline injection, diagnostic-only V2/V4 rows |
| `clustering.py` | D4.1 Regime 1 clustering primitives: embedding loading, L2/no-op normalization, dim reduction (none/PCA/UMAP), HDBSCAN/KMeans/agglomerative wrappers, HDBSCAN noise-policy handling |
| `metrics.py` | D4.1 sweep evaluation: manual triplet accuracy (B/C distance direction, skip exclusion), cluster diagnostics, exploration-evidence status, 1001 no-usable-source handling |

### Driver scripts (`scripts/dj_clustering/`)

| File | Description |
| :--- | :--- |
| `inventory_library.py` | D1.2 build inventory CSV |
| `report_metadata_quality.py` | D1.3 emit metadata quality report |
| `validate_mert_backend.py`, `extract_features.py` | D2.2 MERT validation and feature extraction |
| `build_pair_features.py` | D3.1 build pair feature parquet, manifest, and summary |
| `generate_triplet_questions.py` | D3.2 generate initial manual comparison question queue (40 questions) |
| `ingest_manual_triplets.py` | D3.3 two-mode CLI: create-template (blank answer CSV) and ingest (validate + write manual_triplets.csv + summary); D4.3 adds `create-template --exclude-answered` (active-only template) and `ingest --merge-with` (merge into combined artifacts) |
| `match_1001tracklists.py` | D3.4 two-mode CLI: create-input-template (blank 1001 input CSV) and match (match against canonical tracks + write matching report) |
| `run_similarity_sweep.py` | D4.1 Regime 1 sweep runner: assembles the run plan, optionally executes clustering/scoring, writes leaderboard, component metrics, run metadata, and resolved config under ignored `runs/` |
| `generate_active_triplets.py` | D4.2 active triplet question generator: selects top-3 sweep configs, scores candidate disagreement, fills with boundary cases, extends the ignored queue/M3U, writes the aggregate summary |
| `export_portable_state.py` | EXIT.2–EXIT.4 cluster-exit exporter: inventories critical/optional ignored artifacts, builds `.tar.zst` archives + `SHA256SUMS.txt`, writes the ignored per-file raw-audio checksum manifest plus its verification-safe aggregate, and emits the ignored export manifest JSON |

### Tests (`tests/dj_clustering/`)

| File | Description |
| :--- | :--- |
| `test_inventory.py`, `test_identity.py` | D1 inventory + identity unit tests |
| `test_segments.py`, `test_features.py` | D2 segment + feature unit tests |
| `test_pair_features.py`, `test_similarity_profiles.py` | D3.1 pair feature + profile unit tests |
| `test_triplets.py` | D3.2 triplet generation unit tests |
| `test_triplet_ingest.py` | D3.3 answer template, validation, assembly, and summary report unit tests |
| `test_match_1001.py` | D3.4 1001Tracklists matching: normalization, confidence policy, categorization, no-usable-source path, report privacy unit tests |
| `test_sweep.py` | D4.1 grid expansion, invalid-combination pruning, seed-13 sampling, fixed-baseline injection, diagnostic-only rows |
| `test_metrics.py` | D4.1 triplet accuracy B/C direction, skip exclusion, unscored handling, evidence status, 1001 no-usable-source |
| `test_clustering_membership.py` | D4.1 normalization, dim reduction, clusterer wrappers, ward rejection, HDBSCAN noise-policy behavior |
| `test_active_triplets.py` | D4.2 top-3 config tie-break, disagreement detection / margin / variance ranking, existing-queue de-dup, hash-collision exclusion, active-set assembly |
| `test_export_portable_state.py` | EXIT exporter unit tests: sha256 hashing, item presence/missing resolution, raw-audio aggregate (count/bytes only), per-file manifest determinism, manifest assembly, critical-item coverage |

### Configs (`configs/dj_clustering/`)

| File | Description |
| :--- | :--- |
| `inventory.yaml` | D1.1 library roots and inventory output |
| `features.yaml` | D2.1 feature config; D3.1 adds the `pairs:` block |
| `sweep_regime1.yaml` | D4.1 committed Regime 1 vector grid: axes, conditional clusterer branches, seed-13 cap, fixed baselines, diagnostic references |

### Approved implementation namespace

| Directory | Purpose |
| :--- | :--- |
| `src/dj_clustering/` | Core library modules |
| `scripts/dj_clustering/` | Pipeline entry-point scripts |
| `configs/dj_clustering/` | YAML configuration files |
| `tests/dj_clustering/` | Tests for dj_clustering |
| `reports/dj_clustering/` | Small committed reports |

These directories are created incrementally from D1 onward.

### Frozen V4 policy (D0–D7)

V4 (`src/v4/`, `tests/v4/`, `config/v4.yaml`, `slurm/jobs/v4/`, `docs/v4/`,
`docs/V4_USAGE.md`, `playlists/V4_*/`, `requirements_v4.txt`) is frozen during D0–D7.

- V4 must not be modified, moved, deleted, or renamed.
- New DJ clustering work must not be implemented under `src/v4/`.
- V4 is classified as `v4_diagnostic_only` for the leaderboard (no reusable embeddings in Git).
- Selected V4 utilities are classified `reuse_by_import` or `adapt_into_src_dj_clustering` per `reports/dj_clustering/v4_reuse_map.md`.
- This separate namespace (`src/dj_clustering/` etc.) is authorized by `docs/plans/dj_clustering_plan.md §6` and overrides the CLAUDE.md refactor-preference for D0–D7 only.

### Cluster-exit migration (EXIT.1–EXIT.7)

Operational portability checkpoint taken because Surrey HPC access ends
2026-05-31. Not a science cut; D0–D7 semantics are unchanged and D4.3 stays
blocked as an offline human task.

| File | Description |
| :--- | :--- |
| `docs/migration/dj_clustering_cluster_exit_plan.md` | Cluster-exit plan: preserved vs ignored, critical vs optional payload, archive checksums, raw-audio aggregate, continuation paths |
| `docs/migration/dj_clustering_artifact_manifest.yaml` | Machine-readable archive manifest (member dirs, sha256, sizes) + raw-audio aggregate + verification commands |
| `docs/migration/dj_clustering_restore_guide.md` | Restore on local CPU / paid GPU VM / another HPC; CPU-only continuation; GPU only for re-extraction |

Portable archives, `SHA256SUMS.txt`, the per-file raw-audio checksum manifest,
and the export manifest JSON are written under the ignored
`artifacts/dj_clustering/_export/` and are never committed.
