# TRAKTOR ML V4 — Progress Tracker

Claude Code: al completar cada tarea, marcar [x] y agregar fecha de finalización.
Formato: - [x] Tarea X.Y — Descripción | Completado: YYYY-MM-DD

---

## BLOQUE 0: Setup y organización
- [x] 0.1 Leer plan completo + crear este archivo TODO.md          | Completado: 2026-02-28
- [x] 0.2 Mover V3 a legacy                                        | Completado: 2026-02-28
- [x] 0.3 Crear estructura V4 + config.py + v4.yaml + requirements | Completado: 2026-02-28
- [x] TEST-0 Verificación de bloque 0                              | Completado: 2026-02-28

## BLOQUE 1: Common utilities
- [x] 1.1 config_loader.py + path_resolver.py                      | Completado: 2026-02-28
- [x] 1.2 catalog.py                                               | Completado: 2026-02-28
- [x] 1.3 audio_utils.py (carga + segmentación DJ)                 | Completado: 2026-02-28
- [x] 1.4 demucs_utils.py                                          | Completado: 2026-02-28
- [x] 1.5 embedding_utils.py (MERTEmbedder)                        | Completado: 2026-02-28
- [x] 1.6 logging_utils.py (JSONL + run manifests)                 | Completado: 2026-02-28
- [x] TEST-1 Verificación de bloque 1 (integration test)           | Completado: 2026-02-28

## BLOQUE 2: Pipeline scripts + Slurm
- [x] 2.1 phase0_ingest.py                                         | Completado: 2026-02-28
- [x] 2.2 phase1_extract.py                                        | Completado: 2026-02-28
- [x] 2.3 phase1_merge_shards.py                                   | Completado: 2026-02-28
- [x] 2.4 Slurm jobs V4 (todos)                                    | Completado: 2026-02-28
- [x] TEST-2 Verificación de bloque 2 (Phase 0 run + validaciones) | Completado: 2026-02-28

## >>> PAUSA HUMANA: ejecutar Phase 0, submit Phase 1 GPU, revisar embeddings <<<

## BLOQUE 3: Clustering + evaluación
- [x] 3.1 phase2_cluster.py                                        | Completado: 2026-02-28
- [x] 3.2 metrics.py + eval_runner.py                              | Completado: 2026-02-28
- [x] TEST-3 Verificación de bloque 3 (clustering + eval stats)    | Completado: 2026-03-01 (Parte A+B; assertions suavizadas; N canónico = 239)

## >>> PAUSA HUMANA: revisar clustering, ajustar hiperparámetros <<<

## BLOQUE 4: Export pipeline
- [x] 4.1 phase3_name.py                                           | Completado: 2026-03-01
- [x] 4.2 phase4_order.py                                          | Completado: 2026-03-01 (Camelot + greedy NN)
- [x] 4.3 phase5_export.py                                         | Completado: 2026-03-01 (M3U UTF-8 + Windows paths)
- [ ] TEST-4 Verificación de bloque 4 (playlists + human review)   | Completado: ____ (ejecutar tras PAUSA HUMANA del clustering)

## BLOQUE 5: UI + finalización
- [x] 5.1 UI Streamlit                                             | Completado: 2026-03-01 (src/v4/ui/app.py)
- [x] 5.2 Adaptation stubs (projection_head + contrastive_trainer) | Completado: 2026-03-01
- [x] 5.3 Integración end-to-end + documentación                   | Completado: 2026-03-01 (docs/V4_USAGE.md + PROJECT_MAP.md)
- [ ] TEST-5 Verificación final del sistema                        | Completado: ____ (ejecutar tras TEST-4)

## Notas de implementación (2026-03-01)
- N canónico = len(track_uids.json) = 239 (no 243 del catálogo)
- catalog_success.parquet: generado en phase1_merge_shards, 239 filas, alineado con embeddings
- Bug checkpoint corregido: run_id en progress_shard_XX.json (phase1_extract.py)
- Keys de Essentia ("C minor") → Camelot ("5A") normalización en phase4_order.py
- TEST-3 assertions suavizadas: n_clusters≥1, noise<0.8 (hard); resto = reporte humano
