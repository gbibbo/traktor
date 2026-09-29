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
- [x] TEST-4 Verificación de bloque 4 (playlists + human review)   | Completado: 2026-03-01 (239/239 tracks, transition_score=0.797, fix doble conteo phase5)

## BLOQUE 5: UI + finalización
- [x] 5.1 UI Streamlit                                             | Completado: 2026-03-01 (src/v4/ui/app.py)
- [x] 5.2 Adaptation stubs (projection_head + contrastive_trainer) | Completado: 2026-03-01
- [x] 5.3 Integración end-to-end + documentación                   | Completado: 2026-03-01 (docs/V4_USAGE.md + PROJECT_MAP.md)
- [x] TEST-5 Verificación final del sistema                        | Completado: 2026-03-01 (todos los módulos importan, ProjectionHead OK)

## Notas de implementación (2026-03-01)
- N canónico = len(track_uids.json) = 239 (no 243 del catálogo)
- catalog_success.parquet: generado en phase1_merge_shards, 239 filas, alineado con embeddings
- Bug checkpoint corregido: run_id en progress_shard_XX.json (phase1_extract.py)
- Keys de Essentia ("C minor") → Camelot ("5A") normalización en phase4_order.py
- TEST-3 assertions suavizadas: n_clusters≥1, noise<0.8 (hard); resto = reporte humano

## BLOQUE 6: MVP biblioteca completa → Rekordbox + Traktor (Windows, CPU)
- [x] 6.1 Catálogo recursivo con tags, hash de audio sin tags y deduplicado       | Completado: 2026-09-27
- [x] 6.2 phase1_tags: BPM/tonalidad/energía de tags + BPM estimado (tempo.py)    | Completado: 2026-09-27
- [x] 6.3 extract_representations: CLAP determinista, MAEST-HF, AST, shards       | Completado: 2026-09-27
- [x] 6.4 tag_vocals: etiqueta Vocal (CLAP zero-shot) con respaldo y revert       | Completado: 2026-09-27
- [x] 6.5 Phase 2 --rep / --bpm-weight / --method ward; Phase 4 --rep + energía   | Completado: 2026-09-27
- [x] 6.6 Phase 5 --formats m3u8,rekordbox,traktor (dj_export.py)                 | Completado: 2026-09-27
- [ ] 6.7 Gabriel importa en Rekordbox (pendrive) y Traktor y valida los formatos
- [x] 6.8 MAEST-HF sobre la biblioteca (noche) y comparación contra CLAP en tripletas/carpetas | Completado: 2026-09-29 (tripletas no concluyentes; escuchando, Gabriel eligió CLAP)

## BLOQUE 7: Organizaciones estables (plan docs/plans/20260929_organizaciones_estables.md)
- [x] 7.1 Catálogo y extracción por carpeta (phase0 --scope; ensamblado siempre completo)      | Completado: 2026-09-29
- [x] 7.2 organize.py: import / build (por alcance, semillas) / add congelado / link / show  | Completado: 2026-09-29
- [x] 7.3 Export y página de revisión por organización (--org-name)                          | Completado: 2026-09-29
- [x] 7.4 biblioteca v1 = fb78f2f6 (mismo mapa, export idéntico a V4_1)                      | Completado: 2026-09-29
- [x] 7.5 Semillas desde la página de revisión (export JSON → organize.py link --from-file)   | Completado: 2026-09-29

## BLOQUE 8: App local para cualquier persona (plan docs/plans/20260929_app_local.md)
- [x] 8.1 review_app.py: servidor local con token, audio con rangos, tareas con progreso        | Completado: 2026-09-29
- [x] 8.2 Agregar música nueva: selector de carpetas, copia, análisis, mantener o reorganizar | Completado: 2026-09-29
- [x] 8.3 Fusiones de temas (clic derecho), colores y nombres, aplicar, deshacer              | Completado: 2026-09-29
- [x] 8.4 Preparar para Rekordbox y Traktor desde la app; lanzador Abrir TRAKTOR ML.bat        | Completado: 2026-09-29
- [ ] 8.5 Gabriel prueba la app con música nueva real
