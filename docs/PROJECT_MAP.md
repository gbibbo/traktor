# TRAKTOR ML V4 — Project Map

Inventario de archivos del proyecto. Actualizar al añadir ficheros nuevos.

## Configuración

| Archivo | Descripción |
| :--- | :--- |
| `config/v4.yaml` | Config principal: rutas, datasets (`test_20`, `musica` = biblioteca completa en `Música/`, git-ignorada), clustering, ordering |
| `src/v4/config.py` | Constantes centrales (sample rates, model names, defaults) |

## Common Utilities (`src/v4/common/`)

| Archivo | Descripción |
| :--- | :--- |
| `config_loader.py` | Cargar config con cascada: CLI > env > YAML |
| `path_resolver.py` | Resolver rutas de artifacts, audio, caches (laptop + HPC) |
| `catalog.py` | Construir y cargar catálogo de tracks (catalog.parquet); recursivo, tags, hash `audio`, deduplicado |
| `audio_utils.py` | Carga de audio, segmentación DJ (get_dj_segments, read_dj_segments por seek), get_audio_files |
| `demucs_utils.py` | Separación de stems (Demucs htdemucs) |
| `embedding_utils.py` | MERTEmbedder: embeddings MERT-v1-330M; load_track_embeddings (Phase 1 o representations/) |
| `logging_utils.py` | Logger JSONL + run manifests |
| `harmonic.py` | Compatibilidad armónica Camelot con transposición ±2 st (regla 2026-09-11) |
| `genre_taxonomy.py` | Similitud entre géneros (y subgéneros) de Beatport a partir de una taxonomía propia en JSON (la de `config/genre_taxonomy/taxonomia.json`): familias, género → familia, relaciones explícitas y regla por niveles; `Taxonomy.similarity/matrix`, `compare` (lo que usa el recomendador: alias de subgénero y géneros que no cuentan) y `validate` (DECISIONS 2026-11-06, puntos 10 y 11) |
| `tags.py` | Tags con mutagen: BPM, tonalidad, género, sello, energía de Mixed In Key; hash de audio sin tags; comentario idempotente; `read_release_tags` (Remixer, ISRC, fecha, ID de Spotify) |
| `duplicates.py` | Temas repetidos: qué copia se queda (MP3 320, si no la mejor calidad), decisiones de Gabriel (`duplicate_decisions.json`) y detección por capas (sonido, nombre, parecido, corte) |
| `beatport.py` | Beatport sin API oficial (JSON de la búsqueda pública, caché por consulta): match por niveles A ISRC / B misma versión / C solo otras versiones / D nada, y tags propuestos con la convención de Beatport (Artist sin remixers, Remixers = mezcla u "Original Mix", Label, Genre, Released) |
| `tempo.py` | Estimación de BPM (flujo espectral + autocorrelación) para temas sin BPM en tags |
| `dj_export.py` | Escritores rekordbox.xml, NML de Traktor y M3U8 con rutas absolutas |
| `traktor_nml.py` | Lectura de `collection.nml` solo con la biblioteca estándar: dónde está, si Traktor está abierto, ruta de cada entrada, cues propios (la usan `traktor_collection.py` y `tools/traktor_live/`) |

## Pipeline (`src/v4/pipeline/`)

| Archivo | Descripción |
| :--- | :--- |
| `phase0_ingest.py` | Escaneo + validación + catálogo + metadata merge |
| `phase1_extract.py` | GPU: Demucs + MERT + Essentia (con sharding) |
| `phase1_tags.py` | CPU: BPM/tonalidad/energía desde tags → features/bpm_key.parquet (sin Essentia; Windows); `--estimate-missing` |
| `phase1_merge_shards.py` | CPU: Consolida shards de Phase 1 → mert_perc.npy, mert_full.npy, track_uids.json |
| `phase2_cluster.py` | CPU: HDBSCAN L1/L2 + UMAP 2D → clustering/results_<hash>.parquet |
| `phase3_name.py` | CPU: Naming semántico de clusters (genre voting + fallback genérico) |
| `phase4_order.py` | CPU: Ordering greedy NN (cosine + BPM + Camelot key) → ordered_<hash>.parquet |
| `extract_representations.py` | CPU: representaciones congeladas (EffNet, MAEST, MERT por capas, CLAP, MAEST-HF, AST; full y HPSS) con segmentación V4 → representations/<variante>/ (fase 2 del plan); `--shard`, `--assemble-only`, `--folder`, `--low-priority`, filtros de duración |
| `organize.py` | Organizaciones estables con versiones (`orgs/<nombre>/`): `ingest` por carpeta, `import` de un config_hash, `build` (por alcance, respetando fusiones), `add` congelado, `link` (fusiones de temas: congelado o `--rebuild`), `show`; fusiones con nombre y color guardadas por versión, borradores y deshacer; `reorder` (orden a mano de una playlist) y `remove_fusion` (quitar una fusión sin mover temas) para la app |
| `dedupe.py` | Temas repetidos: `candidates`, `auto` (lo que la regla resuelve sola), `decide`, `import`, `apply` (saca las copias de una organización, versión 'dedupe'), `move-copies` / `restore-copies` (a `Música/_copias/` y de vuelta), `list`. La app lo usa al agregar música. Nunca borra archivos |
| `beatport_lookup.py` | Red: busca cada tema del dataset en Beatport → features/beatport.parquet/.csv + beatport_summary.json (nivel, datos de Beatport, tags actuales y propuestos). No escribe tags; caché en artifacts/v4/beatport_cache; `--folder`/`--limit` para la prueba, `--offline`, `--low-priority` |
| `beatport_subgenres.py` | Red: categoría, género y subgénero de Beatport de cada tema con coincidencia, leídos de la página del tema → features/beatport_genres.parquet/.csv; caché en artifacts/v4/beatport_cache/tracks/; `--limit`, `--offline`, `--low-priority` |
| `genre_model.py` | CPU: género en taxonomía Beatport para temas sin match: gate MAEST (estilos Discogs no electrónicos) + regresión logística sobre MAEST capa 7 ‖ CLAP entrenada con los géneros de Beatport confirmados; `--eval` (validación cruzada agrupada por artista → genre_model_eval.json), `--predict` (→ genre_pred.parquet/.csv). No escribe tags |
| `write_beatport_tags.py` | Escribe Artist, Remixers, Label, Genre y Released de Beatport (A y B confirmado) y el género del modelo con confianza >= 0.6 (C/D), solo MP3/FLAC. Simulación por defecto (→ features/tag_plan.csv); `--write` con respaldo features/tag_backup_<fecha>.csv y verificación del hash de audio; `--revert <csv>`; `--folder` para la prueba |
| `tag_vocals.py` | CPU: etiqueta Vocal (por defecto `clap_probe`: CLAP del tema entero + clasificador lineal de `models/vocal_probe/`, Vocal si >= 5 % con voz; también CLAP zero-shot o AST AudioSet) → features/vocals_<método>.parquet; `--write-tags` agrega " - Vocal" al comentario con respaldo CSV y `--revert` |
| `phase5_export.py` | CPU: Export M3U Traktor (UTF-8, rutas Windows) → playlists/V4_<N>/; `--formats m3u8,rekordbox,traktor`, `--rep`, `--out-root` |
| `usb_package.py` | Paquete portátil para Rekordbox y Traktor (con `--refresh <paquete> --from-org --org-name X`, rehace sus playlists desde la organización sin mover audio, cuando la biblioteca es el paquete): copia la música ordenada por playlist a un pendrive/disco + `rekordbox.xml` y `traktor.nml` con sus plantillas y `Preparar Rekordbox.bat` / `Preparar Traktor.bat`, m3u8 relativos; reanudable, `--verify hash`; `--refresh <paquete>` rehace todo lo que no es audio desde los tags actuales del paquete |
| `file_embeddings.py` | Guarda dentro de cada MP3/FLAC del paquete su embedding (`GEOB «TRAKTOR_ML <rep> v1»`; FLAC: comentario Vorbis en base64) y su lugar en la organización (JSON: versión, carpeta, playlist, posición, punto del mapa, track_uid); respeta la versión ID3, verifica que el hash del audio no cambie y no reescribe lo que ya está; WAV/AIFF no se tocan. `write`/`check --package`, `--folder` para una prueba chica, `--low-priority` |
| `package_library.py` | La biblioteca pasa a ser el paquete del pendrive: `adopt --package` apunta el catálogo a cada archivo del paquete por track_uid (rel_path relativo a `<paquete>/Musica`, source_path real, ruta original en `orig_rel_path`, tags mostrados releídos de los archivos), saca las filas sin archivo, actualiza duplicates.csv y decisiones, respalda todo antes y hace de `Música/` un junction a `<paquete>/Musica`. Nunca mueve ni borra audio |
| `package_tags.py` | Red: etiquetas y carátula de los temas de un paquete (no de la biblioteca): `lookup` (Beatport, Deezer de respaldo) → `rescue` (2.ª búsqueda de los incompletos: Deezer, iTunes, Beatport por título) → `plan` (plan.csv, summary.json, carátulas 600x600) → `write --target` (respaldo JSON, audio verificado) / `revert`; salida en artifacts/v4/package_tags/<paquete> |
| `traktor_collection.py` | Playlists de un paquete directo en `collection.nml` de Traktor (cerrado, con respaldo): `prepare` (saca los temas sin carátula para que Traktor los relea; texto desde los tags) y `playlists` (carpeta con todas las playlists; no escribe si falta un tema) |
| `rekordbox_prepare.ps1` | «Preparar Rekordbox» del paquete (otra PC, sin Python): escribe `rekordbox.xml` con la letra y carpeta del paquete y cuenta los archivos que faltan |
| `traktor_prepare.ps1` | «Preparar Traktor» del paquete (otra PC, sin Python): usa los temas de su `collection.nml` (ficha y análisis tal cual), el paquete solo para faltantes; `_informe.txt` |

## Evaluation (`src/v4/evaluation/`)

| Archivo | Descripción |
| :--- | :--- |
| `metrics.py` | ARI, NMI, Recall@k, MRR, NDCG, pairwise_auc, transition_score, noise_rate |
| `eval_runner.py` | Loop de evaluación: carga artifacts, calcula métricas, guarda JSON |
| `triplet_evidence.py` | Carga respuestas de tripletas del DJ, resuelve a track_uid, baselines clave/BPM |
| `genre_evidence.py` | Evidencia por género de Beatport (solo temas con género de Beatport): BPM, estilos de Discogs de MAEST, subgéneros, centroides y vecinos CLAP (con lift), géneros que Beatport mezcla entre versiones y un borrador de similitud → features/genre_evidence/ |
| `vocal_eval.py` | Evaluación de detectores de voz (protocolo `docs/plans/20260929_deteccion_voz.md`): Electrobyte por segundo (umbral en valid, métricas en test con IC bootstrap) y MTG-Jamendo `voice_instrumental` por tema (`--fetch-jamendo`, `--track-level`); detectores clap, ast y hdemucs; puntajes en artifacts/v4/vocal_eval/ |
| `representation_eval.py` | Instrumento de evaluación de representaciones: accuracy de tripletas por fuente con bootstrap pareado vs BPM, coherencia vs carpetas de v1 (fase 1 del plan) |
| `legacy_crosscheck.py` | Cruza las tripletas del DJ con los clusters versionados de legacy v1/v2, genre_discogs400 y playlists V4_5 (sin re-extraer audio) |

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
| `ui/review_app.py` | App local (http.server, solo 127.0.0.1 y con token) para cualquier persona: sirve la interfaz de revisión y ejecuta agregar música nueva (selector de carpetas, análisis con progreso, mantener o reorganizar), fusiones de temas (crear, aplicar, eliminar), orden a mano de una playlist, export para Rekordbox/Traktor y deshacer; `/cover/<tema>` sirve la carátula del archivo (o cover/folder.jpg de su carpeta). Se abre con `Abrir TRAKTOR ML.bat` o en la pestaña Organización de la app de descargas (`/api/ping`, `--exit-on-stdin-eof`; `?agregar=<carpeta>` abre «Agregar música nueva» con esa carpeta). Sin `Música/`, toca los temas del paquete de `usb_package.py` (`<unidad>:\TRAKTOR ML` o `--package`, por track_uid en `_organizacion.csv`) y queda en solo lectura |
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
| `test_vocal_eval.py` | Unit tests de vocal_eval (etiquetas .lab, grilla de 1 s, AUC, umbral por exactitud balanceada) |
| `test_representation_eval.py` | Unit tests de representation_eval (tripletas, bootstrap pareado, coherencia kNN/MAP) |
| `test_triplet_evidence.py` | Unit tests de triplet_evidence (dedup, resolución, baselines) |
| `test_tags_catalog.py` | Unit tests de tags.py y del catálogo recursivo (hash estable al escribir tags, dedup) |
| `test_beatport.py` | Unit tests sin red de beatport.py y de las consultas de beatport_lookup (niveles A-D, alias de artista, sellos, convención de tags, caché, subgénero y género completo de la página de un tema) |
| `test_genre_taxonomy.py` | Similitud por el árbol de familias, relaciones de género y de subgénero, subgénero faltante, matriz simétrica, validación de forma, `compare` con alias y la taxonomía en uso validada contra los géneros de Beatport |
| `test_genre_model.py` | Unit tests de genre_model.py (gate, etiquetas confirmadas alineadas por track_uids, validación cruzada) |
| `test_write_beatport_tags.py` | Unit tests de write_beatport_tags.py (reglas del plan, escritura real en MP3 ID3 2.3 y FLAC, hash estable, reversión) |
| `test_organize.py` | Unit tests de organize.py: build por alcance, add congelado, link congelado/rebuild, catálogo por alcance, ensamblado con --folder, orden a mano y quitar fusiones |
| `test_duplicates.py` | Unit tests de la regla de temas repetidos, la detección por capas, las decisiones, remove_tracks y la exclusión en Library |
| `test_playlist_review.py` | Unit tests del generador de la página de revisión |
| `test_review_app.py` | Unit tests de la app local: token y Host/Origin, audio con rangos, fusiones (aplicar, eliminar y deshacer), orden a mano, carpetas dentro/fuera de la biblioteca, carátulas || `test_dj_export.py` | Unit tests de rekordbox.xml / NML / M3U8, bpm_key desde tags, orden por energía, ventanas |
| `test_usb_package.py` | Paquete portátil: nombres FAT32, ubicación por playlist, plantillas NML y rekordbox.xml con sus `.ps1` reales, m3u8 relativos, copia, verificación y `--refresh` |
| `test_file_embeddings.py` | Embedding y lugar en los tags: MP3 (versión ID3 respetada) y FLAC, track_uid intacto, sin reescribir, WAV sin tocar, paquete sintético de punta a punta |
| `test_package_library.py` | Paquete sintético: `adopt` (catálogo, respaldo, junction, no toca una carpeta real), un orden nuevo que llega a las playlists del paquete sin mover ni cambiar el audio y conservando el origen, y la app que actualiza el pendrive al preparar |
| `test_traktor_collection.py` | traktor_collection.py con paquete y colección falsos: qué se saca, texto actualizado, rechazo si faltan temas, carpeta en orden y reemplazable |
| `test_sugerencias_itunes.py` | XML de iTunes de la prueba de sugerencias: orden, rutas codificadas, rondas sin repetir, no pisa ni borra un XML ajeno |
| `test_midi_monitor.py` | Monitor MIDI sin hardware: decodificación, mínimo y máximo por control, BPM del reloj, búsqueda de la entrada (K2 no abre el DJM) |
| `test_traktor_live_api.py` | Parche de `D2.qml` (ida y vuelta, idempotente), instalar y quitar en una carpeta de prueba, receptor HTTP con texto no ASCII, playlists `.m3u8`/`.nml` en orden y sin pisar archivos ajenos |
| `test_sugerencias_servicio.py` | Servicio automático: el supervisor sigue a Traktor y reintenta, el bucle termina con el archivo de parada, notificaciones sin sonido y sin repetir, acceso directo de Windows y LaunchAgent de macOS, control de PID |
| `test_tema_principal.py` | Tema principal: deck MASTER (suene o no), master tempo con secciones intro/main/outro, fader >= 90 % y 3 s de espera, faders de la DJM o de Traktor según la mezcla, tocado tras 30 s al aire, mezcla leída de `Traktor Settings.tsi`, registro de `Live`, receptor que pasa los eventos, DJM ausente sin notificación |
| `test_recomendador.py` | Sugerencias reales con colección e índice armados a mano: orden por parecido CLAP, BPM ±20, tonalidad (Traktor sobre tags, sin tonalidad no pasa), fuera el principal y lo tocado (también por ruta original), corte en N, modo sin CLAP, raíz del paquete, índice de `build_index`, lista escrita por `Suggestions`, recálculo por tema principal, pitch y tocado |
| `test_mapeo_tsi.py` | `mapeo_tsi.py` con un `.tsi` armado a mano (leer, reconstruir idéntico, detalle con condiciones, liberar un botón, quitar una asignación, puertos, archivo) y los mapeos del repo (K2 con 125 asignaciones, D0 libre y sin M2; teclado con 56) |
| `test_traktor_live_platforms.py` | Lo que cambia entre Windows y macOS en las sugerencias en vivo: rutas de la colección, camino del árbol, carpeta por defecto, botón nota o CC, camino aprendido, ajustes, python-rtmidi (falso), dónde busca Traktor el instalador; corren en Windows, macOS y Linux |
| `test_package_tags.py` | package_tags.py sin red: limpieza local de artista/título/mezcla, verificación de Deezer y de la 2.ª búsqueda, reglas del plan, escritura y reversión en un MP3 mínimo (audio intacto) |

En `tests/` (raíz): `test_check_staged.py` (chequeo pre-commit sobre repos git temporales),
`test_status_doc.py` (`docs/STATUS.md` con sus tres secciones y a lo sumo 6 KB) y
`test_spotify_soulseek_orchestrator.py` (`download_JIJIJI/`) y `test_soulseek_safety.py`
(tamaño/duración, descarga exacta, límites de procesos, carátulas y antivirus en consolidación), y
`test_descargas_app.py` (app de descargas: estado de cada tema por fase, guardado parcial, run
colgado, historial, ajustes, token y Host, cierre de procesos huérfanos, pestañas Colección y
Organización).

## Arnés de agentes y git

| Archivo | Descripción |
| :--- | :--- |
| `AGENTS.md` | Instrucciones canónicas para cualquier agente |
| `CLAUDE.md` | Importa `AGENTS.md` y agrega solo lo propio de Claude Code (hook de estado, skill, compactación) |
| `.claude/settings.json` | Permisos de auto mode y hook `SessionStart` que inyecta `docs/STATUS.md` (tope `head -c 8000`) |
| `.claude/skills/auditar/SKILL.md` | Protocolo de investigación: tres coordenadas de cada dato, revisión hostil, veredicto etiquetado, reglas del dominio |
| `tools/git_hooks/pre-commit` | Hook de git; se activa con `git config core.hooksPath tools/git_hooks` y corre `check_staged.py` |
| `tools/git_hooks/check_staged.py` | Bloquea audio, arrays, pesos, credenciales, archivos > 5 MB y contenido con credenciales (sin imprimirlas); `--all` revisa el árbol entero |

## Artifacts (generados, no en git)

```
artifacts/v4/datasets/<dataset_name>/
├── catalog.parquet                # Phase 0: catálogo de tracks
├── duplicate_decisions.json       # dedupe.py: veredictos de temas repetidos (qué copia se queda)
├── moved_copies.csv               # dedupe.py move-copies: copias movidas a Música/_copias (restore-copies)
├── ingest_report.json             # Phase 0: estadísticas de ingesta
├── embeddings/
│   ├── mert_perc.npy              # Phase 1: embeddings percusivos (N, 1024)
│   ├── mert_full.npy              # Phase 1: embeddings full mix (N, 1024)
│   ├── track_uids.json            # Phase 1: UIDs en orden (fuente de verdad de alineación)
│   └── shards/                    # Shards temporales Phase 1
├── features/
│   ├── bpm_key.parquet            # Phase 1: BPM, key, beat_confidence (también vía --essentia-only en CPU)
│   ├── beatport.parquet/.csv      # beatport_lookup: match A-D, datos de Beatport, tags actuales y propuestos
│   ├── beatport_summary.json      # beatport_lookup: conteos por nivel y cambios que se propondrían
│   ├── genre_model_eval.json      # genre_model --eval: validación cruzada y gate
│   ├── genre_pred.parquet/.csv    # genre_model --predict: género final (Beatport o modelo) con top-3
│   ├── tag_plan.csv               # write_beatport_tags (simulación): valor actual y nuevo por campo
│   └── tag_backup_<fecha>.csv     # write_beatport_tags --write: valores anteriores (para --revert)
├── evidence/
│   └── manual_triplets.csv        # triplet_evidence: respuestas resueltas a track_uid
├── catalog_success.parquet        # Phase 1 merge: catalog filtrado a N canónico (track_uids.json)
├── clustering/
│   ├── results_<hash>.parquet     # Phase 2: label_l1, label_l2, umap_x, umap_y
│   ├── config_<hash>.json         # Phase 2: parámetros + label_semantics
│   ├── names_<hash>.json          # Phase 3: nombres de clusters
│   └── ordered_<hash>.parquet     # Phase 4: results + columna 'position'
├── representations/<variante>/    # extract_representations: embeddings.npy + track_uids.json + manifest.json (+ cache/)
├── evaluation/
│   ├── <hash>_scores.json         # eval_runner: métricas calculadas
│   └── representations_<fecha>.json  # representation_eval: tabla de tripletas y coherencia
└── logs/
    └── *.jsonl                    # Logs estructurados por phase

playlists/V4_<N>/
├── L1_A_[nombre]/
│   └── L2_A1_[nombre].m3u        # Phase 5: playlists ordenadas por subcluster
├── All_Noise.m3u                  # Phase 5: tracks en ruido L1
└── _summary.txt                   # Phase 5: tabla resumen de clusters y tracks
```

## Trabajo paralelo de Gabriel (`download_JIJIJI/`)

Se incluye en git (sin audio; ver `AGENTS.md`). Lectura de playlists de Spotify
(`spotify_playlist.py`), búsqueda/descarga por Soulseek (`spotify_soulseek_download.py`), grabador
por loopback (`playlist_loopback_recorder.py`) y los reportes JSON/TXT de una prueba
(`soulseek_prueba_descarga/`).

`spotify_soulseek_orchestrator.py` es el punto de entrada único: recibe una URL y va tema a tema
(2026-11-06): el descargador busca cada tema en Soulseek y, si está, lo baja enseguida; lo que no se
consigue entra en la cola de grabación (`recording_queue.py`, `recording_queue.jsonl` del run) y el
orquestador lo graba de Spotify mientras sigue la búsqueda, en tandas de hasta 10 temas con su propia
playlist privada (borrada al terminar la tanda) y su carpeta `recordings/NN/`. El descargador trabaja con
2 temas a la vez por una sola sesión de Soulseek (`SockseekDaemon`: el daemon de sockseek, con `--remote`). Reúne los resultados bajo
`download_JIJIJI/runs/` (git-ignorado). `download_JIJIJI/README.md` documenta la
configuración inicial y el uso; `download_JIJIJI/CONFIGURAR_OTRO_ORDENADOR.md`, la instalación
paso a paso en otra PC (claves de Spotify y Soulseek, `sockseek.exe` en `tools/sockseek/`,
git-ignorado, y las dos autorizaciones de Spotify). La grabación crea checkpoints por tema en
`recording_progress.json` de cada tanda; una ejecución posterior sigue la búsqueda donde quedó y
reanuda la tanda sin terminar desde la primera pista pendiente. El grabador reutiliza el
escritor canónico de metadata y cover de `spotify_soulseek_download.py`, por lo
que ambos orígenes producen MP3 con el mismo criterio de nombres y etiquetas.
De cada tema disponible se prueban hasta 4 archivos (duración y álbum de Spotify primero) y cada
uno se verifica antes de aceptarlo; lo que aun así no se pudo bajar va a la cola de grabación.
[REEMPLAZADO 2026-11-06 por la búsqueda tema a tema: `soulseek_failed.json` y `recorded_failed/`
quedan solo en los runs anteriores.]
Al completar las ramas, el orquestador mueve todo el audio sin duplicados a
`download_JIJIJI/Playlists_DOWNLOAD/<nombre>/`; el run conserva sólo los JSON
principales y un subdirectorio `diagnostics/` con el detalle técnico. Si falta algún tema,
guarda lo conseguido y termina como `partial` (código 3).

| Archivo (`download_JIJIJI/app/`) | Descripción |
| :--- | :--- |
| `descargas_app.py` | App local (http.server, solo 127.0.0.1 y con token; una instancia; se apaga sola sin ventana ni descarga): vista previa de la playlist con autorización propia, salida de audio de Windows (elegir, probar con tono grabado, medidor del loopback), carpetas de destino y de trabajo, lanzar/detener/reanudar el orquestador, estado de cada tema leído de los JSON del run (también de runs lanzados desde la terminal), historial y guardado parcial de un run con error (`consolidate_audio` sin conjunto esperado). Ajustes y token en `%APPDATA%\traktor_descargas`. Además, la pestaña Colección (carpetas de descargas, biblioteca organizada y las agregadas a mano; temas y tamaño por subcarpeta y cuáles se eligen para generar playlists, en `collection.json`) y la pestaña Organización (lanza `src/v4/ui/review_app.py --exit-on-stdin-eof` en su propio proceso, que se cierra con la app después de la tarea en curso) |
| `index.html` | Interfaz (una página, sin dependencias salvo la fuente Figtree): pestañas Descargas, Colección y Organización arriba a la izquierda (la Organización, en un iframe); biblioteca, vista de playlist con tabla de temas y estado, barra inferior con progreso, salida de audio y carpeta |
| `run_job.py` | Ejecuta el orquestador y lo interrumpe como un Ctrl+C cuando aparece un archivo de señal |
| `audio_devices.ps1` | Lista las salidas de audio y fija la predeterminada (IMMDeviceEnumerator + IPolicyConfig) |
| `crear_acceso_directo.ps1` | Crea «Descargar playlists» en el Escritorio (`.venv\Scripts\pythonw.exe`, `icon.ico`) |
| `icon.svg`, `icon.ico` | Icono de la ventana y del acceso directo |

## Taxonomía de géneros (`tools/genre_taxonomy/`)

| Archivo | Descripción |
| :--- | :--- |
| `tools/genre_taxonomy/referencias.py` | Baja las taxonomías de referencia a `artifacts/v4/genre_taxonomy/inputs/`: los géneros de MusicBrainz y su árbol electrónico con relaciones (leídas de las páginas: la API no las da) y la jerarquía de Wikidata bajo «electronic music» |
| `config/genre_taxonomy/taxonomia.json` | La taxonomía en uso (red de agentes, aprobada en la ronda 2): 34 familias, los 47 géneros, 211 relaciones, regla, `sin_genero`, `como_genero`, notas y procedencia. Informe `docs/reports/taxonomia_generos_2026-11-06.md` |
| `config/genre_taxonomy/beatport_generos.json` | Los 47 géneros de Beatport con sus 136 subgéneros (leídos de beatport.com el 2026-11-06): la lista fija contra la que se valida la taxonomía |
| `config/genre_taxonomy/discogs_a_beatport.json` | Mapeo de los 400 estilos de Discogs (MAEST) a géneros de Beatport con pesos. Medido: acierta el 24 % del género (el clasificador propio, 38 %); no se usa |

## Revisión de playlists (`tools/playlist_review/`)

| Archivo | Descripción |
| :--- | :--- |
| `tools/playlist_review/build_review_page.py` | Genera `<carpeta de audio>/_revision_playlists.html`: mapa UMAP, playlists en orden, escucha de la secuencia, veredictos exportables a CSV; organizaciones con nombre (`--org-name X`, `X@N`; sin argumentos, todas) con versión, historial, temas nuevos y semillas; armado y export de semillas para `organize.py link --from-file`; corridas por hash a ciegas (`--org`, `--blind`); mezcla de cada tema (tag Remixer o nombre del archivo) |
| `tools/playlist_review/template.html` | Plantilla autónoma (sin servidor ni red; audio por rutas relativas); mapa con zoom (+/−, Ctrl + rueda, arrastre) y ancho ajustable; búsqueda con filtros por campo; barra de tiempo para avanzar o retroceder; secciones plegables; con la app, carátula en miniatura, ⇅ para reordenar temas arrastrando (el fantasma termina en el cursor) y papelera por fusión |

## Sugerencias en vivo dentro de Traktor (`tools/traktor_live/`, prueba de factibilidad)

| Archivo | Descripción |
| :--- | :--- |
| `tools/traktor_live/sugerencias_itunes.py` | Escribe la playlist "TRAKTOR ML Sugerencias" (temas de la colección de Traktor, con cues propios) en el XML de iTunes que Traktor tiene configurado, para probar si el nodo iTunes se vuelve a leer sin reiniciar; `--round N` cambia los temas, `--remove` borra el XML (solo si lo escribió el script); porcentaje de prueba en Composer, Grouping y Album Artist; `--path-probe`: un tema por carácter especial en la ruta (#, &, +, ;, no ASCII) para ver cuáles reconoce Traktor |
| `tools/traktor_live/midi_monitor.py` | Monitor MIDI (Windows y macOS, por `midi_io.py`; `--backend`): lista entradas, abre una por palabra del nombre (`--port DJM`), muestra CC y notas, mínimo y máximo por control y el BPM del reloj MIDI; dice si el puerto está en uso por otro programa |
| `tools/traktor_live/midi_io.py` | Entrada MIDI de las herramientas en vivo: `winmm` por ctypes (Windows, sin paquetes, por defecto) o `rtmidi` (python-rtmidi: macOS, Linux; también Windows con `--backend rtmidi`); los mensajes llegan empaquetados igual con los dos |
| `tools/traktor_live/sugerencias_explorer.py` | Prueba del refresco sin clics: la ronda como `... M3U.m3u8` y `... NML.nml` (nombres distintos: con el mismo, Traktor muestra un solo nodo; entradas copiadas de la colección) en `~/Music/TRAKTOR ML Sugerencias/` para el nodo Explorer, y con `--archive` en `History/` (Explorer > Archive); `--remove` borra solo sus archivos |
| `tools/traktor_live/traktor_qml/` | Copia sin cambios del módulo `Api/` de traktor-api-client (MIT, commit `1caf11f`): QML del Kontrol D2 que envía a `localhost:8080` el tema de cada deck, play, tiempo, master y canal al aire del mezclador interno. `Screens/ApiBrowserProbe.qml` es propio: agregada a la pantalla del D2 (cuyo navegador es el árbol que ve el DJ), reporta la ruta y los nombres del nivel actual y cumple órdenes del receptor: salir, elegir, entrar a una carpeta y `goto` (una ruta completa, paso a paso dentro de Traktor). Nunca entra a un tema (lo cargaría en el deck) |
| `tools/traktor_live/instalar_api_client.py` | Instala, revisa (`--status`) o quita (`--uninstall`) ese módulo: copia `Api/` y agrega dos líneas al `D2.qml` de la versión instalada; `--sonda` / `--quitar-sonda` agregan o quitan la sonda en `Screens/S8/Views/Screen.qml`; con copia previa en `artifacts/v4/traktor_backups/`; encuentra el Traktor más nuevo instalado en Windows (`Resources64/qml`) o macOS (`Traktor*.app/Contents/Resources/qml`), o `--d2`; pide permiso de administrador (UAC en Windows, contraseña por osascript en macOS); no hace nada con Traktor abierto |
| `tools/traktor_live/traktor_estado.py` | Receptor del módulo (solo biblioteca estándar, 127.0.0.1 y ::1, puerto exclusivo): una línea por tema cargado, play/pausa, canal al aire y master; anota los eventos en `artifacts/v4/traktor_live/*.jsonl`; cola de órdenes para la sonda (`POST /enqueue`, `GET /command/<lado>`, `wait_result`) y `GET /state` |
| `tools/traktor_live/sugerencias_vivo.py` | Sugerencias en vivo (guía: `docs/SUGERENCIAS_EN_VIVO.md`; probado en Windows el 2026-11-05, macOS con tests): un botón del controlador (en el K2, por defecto, mantener Ch15.Note.D#0 y apretar D0; `--aprender` elige otra nota o CC, con o sin modificador, `--midi` otro controlador, `--n`, `--dir`, `--backend`, `--listar`) guarda la carpeta, escribe N temas nuevos en `TRAKTOR ML Sugerencias NML.nml` (Windows: raíz de la unidad; macOS: `/Users/Shared`) y lleva el árbol a esa playlist pasando por su carpeta; otra pulsación vuelve; si una ida o vuelta falla, la siguiente la repite; si Traktor muestra la playlist por otro camino, lo aprende al llegar a mano. Ajustes en `artifacts/v4/traktor_live/sugerencias_vivo.json`. Incluye el receptor (no correr `traktor_estado.py` a la vez). Las listas salen de `recomendador.py` (30 por defecto); al azar sin índice o sin tema principal |
| `tools/traktor_live/recomendador.py` | Sugerencias reales (solo biblioteca estándar): los N temas más parecidos por CLAP al tema principal con BPM ±20 y tonalidad compatible hasta 2 semitonos (corte 0,60), sin el principal ni lo tocado; tonalidad y BPM de la colección de Traktor y, si falta, de los tags; solo la copia del paquete que Traktor tiene; el principal se reconoce también por su ruta original de `Música/`; modo `sin_clap` si no está en la biblioteca. Orden por promedio pesado de componentes `clap`, `bpm`, `clave`, `genero` (similitud de la taxonomía de géneros), `tiv` (`"pesos"` en los ajustes; por defecto solo CLAP), anotados en cada lista. `Recommender.update/refresh/snapshot`, `load_recommender` |
| `tools/traktor_live/indice_recomendacion.py` | Arma `artifacts/v4/traktor_live/indice_recomendacion.json` (con el Python del pipeline; versión 3): por tema del paquete, track_uid, archivo, ruta original, BPM y tonalidad de tags, embedding CLAP normalizado en base64, género (Beatport o clasificador >= 0.6) y subgénero de Beatport. Correrlo cada vez que cambie la biblioteca |
| `tools/traktor_live/tema_principal.py` | Tema principal y temas tocados con las reglas de Gabriel (DECISIONS 2026-11-06, puntos 6 y 7): `Tracker` (lógica pura, eventos con su instante: modo deck MASTER o master tempo con secciones, fader >= 90 %, 3 s de espera; tocado = 30 s al aire con el fader >= 50 %), `Live` (candado, registro `principal`/`tocado`/`fader`/`mezcla`, avisos, `on_principal`), `mixer_mode`/`MixerWatch` (`Routing.Mixer.External` de `Traktor Settings.tsi`). Lo usa `sugerencias_vivo.run_session`, que lee la DJM en un hilo aparte |
| `Sugerencias en vivo.bat` / `Sugerencias en vivo.command` | Lanzadores de doble clic de `sugerencias_vivo.py` (Windows / macOS) en la raíz del repo, con ventana |
| `tools/traktor_live/sugerencias_servicio.py` | Servicio automático: `instalar` deja un acceso directo a `pythonw` en la carpeta Inicio de Windows (o un LaunchAgent en macOS) y lo arranca; el proceso, sin ventana, empieza `sugerencias_vivo.run_session` cuando se abre Traktor y la termina cuando se cierra; notificaciones del sistema sin sonido (Listo y problemas); registro en `artifacts/v4/traktor_live/servicio.log`; `estado`, `quitar`, `correr` |
| `Activar sugerencias automaticas.bat` / `.command` | Activan el servicio automático con doble clic (crean `.venv` si falta); con `estado` o `quitar` como argumento, lo revisan o lo quitan |
| `docs/plans/PROMPT_ETAPA_RECOMENDACION.md` | Prompt para empezar en un chat nuevo la etapa de recomendación en vivo: qué existe, qué definir con Gabriel (tema principal con mezclador interno o DJM, registro de lo que suena, cómo se generan las sugerencias, qué copia sugerir, evaluación), restricciones y la primera respuesta esperada |
| `docs/SUGERENCIAS_EN_VIVO.md` | Guía de las sugerencias en vivo: cómo funciona, requisitos, instalación en Windows y macOS, desde cero en otro ordenador, uso, problemas frecuentes |
| `tools/traktor_live/mapeo_tsi.py` | Exporta un mapeo de controlador o de teclado de Traktor (de `Traktor Settings.tsi` u otro `.tsi`) a un `.tsi` con el formato de NI; `--detalle` muestra cada asignación (comando con nombre, modo, valor, condiciones de modificador), `--liberar "Ch15.Note.D0"` quita todas las de un control y `--quitar "Ch15.CC.021:211"` una sola; `--listar`, `--puertos`; no toca la configuración de Traktor |
| `tools/traktor_live/mapeos/` | Mapeos de Gabriel para importar en otro ordenador: `Xone K2 - TRAKTOR ML (D0 libre).tsi` (125 asignaciones, sin la preescucha: Load Selected y Modifier #2 de D0 y el Seek del encoder derecho con M2) y `Teclado - TRAKTOR ML.tsi` (56); su README dice cómo importarlos y regenerarlos |
| `PROMPT_LLM.md` | Prompt mínimo para que otro LLM guíe a una persona a replicar las sugerencias en vivo: fuentes, pasos con su comprobación, fallas conocidas y límites |

## Herramienta de feedback del DJ (`tools/dj_feedback/`)

| Archivo | Descripción |
| :--- | :--- |
| `tools/dj_feedback/template.html` | Plantilla HTML mínima: cola fija de tripletas + lista de temas con reproducir + CSV |
| `tools/dj_feedback/build_feedback_page.py` | Genera `dj_feedback.html` (embebe filenames del dataset + ruta Windows) y `playlists/feedback/all_tracks.m3u` |
| `tools/dj_feedback/dj_feedback.html` | Página generada para `test_20` (243 temas); se copia a la carpeta de música en Windows |
| `tools/dj_feedback/answers/*.csv` | Respuestas de tripletas exportadas por Gabriel (evidencia humana, una fecha por archivo) |
| `docs/DJ_FEEDBACK_TOOL.md` | Uso, esquemas CSV y requisito de agrupación con semillas |
| `tools/lightning/README.md` | Marca `tools/lightning/` como opcional/legacy e inerte por defecto (ver también `AGENTS.md`) |
| `tools/lightning/switch_machine.py` | OPCIONAL/LEGACY. Cambiar la máquina del Studio (CPU/GPU) vía lightning_sdk; reinicia el Studio y gasta créditos. Nunca automático; solo con aprobación explícita |
| `tools/lightning/gpu_autorun.sh` | OPCIONAL/LEGACY. Hook de arranque del Studio (`~/.lightning_studio/on_start.sh`): con GPU + `artifacts/gpu_autorun.flag` corre fase 1 y vuelve a CPU. No-op fuera de Lightning |

## Documentación (`docs/`)

| Archivo | Descripción |
| :--- | :--- |
| `docs/PROJECT_MAP.md` | Este archivo |
| `docs/V4_USAGE.md` | Instrucciones de uso end-to-end (Phase 0→5, UI, escalar a 2000 tracks) |
| `docs/v4/TODO.md` | Progress tracker con fechas de completado |
| `docs/STATUS.md` | Estado actual, pendientes y recetas de ejecución; se inyecta al abrir cada sesión de Claude Code |
| `docs/v4/JOBS_STATUS.md` | Legado: registro de jobs Slurm de marzo de 2026 (reemplazado por `docs/STATUS.md`) |
| `docs/LESSONS_LEARNED.md` | Base de conocimiento de lecciones aprendidas |
| `reports/Revisión científica de armonía TIV.md` y `research_notes/Revisión científica de armonía TIV/` | Revisión metodológica del informe TIV (2026-11-06): fuentes científicas y comerciales, verificación local y propuesta de escucha; no modifica el protocolo vigente |
| `docs/HARMONIC_COMPATIBILITY.md` | Regla de compatibilidad armónica aprobada (relaciones, transposición, ejemplo 12A) |
| `docs/plans/recomendacion_playlists_plan.md` | Plan de las playlists en vivo (parecidos y dirección: mantener, más, menos, introspectiva): datos de sets (historia de Traktor, djmix-dataset), inventario completo de ideas con origen y etapa, evaluación y pendientes (DECISIONS 2026-11-07) |
| `docs/plans/representation_model_plan.md` | Plan operativo vigente: evaluación, tabla de representaciones, 1001Tracklists, entrenamiento en Kaggle, MVP |
| `docs/DECISIONS.md` | Registro fechado de decisiones de Gabriel que condicionan el trabajo (prevalecen sobre planes anteriores) |
| `docs/reports/` | Informes fechados de resultados (baselines de tripletas; `scientific_review_2026-09-12.md` compara todas las versiones y fija el plan; `mvp_biblioteca_2026-09-27.md` biblioteca completa, CLAP, Vocal y export; `temas_repetidos_2026-09-29.md` versiones frente a duplicados y cómo detectarlos; `organizacion_escritorio_2026-11-04.md` tamaños y opciones para la pestaña Organización en el PC de escritorio; `embeddings_en_los_archivos_2026-11-04.md` si los embeddings pueden ir dentro de los tags de cada archivo) |
| `v4_implementation_plan.md` | Plan de implementación completo (rev.5) |
| `docs/DJ_FEEDBACK_TOOL.md` | Herramienta de escucha y anotación del DJ (tripletas con cola fija) |
