# TRAKTOR ML — Estado actual

Se inyecta al abrir cada sesión de Claude Code (hook `SessionStart` en `.claude/settings.json`).
Es el estado final, no una crónica: se edita en su lugar cuando algo cambia. Tope 6 KB
(`tests/test_status_doc.py`). La historia vive en `git log`, `docs/DECISIONS.md` y `docs/reports/`.

## CURRENT STATE (2026-09-29)

- **Foco: MVP sobre la biblioteca completa** para Rekordbox y Traktor, en CPU local (DECISIONS
  2026-09-27). Dataset `musica` = `Música/` en la raíz del repo (git-ignorada): 3086 archivos, 2303
  temas únicos. `test_20` (243 temas) queda como banco de evaluación con tripletas y carpetas propias.
- **Organización elegida: CLAP + BPM** (peso 0.3), Ward con 15 carpetas y ~35 temas por playlist,
  config `fb78f2f6`, 1786 temas. Export en `artifacts/v4/datasets/musica/exports/V4_1/`
  (`m3u8/`, `rekordbox.xml`, `traktor.nml`). Gabriel la eligió escuchando frente a MAEST-HF capa 7
  (`9e5784c3`), con la salvedad de que no fue del todo a ciegas (DECISIONS 2026-09-29).
- **Organizaciones estables** (`src/v4/pipeline/organize.py`, `orgs/<nombre>/` con versiones): la
  elegida es `biblioteca` v1 (importada de `fb78f2f6`, mismo mapa; su export `exports/biblioteca_v1/`
  es idéntico a `V4_1`). `add` agrega música con lo existente congelado; `link` exige temas juntos
  (congelado o `--rebuild`); `build --scope` organiza una carpeta sola con su propio mapa.
- **Smart App Control activo en la laptop**: bloquea DLLs nuevas; `.venv` fijado en torch 2.7.1 y
  pyarrow 16.1 (`requirements_v4.txt`). CLAP da lo mismo que con torch 2.14.
- **Tripletas (n = 57): ninguna representación supera al BPM de forma concluyente.** Entre
  representaciones decide la escucha de playlists enteras, no el puntaje.
- **Etiqueta Vocal** escrita en el comentario de 542 archivos (CLAP zero-shot, umbral 0.145).
- **Beatport (tabla aparte, tags todavía sin escribir)**: `beatport_lookup.py` encontró 1455 de 1823
  temas (115 por ISRC, 1340 misma versión, 1230 de ellos confirmados por sello, fecha o duración);
  103 solo tienen otro remix y 265 no están. `genre_model.py` (MAEST capa 7 + CLAP, 15 géneros):
  38 % de acierto, 72 % en el top 3 (18 % la clase más frecuente); con confianza >= 0.6, 50 %.
  `write_beatport_tags.py` simula 1087 archivos a cambiar (1338 con datos de Beatport, 171 con
  género del modelo); la escritura real la bloqueó el permiso de auto mode y espera a Gabriel.
- **`download_JIJIJI/`**: descarga de playlists de Spotify por Soulseek, con grabación por loopback
  como respaldo; punto de entrada `spotify_soulseek_orchestrator.py`.
- **Chequeo pre-commit** en `tools/git_hooks/` (audio, arrays, pesos, credenciales, archivos > 5 MB).

## OPEN ITEMS (por prioridad; LOCAL = trabajo del agente, GABRIEL = lo hace o decide Gabriel)

1. GABRIEL: importar y validar `V4_1` en Rekordbox (pendrive) y Traktor (TODO 6.7).
2. GABRIEL: probar la app local (`Abrir TRAKTOR ML.bat`): agregar música nueva, fusiones de temas
   (clic derecho), preparar para Rekordbox/Traktor y deshacer. Plan `docs/plans/20260929_app_local.md`.
3. GABRIEL: decidir cómo tratar los temas repetidos: ≈ 18 copias de más en 16 grupos, todas en la
   misma playlist que el original (propuesta en `docs/reports/temas_repetidos_2026-09-29.md`).
4. GABRIEL: autorizar la escritura de tags (`write_beatport_tags.py --write`, primero `--folder`) y
   decidir el género de los 110 matches sin confirmar (35 tienen género; DECISIONS 2026-09-29).
5. LOCAL: ajustar el umbral Vocal con la lista de chequeo (`tag_vocals.py --check-list`).
6. GABRIEL: 1001Tracklists como fuente del modelo (DECISIONS 2026-09-12, punto 3): fuente de los
   tracklists y descarga del audio se consultan antes de implementar.
7. LOCAL, después del MVP: modelo de representación propio (`docs/plans/representation_model_plan.md`,
   fases 3 a 5).

## RUN RECIPES

- Intérprete en Windows: `.venv/Scripts/python.exe` (este `.venv` es Python 3.12; AGENTS.md declara 3.11).
- Tests: `.venv/Scripts/python.exe -m pytest tests -q`.
- Pipeline del MVP, comandos exactos: `docs/V4_USAGE.md`, sección 0b. La organización elegida sale de
  `phase2_cluster.py --dataset-name musica --rep clap_full --bpm-weight 0.3 --method ward --n-l1 15
  --l2-target-size 35 --pca-dim 50` y después Phases 3 a 5 con `--rep clap_full`.
- **App (uso normal)**: doble clic en `Abrir TRAKTOR ML.bat` (o `src/v4/ui/review_app.py
  [--org-name X] [--port N]`); abre el navegador sobre la organización `biblioteca`.
- Organizaciones: `organize.py ingest --scope "<carpeta>"` · `build --name X [--scope "<carpeta>"]` ·
  `add --name biblioteca --scope "<carpeta>"` · `link --name biblioteca --track "<texto>" --track
  "<texto>" [--rebuild]` o `link --name X --from-file semillas.json` · `show --name X`. Export: `phase5_export.py --org-name X --formats
  m3u8,rekordbox,traktor --out-root artifacts/v4/datasets/musica/exports`; página: `--org-name X`.
- Revisar escuchando: `tools/playlist_review/build_review_page.py --dataset-name musica`
  (varias organizaciones a ciegas: `--org <hash> --org <hash> --blind`).
- Corrida larga: primero `--folder "<subcarpeta>"` y revisar la salida; después la completa con
  `--low-priority`.
- Chequeo pre-commit: activar una vez por clon con `git config core.hooksPath tools/git_hooks`;
  revisar todo el árbol con `python tools/git_hooks/check_staged.py --all`.
