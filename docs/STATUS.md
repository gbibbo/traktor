# TRAKTOR ML — Estado actual

Se inyecta al abrir cada sesión de Claude Code (hook `SessionStart` en `.claude/settings.json`).
Es el estado final, no una crónica: se edita en su lugar cuando algo cambia. Tope 6 KB
(`tests/test_status_doc.py`). La historia vive en `git log`, `docs/DECISIONS.md` y `docs/reports/`.

## CURRENT STATE (2026-10-05)

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
- **Temas repetidos, cerrado** (DECISIONS 2026-09-29, 8-13): una copia por tema; `biblioteca` v11
  (1760 temas); copias en `Música/_copias/` (`dedupe.py restore-copies` las devuelve).
- **Smart App Control activo en la laptop**: bloquea DLLs nuevas; `.venv` fijado en torch 2.7.1 y
  pyarrow 16.1 (`requirements_v4.txt`). CLAP da lo mismo que con torch 2.14.
- **Tripletas (n = 57): ninguna representación supera al BPM de forma concluyente.** Entre
  representaciones decide la escucha de playlists enteras, no el puntaje.
- **Etiqueta Vocal**: método nuevo `tag_vocals.py --method clap_probe` (por defecto): CLAP del tema
  entero + clasificador lineal entrenado con Electrobyte; Vocal si >= 5 % del tema tiene voz (regla de
  Gabriel). En MTG-Jamendo por tema: 90 % de exactitud balanceada, recall 100 %, 20 % de
  instrumentales marcados (con 25 %: 94 %). Cálculo de la colección en curso (~6 h); la escritura
  (`--write-tags`, agrega o quita « - Vocal», con respaldo) la corre Gabriel. Hoy: 542 con la marca vieja.
- **Tags de Beatport escritos y verificados** (Gabriel corrió `write_beatport_tags.py`, 2026-09-29):
  1087 archivos (Genre 785, Released 658, Label 463, Artist 395, Remixers 349), 0 errores, track_uid
  intacto; re-simulación: 0 cambios pendientes. Catálogo refrescado (respaldo
  `catalog_before_beatport_tags_20260929.parquet`). Los 110 sin confirmar conservan su género.
- **`download_JIJIJI/`**: descarga de playlists de Spotify por Soulseek, con grabación por loopback
  como respaldo; punto de entrada `spotify_soulseek_orchestrator.py`. Soulseek exige tamaño
  proporcional, formatos MP3/FLAC/WAV y antivirus antes de incorporar archivos (README del módulo).
- **Chequeo pre-commit** en `tools/git_hooks/` (audio, arrays, pesos, credenciales, archivos > 5 MB).

## OPEN ITEMS (por prioridad; LOCAL = trabajo del agente, GABRIEL = lo hace o decide Gabriel)

1. GABRIEL: «Preparar para Rekordbox y Traktor» de nuevo (`V4_1` aún tiene los temas repetidos) e
   importar y validar en Rekordbox (pendrive) y Traktor (TODO 6.7): ¿se ve el Remixer de cada tema?
2. GABRIEL: probar la app local (`Abrir TRAKTOR ML.bat`): agregar música nueva, fusiones de temas
   (clic derecho, papelera), orden a mano (⇅), preparar para Rekordbox/Traktor y deshacer.
3. LOCAL: terminar `tag_vocals.py --method clap_probe` sobre musica, revisar cambios; GABRIEL: correr
   `tag_vocals.py --dataset-name musica --write-tags` (usa la caché, es rápido).
4. GABRIEL: 1001Tracklists como fuente del modelo (DECISIONS 2026-09-12, punto 3): fuente de los
   tracklists y descarga del audio se consultan antes de implementar.
5. LOCAL, después del MVP: modelo de representación propio (`docs/plans/representation_model_plan.md`,
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
