# TRAKTOR ML — Estado actual

Se inyecta al abrir cada sesión de Claude Code (hook `SessionStart` en `.claude/settings.json`).
Es el estado final, no una crónica: se edita en su lugar cuando algo cambia. Tope 6 KB
(`tests/test_status_doc.py`). La historia vive en `git log`, `docs/DECISIONS.md` y `docs/reports/`.

## CURRENT STATE (2026-11-04)

- **Foco: MVP sobre la biblioteca completa** para Rekordbox y Traktor, en CPU local (DECISIONS
  2026-09-27). Dataset `musica` = `Música/` en la raíz del repo (git-ignorada): 37,8 GB, 2440 archivos
  de audio, 1823 temas en el catálogo. `test_20` (243 temas): evaluación (tripletas, carpetas propias).
- **Organización elegida: CLAP + BPM** (peso 0.3), Ward con 15 carpetas y ~35 temas por playlist,
  config `fb78f2f6`, 1786 temas. Gabriel la eligió escuchando frente a MAEST-HF capa 7
  (`9e5784c3`), con la salvedad de que no fue del todo a ciegas (DECISIONS 2026-09-29).
- **Organizaciones estables** (`src/v4/pipeline/organize.py`, `orgs/<nombre>/` con versiones): la
  elegida es `biblioteca` (v1 importada de `fb78f2f6`, mismo mapa). `add` agrega música con lo
  existente congelado; `link` exige temas juntos (congelado o `--rebuild`); `build --scope`
  organiza una carpeta sola con su propio mapa.
- **Temas repetidos, cerrado** (DECISIONS 2026-09-29, 8-13): una copia por tema; `biblioteca` v11
  (1760 temas); copias en `Música/_copias/` (`dedupe.py restore-copies` las devuelve). Export
  `exports/biblioteca_v11/` (2026-10-08): 51 playlists, 0 rutas faltantes, 907 Vocal, 27 GB.
- **Smart App Control activo en la laptop**: bloquea DLLs nuevas; `.venv` fijado en torch 2.7.1 y
  pyarrow 16.1 (`requirements_v4.txt`).
- **Tripletas (n = 57): ninguna representación supera al BPM de forma concluyente.** Entre
  representaciones decide la escucha de playlists enteras, no el puntaje.
- **Etiqueta Vocal (clap_probe, calculada 2026-10-06)**: CLAP del tema entero + clasificador lineal de
  Electrobyte; Vocal si >= 5 % del tema tiene voz (regla de Gabriel). 936 de 1823 temas Vocal. Escrita
  por Gabriel 2026-10-06 (respaldo `vocal_tags_backup_20261006_111119.csv`): marca y cálculo coinciden
  en los 1792 MP3/FLAC. Desde 11-04 también la usan la app al agregar música y `organize.py ingest`.
- **Tags de Beatport escritos y verificados** (Gabriel corrió `write_beatport_tags.py`, 2026-09-29):
  1087 archivos, 0 errores, track_uid intacto. Catálogo refrescado (respaldo
  `catalog_before_beatport_tags_20260929.parquet`). Los 110 sin confirmar conservan su género.
- **Pendrive `D:\TRAKTOR ML` (copia en `C:\TRAKTOR ML`)** (DECISIONS 2026-10-30, 11-02): los 1760
  temas de `biblioteca` por playlist; `_organizacion.csv` da archivo y track_uid de cada uno
  (verificado en ambas el 11-04). Carátulas 1630/1760. Rekordbox: `Preparar Rekordbox.bat`.
- **`download_JIJIJI/`**: playlists de Spotify por Soulseek, grabando por loopback lo que falte
  (se puede posponer); entrada `spotify_soulseek_orchestrator.py`; otra PC:
  `CONFIGURAR_OTRO_ORDENADOR.md`. App «Descargar playlists» (`app/`): pestañas Descargas,
  Colección y Organización (DECISIONS 11-02); logs: `runs/logs/`.
- **Chequeo pre-commit** en `tools/git_hooks/` (audio, arrays, pesos, credenciales, archivos > 5 MB).

## OPEN ITEMS (por prioridad; LOCAL = trabajo del agente, GABRIEL = lo hace o decide Gabriel)

1. GABRIEL: probar el pendrive en Rekordbox con la controladora. LOCAL: los tags corregidos están
   solo en el paquete; pasarlos a `Música/` antes del próximo `usb_package.py` (si no, los pisa).
   `Preparar Traktor.bat` del pendrive sigue armando un NML: Traktor importa una sola playlist.
   Refrescar Traktor de esta PC (`traktor_collection.py`).
2. GABRIEL: probar la app local (`Abrir TRAKTOR ML.bat`): agregar música nueva, fusiones de temas
   (clic derecho, papelera), orden a mano (⇅), preparar para Rekordbox/Traktor y deshacer.
   Y la app de descargas con pestañas (LOCAL lee su log; probadas en la laptop el 11-04);
   decidir si las playlists usan solo lo elegido en Colección.
   Escritorio (Gabriel, 11-04): `biblioteca` v11 del pendrive y música del paquete: la
   Organización se ve y se escucha, sin cambios (manda la laptop). ¿Permitir cambios aquí?
3. GABRIEL: 1001Tracklists como fuente del modelo (DECISIONS 2026-09-12, punto 3): fuente de los
   tracklists y descarga del audio se consultan antes de implementar.
4. LOCAL, después del MVP: modelo de representación propio (`docs/plans/representation_model_plan.md`,
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
