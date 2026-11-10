# TRAKTOR ML — Estado actual

Se inyecta al abrir cada sesión de Claude Code (hook `SessionStart` en `.claude/settings.json`).
Es el estado final, no una crónica: se edita en su lugar cuando algo cambia. Tope 6 KB
(`tests/test_status_doc.py`). La historia vive en `git log`, `docs/DECISIONS.md` y `docs/reports/`.

## CURRENT STATE (2026-11-05)

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
  `exports/biblioteca_v11/` (2026-10-08): 51 playlists, 0 rutas faltantes.
- `requirements_v4.txt` fija torch 2.7.1 y pyarrow 16.1 (Smart App Control de la laptop).
- **Tripletas (n = 57): ninguna representación supera al BPM de forma concluyente.** Entre
  representaciones decide la escucha de playlists enteras, no el puntaje.
- **Etiqueta Vocal (clap_probe, 2026-10-06)**: Vocal si >= 5 % del tema tiene voz (regla de Gabriel);
  936 de 1823. Escrita y verificada en 1792 MP3/FLAC.
- **Tags de Beatport** (2026-09-29): 1087 archivos, track_uid intacto; los 110 sin confirmar conservan su género.
- **Pendrive `D:\TRAKTOR ML` (copia en `C:\TRAKTOR ML`)** (DECISIONS 2026-10-30, 11-02): los 1760
  temas de `biblioteca` por playlist; `_organizacion.csv` da archivo y track_uid de cada uno
  (verificado en ambas el 11-04). Carátulas 1630/1760. Rekordbox: `Preparar Rekordbox.bat`.
- **`download_JIJIJI/`**: Spotify por Soulseek; lo que falte se graba de a un tema, sin perder audio
  (Voicemeeter; informe grilla 11-07);
  Beatport o modelo de género, Vocal y embedding tema por tema. Después: Traktor (Analyze) y «Agregar
  música nueva» (BPM/tonalidad de Traktor, género; DECISIONS 11-07 p. 18). Logs `runs/logs/`.
- **Chequeo pre-commit** en `tools/git_hooks/` (audio, arrays, pesos, credenciales, archivos > 5 MB).

## OPEN ITEMS (por prioridad; LOCAL = trabajo del agente, GABRIEL = lo hace o decide Gabriel)

1. GABRIEL: probar el pendrive en Rekordbox con la controladora (sus 1752 MP3/FLAC llevan embedding y
   lugar, `file_embeddings.py`). LOCAL: no correr `usb_package.py` desde `Música/` (pisaría el
   paquete, que manda). `Preparar Traktor.bat` arma un NML y Traktor importa una sola playlist:
   refrescar Traktor de esta PC (`traktor_collection.py`).
2. GABRIEL: probar la app (pestaña Organización): agregar música, fusiones, orden a mano (⇅),
   preparar para Rekordbox/Traktor y deshacer; decidir si las playlists usan solo lo elegido en
   Colección. Solo el escritorio (laptop retirada, 11-04), todo en D:: `artifacts/v4/datasets/musica`
   apunta a `D:\TRAKTOR ML datos\musica` y `Música/` a `D:\TRAKTOR ML\Musica` (package_library.py);
   el pendrive manda. LOCAL: probar «Agregar» con Workout Delft.
3. LOCAL: D5/D6 (DECISIONS 11-10; plan § 6): `descarga_continua.py correr` encadena sets, previews,
   rescates y YouTube sin rendirse; `estado`: qué falta. GABRIEL: vetos; ¿YouTube con su sesión?
4. LOCAL, después del MVP: modelo de representación propio (`docs/plans/representation_model_plan.md`,
   fases 3 a 5).
5. Recomendación en vivo (`recomendador.py`, D#0+D0): plan § 12, `PROMPT_FUSION_Y_CONCEPTOS.md`. Unidad 1
   hecha (`fusion_vistas_2026-11-09.md`). Ahora: plan § 12.9. GABRIEL: Traktor,
   «TRAKTOR ML sin tonalidad», Analyze solo Key. TIV aparte.
6. Conceptos del teléfono: circuito con LLM (Worker v5, `conceptos_agente.py`) probado en local.
   GABRIEL: `python cloudflare/publicar.py` (`cloudflare/README.md`).

## RUN RECIPES

- Intérprete: `.venv/Scripts/python.exe`; en el escritorio ese `.venv` no tiene pandas ni yaml (`Abrir
  TRAKTOR ML.bat` falla): usar el Python 3.11 de la Store, con `pytest --basetemp` propio.
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
- Corrida larga: primero `--folder "<subcarpeta>"` y revisar; luego la completa con `--low-priority`.
- Lo pesado (GPU, >1 h de CPU, >10 GB): `tools/gpuq/cola.py add|status|logs`, datos en
  `erclm:traktor_ml/` (`docs/GPUQ.md`; probado en T4 el 11-09).
- Chequeo pre-commit de todo el árbol: `python tools/git_hooks/check_staged.py --all`.
