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
- **Tripletas (n = 57): ninguna representación supera al BPM de forma concluyente.** Entre
  representaciones decide la escucha de playlists enteras, no el puntaje.
- **Etiqueta Vocal** escrita en el comentario de 542 archivos (CLAP zero-shot, umbral 0.145).
- **`download_JIJIJI/`**: descarga de playlists de Spotify por Soulseek, con grabación por loopback
  como respaldo; punto de entrada `spotify_soulseek_orchestrator.py`.
- **Chequeo pre-commit** en `tools/git_hooks/` (audio, arrays, pesos, credenciales, archivos > 5 MB).

## OPEN ITEMS (por prioridad; LOCAL = trabajo del agente, GABRIEL = lo hace o decide Gabriel)

1. GABRIEL: importar y validar `V4_1` en Rekordbox (pendrive) y Traktor (TODO 6.7).
2. LOCAL, en curso: organizaciones estables (por carpeta, incrementales y con semillas), plan
   `docs/plans/20260929_organizaciones_estables.md`.
3. LOCAL: ajustar el umbral Vocal con la lista de chequeo (`tag_vocals.py --check-list`).
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
- Revisar escuchando: `tools/playlist_review/build_review_page.py --dataset-name musica`
  (varias organizaciones a ciegas: `--org <hash> --org <hash> --blind`).
- Corrida larga: primero `--folder "<subcarpeta>"` y revisar la salida; después la completa con
  `--low-priority`.
- Chequeo pre-commit: activar una vez por clon con `git config core.hooksPath tools/git_hooks`;
  revisar todo el árbol con `python tools/git_hooks/check_staged.py --all`.
