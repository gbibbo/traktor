# Prompt: seguridad en Parecidos y coordenadas de dirección (sección 11 del plan)

Para empezar en un chat nuevo de Claude Code (o de otro agente que lea el repo). Lo escribió la sesión
auditora el 2026-11-08 con Gabriel. Las decisiones están en `docs/DECISIONS.md`, entrada 2026-11-08; el
detalle, en la sección 11 de `docs/plans/recomendacion_playlists_plan.md`.

---

Vas a implementar con Gabriel la sección 11 de `docs/plans/recomendacion_playlists_plan.md`: seguridad en
la playlist de parecidos y coordenadas para las playlists de dirección. Respondé en español, con «tú»,
corto y con lo que sirve para avanzar. `AGENTS.md` manda; la skill `auditar`
(`.claude/skills/auditar/SKILL.md`) gobierna cómo se llega a una conclusión.

## Lo que buscamos, en palabras de Gabriel

- **Parecidos**, la playlist que abre el botón D0: «un listado de temas de los que tengo la seguridad de
  que van a sonar bien con lo que estoy tocando». Es precisión de la lista: cada uno de los 30 tiene que
  ser una apuesta segura. No es acertar el tema que él habría puesto después. «Sonar bien» puede ser CLAP
  tal cual o CLAP mejorado con lo que se aprenda de los sets de DJs profesionales ya bajados (D7, D8).
- **Dirección** (mantener, más energía, menos, introspectiva): necesita una coordenada que se mueva como
  se mueven los DJs a lo largo de la noche. La energía de Mixed In Key no se usa. Los modelos de Essentia
  y las direcciones de texto en CLAP son candidatos; el juez es la forma de los sets largos reales y los
  pares de juicio de Gabriel.
- **Entre opciones que sobreviven a las medidas decide Gabriel escuchando.** Un puntaje mejor no es una
  lista mejor. Los sets pasados de Gabriel son referencia con ruido, por sus palabras.

## Lee antes de proponer, en este orden

1. `docs/STATUS.md`, `AGENTS.md`, `.claude/skills/auditar/SKILL.md`.
2. `docs/plans/recomendacion_playlists_plan.md`: la sección 11 primero; después los contratos de la
   sección 3, el inventario de la 4 y las etapas de la 5. Nada de lo anterior se borra: lo que cambia de
   lectura lleva marca en su lugar.
3. `docs/DECISIONS.md`, entradas 2026-11-06, 11-07 y 11-08.
4. Lo medido y auditado: `docs/reports/e1_sets_gabriel_2026-11-07.md` y
   `auditoria_e1_sets_gabriel_2026-11-07.md`; `trayectorias_d7_2026-11-07.md` y
   `auditoria_trayectorias_d7_2026-11-07.md`; `lineas_d1_2026-11-07.md` y
   `auditoria_lineas_d1_2026-11-07.md`. Fíjate qué midió cada uno y con qué medida.
5. Código: `tools/traktor_live/recomendador.py` (componentes, opciones, registro de las listas),
   `src/v4/evaluation/banco_sets.py` y `lineas_d1.py`, `tools/dj_sets/` (extremos y trayectorias de D7),
   `src/v4/pipeline/tag_vocals.py` (codificador de texto de CLAP, ya corre en Windows) y
   `src/v4/pipeline/genre_model.py` (cabeza sobre embeddings congelados, validación por artista).
6. Artefactos, fuera de Git: `artifacts/v4/traktor_live/indice_recomendacion.json` (1760 temas) y, en
   `D:\TRAKTOR ML datos\dj_sets\`, `sets/d1.csv`, los vectores de D7 (por tema validados y discos enteros
   por bloque), D8 y `biblioteca/clap_extremos.npz`. Antes de usarlos, verificá formas, N y valores
   finitos, y alineación por `track_uid`.

## Orden de trabajo (sección 11.4 del plan)

**Etapa A, ahora, en CPU y sin audio nuevo.**
1. Medidas E9 y E10 sobre CLAP crudo: radio de zona (distancia entre temas de un mismo set profesional
   frente a pares al azar del mismo rango de BPM), fracción de los 30 dentro del radio y pureza fuera de
   muestra, con líneas de base al azar y BPM solo. Informe con etiquetas.
2. P8 en el recomendador: campo `seguro` en el registro de cada lista, sin cambiar el orden ni la
   latencia. Listas para la escucha E12: unos 10 temas principales, los 30 crudos frente a los «seguros».
3. T18, direcciones de texto en CLAP: varias frases por concepto, dirección como diferencia de frases
   opuestas o percentil, listas M3U de los 10 primeros y los 10 últimos por concepto para que Gabriel
   escuche, y proyección de los bloques de D7 y D8 y de los sets largos de Gabriel (E11).

**Etapa B, una unidad por vez:** P9 (proyección de CLAP aprendida de sets, discos fuera de muestra), V6 y
T19 (cabezas de Essentia en WSL, prueba chica antes), T20 (cabeza propia con la posición en el set), P10
(consulta compuesta para Mantener). **Etapa C:** perfiles de dirección con la coordenada que sobreviva.

Cada unidad termina con: tests del camino cambiado en verde, artefactos verificados, informe en
`docs/reports/` con cada afirmación etiquetada (EVIDENCIA OBSERVADA, INFERENCIA, HIPÓTESIS, NO
CONCLUYENTE) y los números reproducidos desde los artefactos, estado de la idea actualizado en su lugar en
el plan, DECISIONS y STATUS al día (STATUS no pasa de 6 KB), commit y push con rutas explícitas, y el
bloque CHECKPOINT.

## Reglas que no se negocian

- Una sola variable por comparación, en los mismos sets. Partición por DJ o por disco fuera de la muestra.
- El acierto en 30 sobre los sets de Gabriel (E1) solo descarta. No leas «no mejora el acierto» como «no
  mejora la lista».
- Nada se ajusta solo contra los sets de Gabriel.
- Las listas que escucha Gabriel van a ciegas, enteras y de a dos (CLAP crudo frente a la variante), unas
  10 por ronda. Entre lo que sobrevive, decide él.
- Una coordenada entra a un perfil solo después de pasar E11 (ordena la biblioteca con sentido al oído y
  se mueve con la noche en los sets reales).
- CPU local. Antes de una corrida larga, prueba chica; después, `--low-priority`. Nada de GPU, Lightning
  ni Slurm sin que Gabriel lo pida en el chat. 1001Tracklists se baja con la cuenta de Gabriel por el botón
  de exportar del sitio (DECISIONS 2026-11-08, punto 1); el `robots.txt` no se esquiva de otra forma.
- No romper el lado en vivo: `tools/traktor_live/` y sus tests (`tests/v4/test_traktor_live_*.py`,
  `test_sugerencias_servicio.py`, `test_mapeo_tsi.py`). El botón sigue por debajo de 1 s y el lado vivo
  usa solo la biblioteca estándar; lo pesado se precalcula en el índice.
- Hay una sesión auditora en paralelo sobre el mismo repositorio. Antes de editar sus informes
  (`docs/reports/auditoria_*.md`) avisá. Lo que cambia de lectura se marca en su lugar con
  `[REEMPLAZADO <fecha> por <evidencia>: <lo que rige>]` o `[REFUTADO …]`; no se borra.
- Música, embeddings, `.npy`, `.parquet`, datos de sets y credenciales quedan fuera de Git.

## Tu primera respuesta

Sin código:
1. La sección 11 en tus palabras, en unas diez líneas, para que Gabriel confirme que entendiste qué es
   «seguridad» y por qué el acierto en 30 no la mide.
2. Qué vas a mirar en los artefactos antes de la unidad A1 y qué podría cambiar el diseño (por ejemplo,
   cuántos vectores por tema validados tiene D7).
3. Las preguntas de la sección 11.6 que necesitás contestadas antes de A3: los conceptos iniciales para las
   direcciones de texto, qué DJs y sets para D8, cuántas listas escucha por ronda, cuántos tracklists de
   1001Tracklists bajar con su cuenta.

Después, de a una unidad verificada por vez, empezando por A1.
