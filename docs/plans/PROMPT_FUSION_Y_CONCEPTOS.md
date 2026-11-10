# Prompt: zona con varias vistas y conceptos en dos etapas

Para empezar en un chat nuevo de Claude Code (o de otro agente que lea el repo). Lo escribió la sesión
auditora el 2026-11-09, a pedido de Gabriel. El fundamento, con los números y sus etiquetas, está en
`docs/reports/auditoria_recomendacion_2026-11-09.md`. Sigue a `PROMPT_SEGURIDAD_Y_DIRECCION.md` (sección 11
del plan), del que la unidad A1 ya está hecha; lo que cambia respecto de ese prompt está dicho abajo.

---

Vas a mejorar con Gabriel el algoritmo de recomendación en vivo de TRAKTOR ML. No es trabajo de interfaz:
es qué temas entran en cada lista y en qué orden. Responde en español, con «tú», corto y con lo que sirve
para avanzar. `AGENTS.md` manda; la skill `auditar` (`.claude/skills/auditar/SKILL.md`) gobierna cómo se
llega a una conclusión. Hay una sesión auditora en paralelo que va a revisar cada informe tuyo.

## Lo que buscamos, en palabras de Gabriel (2026-11-09)

- **Parecidos, la playlist de D0**: temas parecidos al que suena y que con seguridad van a sonar bien si se
  mezclan con él. BPM y tonalidad quedan fuera del parecido, porque se ajustan después. Es precisión de la
  lista: cada tema tiene que ser una apuesta segura.
- **Playlists de concepto**: redirigen el set en una dirección escrita en lenguaje natural. Cómo se eligen
  los temas que se acercan a un concepto escrito en palabras es la pregunta abierta, y hay que optimizarla.
  El concepto llega como lo arma el Worker de Cloudflare: de 1 a 4 frases positivas en inglés, frases
  negativas y condiciones de energía, voces y BPM (`docs/reports/traktor_concept_ai_cloudflare_2026-11-09.md`).
- **Entre opciones que sobreviven a las medidas decide Gabriel escuchando.** Un puntaje mejor no es una lista
  mejor.

## La estrategia: una zona segura con varias vistas, y dentro de ella un paso en la dirección pedida

Lo que encontró la auditoría, todo exploratorio y sin escucha todavía:

1. CLAP, MAEST capa 7 y los 400 estilos Discogs de MAEST ya están extraídos para toda la biblioteca y arman
   listas de 30 casi distintas (comparten 3 a 7 temas). Sumadas, superan a CLAP crudo en los sets de Gabriel
   (acierto en 30 de 9,0 % a 11,4-12,7 %) y en sets de otros DJs (AUC de «mismo set» frente a azar del mismo
   BPM, de 0,66 a 0,74-0,77 con 79 pares). Parte de la ventaja es tempo: hay que controlarlo siempre.
2. El agente de conceptos de hoy ordena solo por el coseno entre el texto y el audio. Del tema que suena usa,
   como mucho, el filtro de BPM: la lista no se le parece más que una al azar. Sumar audio y texto tampoco
   alcanza: no cumple «parecido y concepto» mejor
   que el audio solo. Ordenar por el concepto dentro de una zona chica alrededor del tema que suena conserva
   más parecido con el mismo cumplimiento.
3. El texto de CLAP ordena bien por descripciones de sonido (AUC 0,79 contra estilos acústicos) y mal por
   nombres de género (0,58). Una sonda supervisada sobre el mismo embedding llega más lejos (0,76 frente a
   0,64-0,68). Lo que limita es el puntaje de concepto, no la fórmula que lo mezcla con el audio.
4. Del paper «Guided Music Retrieval» (PDF en Descargas de Gabriel: `T-ASL-13226-2026_Proof_hi 1.pdf`;
   código en github.com/sb-lucas/Guided-Music-Retrieval, sin pesos publicados) se toman el protocolo de
   evaluación (parecido, modificación y las dos cosas, con varios blancos válidos), la forma de construir la
   supervisión sola (un modelo independiente de parecido y clasificadores con umbral de alta precisión) y la
   lección de que una fusión chica entrenada sobre CLAP congelado supera a la aritmética de embeddings, sobre
   todo para quitar algo. No hay un modelo listo para usar: sus modelos conocen un conjunto cerrado de
   modificaciones con plantilla sobre música de Jamendo.

De ahí sale el diseño:

- **Parecidos** = la zona: fusión de vistas, con los pesos y una probabilidad de «van juntos» aprendidos de
  pares de sets profesionales, y el acuerdo entre vistas como marca de seguridad.
- **Cada playlist de concepto** = candidatos de esa zona, ordenados por el puntaje del concepto relativo al
  del tema que suena, con un piso de confianza y largo variable.
- **El puntaje de concepto** se elige en un banco local construido como el del paper, subiendo una escalera:
  texto de CLAP con contraste, sondas supervisadas, mapa aprendido de texto a audio y, solo si hace falta,
  una fusión entrenada con el dataset del paper.

Los ids nuevos (P11 a P13, E13, E14, T23 a T28, D9) y los cambios de estado de las ideas viejas están en la
última tabla del informe de auditoría.

## Lee antes de proponer, en este orden

1. `docs/STATUS.md`, `AGENTS.md`, `.claude/skills/auditar/SKILL.md`.
2. `docs/reports/auditoria_recomendacion_2026-11-09.md` entero. Las tablas dan los números que tienes que
   reproducir; los scripts con que se midieron están fuera de Git en
   `D:\TRAKTOR ML datos\dj_sets\auditoria_20261109\` (son exploratorios, con rutas fijas: sirven de
   referencia, no se copian al repo tal cual).
3. `docs/plans/recomendacion_playlists_plan.md`: secciones 3 (contratos), 4 (inventario) y 11. Nada se
   borra; lo que cambia de lectura se marca en su lugar.
4. `docs/DECISIONS.md`, entradas 2026-11-06 a la más reciente.
5. `docs/reports/zona_sets_2026-11-08.md` (E9 y E10, las medidas que vas a extender) y los veredictos de
   `auditoria_e1_sets_gabriel_2026-11-07.md` y `auditoria_lineas_d1_2026-11-07.md`.
6. Código: `src/v4/evaluation/zona_sets.py` (fuentes, pares, azar del mismo BPM, pureza: se reutiliza),
   `tools/traktor_live/recomendador.py` e `indice_recomendacion.py`, `tools/traktor_live/conceptos_agente.py`
   (de otra sesión), `src/v4/pipeline/extract_representations.py` (CLAP de audio y de texto, MAEST, EffNet),
   `src/v4/pipeline/genre_model.py` y `tag_vocals.py` (sondas sobre embeddings congelados).
7. Artefactos, fuera de Git: `artifacts/v4/traktor_live/indice_recomendacion.json` (1760 temas),
   `artifacts/v4/datasets/musica/representations/` (`clap_full`, `maesthf_full_l7`, `maesthf_full_styles`),
   y en `D:\TRAKTOR ML datos\dj_sets\`: `sets/d1.csv`, `sets/tempo_temas.csv`, `clap/`, y `audio/` con los
   originales de Bedrock y D7. Antes de usarlos verifica formas, N, valores finitos y alineación por
   `track_uid`.

## Unidades de trabajo, de a una y verificada

Cada unidad termina con: tests del camino cambiado en verde, artefactos verificados, informe en
`docs/reports/` con cada afirmación etiquetada y los números rehechos desde los artefactos, documentación al
día, commit y push con rutas explícitas, y el bloque CHECKPOINT.

**1. La fusión de vistas, medida como código (P11, P12, P13, E14).**
- Módulo de evaluación con tests (por ejemplo `src/v4/evaluation/fusion_vistas.py`) que reutilice
  `zona_sets.py`. Primero reproduce la tabla «Parecidos: vistas y fusión» del informe. Si un número no sale,
  para y cuéntalo: eso es un hallazgo.
- Suma Bedrock: extrae MAEST (capa 7 y estilos) de sus 179 originales, con la misma segmentación que la
  biblioteca y con una prueba chica antes. Es la única fuente de un DJ profesional con originales y orden.
- Controles de tempo en cada fila: azar del mismo BPM, pool a ±8 BPM, correlación de cada vista con la
  diferencia de BPM, y los estilos con el tempo descontado como variante aparte.
- P12: regresión logística sobre los parecidos por vista, entrenada con pares de un mismo set profesional
  frente a pares al azar del mismo BPM, con los DJs de prueba fuera del entrenamiento. Informa AUC fuera de
  muestra, calibración y la comparación con pesos iguales. Los sets de Gabriel se informan aparte y no
  ajustan nada.
- Cuando la otra sesión termine la importación de 1001Tracklists, repite todo con sus pares cruzados con la
  biblioteca, por DJ. No edites sus archivos.

**2. Ronda de escucha de Parecidos (E12).**
- Unos 10 temas principales; para cada uno, dos listas enteras a ciegas y en orden al azar: CLAP crudo
  frente a la fusión que haya sobrevivido a la unidad 1, con los filtros en vivo tal como están hoy.
- Si Gabriel prefiere la fusión: índice versión 4 con el vector fusionado (las vistas escaladas y
  concatenadas; si lo reduces, demuestra que el orden no cambia), campo `seguro` con la probabilidad de P12 y
  el consenso de P13 en el registro de cada lista. El lado vivo sigue con la biblioteca estándar y por
  debajo de 1 s, con sus tests en verde. El valor por defecto no cambia hasta que Gabriel lo diga.

**3. Essentia en WSL (V6 y T19 del plan).**
- Prueba chica con una subcarpeta en `essentia-tf-venv`, tiempo por tema, y recién después la biblioteca
  entera con `--low-priority`.
- Embeddings EffNet-Discogs y MusiCNN; cabezas de danceability, voz, los estados de ánimo (aggressive,
  happy, party, relaxed, sad, acoustic, electronic), approachability, engagement, arousal y valencia, y las
  de `mtg_jamendo` de ánimo e instrumentos. Los modelos van fuera de Git (CC BY-NC-SA 4.0).
- Por cabeza: distribución en la biblioteca (una que junta casi todo en dos valores no sirve, como la
  energía de Mixed In Key), relación con el BPM, y voz contra la etiqueta Vocal como control.
- Da tres cosas: el modelo independiente para juzgar parecido en el banco, etiquetas de atributos y
  candidatos a coordenada de energía. EffNet entra además como cuarta vista de la unidad 1.

**4. Banco de composición local y escalera de puntajes (E13, T23, T24, T25, T28).**
- Módulo de evaluación con tests (por ejemplo `src/v4/evaluation/banco_conceptos.py`). Consulta = tema de
  referencia más concepto en el formato del Worker. Mide la fracción de la lista que es parecida a la
  referencia, la que cumple el concepto y la que cumple las dos; cuántos blancos válidos había; adición y
  quita por separado; conceptos vistos y no vistos; intervalos remuestreando temas de referencia.
- Etiquetas independientes del método que se mide: género de Beatport (solo los de fuente Beatport), estilos
  Discogs con umbrales de alta precisión, cabezas de Essentia. La etiqueta Vocal sale de una sonda sobre
  CLAP: úsala marcada como no independiente. El parecido lo juzga un modelo que el método no use.
- Frases: al menos tres juegos por concepto (solo el nombre, descriptivas, y las que escribiría el Worker
  con su prompt de `cloudflare/concept-ai-worker.js`), informados por separado. Las de la auditoría son un
  juego más, no la referencia.
- Métodos, cada uno contra el anterior con una sola variable cambiada: solo audio; solo texto crudo (el
  agente de hoy); suma cruda; suma de z; dos etapas con zonas fijadas de antemano; texto con centrado y banco
  de contraste; sondas supervisadas como techo; eje de dos polos para quitar.
- Después, escucha: los conceptos provisionales de Gabriel y dos o tres que escriba él, para unos 5 temas
  principales, la lista de hoy frente a la de dos etapas, a ciegas.
- El puntaje vive en un módulo compartido por el banco y por el agente de conceptos. Ese agente es de otra
  sesión: coordina con Gabriel antes de tocar `conceptos_agente.py`.

**5. Mapa aprendido de texto a audio (T26).** Con los vocabularios de etiquetas de la biblioteca, un modelo
chico en CPU que lleva la descripción de una etiqueta a la región del espacio de audio donde están sus
temas. Se evalúa en el banco con etiquetas que no vio y se compara con el peldaño anterior.

**6. Coordenada de energía (T19 y T20, juez E11)** para la condición `energy` del concepto: un paso típico
respecto del tema que suena, no el tema más extremo.

**7. Condicional: fusión entrenada con el dataset del paper (T27, D9).** Solo si el banco muestra que los
peldaños anteriores no alcanzan. Antes: tamaño real de `original.tar.gz`, tiempo de CLAP medido en 200
recortes, disco necesario (D: tiene unos 50 GB libres) y el OK de Gabriel.

## Reglas que no se negocian

- BPM y tonalidad no entran al puntaje de parecido. Los filtros en vivo (BPM ±20, tonalidad a ±2
  semitonos) quedan como están salvo que Gabriel diga otra cosa.
- Los números de la auditoría son exploratorios: se reproducen, no se dan por buenos.
- Una sola variable por comparación, en los mismos datos y con la misma semilla. Partición por DJ.
- Nada se ajusta contra los sets de Gabriel. Se informan aparte, como referencia con ruido.
- «Mismo set» mide coherencia de estilo, no «suena bien mezclado»: no lo escribas como si fuera lo mismo.
- Las listas que escucha Gabriel van a ciegas, enteras y de a dos, unas 10 por ronda.
- Una coordenada o un puntaje de concepto entra a una playlist solo después de pasar por el banco y por el
  oído de Gabriel.
- CPU local. Prueba chica antes de una corrida larga; después, `--low-priority`. Nada de GPU, Lightning ni
  Slurm sin que Gabriel lo pida en el chat. No llames a `/api/device/claim` del Worker.
- No rompas el lado en vivo: `tools/traktor_live/` y sus tests. Solo biblioteca estándar; lo pesado se
  precalcula en el índice.
- No borres el audio de `D:\TRAKTOR ML datos\dj_sets\audio` antes de sacar de los originales las vistas que
  faltan.
- Hay otras sesiones editando este repo ahora (importación de 1001Tracklists, agente de conceptos y
  Cloudflare, y los documentos `DECISIONS.md`, `PROJECT_MAP.md` y el plan). Mira `git status` antes de
  empezar, haz commit solo de tus rutas y, si un archivo tiene cambios sin commit de otra sesión, avisa a
  Gabriel antes de editarlo.
- Documentación, cuando Gabriel apruebe la estrategia: sección 12 del plan con el inventario del informe de
  auditoría y las marcas en su lugar (G6, P8, P9, T18, V6 y T19, P2); entrada en DECISIONS que diga qué
  decidió Gabriel y qué propuso el agente; `STATUS.md` por debajo de 6 KB; filas nuevas en `PROJECT_MAP.md`.
- Música, embeddings, `.npy`, `.parquet`, datos de sets, modelos y credenciales quedan fuera de Git.

## Tu primera respuesta

Sin código:

1. La estrategia en tus palabras, en unas diez líneas, para que Gabriel confirme que se entendió qué es la
   zona, por qué el concepto va en dos etapas y por qué el banco va antes que el método.
2. Qué vas a verificar en los artefactos antes de la unidad 1 y qué podría cambiar el diseño (por ejemplo,
   que los números de la auditoría no se reproduzcan, o cuántos pares da el cruce de 1001Tracklists).
3. Las preguntas que necesitas contestadas: si los filtros en vivo de BPM y tonalidad quedan como están;
   quién elige los temas principales de la escucha y cuántas listas por ronda; si en la escucha Gabriel quiere
   además marcar tema por tema lo que no pondría.

Después, de a una unidad verificada por vez, empezando por la 1.
