# Prompt: zona con varias vistas y conceptos en dos casos

Para empezar en un chat nuevo de Claude Code (o de otro agente que lea el repo). Lo escribió la sesión
auditora el 2026-11-09, a pedido de Gabriel, que aprobó la estrategia ese mismo día (DECISIONS 2026-11-09,
«algoritmo de recomendación», puntos 4 a 8). El fundamento, con los números y sus etiquetas, está en
`docs/reports/auditoria_recomendacion_2026-11-09.md`; el inventario de ideas y el orden, en la sección 12 de
`docs/plans/recomendacion_playlists_plan.md`. Sigue a `PROMPT_SEGURIDAD_Y_DIRECCION.md` (sección 11 del plan),
del que la unidad A1 ya está hecha.

---

Vas a mejorar con Gabriel el algoritmo de recomendación en vivo de TRAKTOR ML. No es trabajo de interfaz:
es qué temas entran en cada lista y en qué orden. Responde en español, con «tú», corto y con lo que sirve
para avanzar. `AGENTS.md` manda; la skill `auditar` (`.claude/skills/auditar/SKILL.md`) gobierna cómo se
llega a una conclusión. Hay una sesión auditora en paralelo que va a revisar cada informe tuyo.

## Lo que buscamos, en palabras de Gabriel (2026-11-09)

- **Parecidos, la playlist de D0**: temas parecidos al que suena y que con seguridad van a sonar bien si se
  mezclan con él. Es precisión de la lista: cada tema tiene que ser una apuesta segura.
- **El puntaje de Parecidos es una ecuación que pondera varios factores**: modelos como CLAP y los demás,
  género, tonalidad, BPM y otros, cada uno con su peso relativo. BPM y tonalidad son factores propios y se
  usan después: no se le piden a CLAP ni a los demás modelos. Es la arquitectura que ya tiene el recomendador
  (componentes con peso).
- **Los filtros en vivo siguen como están**: BPM ±20 y tonalidad a ±2 semitonos.
- **Playlists de concepto**, con dos casos de uso:
  - **Sin tema de referencia.** El DJ todavía no empezó a tocar y pide una playlist por concepto («vibras
    africanas», «música de muertos»). Bajo esa etiqueta se agrupan los temas que transmiten el concepto.
  - **Con el set en curso.** Hay tema principal y un historial de lo tocado, que conviene usar como fuente
    para la trayectoria del tema siguiente. Si se filtra por concepto, primero se eligen los parecidos que van
    bien con el principal y sobre esa lista se filtra por el concepto.
  - Si el DJ empieza por el primer caso, elige un tema y la iteración siguiente ya es el segundo.
- El concepto llega como lo arma el Worker de Cloudflare: de 1 a 4 frases positivas en inglés, frases
  negativas y condiciones de energía, voces y BPM (`docs/reports/traktor_concept_ai_cloudflare_2026-11-09.md`).
- **Lo pesado en procesamiento o en disco se procesa con las GPU de gpuq y con el Google Drive de Gabriel.**
- **Entre opciones que sobreviven a las medidas decide Gabriel escuchando.** Un puntaje mejor no es una lista
  mejor.

## La estrategia aprobada: una zona segura con varias vistas, y dentro de ella un paso en la dirección pedida

Lo que encontró la auditoría, todo exploratorio y sin escucha todavía:

1. CLAP, MAEST capa 7 y los 400 estilos Discogs de MAEST ya están extraídos para toda la biblioteca y arman
   listas de 30 casi distintas (comparten 3 a 7 temas). Sumadas, superan a CLAP crudo en los sets de Gabriel
   (acierto en 30 de 9,0 % a 11,4-12,7 %) y en sets de otros DJs (AUC de «mismo set» frente a azar del mismo
   BPM, de 0,66 a 0,74-0,77 con 79 pares). Los estilos llevan tempo (−0,37 con la diferencia de BPM): como el
   BPM tiene su propio factor, los modelos se miden con el tempo controlado para no contarlo dos veces.
2. El agente de conceptos de hoy ordena solo por el coseno entre el texto y el audio. Eso es lo que pide el
   caso sin tema de referencia. Para el caso con el set en curso no alcanza: esa lista no se parece al tema
   que suena más que una al azar, y sumar audio y texto tampoco cumple «parecido y concepto» mejor que el
   audio solo. Ordenar por el concepto dentro de una zona chica alrededor del tema que suena conserva más
   parecido con el mismo cumplimiento.
3. El texto de CLAP ordena bien por descripciones de sonido (AUC 0,79 contra estilos acústicos) y mal por
   nombres de género (0,58). Una sonda supervisada sobre el mismo embedding llega más lejos (0,76 frente a
   0,64-0,68). En los dos casos de uso, lo que limita es el puntaje de concepto.
4. Del paper «Guided Music Retrieval» (PDF en Descargas de Gabriel: `T-ASL-13226-2026_Proof_hi 1.pdf`;
   código en github.com/sb-lucas/Guided-Music-Retrieval, sin pesos publicados) se toman el protocolo de
   evaluación (parecido, modificación y las dos cosas, con varios blancos válidos), la forma de construir la
   supervisión sola (un modelo independiente de parecido y clasificadores con umbral de alta precisión) y la
   lección de que una fusión chica entrenada sobre CLAP congelado supera a la aritmética de embeddings, sobre
   todo para quitar algo. No hay un modelo listo para usar: sus modelos conocen un conjunto cerrado de
   modificaciones con plantilla sobre música de Jamendo.

De ahí sale el diseño:

- **Parecidos** = la zona. Los modelos (CLAP, MAEST, estilos y, cuando exista, EffNet) entran como factores
  nuevos de la ecuación, junto a género, tonalidad y BPM. Los pesos de todos los factores y una probabilidad
  de «van juntos» se aprenden de pares de sets profesionales; el acuerdo entre vistas es la marca de seguridad.
- **Concepto sin tema de referencia** = toda la biblioteca ordenada por el puntaje del concepto, con un piso de
  confianza y largo variable.
- **Concepto con el set en curso** = candidatos de la zona de Parecidos (con el historial del set), ordenados
  por el puntaje del concepto relativo al del tema que suena.
- **El puntaje de concepto**, que sirve a los dos casos, se elige en un banco local construido como el del
  paper, subiendo una escalera: texto de CLAP con contraste, sondas supervisadas, mapa aprendido de texto a
  audio y, solo si hace falta, una fusión entrenada con el dataset del paper.

## Lee antes de proponer, en este orden

1. `docs/STATUS.md`, `AGENTS.md`, `.claude/skills/auditar/SKILL.md`.
2. `docs/DECISIONS.md`, de la entrada 2026-11-06 a la más reciente. En «2026-11-09 (algoritmo de
   recomendación)» el punto 1 está refutado por el 8: léelos juntos.
3. `docs/plans/recomendacion_playlists_plan.md`: la sección 12 primero (casos de uso, ideas P11 a P13, T23 a
   T30, M6, D9 y H1, medidas E13 y E14, orden, cómputo pesado, y en 12.8 lo que cambió con el piloto de
   1001Tracklists); después la 3 (contratos), la 4 (inventario) y la
   11. Nada se borra; lo que cambia de lectura se marca en su lugar.
4. `docs/reports/auditoria_recomendacion_2026-11-09.md` entero, con su adenda. Las tablas dan los números que
   tienes que reproducir; los scripts con que se midieron están fuera de Git en
   `D:\TRAKTOR ML datos\dj_sets\auditoria_20261109\` (son exploratorios, con rutas fijas: sirven de
   referencia, no se copian al repo tal cual).
5. `docs/reports/zona_sets_2026-11-08.md` (E9 y E10, las medidas que vas a extender) y los veredictos de
   `auditoria_e1_sets_gabriel_2026-11-07.md` y `auditoria_lineas_d1_2026-11-07.md` (qué se sabe ya de los
   factores BPM, tonalidad y género).
6. Código: `src/v4/evaluation/zona_sets.py` (fuentes, pares, azar del mismo BPM, pureza: se reutiliza),
   `tools/traktor_live/recomendador.py` (los componentes y sus pesos: la ecuación) e `indice_recomendacion.py`,
   `tools/traktor_live/conceptos_agente.py` y `src/v4/evaluation/clap_texto.py` (los dos de otra sesión; el
   segundo mide cómo conviene redactar las frases para CLAP: reutilízalo, no lo dupliques),
   `src/v4/pipeline/extract_representations.py` (CLAP de audio y de texto, MAEST, EffNet),
   `src/v4/pipeline/genre_model.py` y `tag_vocals.py` (sondas sobre embeddings congelados).
7. Artefactos, fuera de Git: `artifacts/v4/traktor_live/indice_recomendacion.json` (1760 temas),
   `artifacts/v4/datasets/musica/representations/` (`clap_full`, `maesthf_full_l7`, `maesthf_full_styles`),
   y en `D:\TRAKTOR ML datos\dj_sets\`: `sets/d1.csv`, `sets/tempo_temas.csv`, `clap/`, y `audio/` con los
   originales de Bedrock y D7. Antes de usarlos verifica formas, N, valores finitos y alineación por
   `track_uid`.
8. La documentación de gpuq (github.com/gbibbo/gpuq: `README.md`, `docs/WRITING_JOBS.md`,
   `docs/HOW_IT_WORKS.md`) antes del primer trabajo pesado.

## Unidades de trabajo, de a una y verificada

Cada unidad termina con: tests del camino cambiado en verde, artefactos verificados, informe en
`docs/reports/` con cada afirmación etiquetada y los números rehechos desde los artefactos, el estado de
cada idea actualizado en su lugar en la sección 12 del plan, DECISIONS y STATUS al día, commit y push con
rutas explícitas, y el bloque CHECKPOINT.

**1. La fusión de vistas y la ecuación, medidas como código (P11, P12, P13, E14).**
- Módulo de evaluación con tests (por ejemplo `src/v4/evaluation/fusion_vistas.py`) que reutilice
  `zona_sets.py`. Primero reproduce la tabla «Parecidos: vistas y fusión» del informe. Si un número no sale,
  para y cuéntalo: eso es un hallazgo.
- Suma Bedrock: extrae MAEST (capa 7 y estilos) de sus 179 originales, con la misma segmentación que la
  biblioteca y con una prueba chica antes. Es la única fuente de un DJ profesional con originales y orden.
- Los modelos, con el tempo controlado en cada fila: azar del mismo BPM, pool a ±8 BPM, correlación de cada
  vista con la diferencia de BPM, y los estilos con el tempo descontado como variante aparte.
- La ecuación entera (P12): regresión logística sobre todos los factores (parecido por vista, género,
  tonalidad, BPM), entrenada con pares de un mismo set profesional frente a pares al azar tomados dentro de
  los filtros en vivo, con los DJs de prueba fuera del entrenamiento. Informa el peso de cada factor, AUC
  fuera de muestra, calibración y la comparación con pesos iguales y con solo CLAP. Los sets de Gabriel se
  informan aparte y no ajustan nada.
- No esperes un juez de 1001Tracklists cruzado con la biblioteca: en el piloto de 30 sets hay 11 de 1206
  temas en la biblioteca, ningún par consecutivo y 5 pares de un mismo set (plan, sección 12.8). Para los
  modelos, el juez con otros DJs es chico: Bedrock y los 79 pares de D3. Dilo así en el informe.
- Lo que 1001Tracklists sí da son pares consecutivos de profesionales en texto. Con los metadatos de Beatport
  de esos temas se aprenden los pesos de BPM, tonalidad y género (P12a). Antes de pedir nada a Beatport,
  pregunta a Gabriel: su `robots.txt` declara `ai-train=no` y la decisión es suya. No edites los archivos de
  la sesión de 1001Tracklists.
- El audio de los temas de los sets (D6) destrabaría el juez de los modelos. Es decisión de Gabriel y trabajo
  pesado: no lo empieces sin que lo pida.

**2. Ronda de escucha de Parecidos (E12).**
- Unos 10 temas principales; para cada uno, dos listas enteras a ciegas y en orden al azar: la de hoy (solo
  CLAP) frente a la ecuación que haya sobrevivido a la unidad 1, con los filtros en vivo.
- Si Gabriel prefiere la nueva: índice versión 4 con las vistas (escaladas y concatenadas; si las reduces,
  demuestra que el orden no cambia), pesos nuevos en el recomendador y campo `seguro` con la probabilidad de
  P12 y el consenso de P13 en el registro de cada lista. El lado vivo sigue con la biblioteca estándar y por
  debajo de 1 s, con sus tests en verde. Los pesos por defecto no cambian hasta que Gabriel lo diga.

**3. Essentia (V6 y T19 del plan).**
- Prueba chica con una subcarpeta en `essentia-tf-venv` (WSL) y tiempo por tema. Si la biblioteca entera pasa
  de alrededor de una hora, es trabajo pesado: va a gpuq (mira la sección de cómputo pesado).
- Embeddings EffNet-Discogs y MusiCNN; cabezas de danceability, voz, los estados de ánimo (aggressive,
  happy, party, relaxed, sad, acoustic, electronic), approachability, engagement, arousal y valencia, y las
  de `mtg_jamendo` de ánimo e instrumentos. Los modelos van fuera de Git (CC BY-NC-SA 4.0).
- Por cabeza: distribución en la biblioteca (una que junta casi todo en dos valores no sirve, como la
  energía de Mixed In Key, que Gabriel descartó), relación con el BPM, y voz contra la etiqueta Vocal como
  control.
- Da tres cosas: el modelo independiente para juzgar parecido en el banco, etiquetas de atributos y
  candidatos a coordenada de energía. EffNet entra además como cuarta vista de la unidad 1.

**4. Banco de conceptos en sus dos modos y escalera de puntajes (E13, T23, T24, T25, T28, T29).**
- Módulo de evaluación con tests (por ejemplo `src/v4/evaluation/banco_conceptos.py`), con el concepto en el
  formato del Worker.
  - Modo sin referencia: fracción de los primeros de la lista que cumple el concepto.
  - Modo con referencia: fracción que es parecida a la referencia, que cumple el concepto y que cumple las
    dos cosas, con el número de blancos válidos al lado.
  - En los dos: adición y quita por separado, conceptos vistos y no vistos, intervalos remuestreando.
- Etiquetas independientes del método que se mide: género de Beatport (solo los de fuente Beatport), estilos
  Discogs con umbrales de alta precisión, cabezas de Essentia. La etiqueta Vocal sale de una sonda sobre
  CLAP: úsala marcada como no independiente. El parecido lo juzga un modelo que el método no use.
- Frases: al menos tres juegos por concepto (solo el nombre, descriptivas, y las que escribiría el Worker
  con su prompt de `cloudflare/concept-ai-worker.js`), informados por separado, apoyándote en lo que mida
  `clap_texto.py`. Las de la auditoría son un juego más, no la referencia.
- Métodos, cada uno contra el anterior con una sola variable cambiada. Sin referencia: texto crudo (el
  agente de hoy); texto con centrado y banco de contraste; sondas supervisadas como techo. Con referencia:
  solo audio; suma cruda; suma de z; dos etapas con zonas fijadas de antemano; eje de dos polos para quitar.
- Después, escucha a ciegas: conceptos escritos por Gabriel (entre ellos «vibras africanas» y «música de
  muertos») sin tema de referencia, la lista de hoy frente a la del mejor puntaje; y unos 5 temas principales
  con concepto, la lista de hoy frente a la de dos etapas.
- El puntaje vive en un módulo compartido por el banco y por el agente de conceptos, que pasa a elegir el
  caso por el estado: sin tema principal, sin referencia; con tema principal, dos etapas. Ese agente es de
  otra sesión: coordina con Gabriel antes de tocar `conceptos_agente.py`.

**5. Mapa aprendido de texto a audio (T26).** Con los vocabularios de etiquetas de la biblioteca, un modelo
chico que lleva la descripción de una etiqueta a la región del espacio de audio donde están sus temas. Se
evalúa en el banco con etiquetas que no vio y se compara con el peldaño anterior.

**6. Coordenada de energía y zona con historial (T19, T20, T30; juez E11).** La condición `energy` del
concepto es un paso típico respecto del tema que suena, no el tema más extremo. La zona del caso con el set
en curso suma el centro de lo tocado y el paso reciente (T17 y P10 del plan) sobre las vistas nuevas, medida
como zona y no como acierto del tema siguiente.

**7. Condicional: fusión entrenada con el dataset del paper (T27, D9).** Solo si el banco muestra que los
peldaños anteriores no alcanzan. Es trabajo pesado: va a gpuq.

## Cómputo pesado: gpuq y Drive

- Regla de Gabriel: lo pesado en procesamiento o en disco no corre en este escritorio. Va a gpuq
  (github.com/gbibbo/gpuq, una cola sobre los niveles gratuitos de Colab y Kaggle) con los datos en su Google
  Drive, por el remoto de `rclone` de gpuq. Detalle: DECISIONS 2026-11-09, «algoritmo de recomendación»,
  punto 7; `AGENTS.md`, «Compute resource discipline»; plan, sección 12.6.
- Liviano, en local: tests, bancos sobre embeddings ya calculados, texto de CLAP, MAEST de los 179 originales
  de Bedrock, sondas y modelos lineales. Pesado: lo que pase de alrededor de una hora de CPU o de unos 10 GB
  de disco nuevo (umbral del agente; Gabriel lo corrige).
- En este escritorio no se encontraron `gpuq` ni `rclone` instalados el 2026-11-09. Antes del primer trabajo
  pesado, pregunta a Gabriel dónde corre el despachador y cuánto espacio libre tiene el Drive. Las
  credenciales las pone él con `gpuq setup`: no las pidas, no las aceptes en el chat y no las escribas en
  ningún archivo.
- Antes del primer envío de cada tipo de trabajo: prueba chica en CPU, qué necesita la máquina remota, horas
  de GPU estimadas frente a la cuota, espacio en Drive, y el OK de Gabriel en el chat.
- Los trabajos se retoman desde un punto de control: la máquina remota es nueva en cada intento y se puede
  perder. Lo que tiene que sobrevivir se sube al remoto; a este escritorio vuelven resultados chicos, que se
  verifican como cualquier artefacto.
- Datos públicos (MTIQ): se bajan directo a la máquina remota y a Drive va solo lo calculado. De la música
  privada, embeddings antes que audio; si hace falta subir audio, primero di a Gabriel qué y cuánto.
- Nada de lo subido es público: ni datasets ni notebooks públicos en Kaggle, ni enlaces compartidos de Drive.

## Reglas que no se negocian

- BPM y tonalidad son factores de la ecuación, no propiedades que se le pidan a los modelos. Los filtros en
  vivo quedan como están.
- Los números de la auditoría son exploratorios: se reproducen, no se dan por buenos.
- Una sola variable por comparación, en los mismos datos y con la misma semilla. Partición por DJ.
- Nada se ajusta contra los sets de Gabriel. Se informan aparte, como referencia con ruido.
- «Mismo set» mide coherencia de estilo, no «suena bien mezclado»: no lo escribas como si fuera lo mismo.
- Las listas que escucha Gabriel van a ciegas, enteras y de a dos, unas 10 por ronda.
- Un peso, una coordenada o un puntaje de concepto entra a una playlist solo después de pasar por las medidas
  y por el oído de Gabriel.
- La energía de Mixed In Key no se usa (DECISIONS 2026-11-07, punto 3).
- CPU local para lo liviano, con prueba chica antes de una corrida larga y `--low-priority`. Lo pesado, a
  gpuq, como dice la sección anterior. Nada de Lightning ni Slurm. No llames a `/api/device/claim` del Worker.
- No rompas el lado en vivo: `tools/traktor_live/` y sus tests. Solo biblioteca estándar; lo pesado se
  precalcula en el índice.
- No borres el audio de `D:\TRAKTOR ML datos\dj_sets\audio` antes de sacar de los originales las vistas que
  faltan.
- Hay otras sesiones editando este repo (importación de 1001Tracklists, agente de conceptos y Cloudflare).
  Comparten el mismo árbol de trabajo y el mismo índice de Git. Antes de preparar un commit comprueba que
  `git diff --cached` esté vacío; agrega solo tus rutas; haz commit y push en un solo paso; edita los
  documentos compartidos (DECISIONS, STATUS, PROJECT_MAP, el plan) justo antes del commit. Si un archivo tiene
  cambios sin commit de otra sesión, avisa a Gabriel antes de editarlo.
- `STATUS.md` por debajo de 6 KB; filas nuevas en `PROJECT_MAP.md` para cada archivo que agregues.
- Música, embeddings, `.npy`, `.parquet`, datos de sets, modelos y credenciales quedan fuera de Git. El
  repositorio es público: no escribas en él direcciones de correo ni nombres de cuentas.

## Tu primera respuesta

Sin código:

1. La estrategia en tus palabras, en unas diez líneas, para que Gabriel confirme que se entendió: la ecuación
   de factores y por qué los modelos se miden con el tempo controlado, los dos casos de uso de los conceptos, y
   por qué el banco va antes que el método.
2. Qué vas a verificar en los artefactos antes de la unidad 1 y qué podría cambiar el diseño (por ejemplo,
   que los números de la auditoría no se reproduzcan, o que Bedrock con MAEST contradiga a los sets de Gabriel).
3. Las preguntas que necesitas contestadas: quién elige los temas principales de la escucha y si Gabriel
   quiere además marcar tema por tema lo que no pondría; dónde corre el despachador de gpuq y cuánto espacio
   tiene el Drive.

Después, de a una unidad verificada por vez, empezando por la 1.
