# Plan de las playlists en vivo: informe para auditoría externa

Fecha: 2026-11-07. Autor: agente de Claude Code (sesión traktor-github-3a), con las decisiones de Gabriel
del 2026-11-06 y el 2026-11-07 (`docs/DECISIONS.md`).

## 0. Qué se pide auditar

El auditor tiene que verificar tres cosas:

1. **Compatibilidad**: las ideas del inventario (sección 4) no se contradicen entre sí ni con las
   decisiones de Gabriel (sección 1) ni con las reglas del repositorio (`AGENTS.md`).
2. **Orden ejecutable**: las etapas (sección 5) se pueden ejecutar en ese orden. Cada etapa tiene
   entradas que existen al empezarla y salidas verificables al terminarla.
3. **Integración orgánica**: al final, el resultado positivo de cada idea se puede combinar con los
   demás, sin quedar en un compartimento estanco. La sección 3 dice cómo se garantiza y la sección 7, qué
   mirar para comprobarlo.

Cómo leer las afirmaciones (skill `auditar`):
- **EVIDENCIA OBSERVADA**: medido o leído en esta sesión, con dónde.
- **INFERENCIA**: deducido, con el paso explícito.
- **HIPÓTESIS**: plausible, con la prueba que la decide.
- **NO CONCLUYENTE**: falta algo.

Los números de la sección 2 son de esta sesión. El auditor puede rehacerlos con los artefactos que se
citan, que están fuera de Git en `artifacts/`.

## 1. Objetivo y decisiones vigentes

### Playlists (Gabriel, DECISIONS 2026-11-07, punto 1)

- **Parecidos**: los temas más parecidos al tema principal que suena, ordenados por parecido. El primero
  es el más parecido de la biblioteca.
- **Dirección**: mantener la energía, más energía, menos energía, más introspectiva. Tienen en cuenta la
  trayectoria de todo el set.

### Botón (Gabriel, 2026-11-07; DECISIONS 2026-11-07, punto 8)

Gabriel lo llama D0. En la mezcla externa el gesto es D#0 + D0 (DECISIONS 2026-11-06, punto 9).
- **Primera pulsación**: refresca todas las playlists respecto del tema principal y abre Parecidos.
- **Las de dirección** quedan guardadas en Favoritos de Traktor, al lado. Gabriel las elige con el mouse.
- **Otra pulsación**: vuelve a la carpeta del navegador que estaba elegida antes de la primera
  pulsación.

### Introspectivo (Gabriel; DECISIONS 2026-11-07, punto 2)

- Suele pasar en sets de 3 horas o más, en general después de la segunda hora.
- No depende necesariamente de la energía.
- En general no tiene grandes saltos.

### Restricciones

| Regla | Origen |
| :--- | :--- |
| La energía de Mixed In Key no se usa en nada nuevo | Gabriel, DECISIONS 2026-11-07, punto 3 |
| La fase 4 del pipeline (`phase4_order.py`) sigue arrancando por la menor energía de Mixed In Key: no se toca por ahora | Gabriel, 2026-11-07 |
| Se usan las listas de sets de Kim et al. (texto, sin audio) y sets largos de 1001Tracklists | Gabriel, DECISIONS 2026-11-07, puntos 4 y 9 |
| Los sets pasados de Gabriel son referencia con ruido: ninguna medida se ajusta solo contra ellos | Gabriel, DECISIONS 2026-11-07, punto 5 |
| GPU y disco, cuando haga falta audio, con una estimación medida antes | Gabriel, DECISIONS 2026-11-07, punto 6 |
| Entre opciones que sobreviven a las medidas, decide Gabriel escuchando; un puntaje mejor no es una organización mejor | DECISIONS 2026-09-29; skill `auditar` |
| CPU local por defecto; ninguna corrida larga sin una prueba chica antes; `--low-priority` | `AGENTS.md` |
| La alineación es por `track_uid`, nunca por orden de archivos; música, embeddings y datos de sets fuera de Git | `AGENTS.md` |
| El lado en vivo usa solo la biblioteca estándar: todo lo pesado se precalcula en el índice | `recomendador.py`; DECISIONS 2026-11-06, punto 7 |
| Filtros: BPM ±20 y tonalidad compatible a ±2 semitonos; fuera lo tocado en la sesión | DECISIONS 2026-11-06, punto 7 |

## 2. Estado actual verificado

| Pieza | Dónde | Estado (EVIDENCIA OBSERVADA salvo indicación) |
| :--- | :--- | :--- |
| Tema principal y temas tocados | `tools/traktor_live/tema_principal.py` | Registra `principal` y `tocado` (30 s al aire con fader >= 50 %) en `artifacts/v4/traktor_live/*.jsonl`. En esta PC, todavía 0 eventos `tocado`: no hubo una sesión desde ese código |
| Recomendador | `tools/traktor_live/recomendador.py` | Componentes `clap`, `bpm`, `clave`, `genero` (taxonomía) y `tiv` (reservado), cada uno de 0 a 1 o None. Puntaje = promedio pesado de los conocidos. Pesos por defecto: solo CLAP. Unos 35 ms para 968 candidatos (colección real, 2026-11-06) |
| Índice | `artifacts/v4/traktor_live/indice_recomendacion.json` (`indice_recomendacion.py`, versión 3) | 1760 temas: embedding CLAP, BPM y tonalidad de tags, género (1602) y subgénero (30) |
| Taxonomía de géneros | `config/genre_taxonomy/taxonomia.json`; `src/v4/common/genre_taxonomy.py` | 47 géneros, 34 familias, 211 relaciones; aprobada por la red de agentes (informe `docs/reports/taxonomia_generos_2026-11-06.md`) |
| Botón | `sugerencias_vivo.py`, clase `Toggle` | Primera pulsación: guarda la carpeta y va a la playlist. Segunda, estando en la playlist: vuelve. **Falta**: si el DJ pasó con el mouse a otra playlist de sugerencias, hoy la pulsación guarda esa playlist como lugar de regreso. Con varias playlists, todas tienen que contar como «dentro» |
| Historia de Traktor (D1) | `Documents/Native Instruments/Traktor 4.4.1/History` y `.../Traktor 4.2.0/History` | Por tema: deck, `STARTTIME`, `DURATION` y `PLAYEDPUBLIC`. Ver el detalle debajo |
| Sets públicos (D3) | `artifacts/v4/dj_sets/djmix/djmix-dataset.json` (LEEME al lado) | Ver el detalle debajo |
| Cruce de D3 con la biblioteca | Normalización artista-título, sin tocar audio | Ver el detalle debajo |
| Escala de CLAP | Colección real, 40 temas principales | Entre el 1.º y el 30.º sugerido, CLAP va de 0.93 a 0.88. Con un peso de género de 0.5, las 30 sugerencias tras un Tech House son Tech House |
| Género frente a CLAP | 60 temas principales | Correlación entre candidatos 0.22 (mediana). Los 106 estilos de Discogs de MAEST como vector: 0.35 con CLAP y 0.33 con la taxonomía. Solo 3 de los 30 primeros coinciden con los de CLAP |
| Energía de Mixed In Key | `features/bpm_key.parquet` | 1424 de 1823 temas; el 83 % en 5 o 6 |

**Historia de Traktor (D1).**
- Hay 29 sets con 8 o más temas al aire por 2 minutos o más, sumando las dos versiones. Hay al menos 2
  repetidos entre 4.2.0 y 4.4.1.
- Sets de 3 h o más (del primer al último tema al aire): 3 limpios (3,5 a 3,9 h) y uno de 130 temas.
  Otros dos de 13 temas o menos dan unas 24 h por el horario: hay que tratar el cruce de medianoche y
  las sesiones dejadas abiertas.
- Temas después de la segunda hora en esos sets: entre 60 y 150, según cómo se cuente el de 130 temas.

**Sets públicos (D3).**
- 5040 sets de MixesDB, con 102 932 apariciones de temas, 63 038 temas con ID de YouTube y 64 764
  transiciones.
- Categorías por set: House 1555, Techno 1547, Tech House 1410.
- 779 sets con minuto por tema, ninguno de más de 150 minutos. Sin género por tema y sin licencia
  publicada.

**Cruce de D3 con la biblioteca.**
- 355 temas de la biblioteca (21 % de 1698 con artista y título) aparecen en D3, en 851 apariciones y
  696 sets.
- Solo 25 transiciones tienen los dos temas en la biblioteca, y 1539 tienen uno solo.

## 3. Arquitectura de integración

**Principio.** Toda idea entra por uno de cinco contratos compartidos, y ninguna tiene un camino propio
hasta Traktor. Una idea que da resultado positivo se activa con un peso, no con código nuevo. Una que da
negativo queda con peso 0, pero sigue registrada y medida.

```
          Almacén de temas (por track_uid)        Almacén de sets (un solo esquema)
          CLAP, BPM, tonalidad, géneros,          D1 Traktor · D2 registro en vivo ·
          estilos, energía, vectores...           D3/D4 MixesDB · D5 1001Tracklists
                     │                                       │
                     └──────────────┬────────────────────────┘
                                    ▼
          Componentes de 0 a 1 (registro con nombre; None = no cuenta)
                                    │
             ┌──────────────────────┼──────────────────────┐
             ▼                      ▼                      ▼
     Perfiles de playlist     Estado de la           Banco de evaluación
     (filtros + objetivo +    trayectoria            único (E1): repite sets
     pesos sobre los mismos   (sale de D2)           y mide cada componente
     componentes)                                    y cada perfil
             │
             ▼
     Un solo cálculo en vivo: los mismos candidatos para todas las playlists,
     el mismo registro de componentes; el botón escribe todas
```

### 3.1 Almacén de temas

- **Qué es.** El índice de `indice_recomendacion.py` pasa a versión 4, con campos opcionales por tema:
  - géneros blandos (G3, G4);
  - estilos de Discogs (G6);
  - coordenadas de energía y de texto CLAP (T3);
  - estadísticas de fase (T2);
  - vector de sets (M1) y vector aprendido (M4).
- **Reglas.**
  - Un campo que falta vale None y el componente no cuenta (ya es así en `weighted`).
  - La alineación es por `track_uid`, con `track_uids.json` de cada representación.
  - El lado en vivo lee números precalculados, sin numpy.
- **Lo que evita.** Que una coordenada o un vector viva solo en el script que lo calculó.

### 3.2 Almacén de sets

- **Esquema único**, una fila por aparición de un tema en un set: `set_id`, `fuente` (D1 a D5), `dj`,
  `posicion`, `minuto` (si hay), `duracion_set`, `segundos_al_aire` (D1, D2) y la referencia del tema.
- **Referencia del tema**, de mejor a peor:
  1. `track_uid` si está en la biblioteca;
  2. id de Beatport con género, subgénero, BPM y tonalidad (la página del tema de Beatport los da;
     cliente `src/v4/common/beatport.py`, con caché);
  3. id externo (YouTube) con artista y título.
- **Lo que evita.** Que D1, D3 y D5 tengan lectores y estadísticas por separado. T1, T2, T7, G7, M1, T9
  y E1 leen la misma tabla.

### 3.3 Componentes

Cada componente es una función (tema principal o estado del set, candidato) → número de 0 a 1 o None,
registrada con nombre. Las variantes de una misma idea son componentes con nombre propio, para que el
ranker aprendido (M3) pueda usarlas todas. `genero` sigue siendo el nombre del componente de género que
se use por defecto.

| Componente | Ideas que lo alimentan | Necesita |
| :--- | :--- | :--- |
| `clap` | P1, P2 | Embedding CLAP |
| `bpm`, `clave`, `tiv` | P3 a P5 | BPM y tonalidad (Traktor, tags o Beatport); TIV |
| `genero_tax`, `genero_blando`, `estilos_discogs`, `genero_cooc`, `genero_aprendido` | G1 a G4, G6, G7, G9 | Género o estilos del tema; estadísticas de sets |
| `transicion` | T1 | Género del principal y del candidato; tabla dirigida (puede depender de la dirección del perfil) |
| `fase` | T2, T7 | Minuto del set en vivo (D2) y estadística de fase del candidato |
| `energia_objetivo` | T3, T5, T6, T8 | Coordenada del candidato y objetivo del perfil |
| `suavidad` | T6, T7 | Tamaño del paso respecto de los últimos temas tocados (en CLAP y en la coordenada) |
| `texto_clap` | T3, T7 | Proyección en direcciones de texto de CLAP |
| `set2vec` | M1, M2 | Vector de sets del tema (o el predicho desde CLAP) |
| `aprendido` | M4 | Embedding del modelo de audio entrenado con transiciones |

### 3.4 Perfiles de playlist

Un perfil son los filtros comunes, un objetivo y pesos sobre los mismos componentes:

| Perfil | Objetivo | Componentes principales |
| :--- | :--- | :--- |
| Parecidos | Ninguno: el más parecido primero | `clap`; el género que gane en E1 y en la escucha |
| Mantener | Coordenada en el nivel actual | `energia_objetivo` (paso 0), `suavidad`, `clap` |
| Más / Menos | Nivel actual ± un paso típico (T6) | `energia_objetivo`, `transicion` según la dirección, `clap` |
| Introspectiva | Sin saltos y típico de la parte tardía de sets largos | `suavidad`, `fase`, `texto_clap`, `clap`; la energía no manda |

Los pesos de cada perfil salen de E1 y de la escucha; en la etapa 4, del ranker aprendido (M3). Añadir
una playlist es añadir un perfil, no un camino nuevo.

### 3.5 Estado de la trayectoria

Lo calcula el lado en vivo desde D2:
- temas tocados con su hora;
- minuto del set;
- nivel y pendiente de cada coordenada en los últimos temas.

Es el mismo estado para todos los perfiles. El banco E1 lo reconstruye igual desde D1, D3 y D5 al
repetir un set.

### 3.6 Banco de evaluación único (E1)

- **Qué hace.** Repite sets reales. En cada paso, el estado son los temas anteriores. Calcula los
  candidatos con los mismos filtros que en vivo y mide en qué puesto queda el tema que el DJ tocó
  después.
- **Qué informa.** Cada componente por separado, cada perfil y las líneas de base: al azar, solo BPM y
  solo CLAP.
- **Una regla.** Una sola variable por comparación, en los mismos sets.

Qué se puede evaluar sobre qué fuente (INFERENCIA desde el cruce de la sección 2: con 25 transiciones con
los dos temas en la biblioteca, lo que necesita audio no se mide en sets de otros DJs sin D6):

| Componentes | D1, D2 (Gabriel) | D3, D4, D5 sin audio | D3, D4, D5 con audio (D6) |
| :--- | :--- | :--- | :--- |
| `bpm`, `clave` | Sí | Sí, con BPM y tonalidad de Beatport | Sí |
| `genero_*`, `transicion`, `fase` | Sí | Sí, con el género de Beatport | Sí |
| `set2vec` | Sí, si sus temas aparecen en sets | Sí | Sí |
| `clap`, `estilos_discogs`, `texto_clap`, `suavidad` en CLAP, `energia_objetivo` de audio, `aprendido` | Sí | Solo en las 25 transiciones del cruce | Sí |

### 3.7 Puentes entre el espacio de los sets y el de la biblioteca

Es el riesgo principal de compartimentos estancos: los sets están en texto y la biblioteca en audio.
Hay tres puentes, uno por etapa:

1. **Etapa 2: los metadatos de Beatport.** Con el género, BPM y tonalidad de cada tema de los sets, las
   estadísticas de transición y de fase se calculan por género y se aplican a cualquier tema de la
   biblioteca por su género.
2. **Etapa 4: los 355 temas en común.**
   - Sirven de ancla para aprender a predecir el vector de sets (M1) desde CLAP (M2). Así los 1405
     temas restantes de la biblioteca reciben `set2vec`.
   - Sirven para comprobar que lo aprendido de los sets vale para la biblioteca.
3. **Etapa 5: el audio de los temas de los sets (D6).** Da CLAP y las coordenadas para todos y permite
   E1 completo sobre sets de otros DJs.

### 3.8 Combinación final

- **Hasta la etapa 3**, los pesos de cada perfil se ajustan con E1 y se confirman escuchando.
- **En la etapa 4 (M3)**, un ranker aprendido (learning to rank) toma el vector completo de componentes
  de cada candidato. Ese vector ya queda en el registro de cada lista desde hoy. Lo entrena con «qué tocó
  el DJ después» en D3, D5 y D1, por perfil:
  - la dirección de cada transición real se etiqueta por el cambio de la coordenada;
  - sobre D1 entra con peso menor, por el ruido.
- **Resultado.** Ninguna idea se descarta por no haber sido elegida a mano: el ranker decide su aporte.
  Cada decisión se confirma escuchando.

## 4. Inventario de ideas

Contrato: Tema = almacén de temas; Sets = almacén de sets; Comp = componente; Perfil; Eval = evaluación;
Traktor = plataforma en vivo.

### Parecido entre temas

| Id | Idea | Origen | Contrato | Depende de | Etapa | Estado |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| P1 | CLAP como base: el timbre es lo que más se parece entre temas consecutivos de un set | Kell y Tzanetakis 2013; elección de Gabriel (DECISIONS 2026-09-29) | Comp `clap` | — | — | Vigente |
| P2 | CLAP en escala comparable (percentil o rango entre los candidatos) | Medido 2026-11-06 | Comp `clap` | — | 1 | Pendiente |
| P3 | Filtros BPM ±20 y tonalidad ±2 semitonos. En Kell 2013 la tonalidad no mostró efecto en el orden de los DJs, pero es la regla de Gabriel | DECISIONS 2026-11-06, punto 7 | Filtro común | — | — | Vigente |
| P4 | La misma tonalidad original puntúa más que una que pide mover el pitch | DECISIONS 2026-11-06, punto 9 | Comp `clave` | — | — | Vigente (peso 0) |
| P5 | Vector TIV de compatibilidad armónica | Rama `harmonic-tiv` (otra sesión) | Comp `tiv`; Tema | Esa rama | Aparte | En curso |
| P6 | Pesos por componente y registro de componentes en cada lista | DECISIONS 2026-11-06, punto 9 | Perfil; Eval | — | — | Vigente |

### Género

| Id | Idea | Origen | Contrato | Depende de | Etapa | Estado |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| G1 | Taxonomía propia de Beatport → similitud escalar entre géneros | DECISIONS 2026-11-06, punto 11 | Comp `genero_tax` | — | — | HECHO 2026-11-06 |
| G2 | Aplicar las 17 objeciones menores de la red | Informe de la taxonomía | Comp `genero_tax` | E1 (para ver si cambian algo) | 2 | Pendiente |
| G3 | Género blando: probabilidades (Beatport con peso alto, las 3 primeras del clasificador para el resto) y similitud esperada; cubre los 158 sin género | Propuesta 2026-11-07 | Tema; Comp `genero_blando` | G1 | 1 | Pendiente |
| G4 | Las versiones con géneros distintos en Beatport (267 temas) dan una distribución | `genre_evidence/versiones.csv` | Tema (con G3) | G3 | 1 | Pendiente |
| G5 | El género de Beatport es una decisión editorial por lanzamiento: una señal entre otras | Gabriel; centro de ayuda de Beatport; DECISIONS 2026-11-07, punto 7 | Criterio | — | — | Vigente |
| G6 | Estilos de Discogs de MAEST como vector de género (coseno) | Alonso-Jiménez et al., ISMIR 2023 (UPF); medido 2026-11-07 | Tema; Comp `estilos_discogs` | — | 2 | Pendiente |
| G7 | Similitud entre géneros por aparición en los mismos sets | Knees y Schedl 2013; arXiv 2604.26119 | Sets; Comp `genero_cooc` | C1 sobre D3 | 2 | Pendiente |
| G8 | Una sola coordenada pierde estructura: hacen falta al menos dos (cierre hacia afuera y diferenciación interna) | arXiv 2604.26119; Sordo et al. 2008 | Criterio para G7 y G9 | — | — | Vigente |
| G9 | Embeddings de géneros aprendidos (audio o texto) | Hennequin et al. 2018; Epure et al. 2019 y 2020 | Comp `genero_aprendido` | E1 con G1, G3, G6 y G7 medidos | 4 | Exploratorio |
| G10 | Embeddings hiperbólicos de la jerarquía | Nickel y Kiela 2017 | Comp `genero_aprendido` | G9 muestra que la jerarquía importa | 4+ | Condicional |
| G11 | Mapeo de estilos de Discogs a Beatport | Red de agentes | — | — | — | DESCARTADO 2026-11-06: acierta el 24 % (el clasificador, 38 %); queda de referencia |
| G12 | Chequeo contra el audio: Tech House con Minimal y Progressive con MH&T se funden acústicamente (en la taxonomía, 0.80 y 0.85) | arXiv 2509.11474 | Criterio para G2 | — | — | Vigente |

### Dirección y trayectoria

| Id | Idea | Origen | Contrato | Depende de | Etapa | Estado |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| T1 | Transiciones dirigidas entre géneros: cuántas veces más que por azar se pasa de g a h; la taxonomía (G1) como respaldo donde hay pocos datos | Propuesta 2026-11-07 (no hay un estudio publicado) | Sets; Comp `transicion` | C1 sobre D3 y D5 | 2 | Pendiente |
| T2 | Fase del set por tema y por género (apertura, pico, cierre, después de la 2.ª hora) | Propuesta 2026-11-07 | Sets; Tema; Comp `fase` | C1; D5 para sets largos | 2 | Pendiente |
| T3 | Coordenada de energía sin Mixed In Key. Candidatos: arousal, engagement, danceability y estados de ánimo de Essentia (UPF); direcciones de texto en CLAP; la fase aprendida (T2) | Essentia; arXiv 2510.20759; T2 | Tema; Comp `energia_objetivo`, `texto_clap` | Extracción en WSL para Essentia; codificador de texto de CLAP | 3 | Pendiente |
| T4 | Validar las coordenadas con juicios por pares de Gabriel (unos 20 como mucho) | MusAV (ISMIR 2022); DECISIONS 2026-11-06, punto 10 | Eval | T3 | 3 | Pendiente |
| T5 | Trayectoria en vivo: nivel y pendiente de los últimos temas tocados | Kell 2013 (el orden importa) | Estado de la trayectoria | D2; T3 | 3 | Pendiente |
| T6 | Paso parejo: el nivel actual más un paso típico, medido en sets reales, no el tema más extremo | Kell 2013 (los pasos parejos se acercan más al orden real) | Comp `energia_objetivo`, `suavidad` | T3, T5; sets para el paso típico | 3 | Pendiente |
| T7 | Introspectiva: saltos chicos (`suavidad`), temas típicos de la parte tardía de sets largos (`fase`) y quizá una dirección de texto («hipnótico», «introspectivo»). Se calibra con los 3 o 4 sets largos de Gabriel y con D5 | Definición de Gabriel, 2026-11-07 | Perfil Introspectiva | T2, T6; D5 | 3 | Pendiente |
| T8 | Más y menos energía: un paso en T3 dentro del vecindario CLAP, con T1 según la dirección | Propuesta 2026-11-07 | Perfiles Más y Menos | T1, T3, T6 | 3 | Pendiente |
| T9 | Modelo que aprende durante la sesión qué transiciones le gustan a Gabriel | DJ-MC, Liebman et al. 2015 | Perfil (ajuste en vivo) | D2 con sesiones; M3 | 4 | Pendiente |
| T10 | La dirección se aprende de sets de DJ, no de playlists de oyentes, donde el orden casi no ayuda | Vall et al. 2019 frente a Kell 2013 y DJ-MC | Criterio para las fuentes | — | — | Vigente |

### Representación aprendida de sets

| Id | Idea | Origen | Contrato | Depende de | Etapa | Estado |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| M1 | Vector de cada tema aprendido de los sets (word2vec sobre tracklists), solo texto | Deej-AI; recomendación por playlists | Tema; Comp `set2vec` | Sets (D3, D5, D1) | 4 | Pendiente |
| M2 | Predecir el vector de sets desde CLAP, anclado en los 355 temas en común, para los temas que no aparecen en sets | Vall et al. (híbridos) | Tema; Comp `set2vec` | M1 | 4 | Pendiente |
| M3 | Ranker aprendido sobre el vector de componentes, por perfil; el género y todo lo demás pasan a ser entradas | Propuesta 2026-11-07 | Perfil | E1; componentes de las etapas 1 a 3 | 4 | Pendiente |
| M4 | Modelo de audio entrenado con pares «tema y el que vino después» (aprendizaje contrastivo, la receta de la UPF con playlists) | Alonso-Jiménez et al., ICASSP 2023 | Tema; Comp `aprendido` | D6; GPU y disco | 5 | Pendiente |

### Evaluación

| Id | Idea | Etapa |
| :--- | :--- | :--- |
| E1 | Banco único de sets no vistos (sección 3.6) | 1 |
| E2 | Lo que dice E1 sirve para descartar lo claramente peor; entre lo que queda, escucha a ciegas de listas enteras | Siempre |
| E3 | Unas 20 preguntas a Gabriel como mucho (tripletas o pares) | 3 |
| E4 | D3 está cargado hacia House y Techno: informar los géneros chicos aparte | 2 |
| E5 | Una sola variable por comparación, en los mismos sets | Siempre |
| E6 | Todo resultado se informa sobre los sets de otros DJs y sobre los de Gabriel por separado | Siempre |

### Plataforma en vivo (Traktor)

| Id | Idea | Etapa |
| :--- | :--- | :--- |
| I1 | Un `.nml` por perfil en la carpeta de sugerencias; todos se calculan con los mismos candidatos y se escriben en la misma pulsación | 1 (estructura), 3 (perfiles de dirección) |
| I2 | `Toggle`: todas las playlists de sugerencias cuentan como «dentro». Pulsar desde cualquiera vuelve a la carpeta guardada antes de la primera pulsación; pulsar desde fuera refresca y abre Parecidos | 1 |
| I3 | Cada lista anota en el registro sus componentes, su perfil y su objetivo | 1 (ya existe para Parecidos) |
| I4 | Costo en vivo con 5 perfiles por debajo de unos 300 ms; el cálculo sigue siendo antes de apretar y se rehace al apretar | 1 |

## 5. Etapas, dependencias y criterios de salida

```
Etapa 0 (verificaciones) ──► Etapa 1 (bases) ──► Etapa 2 (sets de otros DJs, texto)
                                    │                     │
                                    └─────────┬───────────┘
                                              ▼
                              Etapa 3 (dirección e introspectiva)
                                              ▼
                              Etapa 4 (aprendido: M1-M3, G9, T9)
                                              ▼
                              Etapa 5 (audio de los sets, M4)
```

**Criterios de salida comunes a todas las etapas:**
- tests del camino cambiado en verde;
- artefactos con cantidad de filas, tipos y valores finitos verificados;
- tabla de E1 contra las líneas de base, cuando corresponde;
- DECISIONS y STATUS al día;
- commit y push;
- la escucha que corresponda.

### Etapa 0: verificaciones que podrían cambiar el diseño

| Id | Qué | Cómo | Si falla |
| :--- | :--- | :--- | :--- |
| V1 | Traktor acepta en Favoritos un `.nml` de la carpeta de sugerencias y lo relee al elegirlo con el mouse | Dos playlists de prueba; Gabriel las pone en Favoritos y cambia entre ellas después de reescribirlas | Una playlist por carpeta, o Favoritos de carpetas (HIPÓTESIS: hoy Traktor relee un `.nml` del Explorer al entrar desde otra carpeta, según `sugerencias_vivo.py`) |
| V2 | El cambio de `Toggle` (I2) | Test unitario con caminos de varias playlists | — |
| V3 | El costo en vivo con 5 perfiles (I4) | Medir sobre la colección real | Precalcular más en el índice |
| V4 | Acceso a sets largos de 1001Tracklists (D5) | Ver sección 6 | Sets largos de MixesDB y de Gabriel |
| V5 | La lectura de la historia de Traktor | Sacar los repetidos entre 4.2.0 y 4.4.1 y tratar el horario (cruce de medianoche y sesiones dejadas abiertas); comparar a mano con 2 sets | — |

### Etapa 1: bases

1. Almacén de sets (3.2) con lector de D1 y de D2.
   - Sale: tabla en `artifacts/v4/dj_sets/`, con los conteos de la sección 2 reproducidos.
2. E1 (3.6) sobre D1, con líneas de base al azar, solo BPM y solo CLAP.
   - Sale: tabla base.
3. P2, G3 y G4 como componentes, medidos en E1 uno por vez.
4. Índice versión 4 con campos opcionales (3.1) y registro de componentes con nombre (3.3).
5. I1 a I4 con un perfil: Parecidos.
   - Sale: la estructura de varias playlists funcionando en Traktor (con V1), sin perfiles de dirección
     todavía.

### Etapa 2: sets de otros DJs, solo texto

1. C1: resolver los temas de D3 con Beatport (género, subgénero, BPM, tonalidad).
   - Muestra chica y después completo, a una consulta por segundo con caché y `--low-priority`. Primero
     los sets de House, Techno y Tech House.
   - Con D5 también, si se consiguió.
2. T1, T2, G6, G7 y G2 medidos en E1 sobre D3 (lo que se puede medir sin audio, 3.6) y sobre D1.
3. Cuando un resultado venga de D3 y se aplique a la biblioteca, comprobarlo en los 355 temas en común.

### Etapa 3: dirección e introspectiva

1. T3: coordenadas candidatas.
   - Essentia corre en WSL (no tiene versión nativa para Windows); necesita embeddings EffNet-Discogs de
     la biblioteca, con una corrida chica antes.
   - Las direcciones de texto de CLAP corren en Windows.
2. T4: hasta unos 20 juicios de Gabriel por pares.
3. T5, T6, T7 y T8 como perfiles.
   - Pesos con E1 sobre D1 y D5 (sets largos), más D3 para lo de texto.
   - Escucha a ciegas de Gabriel.
4. Los perfiles de dirección en Traktor (I1).

### Etapa 4: aprendido

- M1 (D3, D5, D1), M2 (anclas en común) y M3 (ranker por perfil sobre el vector de componentes).
- G9, y T9 cuando D2 tenga sesiones.
- Sale: E1 con el ranker frente a los pesos a mano, y la escucha.

### Etapa 5: audio

- D6: la fuente del audio se decide con Gabriel (DECISIONS 2026-09-12, punto 3).
- Estimación de disco y GPU medida en una muestra. Orden de magnitud: 13 728 temas en avances de
  2 minutos son unos 66 GB.
- CLAP y coordenadas para los temas de los sets: E1 completo sobre sets de otros DJs.
- M4 con GPU.
- Se rehace E1 de todo.

## 6. Datos: acceso, licencias y volumen

| Id | Fuente | Acceso | Licencia y condiciones | Volumen |
| :--- | :--- | :--- | :--- | :--- |
| D1 | Historia de Traktor de Gabriel | Local | De Gabriel | 29 sets entre las dos versiones, con repetidos: 592 temas al aire en los 21 sets de 4.4.1 y 201 en las 12 sesiones de 4.2.0 |
| D2 | Registro en vivo | Local | De Gabriel | Crece con cada sesión |
| D3 | djmix-dataset | Bajado (JSON, 11,6 MB) | Sin licencia publicada: uso de investigación, no se redistribuye | Sección 2 |
| D4 | Raveform (Hugging Face `taejunkim/raveform`) | Sin bajar | Sin licencia publicada en la página | 4902 sets; 38 % con horarios; 1423 temas con género de Beatport puesto a mano |
| D5 | 1001Tracklists, sets largos | Ver debajo | Ver debajo | A definir |
| D6 | Audio de los temas de los sets | Decisión de Gabriel | Depende de la fuente | Etapa 5 |

**1001Tracklists (D5).** Lo observado el 2026-11-07:
- No tiene API pública.
- Su `robots.txt` bloquea por completo, entre otros, al agente «CPython» (los scripts de Python) y a
  muchos robots. Al resto le pide 8 segundos entre páginas.
- Su página de privacidad menciona unos «Terms & Conditions» que no encontré publicados.
- Su contacto es info@1001tracklists.com; el operador es Sacks Music Services LLC.
- El dataset del paper de 2020 (Kim et al.) se obtuvo «por comunicación directa con 1001Tracklists». Uno
  de sus autores se apellida Sacks (INFERENCIA: es probable que sea del sitio, no verificado).

Ruta propuesta:
1. Que Gabriel pida acceso de investigación a info@1001tracklists.com: sets de 3 h o más, con minutos.
   El agente le prepara el correo, pero lo manda Gabriel.
2. No esquivar el `robots.txt`: no se cambia el agente para pasar por navegador.
3. Mientras tanto:
   - sets largos de MixesDB, cuyo `robots.txt` permite leer páginas a una cada 4 segundos y bloquea las
     consultas a la API (HIPÓTESIS: hay suficientes sets largos con lista completa; se decide con una
     muestra);
   - Raveform;
   - los sets largos de Gabriel.

## 7. Lista de comprobación para el auditor

1. ¿Cada idea del inventario tiene un contrato de la sección 3? ¿Hay alguna con un camino propio hasta
   Traktor o con datos que solo lee ella?
2. ¿Hay ideas incompatibles?
   - Ejemplos a mirar: la regla de tonalidad de Gabriel frente a Kell 2013 (P3); la fase 4 con Mixed In
     Key frente a «no usar Mixed In Key» (son ámbitos distintos: la organización frente a lo nuevo en
     vivo); la introspectiva sin energía frente a los perfiles de energía.
3. ¿Las dependencias de la sección 5 están en orden? ¿Alguna etapa usa algo que se produce después?
4. ¿La tabla de la sección 3.6 es correcta? ¿Algo se promete medir en sets de otros DJs sin audio cuando
   necesita audio?
5. ¿Los puentes de la sección 3.7 alcanzan para que lo aprendido en sets se use en la biblioteca?
6. ¿La regla de ruido de los sets de Gabriel (E6, sin ajustar solo contra D1) se cumple en cada etapa?
7. ¿Las condiciones de D3, D4 y D5 (sección 6) están respetadas?
8. ¿Las invariantes de `AGENTS.md` (`track_uid`, nada de datos en Git, CPU primero, prueba chica antes
   de una corrida larga) se cumplen en cada etapa?
9. ¿El lado en vivo sigue sin dependencias fuera de la biblioteca estándar?

## 8. Afirmaciones

| Afirmación | Etiqueta | Evidencia |
| :--- | :--- | :--- |
| D3 tiene 5040 sets, 63 038 temas con ID, 64 764 transiciones y ninguno con minutos pasa de 150 min | EVIDENCIA OBSERVADA | `djmix-dataset.json`, conteo de esta sesión |
| 355 temas de la biblioteca aparecen en D3, y solo 25 transiciones tienen los dos | EVIDENCIA OBSERVADA | Cruce por artista y título normalizados (puede subestimar: remixes con otro nombre) |
| Sin audio de los sets, los componentes de audio no se pueden medir en sets de otros DJs | INFERENCIA | De la fila anterior |
| Gabriel tiene 3 o 4 sets de 3 h o más | EVIDENCIA OBSERVADA, con reservas | Historia de Traktor; horarios con cruce de medianoche sin limpiar (V5) |
| El `Toggle` actual no trata varias playlists como «dentro» | EVIDENCIA OBSERVADA | `sugerencias_vivo.py`, clase `Toggle`, método `at_target` |
| Traktor relee en Favoritos un `.nml` del Explorer | HIPÓTESIS | V1 |
| 1001Tracklists bloquea los scripts de Python y no tiene API pública | EVIDENCIA OBSERVADA | Su `robots.txt` y su sitio, 2026-11-07 |
| El acceso de investigación a 1001Tracklists se consigue pidiéndolo | HIPÓTESIS | Kim et al. lo obtuvieron así; se decide con el correo de Gabriel |
| CLAP varía unos 0.05 entre los 30 primeros | EVIDENCIA OBSERVADA | Colección real, 40 temas principales |
| El género (taxonomía y estilos de Discogs) aporta información que CLAP no tiene | EVIDENCIA OBSERVADA (correlaciones 0.22 y 0.35); que eso mejore las listas es NO CONCLUYENTE | Falta E1 y la escucha |

## 9. Fuentes

- Kell y Tzanetakis, «Empirical Analysis of Track Selection and Ordering in Electronic Dance Music using
  Audio Feature Extraction», ISMIR 2013.
- Liebman, Saar-Tsechansky y Stone, «DJ-MC», AAMAS 2015 (arXiv 1401.1880).
- Vall, Quadrana, Schedl y Widmer, «Order, context and popularity bias in next-song recommendations»,
  IJMIR 2019.
- Kim, Choi, Sacks, Yang y Nam, «A Computational Analysis of Real-World DJ Mixes», ISMIR 2020
  (arXiv 2008.10267).
- djmix-dataset (github.com/mir-aidj/djmix-dataset); Raveform (mir-aidj.github.io/raveform).
- «Acoustic Overspecification in Electronic Dance Music Taxonomy», arXiv 2509.11474.
- «Two-Dimensional Structural Characterization of Music Genre Communities in Playlist Co-occurrence
  Networks», arXiv 2604.26119.
- Sordo, Celma, Blech y Guaus, «The Quest for Musical Genres», ISMIR 2008.
- Knees y Schedl, «A survey of music similarity and recommendation from music context data», ACM TOMM
  2013.
- Hennequin, Royo-Letelier y Moussallam, «Audio based disambiguation of music genre tags», ISMIR 2018.
- Epure, Khlif y Hennequin, ISMIR 2019; Epure, Salha, Moussallam y Hennequin, EMNLP 2020.
- Nickel y Kiela, «Poincaré Embeddings for Learning Hierarchical Representations», NeurIPS 2017.
- Alonso-Jiménez, Serra y Bogdanov, «Efficient Supervised Training of Audio Transformers for Music
  Representation Learning» (MAEST), ISMIR 2023.
- Alonso-Jiménez et al., «Pre-Training Strategies Using Contrastive Learning and Playlist Information
  for Music Classification and Similarity», ICASSP 2023.
- Modelos de Essentia (MTG-UPF): arousal y valencia, engagement, approachability, danceability y
  estados de ánimo; MusAV (ISMIR 2022).
- «Controllable Embedding Transformation for Mood-Guided Music Retrieval», arXiv 2510.20759.
- Bittner et al., «Automatic Playlist Sequencing and Transitions», ISMIR 2017.
- Centro de ayuda de Beatport: el equipo de curación revisa y asigna el género.
- `robots.txt` de 1001tracklists.com y de mixesdb.com, y la página de privacidad de 1001Tracklists,
  leídos el 2026-11-07.
