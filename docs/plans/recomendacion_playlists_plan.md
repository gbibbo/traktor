# Plan: playlists de recomendación en vivo (parecidos y dirección)

Fecha: 2026-11-07. Decisiones de Gabriel en DECISIONS 2026-11-07. Lo anterior: DECISIONS 2026-11-06,
puntos 7 a 11; informes `docs/reports/generos_beatport_2026-11-06.md` y
`docs/reports/taxonomia_generos_2026-11-06.md`.

Este plan es el inventario completo de las ideas que salieron al pensar la recomendación. Cada idea
tiene un identificador, su origen, lo que necesita, cómo se evalúa y su etapa. Una idea que se deja para
después, o que se descarta, sigue acá con el motivo. Al cerrar una etapa se marca en su lugar
(`HECHO <fecha>`, `DESCARTADO <fecha>: motivo`); no se borra.

## Objetivo (Gabriel, 2026-11-07)

Varias playlists de Traktor que se actualizan solas mientras suena el tema principal:
- **Parecidos**: los temas más parecidos al tema principal, ordenados por parecido. El primero es el más
  parecido de la biblioteca. Es la lista que ya existe (`recomendador.py`).
- **Dirección**: playlists que llevan el set hacia un lado: mantener la energía, más energía, menos
  energía, más introspectiva. Tienen en cuenta la trayectoria de todo el set, no solo el tema que suena.

«Introspectivo», en palabras de Gabriel:
- suele pasar en sets de 3 horas o más, en general después de la segunda hora;
- no depende necesariamente de la energía;
- en general no tiene grandes saltos.

## Restricciones

- No se usa la energía de Mixed In Key: Gabriel la considera prácticamente inútil. En los datos, el 83 %
  de los 1424 temas que la tienen está en 5 o 6.
- Los sets pasados de Gabriel sirven de referencia, pero pueden tener ruido. No se ajusta nada solo
  contra ellos: cada medida se informa también sobre sets de otros DJs.
- Entre opciones que sobreviven a las medidas, decide Gabriel escuchando (skill `auditar`).
- Hasta la etapa 5 todo es texto y CPU local. GPU y disco los consigue Gabriel cuando hagan falta, con
  una estimación medida antes sobre una muestra chica.
- Los datos de sets van fuera de Git (`artifacts/v4/dj_sets/`).

## Datos

| Id | Fuente | Qué trae | Estado |
| :--- | :--- | :--- | :--- |
| D1 | Historia de Traktor de Gabriel (`Documents/Native Instruments/Traktor 4.4.1/History`, también la de 4.2.0) | Por sesión: temas cargados, deck, hora de inicio, segundos que sonó y si salió al aire (`PLAYEDPUBLIC`). 4.4.1: 36 sesiones, 21 sets reales (8 o más temas al aire por 2 min o más), 592 temas. 4.2.0: 12 sesiones, 201 temas al aire. Tiene sets largos (varios de más de 6 h) | Existe; falta leerla como tabla de sets |
| D2 | Registro en vivo (`artifacts/v4/traktor_live/*.jsonl`): eventos `principal` y `tocado` (30 s al aire con fader >= 50 %) | Lo que suena y lo que se tocó en cada sesión, con faders | Código desde el 2026-11-06; todavía sin sesiones registradas en esta PC |
| D3 | djmix-dataset (grupo de Kim, KAIST), `artifacts/v4/dj_sets/djmix/` | 5040 sets de MixesDB y 63 038 temas únicos (ID de YouTube y título «Artista - Título [Sello]»). 64 764 transiciones. Categorías por set (House 1555, Techno 1547, Tech House 1410). 779 sets con minuto por tema, ninguno de más de 150 min. Sin género por tema ni audio; sin licencia publicada | Bajado el 2026-11-07 |
| D4 | Raveform (mismo grupo; Hugging Face `taejunkim/raveform`) | 4902 sets de MixesDB con géneros y tags; 38 % con horarios; 1423 temas con género de Beatport puesto a mano y estructura | Sin bajar; se suma a D3 si D3 queda corto |
| D5 | 1001Tracklists propiamente dicho | Sets largos con orden y horarios: lo que falta para la fase introspectiva. El dataset del paper de 2020 no es público | Sin decidir: fuente y condiciones de uso se consultan con Gabriel (DECISIONS 2026-09-12, punto 3) |
| D6 | Audio de los temas de los sets (D3 y D4 traen enlaces) | Para aprender un modelo de audio de las transiciones (M4) | Etapa 5; decisión aparte de Gabriel |

**Cruce de temas** (C1): para usar D3 y D4 hay que saber el género y, si está, el tema de la biblioteca.
- Opción 1: por artista y título contra la biblioteca (`_organizacion.csv` y tags).
- Opción 2: búsqueda en Beatport con el cliente que ya existe (`src/v4/common/beatport.py`, con caché).

Son decenas de miles de temas a una consulta por segundo: es una corrida larga. Primero una muestra
chica, después la completa con `--low-priority`, y empezando por los sets de House, Techno y Tech House.

## Inventario de ideas

### Parecido entre temas (playlist Parecidos y base de las demás)

| Id | Idea | Origen | Etapa |
| :--- | :--- | :--- | :--- |
| P1 | CLAP es la base del parecido: lo que más se parece entre temas consecutivos de un set es el timbre | Kell y Tzanetakis, ISMIR 2013; elección de Gabriel escuchando (DECISIONS 2026-09-29) | Vigente |
| P2 | Llevar CLAP a una escala comparable (percentil o rango entre los candidatos). Hoy, entre el 1.º y el 30.º sugerido varía unos 0.05, y cualquier peso de género lo tapa | Medido 2026-11-06 | 1 |
| P3 | Filtros de BPM ±20 y tonalidad a ±2 semitonos. En Kell 2013 la tonalidad no mostró diferencias en el orden de los DJs, pero es la regla de Gabriel y se mantiene | DECISIONS 2026-11-06, punto 7 | Vigente |
| P4 | Componente de tonalidad: la misma tonalidad original puntúa más que una que pide mover el pitch | DECISIONS 2026-11-06, punto 9 | Vigente (peso 0) |
| P5 | TIV: vector de compatibilidad armónica | Otra sesión, rama `harmonic-tiv` | En curso aparte |
| P6 | Pesos ajustables por componente (`"pesos"`) y registro de los componentes de cada lista | DECISIONS 2026-11-06, punto 9 | Vigente |

### Género

| Id | Idea | Origen | Etapa |
| :--- | :--- | :--- | :--- |
| G1 | Taxonomía propia de los 47 géneros de Beatport → similitud escalar entre géneros | Red de agentes, DECISIONS 2026-11-06, punto 11 | HECHO 2026-11-06 |
| G2 | Aplicar las 17 objeciones menores de la ronda 2 (una ronda más de la red, o a mano con un evaluador) | Informe de la taxonomía | 2 |
| G3 | Género blando: probabilidad de cada género (Beatport con peso alto, las 3 primeras del clasificador para el resto) y similitud esperada. Cubre los 158 temas sin género | Propuesta 2026-11-07 | 1 |
| G4 | Las versiones de un tema con géneros distintos en Beatport (267 temas) dan una distribución, no una etiqueta: se usa como género blando del tema | Evidencia `versiones.csv` | 1 (con G3) |
| G5 | La etiqueta de Beatport es una decisión editorial y comercial, por lanzamiento, de su equipo de curación (no un modelo de audio, según su centro de ayuda). Es una señal entre otras, no la verdad | Gabriel y fuentes, 2026-11-07 | Criterio |
| G6 | Estilos de Discogs de MAEST (106 electrónicos) como vector de género, comparado por coseno. Medido: correlación 0.35 con CLAP y 0.33 con la taxonomía; de los 30 primeros, solo 3 coinciden con los de CLAP. Discogs: etiquetas de la comunidad, por lanzamiento | MAEST, Alonso-Jiménez et al., ISMIR 2023 (UPF); medido 2026-11-07 | 2 (contra G1 y G3) |
| G7 | Similitud entre géneros por aparición conjunta en sets de DJ (dos géneros que se tocan en el mismo set) | Knees y Schedl 2013; arXiv 2604.26119 | 2 |
| G8 | Una sola coordenada pierde estructura: las comunidades de géneros necesitan al menos dos dimensiones (cierre hacia afuera y diferenciación interna) | arXiv 2604.26119; Sordo et al. 2008 | Criterio para G7 y G9 |
| G9 | Embeddings de géneros aprendidos (desde audio o texto) en lugar de una matriz armada a mano | Hennequin et al. 2018; Epure et al. 2019 y 2020 (Deezer) | 4 (exploratorio) |
| G10 | Embeddings hiperbólicos para la jerarquía de géneros | Nickel y Kiela 2017 | Después de G9, solo si G9 muestra que la jerarquía importa |
| G11 | Mapeo de estilos de Discogs a Beatport | Red de agentes | DESCARTADO 2026-11-06: acierta el 24 % (el clasificador, 38 %); queda como referencia |
| G12 | Chequeo de la taxonomía contra el audio: Tech House con Minimal y Progressive con MH&T se funden acústicamente (en la taxonomía, 0.80 y 0.85) | arXiv 2509.11474 | Criterio |

### Dirección y trayectoria (playlists de dirección)

| Id | Idea | Origen | Etapa |
| :--- | :--- | :--- | :--- |
| T1 | Transiciones dirigidas entre géneros: cuántas veces más que por azar se pasa del género g al h, de D3 (y D1). Donde hay pocos datos, la taxonomía (G1) sirve de respaldo | Propuesta 2026-11-07; no hay un estudio publicado de esto | 2 |
| T2 | Fase del set por tema y por género: dónde suele aparecer (apertura, pico, cierre, después de la 2.ª hora) | Propuesta 2026-11-07 | 2 (D3 para sets de hasta 2,5 h; D1 y D5 para sets largos) |
| T3 | Coordenada de energía por tema, sin Mixed In Key. Candidatos: arousal, engagement, danceability y estados de ánimo de Essentia (UPF); direcciones de texto en CLAP; la fase del set aprendida (T2) | Essentia; trabajo de Pandora arXiv 2510.20759; T2 | 3 |
| T4 | Validar las coordenadas con juicios por pares de Gabriel («¿cuál tiene más energía?»), como mucho unos 20 | MusAV (ISMIR 2022); DECISIONS 2026-11-06, punto 10 | 3 |
| T5 | Trayectoria del set en vivo: nivel y pendiente de los últimos temas tocados (D2) | Kell 2013 (el orden importa) | 3 |
| T6 | Paso parejo: el objetivo es el nivel actual más un paso en la dirección elegida, no el tema más extremo. «Mantener» = pasos chicos alrededor del nivel actual | Kell 2013 (las heurísticas de pasos parejos se acercan más al orden real que el parecido decreciente) | 3 |
| T7 | Introspectiva como fase y como forma de la trayectoria: saltos chicos en CLAP, más temas típicos de la parte tardía de los sets largos (T2), y quizá una dirección de texto en CLAP («hipnótico», «introspectivo»). Se calibra con los sets largos de Gabriel (D1) y, si se consigue, con D5 | Definición de Gabriel, 2026-11-07 | 3 |
| T8 | Más energía y menos energía: un paso en T3 dentro del vecindario CLAP, con BPM y tonalidad filtrados y T1 como guía de qué género suele venir | Propuesta 2026-11-07 | 3 |
| T9 | Modelo de secuencia que aprende qué transiciones le gustan a Gabriel durante la sesión (planear el tema siguiente como un DJ) | DJ-MC, Liebman et al. 2015 | 4 |
| T10 | En playlists de oyentes comunes el orden casi no ayuda a predecir el tema siguiente. En sets de DJ sí, según Kell 2013 y DJ-MC. Por eso la dirección se aprende de sets de DJ, no de playlists | Vall et al. 2019 | Criterio |

### Representación aprendida de sets (largo plazo, modelo de representación propio)

| Id | Idea | Origen | Etapa |
| :--- | :--- | :--- | :--- |
| M1 | Representación de cada tema aprendida de los sets, tratando los temas como palabras y los sets como frases (word2vec sobre tracklists). Solo texto | Deej-AI; prácticas de recomendación por playlists | 4 |
| M2 | Combinación con CLAP para temas que no aparecen en ningún set (arranque en frío) | Vall et al. (híbridos de contenido y aparición conjunta) | 4 |
| M3 | El género deja de pesarse a mano y pasa a ser una entrada más del modelo | Propuesta 2026-11-07 | 4 |
| M4 | Modelo de audio entrenado con pares «tema y el que vino después» de sets reales: aprendizaje contrastivo, la receta de la UPF con playlists | Alonso-Jiménez et al., ICASSP 2023 | 5 (necesita audio D6, GPU y disco) |

### Evaluación

| Id | Idea | Etapa |
| :--- | :--- | :--- |
| E1 | Banco de prueba con sets que el modelo no vio: en qué puesto queda el tema que el DJ tocó después. Se informa por separado sobre D3 (otros DJs) y D1 (Gabriel, con ruido) | 1 |
| E2 | Sirve para descartar lo claramente peor. Entre lo que queda decide la escucha a ciegas de listas enteras | Siempre |
| E3 | Como mucho unas 20 preguntas a Gabriel: tripletas o pares | 3 |
| E4 | D3 está cargado hacia House y Techno: es el mundo de Gabriel, pero hay que mirar los géneros chicos aparte | 2 |
| E5 | Una sola variable por comparación y el mismo conjunto de sets (skill `auditar`) | Siempre |

### Implementación en Traktor

| Id | Idea | Etapa |
| :--- | :--- | :--- |
| I1 | Varias playlists en el `.nml` de sugerencias (Parecidos, Mantener, Más, Menos, Introspectiva), recalculadas cuando cambia el tema principal o un tema pasa a tocado | 3 |
| I2 | Cómo se elige cuál abrir con la K2 o el teclado (hoy D#0 + D0 abre una sola). Se define con Gabriel | 3 |
| I3 | Cada lista anota sus componentes y su objetivo en el registro, para ajustar después | 3 |

### Pendientes de decidir con Gabriel

- **Fase 4 del pipeline** (`phase4_order.py`): arranca cada playlist de la organización por el tema de
  menor energía de Mixed In Key. Quitarlo cambia el orden de las playlists ya exportadas al pendrive. No
  se toca sin su OK.
- **D5**: si se usa 1001Tracklists propiamente dicho para sets largos, de qué fuente.
- **I2**: el gesto para elegir la playlist.

## Etapas

1. **Bases, local:**
   - P2: escala de CLAP.
   - G3 y G4: género blando.
   - D1 leída como tabla de sets: un tema por fila, con su set, orden, hora, duración al aire y
     track_uid.
   - E1 armado sobre D1 y una primera prueba con P1, P2 y G1.
2. **Sets de otros DJs:**
   - C1 sobre D3: muestra, después completo.
   - T1, T2, G6 y G7 medidos con E1.
   - G2 si las objeciones cambian algo medible.
3. **Playlists de dirección:**
   - T3 y T4: coordenadas candidatas y validación con Gabriel.
   - T5 a T8: trayectoria, paso, introspectiva, más y menos.
   - I1 a I3 en Traktor.
4. **Representación aprendida de sets:** M1 a M3, G9, T9.
5. **Audio y GPU:** D6 y M4, con la estimación de GPU y disco antes de pedirlos.

Cada etapa cierra con tests, registro en DECISIONS si hubo decisiones, STATUS, commit y la escucha que
corresponda.

## Fuentes

- Kell y Tzanetakis, «Empirical Analysis of Track Selection and Ordering in Electronic Dance Music using
  Audio Feature Extraction», ISMIR 2013.
- Liebman, Saar-Tsechansky y Stone, «DJ-MC», AAMAS 2015 (arXiv 1401.1880).
- Vall, Quadrana, Schedl y Widmer, «Order, context and popularity bias in next-song recommendations»,
  IJMIR 2019.
- Kim, Choi, Sacks, Yang y Nam, «A Computational Analysis of Real-World DJ Mixes», ISMIR 2020
  (arXiv 2008.10267); djmix-dataset (github.com/mir-aidj/djmix-dataset); Raveform
  (mir-aidj.github.io/raveform).
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
