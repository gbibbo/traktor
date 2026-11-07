# Plan de las playlists en vivo: informe para auditoría externa

Fecha: 2026-11-07. Autor: agente de Claude Code (sesión traktor-github-3a), con las decisiones de Gabriel
del 2026-11-06 y el 2026-11-07 (`docs/DECISIONS.md`).

Auditado el 2026-11-07 (skill `auditar`, otra sesión): `docs/reports/auditoria_plan_playlists_2026-11-07.md`.
Las correcciones están en su lugar, con `[REFUTADO …]` o `[REEMPLAZADO …]`; los refuerzos llevan la marca
«(auditoría 11-07)». La estructura del plan no cambió.

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
| Cobertura de D1 en el índice (auditoría 11-07) | Historia 4.4.1 cruzada con `indice_recomendacion.json` | 562 de 592 temas al aire (95 %); 535 de 571 transiciones con los dos temas |
| Representaciones guardadas de `musica` (auditoría 11-07) | `artifacts/v4/datasets/musica/representations/` | `clap_full` (512), `maesthf_full_l7` (768) y `maesthf_full_styles` (400), 1786 uids cada una. **No hay EffNet-Discogs ni MusiCNN guardados** |
| Modelos de Essentia en disco (auditoría 11-07) | `models/essentia/` y el home de WSL | Solo los troncos `discogs-effnet` y `discogs-maest` y la cabeza `genre_discogs400`. Las cabezas de T3 (engagement, danceability, mood_*) no están bajadas; arousal/valencia no existe sobre EffNet, solo sobre MusiCNN y VGGish (essentia.upf.edu/models, 2026-11-07) |

**Historia de Traktor (D1).**
- [REFUTADO 2026-11-07 por el md5 de los 12 archivos (auditoría): la carpeta `Traktor 4.2.0/History` es
  copia byte a byte de 12 archivos de `4.4.1/History` y no suma ningún set] Hay 29 sets con 8 o más temas
  al aire por 2 minutos o más, sumando las dos versiones. Hay al menos 2 repetidos entre 4.2.0 y 4.4.1.
  Lo que vale: **21 sets** en 4.4.1 (36 archivos; 15 son pruebas de menos de 8 temas), 592 temas al aire
  por 2 minutos o más y 571 transiciones.
- Sets de 3 h o más (del primer al último tema al aire, leyendo `STARTDATE` + `STARTTIME`): 3 limpios
  (3,4, 3,4 y 3,8 h) y uno de 130 temas (8,5 h). [REFUTADO 2026-11-07 por esa lectura: con `STARTDATE`
  ningún set cruza mal la medianoche; las «24 h» salían de leer solo `STARTTIME`] Otros dos de 13 temas o
  menos dan unas 24 h por el horario: hay que tratar el cruce de medianoche y las sesiones dejadas
  abiertas.
- Temas después de la segunda hora en esos sets: entre 60 y 150, según cómo se cuente el de 130 temas.
- **Cobertura en el índice (auditoría 11-07)**: 562 de los 592 temas al aire (95 %) están en el índice de
  recomendación (por nombre de archivo o por artista-título), y 535 de las 571 transiciones (94 %) tienen
  los dos temas. E1 sobre D1 es viable; un set (2026-01-08, 22:59) tiene solo 17 de 37.

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
| `siguiente` (adenda 11-07) | T13 | Punto del espacio CLAP que predice el modelo de trayectorias a partir de lo tocado |

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

**Sesión (auditoría 11-07).** El minuto del set y la pendiente necesitan saber cuándo empezó el set:
empieza con el primer `tocado` de la sesión de Traktor y se reinicia si pasan más de 20 minutos sin
nada al aire (Gabriel probando temas no es un set). El mismo criterio corta las sesiones de D1 dejadas
abiertas. Los 20 minutos son un valor inicial: se miran contra los 21 sets de D1 antes de fijarlos.

### 3.6 Banco de evaluación único (E1)

- **Qué hace.** Repite sets reales. En cada paso, el estado son los temas anteriores. Calcula los
  candidatos con los mismos filtros que en vivo y mide en qué puesto queda el tema que el DJ tocó
  después.
- **Qué informa.** Cada componente por separado, cada perfil y las líneas de base: al azar, solo BPM y
  solo CLAP.
- **Una regla.** Una sola variable por comparación, en los mismos sets.
- **Medidas (auditoría 11-07).** La principal es el acierto en los 30 primeros (lo que Gabriel ve); además
  el rango recíproco medio y el percentil del rango, porque el número de candidatos cambia con los
  filtros. El tema que el DJ tocó y que los filtros de P3 dejaron fuera no tiene rango: se informa
  aparte como «recuperación de los filtros», que es la medida de P3 en sí.
- **Candidatos por fuente (auditoría 11-07).** En D1 y D2, el índice menos lo tocado, como en vivo. En
  D3, D4 y D5 sin audio, los temas de esas fuentes con metadatos de Beatport (C1) menos lo tocado en el
  set: el tema siguiente no está en la biblioteca, así que la biblioteca no sirve de pool ahí.
- **Partición (auditoría 11-07).** Las estadísticas de T1, T2, G7 y M1 se calculan en sets de
  entrenamiento y se miden en sets de otros DJs (D3 trae 1754 categorías de DJ como primera etiqueta):
  partir por DJ, nunca solo por set, para que nada aprenda la identidad del DJ ni el set que ya vio.
- **Línea de base adicional (auditoría 11-07).** En D3, D4 y D5, la popularidad (en cuántos sets aparece
  el candidato): Vall et al. 2019 muestran que domina la predicción del siguiente tema.

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
2. **Etapa 4: los temas en común** (355 con la normalización de la sección 2; 316 con otra más estricta,
   auditoría 11-07: el número depende de cómo se normalicen artista y título).
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

**Lo que esta combinación no valida (auditoría 11-07).** Etiquetar la dirección de cada transición real
por el cambio de la coordenada (T3) hace que E1, en los perfiles de dirección, mida la coherencia con la
coordenada, no si la coordenada es «energía». Eso lo deciden T4 y la escucha (E2). T4 con unos 20 pares
puede tumbar una coordenada (16 o más aciertos de 20 la separan del azar con p < 0,01), no elegir entre
dos razonables: entre las que sobreviven, decide Gabriel escuchando.

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
| T3 | Coordenada de energía sin Mixed In Key. Candidatos, en este orden (auditoría 11-07): direcciones de texto en CLAP (los embeddings ya existen); una coordenada sobre los 400 estilos de MAEST (`maesthf_full_styles`, ya extraído); la fase aprendida (T2); cabezas de Essentia sobre EffNet-Discogs (engagement, danceability, mood_*; arousal/valencia solo existe sobre MusiCNN o VGGish) | Essentia; arXiv 2510.20759; T2 | Tema; Comp `energia_objetivo`, `texto_clap` | Codificador de texto de CLAP (Windows). Para Essentia: bajar las cabezas (CC BY-NC-SA 4.0) y una pasada de audio en WSL (`extract_representations.py --models effnet`, venv `essentia-tf-venv`) con prueba chica (V6): no hay EffNet guardado de `musica` | 3 | Pendiente |
| T4 | Validar las coordenadas con juicios por pares de Gabriel (unos 20 como mucho) | MusAV (ISMIR 2022); DECISIONS 2026-11-06, punto 10 | Eval | T3 | 3 | Pendiente |
| T5 | Trayectoria en vivo: nivel y pendiente de los últimos temas tocados | Kell 2013 (el orden importa) | Estado de la trayectoria | D2; T3 | 3 | Pendiente |
| T6 | Paso parejo: el nivel actual más un paso típico, medido en sets reales, no el tema más extremo | Kell 2013 (los pasos parejos se acercan más al orden real) | Comp `energia_objetivo`, `suavidad` | T3, T5; sets para el paso típico | 3 | Pendiente |
| T7 | Introspectiva: saltos chicos (`suavidad`), temas típicos de la parte tardía de sets largos (`fase`) y quizá una dirección de texto («hipnótico», «introspectivo»). Se calibra con los 3 o 4 sets largos de Gabriel y con D5 | Definición de Gabriel, 2026-11-07 | Perfil Introspectiva | T2, T6; D5 | 3 | Pendiente |
| T8 | Más y menos energía: un paso en T3 dentro del vecindario CLAP, con T1 según la dirección | Propuesta 2026-11-07 | Perfiles Más y Menos | T1, T3, T6 | 3 | Pendiente |
| T9 | Modelo que aprende durante la sesión qué transiciones le gustan a Gabriel | DJ-MC, Liebman et al. 2015 | Perfil (ajuste en vivo) | D2 con sesiones; M3 | 4 | Pendiente |
| T10 | La dirección se aprende de sets de DJ, no de playlists de oyentes, donde el orden casi no ayuda | Vall et al. 2019 frente a Kell 2013 y DJ-MC | Criterio para las fuentes | — | — | Vigente |
| T11 | Trayectorias en el espacio CLAP de sets profesionales: un punto por tema en D7 (los límites vienen dados) y uno cada 30 s en D8. Es la misma unidad que la biblioteca: tramos de 30 s en ventanas de 10 s (`extract_representations.py`). Da el paso típico (T6), la deriva y la dispersión por hora del set, incluida la parte después de la segunda hora (T7) (adenda 11-07) | Gabriel, 2026-11-07 | Sets (filas con embedding); Comp `suavidad`, `fase` | D7 o D8; CLAP en CPU con prueba chica | 2 | Pendiente |
| T12 | Separar en temas los sets crudos de D8 (cambios en la secuencia CLAP), comprobado contra las tracklists cuando existan (D5, MixesDB, descripciones). En D7 no hace falta (adenda 11-07) | Kim et al. 2020 (alineación set-tema); adenda 11-07 | Sets | D8 | 2 | Pendiente |
| T13 | Predictor del paso siguiente en el espacio CLAP, entrenado con lo que el DJ puso después (autosupervisado, sin reentrenar CLAP). En vivo da el punto hacia donde iría un profesional; los perfiles de dirección corren ese punto en la dirección elegida (adenda 11-07) | Gabriel, 2026-11-07; DJ-MC (transiciones) | Comp `siguiente` (nuevo, en 3.3) | T11; E1 con D7 | 3 | Pendiente |
| T14 | Fases del set detectadas sin etiquetas (por ejemplo, un modelo de estados ocultos sobre la secuencia CLAP): apertura, pico, introspectiva. Alimenta `fase` (adenda 11-07) | Adenda 11-07 | Sets; Comp `fase` | T11; sets largos de D7 y D8 | 3 | Pendiente |
| T15 | Direcciones de energía e introspección aprendidas de las trayectorias profesionales (hacia dónde se mueve CLAP en el pico y después de la segunda hora), como candidatas de T3 sin etiquetas (adenda 11-07) | Adenda 11-07 | Tema; Comp `energia_objetivo`, `texto_clap` | T11, T14 | 3 | Pendiente |

### Representación aprendida de sets

| Id | Idea | Origen | Contrato | Depende de | Etapa | Estado |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| M1 | Vector de cada tema aprendido de los sets (word2vec sobre tracklists), solo texto. Condición (auditoría 11-07): en D3, 50 696 de los 63 038 temas aparecen una sola vez; 12 342 aparecen 2 o más veces, 4186 3 o más y 749 5 o más. Con `min_count` 3 el vector existe para unos 4200 temas: M1 se hace si D5 sube esos conteos o se acepta esa cobertura, y M2 se evalúa con validación cruzada sobre las anclas contra «el ancla más cercana por CLAP» | Deej-AI; recomendación por playlists | Tema; Comp `set2vec` | Sets (D3, D5, D1) | 4 | Pendiente |
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
| E7 | Partición por DJ entre entrenamiento y medida, con la popularidad como línea de base en D3, D4 y D5 (auditoría 11-07; sección 3.6) | 2 |
| E8 | Medidas: acierto en los 30 primeros, rango recíproco medio, percentil del rango y recuperación de los filtros, por fuente (auditoría 11-07; sección 3.6) | 1 |

### Plataforma en vivo (Traktor)

| Id | Idea | Etapa |
| :--- | :--- | :--- |
| I1 | Un `.nml` por perfil en la carpeta de sugerencias; todos se calculan con los mismos candidatos y se escriben en la misma pulsación | 1 (estructura), 3 (perfiles de dirección) |
| I2 | `Toggle`: todas las playlists de sugerencias cuentan como «dentro». Pulsar desde cualquiera vuelve a la carpeta guardada antes de la primera pulsación; pulsar desde fuera refresca y abre Parecidos | 1 |
| I3 | Cada lista anota en el registro sus componentes, su perfil y su objetivo | 1 (ya existe para Parecidos) |
| I4 | Costo en vivo con 5 perfiles por debajo de unos 300 ms; el cálculo sigue siendo antes de apretar y se rehace al apretar | 1 |
| I5 | En cada `tocado`, el registro anota el vector de componentes de ese tema respecto del tema principal anterior y su rango entre los candidatos, estuviera o no entre los 30 (auditoría 11-07). Sin eso M3 y T9 no tienen la etiqueta de D2: hoy solo se anotan los componentes de los 30 primeros | 1 |

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
| V1 | Traktor acepta en Favoritos un `.nml` de la carpeta de sugerencias y lo relee al elegirlo con el mouse | [REEMPLAZADO 2026-11-07 por la prueba de Gabriel: hecho] Dos playlists de prueba; Gabriel las pone en Favoritos y cambia entre ellas después de reescribirlas. Resultado: Traktor las acepta en Favoritos y las relee al elegirlas viniendo de otra playlist o carpeta. Volver a tocar el favorito de la playlist que ya está abierta no la relee (como en LESSONS_LEARNED, prueba del 2026-11-04). Cambiar entre las playlists de sugerencias con el mouse muestra siempre la versión nueva. Con una playlist de dirección abierta, ver una lista nueva pide pasar por otra | Una playlist por carpeta, o Favoritos de carpetas |
| V2 | El cambio de `Toggle` (I2) | Test unitario con caminos de varias playlists | — |
| V3 | El costo en vivo con 5 perfiles (I4) | Medir sobre la colección real | Precalcular más en el índice |
| V4 | Acceso a sets largos de 1001Tracklists (D5) | Ver sección 6 | Sets largos de MixesDB y de Gabriel |
| V5 | La lectura de la historia de Traktor | [REEMPLAZADO 2026-11-07 por la auditoría: hecho. La carpeta 4.2.0 es copia byte a byte de 12 archivos de 4.4.1 (md5), así que D1 son los 36 archivos de 4.4.1, 21 de ellos sets; con `STARTDATE` + `STARTTIME` no hay cruce de medianoche mal leído. Queda cortar las sesiones dejadas abiertas con la regla de la sección 3.5] Sacar los repetidos entre 4.2.0 y 4.4.1 y tratar el horario (cruce de medianoche y sesiones dejadas abiertas); comparar a mano con 2 sets | — |
| V6 | Costo de la pasada de EffNet en WSL para T3 (auditoría 11-07) | `extract_representations.py --models effnet --folder "<subcarpeta>"` en `essentia-tf-venv`; tiempo por tema × 1786 | Empezar T3 por las direcciones de texto de CLAP y los estilos de MAEST, que ya están |

### Etapa 1: bases

Aviso (auditoría 11-07): en esta etapa el único banco es D1, los sets de Gabriel. Por la regla E6, lo que
E1 mida aquí (P2, G3, G4) orienta pero no fija pesos: los pesos esperan a la etapa 2, con D3 resuelto.
I5 entra con I3, para que D2 guarde la etiqueta de M3 desde la primera sesión.

1. Almacén de sets (3.2) con lector de D1 y de D2.
   - Sale: tabla en `artifacts/v4/dj_sets/`, con los conteos de la sección 2 reproducidos.
2. E1 (3.6) sobre D1, con líneas de base al azar, solo BPM y solo CLAP.
   - Sale: tabla base.
3. P2, G3 y G4 como componentes, medidos en E1 uno por vez.
4. Índice versión 4 con campos opcionales (3.1) y registro de componentes con nombre (3.3).
5. I1 a I5 con un perfil: Parecidos.
   - Sale: la estructura de varias playlists funcionando en Traktor (con V1), sin perfiles de dirección
     todavía.

### Etapa 2: sets de otros DJs, solo texto

1. C1: resolver los temas de D3 con Beatport (género, subgénero, BPM, tonalidad).
   - Volumen medido (auditoría 11-07): 40 790 temas distintos en los sets de House, Techno y Tech House
     (63 038 en total). A dos pedidos por tema (búsqueda y página) y 1,2 s entre pedidos
     (`BeatportClient.min_interval`), unas 27 h para esos tres géneros y 42 h para todo. Orden: primero
     los temas que aparecen 3 o más veces (4186), después los de 2 (12 342), y solo sets de 2010 en
     adelante (4741 de 5040): los anteriores casi no están en Beatport. La muestra chica mide, además
     del tiempo, cuántos títulos de D3 se resuelven.
   - Muestra chica y después completo, a una consulta por segundo con caché y `--low-priority`. Primero
     los sets de House, Techno y Tech House.
   - Con D5 también, si se consiguió.
2. T1, T2, G6, G7 y G2 medidos en E1 sobre D3 (lo que se puede medir sin audio, 3.6) y sobre D1.
3. Cuando un resultado venga de D3 y se aplique a la biblioteca, comprobarlo en los 355 temas en común.
4. Piloto de audio con D7 y D8 (adenda 11-07). Unas 5 compilaciones y unos 5 sets crudos, al menos 2 de
   3 h o más, en CPU con `--low-priority`. Mide:
   - el tiempo de CLAP por hora de audio;
   - el disco temporal;
   - si los tramos de un tema «(Mixed)» o de un set crudo caen cerca del mismo tema de la biblioteca
     (temas en común);
   - qué tan bien T12 separa los temas.

   Con eso se decide la escala y si hace falta GPU.
   - HECHO 2026-11-07 con John Digweed: `docs/reports/piloto_trayectorias_digweed_2026-11-07.md`.
     Fueron 3 compilaciones de Bedrock (17,7 h de sets, 179 temas con original) y 2 sets crudos (7,2 h).
     - CLAP de esta PC es el de la biblioteca.
     - Por tema, el original se identifica en el 79-86 %; bloque a bloque y contra toda la biblioteca,
       en el 26-34 %.
     - La separación simple (T12) no supera a cortes fijos.
     - CLAP tarda 1,3-1,9 min por hora de audio: no hace falta GPU para el piloto.
5. Con D7 a escala: T11, y E1 con los componentes de audio sobre sets de otros DJs (adenda 11-07, ver
   sección 10).
   - Según el piloto, D7 da trayectorias por tema con los originales (limpias, sin separar el set), y
     D8 queda en el nivel de bloque hasta que T12 funcione (alineación contra los originales).
   - Entre temas consecutivos, CLAP casi no muestra el orden dentro de la noche (0,16 frente a 0,18 al
     azar del mismo disco). Para las playlists de dirección pesan T3 y T13.

### Etapa 3: dirección e introspectiva

1. T3: coordenadas candidatas.
   - Orden (auditoría 11-07): primero las direcciones de texto de CLAP y una coordenada sobre los estilos
     de MAEST, que no necesitan audio nuevo; las cabezas de Essentia después de V6.
   - Essentia corre en WSL (no tiene versión nativa para Windows); necesita embeddings EffNet-Discogs de
     la biblioteca, con una corrida chica antes.
   - Las direcciones de texto de CLAP corren en Windows.
2. T4: hasta unos 20 juicios de Gabriel por pares.
3. T5, T6, T7 y T8 como perfiles.
   - Introspectiva (auditoría 11-07): hasta tener D5, `fase` para «después de la segunda hora» solo
     tiene los 4 sets largos de Gabriel y, en D3, 13 sets con minutos que pasan de 120 (312 pasan de
     90). El perfil arranca con `suavidad` y `clap` respecto de los últimos temas tocados; `fase` y
     `texto_clap` toman peso cuando haya sets largos para medirlos.
   - Pesos con E1 sobre D1 y D5 (sets largos), más D3 para lo de texto.
   - Escucha a ciegas de Gabriel.
4. Los perfiles de dirección en Traktor (I1).

### Etapa 4: aprendido

- M1 (D3, D5, D1), M2 (anclas en común) y M3 (ranker por perfil sobre el vector de componentes).
- G9, y T9 cuando D2 tenga sesiones.
- Sale: E1 con el ranker frente a los pesos a mano, y la escucha.

### Etapa 5: audio

- D6: la fuente del audio se decide con Gabriel (DECISIONS 2026-09-12, punto 3).
- Estimación de disco y GPU medida en una muestra. Orden de magnitud [REEMPLAZADO 2026-11-07 por los
  conteos de la auditoría: el 13 728 no tenía origen]: 63 038 temas de D3 (40 790 en House, Techno y Tech
  House) en avances de 2 minutos a 320 kb/s son unos 300 GB (unos 200 GB para los tres géneros); los
  4186 que aparecen 3 o más veces, unos 20 GB.
- CLAP y coordenadas para los temas de los sets: E1 completo sobre sets de otros DJs.
- M4 con GPU.
- Se rehace E1 de todo.

## 6. Datos: acceso, licencias y volumen

| Id | Fuente | Acceso | Licencia y condiciones | Volumen |
| :--- | :--- | :--- | :--- | :--- |
| D1 | Historia de Traktor de Gabriel | Local | De Gabriel | 21 sets (36 archivos en 4.4.1; la carpeta 4.2.0 repite 12 de ellos byte a byte), 592 temas al aire, 571 transiciones; 562 temas y 535 transiciones en el índice (auditoría 11-07) |
| D2 | Registro en vivo | Local | De Gabriel | Crece con cada sesión |
| D3 | djmix-dataset | Bajado (JSON, 11,6 MB) | Sin licencia publicada: uso de investigación, no se redistribuye | Sección 2 |
| D4 | Raveform (Hugging Face `taejunkim/raveform`) | Sin bajar | Sin licencia publicada en la página | 4902 sets; 38 % con horarios; 1423 temas con género de Beatport puesto a mano |
| D5 | 1001Tracklists, sets largos | Ver debajo | Ver debajo | A definir |
| D6 | Audio de los temas de los sets | Decisión de Gabriel | Depende de la fuente | Etapa 5 |
| D7 | Compilaciones mezcladas: álbumes «(DJ Mix)» con un tema por pista, titulado «(Mixed)» (adenda 11-07, idea de Gabriel) | Lista y orden: MusicBrainz (tipo `DJ-mix`, sin `Broadcast`; API con 1,2 s entre pedidos). Audio: YouTube, con el OK de Gabriel (DECISIONS 2026-11-07, punto 12) | De Gabriel para este uso; el audio queda fuera de Git y se borra después de sacar los embeddings | 26 438 grupos `DJ-mix` que no son radio; 4817 con etiqueta techno, tech house, progressive o deep house; 49 de John Digweed (Global Underground, Renaissance, «Live in Tokyo», «Last Night at Output») |
| D8 | Sets crudos de YouTube de DJs que elige Gabriel (adenda 11-07) | YouTube (`yt-dlp`), con el OK de Gabriel | Igual que D7 | Unos 2 KB por cada 30 s de embedding; un set de 4 h, alrededor de 1 MB |
| C1 | Metadatos de Beatport para los temas de los sets (auditoría 11-07) | Web, `src/v4/common/beatport.py` (1,2 s entre pedidos, caché) | `robots.txt` (2026-11-07): `Allow: /` para todos, sin `Crawl-delay`; bloquea unos 35 robots de entrenamiento y declara `Content-Signal: search=yes, ai-train=no`. Es una señal de política, no una licencia: antes de que M3 o M4 entrenen con etiquetas de Beatport, lo decide Gabriel | 40 790 temas en House, Techno y Tech House; 63 038 en total |

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
10. ¿La adenda de la sección 10 entra por los contratos de la sección 3, sin abrir un camino propio?

## 10. Adenda posterior a la auditoría (2026-11-07)

Gabriel autorizó bajar de YouTube para esta aplicación y propuso las compilaciones mezcladas como
alternativa a 1001Tracklists (DECISIONS 2026-11-07, puntos 12 y 13). Se suman dos fuentes:

- **D7, compilaciones mezcladas.** Discos de un DJ subidos tema por tema: el álbum se titula
  «(DJ Mix)» y cada tema «(Mixed)». Es la convención que piden los distribuidores (RouteNote, Label
  Worx). Se oye el set mezclado, con transiciones cortas, y cada tema conserva su identidad.
- **D8, sets crudos de YouTube** de DJs que elige Gabriel.

**Qué cambia en la arquitectura** (los contratos son los mismos):
- **Almacén de sets.** D7 y D8 son fuentes nuevas. D7 trae el límite y la identidad de cada tema
  (MusicBrainz). D8 trae el minuto exacto, y sus temas salen de T12.
- **Un componente nuevo**, `siguiente` (T13): la cercanía del candidato al punto que predice el modelo.
  Se agrega a la tabla de 3.3.
- **Lo que se puede evaluar (3.6).** Con D7, los componentes de audio (`clap`, `suavidad`, `texto_clap`,
  `energia_objetivo` de audio, `siguiente`) se miden sobre sets de otros DJs desde la etapa 2. Ya no hace
  falta esperar a la etapa 5 ni a los avances de Beatport.
- **El candidato que se busca en E1**, en D7 y D8, es el tema que el DJ puso después, con su CLAP. Los
  candidatos son los temas de esas fuentes con audio, menos lo ya tocado. En los temas en común se
  comprueba que todo cae en el mismo espacio que la biblioteca.
- **Puentes (3.7).** El puente 3 (el audio de los sets) pasa de la etapa 5 a la etapa 2 para D7 y D8. M4
  y los avances de Beatport (D6) quedan para lo que D7 y D8 no cubran.

**Riesgos** (HIPÓTESIS hasta el piloto, etapa 2, punto 4):
- **Audio de un set frente a tema limpio.** Hay encimados, ecualización, efectos y público. Se mide con
  los temas en común antes de usar D8 para nada fino.
- **Compilaciones frente a sets en vivo.** Las compilaciones son mezclas de estudio de 70 a 80 minutos
  por disco, y pueden no mostrar la parte tardía de un set largo. Para eso cuentan las de varios discos
  («Last Night at Output», los «Live in…») y D8.
- **Muchos «DJ-mix» de MusicBrainz son emisiones de radio en un solo bloque** (por ejemplo, los
  Essential Mix). Se filtran por tipo y por cantidad de pistas.

## 8. Afirmaciones

| Afirmación | Etiqueta | Evidencia |
| :--- | :--- | :--- |
| D3 tiene 5040 sets, 63 038 temas con ID, 64 764 transiciones y, salvo uno con una marca de 404 min (error probable del dato), ninguno con minutos pasa de 150 | EVIDENCIA OBSERVADA | `djmix-dataset.json`, conteo de esta sesión; reproducido en la auditoría 11-07 (13 sets pasan de 120 min, 312 de 90) |
| 355 temas de la biblioteca aparecen en D3, y solo 25 transiciones tienen los dos | EVIDENCIA OBSERVADA | Cruce por artista y título normalizados (puede subestimar: remixes con otro nombre) |
| Sin audio de los sets, los componentes de audio no se pueden medir en sets de otros DJs | INFERENCIA | De la fila anterior |
| Gabriel tiene 4 sets de 3 h o más (3,4, 3,4, 3,8 y 8,5 h) y 21 sets en total | EVIDENCIA OBSERVADA | Historia 4.4.1 con `STARTDATE` + `STARTTIME` (auditoría 11-07); la carpeta 4.2.0 es copia byte a byte |
| El `Toggle` actual no trata varias playlists como «dentro» | EVIDENCIA OBSERVADA | `sugerencias_vivo.py`, clase `Toggle`, método `at_target` |
| Traktor relee en Favoritos un `.nml` del Explorer al llegar desde otra playlist, no al volver a tocar el que está abierto | EVIDENCIA OBSERVADA | Prueba de Gabriel con «Prueba Favoritos A y B», 2026-11-07 (V1) |
| 1001Tracklists bloquea los scripts de Python y no tiene API pública | EVIDENCIA OBSERVADA | Su `robots.txt` y su sitio, 2026-11-07 |
| El acceso de investigación a 1001Tracklists se consigue pidiéndolo | HIPÓTESIS | Kim et al. lo obtuvieron así; se decide con el correo de Gabriel |
| CLAP varía unos 0.05 entre los 30 primeros | EVIDENCIA OBSERVADA | Colección real, 40 temas principales |
| El género (taxonomía y estilos de Discogs) aporta información que CLAP no tiene | EVIDENCIA OBSERVADA (correlaciones 0.22 y 0.35); que eso mejore las listas es NO CONCLUYENTE | Falta E1 y la escucha |
| 562 de 592 temas al aire de D1 y 535 de 571 transiciones están en el índice | EVIDENCIA OBSERVADA | Cruce de la historia 4.4.1 con `indice_recomendacion.json` por archivo o artista-título (auditoría 11-07) |
| El cruce D3-biblioteca depende de la normalización: 355/851/696/25 con la de la sección 2, 316/777/636/17 con otra más estricta | EVIDENCIA OBSERVADA | Dos cruces independientes (auditoría 11-07); las conclusiones de 3.6 y 3.7 valen con los dos |
| En D3, 50 696 de 63 038 temas aparecen una sola vez; 4186 aparecen 3 o más veces | EVIDENCIA OBSERVADA | Conteo de `djmix-dataset.json` (auditoría 11-07); condiciona M1 y M2 |
| Las cabezas de Essentia para T3 no están en disco, y arousal/valencia no existe sobre EffNet | EVIDENCIA OBSERVADA | `models/essentia/`, home de WSL y essentia.upf.edu/models (auditoría 11-07) |
| El lado en vivo usa solo la biblioteca estándar | EVIDENCIA OBSERVADA | Importaciones de `recomendador.py`, `sugerencias_vivo.py`, `tema_principal.py` y de los módulos de `src/v4/common` que usan (auditoría 11-07) |
| El índice se carga en 0,02 s y se decodifica en 0,01 s: los campos de la versión 4 no comprometen I4 | EVIDENCIA OBSERVADA | Medido sobre `indice_recomendacion.json`, 1760 temas, 5,8 MB (auditoría 11-07) |
| MusicBrainz tiene 26 438 grupos `DJ-mix` que no son radio, 4817 de ellos con etiqueta techno, tech house, progressive o deep house, y 49 de John Digweed | EVIDENCIA OBSERVADA | Consultas a la API de búsqueda de MusicBrainz, 2026-11-07 (adenda) |
| En las tiendas, el álbum de una mezcla lleva «(DJ Mix)» y cada tema «(Mixed)» | EVIDENCIA OBSERVADA para los distribuidores (RouteNote, Label Worx) y ejemplos de Apple Music y Deezer; no verificada como regla de Spotify | Fuentes de la adenda |
| Los tramos de un tema «(Mixed)» o de un set crudo caen cerca del mismo tema de la biblioteca | [REEMPLAZADO 2026-11-07 por el piloto de Digweed] HIPÓTESIS. Lo medido: EVIDENCIA OBSERVADA. Por tema, el original sale primero en el 79-86 % entre los de su disco. Bloque a bloque y contra toda la biblioteca, en el 26-34 % | `docs/reports/piloto_trayectorias_digweed_2026-11-07.md` |

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
- Adenda: MusicBrainz, tipo de grupo de lanzamiento «DJ-mix» y su API de búsqueda; RouteNote, «How do I
  format a DJ mix release»; Label Worx, «How do I create a Mixed Compilation Album»; Wikipedia, «Category:DJ
  mix album series».
