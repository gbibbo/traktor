# Plan de las playlists en vivo: informe para auditoría externa

Fecha: 2026-11-07. Autor: agente de Claude Code (sesión traktor-github-3a), con las decisiones de Gabriel
del 2026-11-06 y el 2026-11-07 (`docs/DECISIONS.md`).

Auditado el 2026-11-07 (skill `auditar`, otra sesión): `docs/reports/auditoria_plan_playlists_2026-11-07.md`.
Las correcciones están en su lugar, con `[REFUTADO …]` o `[REEMPLAZADO …]`; los refuerzos llevan la marca
«(auditoría 11-07)». La estructura del plan no cambió.

Adenda 2026-11-08 (auditoría, con Gabriel; DECISIONS 2026-11-08): sección 11, con las medidas E9 a E12 y las
ideas P8 a P10, T18 a T22 y M5. Las marcas «(adenda 11-08)» son suyas. Para implementarla desde un chat nuevo:
`docs/plans/PROMPT_SEGURIDAD_Y_DIRECCION.md`.

Adenda 2026-11-09 (auditoría, aprobada por Gabriel; DECISIONS 2026-11-09, «algoritmo de recomendación»):
sección 12. Parecidos como zona con varias vistas, conceptos en dos casos de uso (sin tema de referencia y con
el set en curso), un banco local al estilo del paper «Guided Music Retrieval» y el cómputo pesado en gpuq y
Drive. Ideas P11 a P13 (con P12a y P12b), T23 a T30, M6, D9 y H1; medidas E13 y E14. Las marcas «(auditoría 11-09)» y «(adenda
11-09)» son suyas. Fundamento: `docs/reports/auditoria_recomendacion_2026-11-09.md`. Para implementarla desde
un chat nuevo: `docs/plans/PROMPT_FUSION_Y_CONCEPTOS.md`.

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

**Escala de los componentes (auditoría E1, 11-07).** Entre los candidatos de una consulta, los
componentes se dispersan distinto (desvío mediano en D1: `clap` 0,08, `bpm` 0,15, `clave` 0,12,
`genero` 0,24): un peso igual no es igual importancia, y `clap+bpm` con pesos 1 y 1 pesa más el BPM.
Antes de fijar pesos (etapa 2), o se normaliza cada componente por consulta (percentil o rango entre
los candidatos) o se declara que los pesos van sobre la escala cruda. Los perfiles de un solo
componente con pocos valores distintos (`bpm`, `clave`, `genero`) necesitan desempate al azar en E1;
hoy el banco desempata por el orden del índice.

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
- **Medidas de seguridad (adenda 11-08).** El acierto en 30 solo descarta: con unos 1700 candidatos y decenas
  de temas igual de buenos, no distingue una lista buena de una regular. La calidad de Parecidos se mide con
  E9 a E12 (sección 11.2): radio de zona, fracción de los 30 dentro del radio y escucha a ciegas de listas
  enteras de a dos.

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
| P2 | CLAP en escala comparable (percentil o rango entre los candidatos) | Medido 2026-11-06 | Comp `clap` | — | 1 | Pendiente (adenda 11-09: lo cubre la suma de z de P11, que pone cada vista en desvíos de su propia distribución) |
| P3 | Filtros BPM ±20 y tonalidad ±2 semitonos. En Kell 2013 la tonalidad no mostró efecto en el orden de los DJs, pero es la regla de Gabriel | DECISIONS 2026-11-06, punto 7 | Filtro común | — | — | Vigente |
| P4 | La misma tonalidad original puntúa más que una que pide mover el pitch | DECISIONS 2026-11-06, punto 9 | Comp `clave` | — | — | Vigente (peso 0) |
| P5 | Vector TIV de compatibilidad armónica | Rama `harmonic-tiv` (otra sesión) | Comp `tiv`; Tema | Esa rama | Aparte | En curso |
| P6 | Pesos por componente y registro de componentes en cada lista | DECISIONS 2026-11-06, punto 9 | Perfil; Eval | — | — | Vigente |
| P7 | Valores del componente de tonalidad aprendidos de sets: cuántas veces más que al azar aparece cada relación (misma, relativa, vecina, paralela; sin transponer o a 1-2 semitonos) en lugar de los valores de la regla armónica, que en D1 ordenan al revés la vecina sin transponer (1,5 veces el azar, vale 0,70) y la misma transpuesta (0,76 veces, vale 0,95). Valor explícito para los temas sin tonalidad. Evaluado fuera de muestra (otros DJs o dejando sets fuera), con D2 para la tonalidad que sonó (adenda 11-07) | Gabriel (DECISIONS 2026-11-07, punto 17); auditoría de E1 | Comp `clave` | D2 con sesiones; tonalidades de los temas de los sets | 2-3 | En curso: medido en D1 dejando un set fuera (`lineas_d1_2026-11-07.md`): la regla actual empata con CLAP si los temas sin tonalidad valen 0; los valores aprendidos con peso 0,25, 11,7 % frente a 8,9 % (p = 0,09, NO CONCLUYENTE); falta D2 |

### Género

| Id | Idea | Origen | Contrato | Depende de | Etapa | Estado |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| G1 | Taxonomía propia de Beatport → similitud escalar entre géneros | DECISIONS 2026-11-06, punto 11 | Comp `genero_tax` | — | — | HECHO 2026-11-06 |
| G2 | Aplicar las 17 objeciones menores de la red | Informe de la taxonomía | Comp `genero_tax` | E1 (para ver si cambian algo) | 2 | Pendiente |
| G3 | Género blando: probabilidades (Beatport con peso alto, las 3 primeras del clasificador para el resto) y similitud esperada; cubre los 158 sin género | Propuesta 2026-11-07 | Tema; Comp `genero_blando` | G1 | 1 | Pendiente |
| G4 | Las versiones con géneros distintos en Beatport (267 temas) dan una distribución | `genre_evidence/versiones.csv` | Tema (con G3) | G3 | 1 | Pendiente |
| G5 | El género de Beatport es una decisión editorial por lanzamiento: una señal entre otras | Gabriel; centro de ayuda de Beatport; DECISIONS 2026-11-07, punto 7 | Criterio | — | — | Vigente |
| G6 | Estilos de Discogs de MAEST como vector de género (coseno) | Alonso-Jiménez et al., ISMIR 2023 (UPF); medido 2026-11-07 | Tema; Comp `estilos_discogs` | — | 2 | Pendiente (auditoría 11-09, exploratorio: sola no se distingue de CLAP en D1, 8,3 % frente a 9,0 %; sumada a CLAP, 12,7 %, y separa mejor los pares de un mismo set de D3, AUC 0,75 frente a 0,66; lleva tempo, −0,37 con la diferencia de BPM. `auditoria_recomendacion_2026-11-09.md`, idea P11) |
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
| T3 | Coordenada de energía sin Mixed In Key. Candidatos, en este orden (auditoría 11-07): direcciones de texto en CLAP (los embeddings ya existen); una coordenada sobre los 400 estilos de MAEST (`maesthf_full_styles`, ya extraído); la fase aprendida (T2); cabezas de Essentia sobre EffNet-Discogs (engagement, danceability, mood_*; arousal/valencia solo existe sobre MusiCNN o VGGish) | Essentia; arXiv 2510.20759; T2 | Tema; Comp `energia_objetivo`, `texto_clap` | Codificador de texto de CLAP (Windows). Para Essentia: bajar las cabezas (CC BY-NC-SA 4.0) y una pasada de audio en WSL (`extract_representations.py --models effnet`, venv `essentia-tf-venv`) con prueba chica (V6): no hay EffNet guardado de `musica` | 3 | Pendiente (adenda 11-08: lo concretan T18, direcciones de texto, y T19, cabezas de Essentia; el juez es E11) |
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
| T16 | Comienzos y finales: el CLAP del final del tema actual frente al del comienzo del candidato, porque la biblioteca solo tiene el centro (3 tramos del 35-65 %) y en D7 lo que predecía el siguiente estaba cerca de los cambios. Componente `extremos` (`tools/dj_sets/clap_extremos.py`), con el sentido inverso como control (adenda 11-07, revalidación de D7) | Revalidación de D7, 2026-11-07 | Comp `extremos` | CLAP de los extremos de la biblioteca | 2 | En curso (`lineas_d1_2026-11-07.md`; auditoría líneas D1: lo que predice es el comienzo del candidato; el final del actual no agrega sobre su comienzo o su centro) |
| T17 | Zona de la noche: la consulta CLAP suma el centro de lo ya tocado (todo el set o los últimos n), porque los temas de una compilación forman una zona compacta en CLAP (D7). Opciones `clap_historia` y `clap_historia_n` del recomendador | Revalidación de D7, 2026-11-07 | Perfil Mantener; Comp `clap` | Nada nuevo | 2 | Medido en D1: no mejora el tema siguiente (p ≥ 0,72); falta probarlo para «mantener» (adenda 11-08: se vuelve a medir como zona, con E10 y P10, no con el acierto del tema siguiente) (adenda 11-09: es la zona del caso con el set en curso, T30, con las vistas de P11) |

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
| E8 | Medidas: acierto en los 30 primeros, rango recíproco medio, percentil del rango y recuperación de los filtros, por fuente (auditoría 11-07; sección 3.6); desempate al azar donde el puntaje empata, y el acierto por set además del total (auditoría E1, 11-07) | 1 |

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
   - D1 HECHO 2026-11-07: `tools/dj_sets/sets_traktor.py` → `d1.csv`. Por archivo reproduce los 21 sets y
     592 temas de la auditoría; con la regla de 20 minutos son 22 sets, 602 temas y 543 transiciones con
     los dos temas en el índice. Falta el lector de D2.
2. E1 (3.6) sobre D1, con líneas de base al azar, solo BPM y solo CLAP.
   - Sale: tabla base.
   - HECHO 2026-11-07: `src/v4/evaluation/banco_sets.py`, informe `docs/reports/e1_sets_gabriel_2026-11-07.md`.
     - Acierto en 30: azar 2,0 %, BPM 4,3 %, CLAP 9,3 %, CLAP + BPM 10,2 %. Auditoría E1
       (`docs/reports/auditoria_e1_sets_gabriel_2026-11-07.md`): con desempate al azar, BPM 4,6 %,
       tonalidad 4,2 %, género 5,0 %, popularidad 4,8 %; CLAP frente a CLAP + BPM no se distingue
       (p = 0,61); 48 de los 50 aciertos de CLAP son entre artistas distintos.
     - El filtro de tonalidad deja fuera el 22 % de las transiciones reales de Gabriel; se le consulta
       (P3 es su regla).
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
   - HECHO 2026-11-07 (T11 y E1 con D7): `docs/reports/trayectorias_d7_2026-11-07.md`. Fueron 83
     compilaciones de Sasha, Carl Cox, Nick Warren y Hernán Cattáneo: 118 h, 1119 temas con vector en 94
     discos. Los vectores salen del audio mezclado: estas compilaciones no traen los originales.
     - [REEMPLAZADO 2026-11-07 por la revalidación (`validar_d7.py`, el mismo informe): 53 de 96
       archivos de discos enteros no eran la compilación, y sobre audio mezclado los límites falsos dan un
       «siguiente» más parecido que los reales (24,0 % contra 35,6-46,4 %). Con pistas en archivos
       separados (las «por_tema» validadas), CLAP no supera al azar: 17,9 % contra 15,3 %, 123
       transiciones] Dentro del disco, CLAP elige el siguiente el 26 % de las veces (17 % al azar). El
       siguiente queda a 0,114 y uno al azar del mismo disco a 0,131.
     - [REEMPLAZADO 2026-11-07 por la revalidación: la compilación sigue agrupando (sus discos a 0,049;
       14,5 % de los 30 primeros contra 3,0 % de los candidatos); DJ frente a época queda NO CONCLUYENTE
       con 32 discos] Lo que más agrupa es la compilación (sus discos a 0,034), no el DJ ni la época
       (0,098-0,118). E1 sobre D7 se mide entonces dentro del disco.
     - Desde la revalidación:
       - el «siguiente» (T13, E1 tema a tema) se mide con vectores de archivos separados: las «por_tema»
         validadas, los originales y las compilaciones con originales sueltos;
       - los discos enteros sirven para centroides, deriva a nivel de bloque y el pool «d7» con la
         compilación controlada;
       - toda medida de orden sobre audio mezclado va con el control de límites falsos;
       - `youtube_buscar.py` versión 4 exige el título además de la duración.
     - Extrapolar el último paso empeora: T13 tiene que superar a CLAP dentro del disco con discos fuera
       de la muestra. [REFUTADO 2026-11-07 noche, adenda de la auditoría D7: parte del audio continua es otra
       mezcla, y con límites falsos el «siguiente» gana más que con los reales; el 26 % dentro del disco es
       continuidad de la mezcla, no elección del DJ; se recalcula con lo validado] Auditoría D7
       (`docs/reports/auditoria_trayectorias_d7_2026-11-07.md`): empeora con
       todo D7 como pool; dentro del disco no se distingue de CLAP (p = 0,10). CLAP dentro del disco
       supera al azar con prueba apareada (z = 7,5; 52 de 74 discos; los 4 DJs); el borde de 60 s saca la
       señal de la mezcla (31 % con borde 0).
     - D7 no informa sobre la parte después de la segunda hora (discos de 75 min): T7 sale de D8 y de
       los sets largos de Gabriel.

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
| D5 | [REEMPLAZADO 2026-11-09 por pedido de Gabriel: sets de 3 h o más para trayectorias y sets de cualquier largo para pares; sin comerciales] 1001Tracklists, sets largos | Ver debajo | Ver debajo | [REEMPLAZADO 2026-11-09 por la selección de abajo: 1549 candidatos con las 158 búsquedas] A definir |
| D6 | Audio de los temas de los sets [REEMPLAZADO 2026-11-09 por DECISIONS 2026-11-09 (sets de 1001Tracklists), punto 4: previews de Beatport en `erclm:traktor_ml/beatport_previews/`, piloto de 100 hecho] | Decisión de Gabriel | Depende de la fuente | Etapa 5 |
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
1. [REEMPLAZADO 2026-11-08 por decisión de Gabriel (DECISIONS 2026-11-08, punto 1): el correo ya se mandó y no se depende de la respuesta] Que Gabriel pida acceso de investigación a info@1001tracklists.com: sets de 3 h o más, con minutos.
   El agente le prepara el correo, pero lo manda Gabriel.
2. [REEMPLAZADO 2026-11-08 por decisión de Gabriel (DECISIONS 2026-11-08, punto 1): se exporta por navegador con su cuenta, iniciada a mano; probado con un tracklist; también se bajan muchos con su cuenta] No esquivar el `robots.txt`: no se cambia el agente para pasar por navegador.
3. Mientras tanto:
   - sets largos de MixesDB, cuyo `robots.txt` permite leer páginas a una cada 4 segundos y bloquea las
     consultas a la API (HIPÓTESIS: hay suficientes sets largos con lista completa; se decide con una
     muestra);
   - Raveform;
   - los sets largos de Gabriel.

**Selección de sets para D5 (2026-11-09; DECISIONS 2026-11-09, sets de 1001Tracklists).** Herramienta:
`tools/dj_sets/tracklists_1001.py`. Reglas y semillas: `config/dj_sets/tracklists_1001.json`. Datos:
`<root>/1001tracklists/` (en este PC, `D:\TRAKTOR ML datos\dj_sets`).

Dos usos, cada set con el suyo (Gabriel):

| Uso | Para qué | Condiciones (fila del listado) | Después de abrir la página |
| :--- | :--- | :--- | :--- |
| Trayectoria | Cómo cambia la elección de música a lo largo de la noche: T2, T7, T14, T20, E11 | Evento real (`@` en el título), no stream ni «+ More», 3 h o más, 60 % identificado o más | Con tiempos en el 80 % de los temas o más; sin tiempos, la posición sale del orden |
| Pares | Qué temas van juntos: temas consecutivos de sets de cualquier largo, radio incluida | 1 h o más, 70 % identificado o más, 8 pares consecutivos esperados o más | Cuentan los pares con los dos vecinos identificados; las filas tocadas encima de otra no cuentan |

En los dos: 7 a 26 temas por hora (una lista de 8 h con 17 temas está incompleta). Un set puede tener los dos usos.

Inventario de ideas de la selección:

| Id | Idea | Origen | Etapa |
| :--- | :--- | :--- | :--- |
| S1 | Dos usos separados: trayectoria y pares | Gabriel 11-09 | Hecho en `candidatos` |
| S2 | [REFUTADO 2026-11-09 por Gabriel: comercial es «David Guetta y Martin Garrix»; rige ante la duda, adentro: afuera el EDM de ese tipo por nombre, y lo etiquetado Mainstage o similar si el DJ no es semilla; un festival no cuenta] Sin sets comerciales; ante la duda, afuera: género, DJ (comerciales y dudosos), evento o escenario. `descartados.csv` dice el motivo y se puede rescatar | Gabriel 11-09; las listas, agente | Corregido |
| S3 | % de temas con género comercial dentro del set, como segunda señal | Agente | `medir` lo calcula; sin umbral hasta ver el piloto |
| S4 | Curar sin abrir el set: el listado ya da fecha, duración, identificados, géneros y grabación | Agente | Hecho |
| S5 | Filtros del buscador (duración, tipo, orden, ventanas de fecha) en vez de scroll infinito, que es donde el sitio pide captcha | Agente | Hecho |
| S6 | Semillas: Spotify de Gabriel, artistas con 5 temas o más en la biblioteca, DJs de noches largas vistos en la muestra, y los que nombró Gabriel (ARTBAT, Argy, Boris Brejcha, Solomun, Tale Of Us, Charlotte de Witte, Amelie Lens, Nina Kraviz). Una etiqueta del sitio no saca a una semilla | Agente; Gabriel aprobó la lista y sumó esos nombres | Hecho |
| S7 | Términos y eventos: All Night Long, Extended, Closing, Opening, Sunrise; Awakenings, Time Warp, Stereo y otros | Agente | Hecho 11-09 (las 158 búsquedas) |
| S8 | Bola de nieve por afinidad de artistas, no por temas en común (solo el 2,4 % de los temas está en la biblioteca, frente a 34 % de artistas) | Agente | Después del piloto |
| S9 | Topes: 15 sets por DJ y 30 por programa, por la partición por DJ (E7) | Agente | Hecho |
| S10 | Tabla con columna `veto`, que `bajar` respeta y `candidatos` conserva | Agente | Hecho; Gabriel veta |
| S11 | Leer la página en lugar del botón de exportar: trae lo mismo con un pedido por set, y además sello, enlaces y grabación | Agente | Hecho (832 temas y 651 identificados, igual que la medición previa) |
| S12 | Piloto de 30 sets, después de a 50 por día; desde la conexión de casa, sin VPN; parada al primer rechazo | Agente; Gabriel aprobó | [REEMPLAZADO 2026-11-10 por la noche del 9 al 10 (DECISIONS 2026-11-09, punto 5): Gabriel pidió bajar todos] Piloto hecho 11-09 (30 sets, uno por DJ, sin cortes); el resto, de a 50 por corrida. 838 de 1549 bajados; paró en un 403 tras unas 830 páginas seguidas; faltan 711 |
| S13 | Unir noches publicadas por partes | Agente | `grupo_noche` cubre «Part N»; falta el caso de episodios seguidos de un programa (Eelke Kleijn) |
| S14 | Enlaces por tema (YouTube, Beatport) al final, solo de los temas que se usen y una vez cada uno | Gabriel aprobó 11-08 | Piloto 11-09 (`beatport_previews.py`): el botón de compra lleva a Beatport en 96 de 100, a 8,4 s por tema; [REEMPLAZADO 2026-11-09 por Gabriel: primero la búsqueda, y el enlace del tracklist si el tema no sale] la búsqueda por nombre en Beatport da la misma versión con 2 errores en 92, así que queda de respaldo. Noche del 9 al 10: 3703 previews por la búsqueda; 899 sin id esperan el enlace (`enlaces`) |
| S15 | Puente con la biblioteca: con 2,4 % de temas en común, hacen falta metadatos de Beatport o audio de los temas de los sets (D6, C1, sección 3.7) | Agente | Pendiente. El piloto lo confirma: con la versión incluida, 11 de 1206 temas (0,9 %), ningún par consecutivo y 5 pares en el mismo set |

Cómo se retoma la bajada (D5 y D6), desde este PC, con `--root "D:\TRAKTOR ML datos\dj_sets"`:
- Sets: `tracklists_1001.py bajar --n 50`, repetido, con un Python que tenga Playwright (no está en
  `requirements_v4.txt`) y el perfil de `~/.traktor_ml/`. Después de un corte: esperar, probar con `--n 1`, y no
  insistir si vuelve a parar. La página donde paró queda en `1001tracklists/paros/`.
- Previews: por cada lote nuevo, `beatport_previews.py --corrida lote_NN elegir --todos --n 1000000 --excluir piloto
  --excluir lote_01 ...` (todas las corridas anteriores) y después `bajar --temporal "<carpeta en D:>"`. Un lote cortado
  se retoma con el mismo `bajar`: salta las tandas que ya están en el Drive. Kaggle no sirve: Beatport le pide la
  verificación de Cloudflare.
- Respaldo para los temas sin id: falta armar un lote con los `sin_id` de los manifiestos de cada lote, pasarle
  `enlaces` (un pedido a 1001Tracklists por tema) y bajarlo.

[REEMPLAZADO 2026-11-09 por las 158 búsquedas y el piloto: 2540 sets distintos, **1549 candidatos** (385 para trayectoria, 1164 solo para pares; 354 DJs; 1489 con grabación); piloto de 30 sets sin cortes (DECISIONS 2026-11-09, sets de 1001Tracklists)] [REEMPLAZADO 2026-11-09 por el criterio de comercial de Gabriel: 84 de 158 búsquedas, 1269 sets distintos, **831 candidatos** (142 para trayectoria, 689 solo para pares; 156 DJs; 798 con grabación) y faltan 74 búsquedas] Estado del 2026-11-09: 84 de 147 búsquedas, 1269 sets distintos, **740 candidatos** (135 para trayectoria, 605 solo para
pares; 139 DJs; 709 con grabación). Lo que falta incluye casi todos los términos y eventos y a varios DJs de noches largas
(Massano, Eli Brown, Soul Button, Voltaire, Innellea), así que 135 es un piso. La conexión se cortó en la búsqueda 81 de
esa tanda (`ERR_CONNECTION_RESET`, con el resto de internet funcionando): no se reintentó, y `buscar` quedó con un tope
de 40 por corrida.

Cómo seguir (Playwright en un entorno aparte, no en `requirements_v4.txt`; la sesión queda en `~/.traktor_ml/`):

```text
python tools/dj_sets/tracklists_1001.py --root "<datos>" buscar                  (hasta 40 por corrida; salta lo hecho)
python tools/dj_sets/tracklists_1001.py --root "<datos>" candidatos --biblioteca "<_organizacion.csv>"
python tools/dj_sets/tracklists_1001.py --root "<datos>" bajar --n 30
python tools/dj_sets/tracklists_1001.py --root "<datos>" medir --biblioteca "<_organizacion.csv>"
```

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
| En D1, CLAP solo acierta en 30 el 9,3 % y supera a azar, BPM, género y popularidad con prueba apareada; CLAP + BPM no se distingue de CLAP | EVIDENCIA OBSERVADA | `banco_sets.py` y la auditoría E1 (539 transiciones, 22 sets) |
| La regla de tonalidad (P3) deja fuera el 23,5 % de las transiciones reales de Gabriel y deja pasar el 69 % de los pares al azar; los ±1 semitono ocurren a la tasa del azar | EVIDENCIA OBSERVADA | Auditoría E1: 387 de 506 frente a 349,5 esperadas; decide Gabriel |
| [REFUTADO 2026-11-07 noche por la adenda de la auditoría D7: con límites falsos sobre el mismo audio el «siguiente» gana más (34,6-41,5 %) que con los reales (27,1 %); es continuidad de la mezcla] En D7, dentro del disco, CLAP elige el siguiente el 26 % (azar exacto 17,4 %), apareado y en los 4 DJs; una parte de la señal está en los 60 s junto a los límites (31 % con borde 0) y el borde de 60 s la saca | EVIDENCIA OBSERVADA | `trayectorias_d7.py` y la auditoría D7 (920 transiciones, 74 discos) |
| En D7 lo que agrupa en CLAP es la compilación (sus discos a 0,034), no el DJ (0,098-0,100) ni la época (suma 0,013-0,020); E1 sobre D7 se mide dentro del disco [REEMPLAZADO 2026-11-07 noche: números contaminados por audio ajeno, se recalculan con lo validado; dentro del disco con audio mezclado no se mide la elección del siguiente] | EVIDENCIA OBSERVADA | `trayectorias_d7.py`, `entre_discos`, 74 discos; con peso, 3 DJs |

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

- Vande Veire y De Bie, «From raw audio to a seamless mix: creating an automated DJ system for Drum and
  Bass», EURASIP Journal on Audio, Speech, and Music Processing 2018:13 (adenda 11-08; acceso abierto, CC BY
  4.0; código en bitbucket.org/ghentdatascience/dj).

## 11. Adenda 2026-11-08: seguridad en Parecidos y coordenadas de dirección

Escrita por la sesión auditora con Gabriel (DECISIONS 2026-11-08). Entra por los contratos de la sección 3:
no abre ningún camino propio hasta Traktor. Las ideas se suman al inventario de la sección 4 con sus ids;
nada de lo anterior se borra. Para implementarla desde un chat nuevo: `PROMPT_SEGURIDAD_Y_DIRECCION.md`.

### 11.0 Qué cambió y por qué

- **Parecidos (la playlist de D0) es, en palabras de Gabriel, «un listado de temas de los que tengo la
  seguridad de que van a sonar bien con lo que estoy tocando».** Es precisión de la lista: cada uno de los
  30 tiene que ser una apuesta segura. No es acertar el tema que él habría puesto después. «Sonar bien»
  puede ser CLAP tal cual o CLAP mejorado con lo que se aprenda de los sets de DJs profesionales.
- **Lectura de los resultados de la etapa 1 y de las líneas D1** (INFERENCIA, a partir de los informes
  E1, D7 y líneas D1 y sus auditorías): todo lo medido hasta hoy usa una sola medida, el acierto en 30 del
  tema que Gabriel tocó después, que con unos 1700 candidatos y decenas de temas igual de buenos da entre
  9 y 12 % para cualquier variante y no distingue una lista buena de una regular. Ninguna variante nueva
  pasó por la escucha de Gabriel. «No mejora el acierto» es, para la calidad de la lista, NO CONCLUYENTE.
  El acierto en 30 queda como línea de descarte (E2), que es lo que la sección 3.6 ya decía.
- **El material de DJs profesionales ya bajado (D7, D8) se reutiliza para Parecidos**, no solo para las
  playlists de dirección (Gabriel).
- **Las playlists de dirección necesitan una coordenada que se mueva como se mueven los DJs a lo largo de
  la noche.** Los modelos de Essentia son, para Gabriel, los candidatos naturales a juzgar la trayectoria;
  el juez final es la forma de los sets largos reales y sus pares de juicio (E11), porque una etiqueta que
  no se mueve con la noche no sirve, como la energía de Mixed In Key (83 % de la biblioteca en 5 o 6).
- **Texto en CLAP como dirección** (pregunta de Gabriel): parecidos al tema principal y a una palabra o
  concepto a la vez. El codificador de texto ya está en el repo (`tag_vocals.py`) y corre en Windows.
- **«Fine-tuning de Essentia»** (pregunta de Gabriel): lo realista es entrenar cabezas sobre los embeddings
  congelados, como ya hacen `genre_model.py` (regresión logística sobre MAEST capa 7 y CLAP) y `clap_probe`
  (lineal sobre CLAP, entrenado con 54 temas de Electrobyte). Las etiquetas abundantes y gratis son la
  posición o fase de cada tema en los sets profesionales. Afinar el tronco (MAEST en torch) es M4.

### 11.1 Lo que se toma de Vande Veire y De Bie 2018 (ideas, no copia)

Leído del PDF de acceso abierto el 2026-11-08 (EVIDENCIA OBSERVADA del texto; secciones entre paréntesis).

| Idea del paper | Lo que dice | Dónde entra acá |
| :--- | :--- | :--- |
| El set es una zona (4.2.2) | En mezclas profesionales (90 s cada 120 s) «el estilo se queda en un vecindario relativamente chico comparado con toda la colección» | P8: la seguridad de Parecidos es estar dentro de esa zona; coincide con D7 (el disco es compacto en CLAP) |
| Objetivo del tema siguiente = centro de la zona + paso chico (ec. 34) | Centro = el cuarto de la biblioteca más cercano al primer tema; objetivo = α·centro + (1−α)·(β·anterior + (1−β)·actual) con α = 0,4 y β = −0,1; 6 candidatos tras el filtro de tonalidad | P10 (Mantener); T17 vuelve a medirse como zona |
| Validación escuchando, no adivinando (5.6) | Dos mezclas de 15 min, con y sin descriptor de estilo: 17 de 18 oyentes prefirieron la del estilo (3,89 frente a 2,39; p 7e-5). La similitud rítmica del tramo superpuesto apenas pesó (10 de 15, p 0,15). La mezcla sin estilo sonaba «apurada» | E12: la medida de calidad de una lista es la escucha a ciegas de listas enteras de a dos |
| Descriptor simple sobre los tramos altos (4.2.2) | 48 estadísticas de contraste espectral solo en los segmentos «altos» (drops), PCA a 3 dimensiones sobre 160 temas | T22; y aviso de que una coordenada no necesita un modelo profundo para que la escucha la note |
| Tonalidad (4.2.1) | Misma, quinta arriba o abajo, relativa; además los temas a un semitono de una compatible, transpuestos si salen elegidos | P7 y DECISIONS 11-07 punto 17: es la práctica de Gabriel |
| Energía por tipos de transición (4.2.3) | Double drop, rolling, relajada, elegidos por una máquina de estados para variar el flujo | T21 (no aplica por ahora: la dirección la elige Gabriel con el mouse) |
| Su propio límite (6) | «Podría ser una simplificación excesiva de cómo los DJs profesionales ... crean una progresión deliberada de energía y ánimo»; proponen aprenderla de las mezclas profesionales que hay en internet | Es D7 y D8 |

### 11.2 Medidas nuevas (se suman a la tabla de evaluación de la sección 4)

| Id | Idea | Etapa |
| :--- | :--- | :--- |
| E9 | Compacidad de los sets en CLAP: distribución de la distancia entre temas de un mismo set profesional validado (y de un mismo set de D1) frente a pares al azar del mismo rango de BPM. Da el radio de zona. Con pocos vectores por tema validados en D7 (123 transiciones en la revalidación), se calcula además dentro de los discos enteros a nivel de bloque y crece con D8 cuando T12 funcione. MEDIDO 2026-11-08 (`docs/reports/zona_sets_2026-11-08.md`): en el espacio de la biblioteca, mediana 0,176 entre temas de un mismo set profesional (D3 × biblioteca, 79 pares de 48 DJs, y originales de Bedrock, un DJ) frente a 0,233 al azar del mismo BPM; percentil 10: 0,098-0,104; percentil 25: 0,134-0,147. La mezcla junta los mismos pares 0,015 | A |
| E10 | Precisión de zona de Parecidos: fracción de los 30 primeros dentro del radio de E9. Y pureza fuera de muestra: con un tema principal de un set profesional y el pool de los temas de esa fuente más la biblioteca menos lo tocado, fracción de los 30 que son del mismo set (o del mismo DJ en la misma época), contra CLAP crudo, al azar y BPM solo. MEDIDO 2026-11-08 (`zona_sets_2026-11-08.md`): dentro de 0,176 cae el 94 % de los 30, dentro de 0,147 el 89 % y dentro de 0,098 el 49 %; al azar, 23, 12 y 2 %. Pureza con pools de un solo dominio: Bedrock 5,7 % (azar 1,6 %, BPM 2,0 %), D1 1,8 % (0,7 y 1,1 %), D3 no concluyente; sobre audio mezclado BPM empata o gana a CLAP porque el tempo identifica el disco | A |
| E11 | Validación de una coordenada de dirección en tres pasos, en este orden: (1) ordena la biblioteca con sentido: Gabriel escucha los 10 primeros y los 10 últimos (E3); (2) se mueve con la noche: correlación con la posición normalizada en los bloques de D7 y D8 y en los 4 sets largos de Gabriel, por set y fuera de muestra; sube hacia el pico y cambia después de la 2.ª hora; (3) acuerda con al menos 16 de 20 pares de Gabriel (T4). Lo que no pasa (1) y (2) no entra a un perfil | A-B |
| E12 | Escucha a ciegas de listas enteras, de a dos (CLAP crudo frente a la variante), para unos 10 temas principales por ronda; Gabriel dice cuál prefiere. Es E2 hecha concreta, con el diseño del test de Vande Veire | Siempre |

### 11.3 Ideas nuevas (se suman al inventario de la sección 4; los ids siguen la numeración)

| Id | Idea | Origen | Contrato | Depende de | Etapa | Estado |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| P8 | Radio de zona: los 30 de Parecidos se marcan (o se cortan) por caer dentro del radio de E9, no por el número 30. En el registro de cada lista, un campo `seguro` por tema; el orden y la latencia no cambian | Gabriel 11-08 («seguridad»); Vande Veire y De Bie 2018 | Perfil Parecidos; Eval E9, E10; registro I3 | E9; vectores por tema validados de D7; `d1.csv` | A | Pendiente. Con E9 y E10 medidos (`zona_sets_2026-11-08.md`), el radio de la mediana casi no distingue y, como la lista ya está ordenada por parecido, el radio solo decide dónde cortarla: «los 30 frente a los seguros» compara una lista con su comienzo, y E12 para P8 necesita otro diseño antes de A2 [REEMPLAZADO 2026-11-09 por la adenda de esa fecha (sección 12): `seguro` se define con la probabilidad de «van juntos» (P12) y el consenso entre vistas (P13), que sí cambian el orden; el radio queda como referencia] |
| P9 | Proyección liviana de CLAP aprendida de sets: lineal y de pocas dimensiones (LDA o pérdida contrastiva sobre CLAP congelado), acerca temas del mismo set y aleja los de sets distintos; discos fuera de la muestra; se compara con CLAP crudo por E10 y después por E12. Es M4 sin GPU ni reentrenar el tronco | Gabriel 11-08 (reutilizar D7 y D8); Alonso-Jiménez et al. ICASSP 2023 | Tema (vector `clap_sets`); Comp `clap` variante | P8 (la medida); D7 validado; D1 como apoyo | B | Pendiente (adenda 11-09: es el paso siguiente de P12, cuando haya miles de pares de sets profesionales en el espacio de la biblioteca; con los cientos de hoy se aprenden 3 o 4 pesos, no una proyección) |
| P10 | Consulta compuesta para Mantener: centro de lo tocado con peso α más el tema actual con un paso chico en la dirección del último paso (β negativo), como la ec. 34 del paper (0,4 y −0,1 de partida). Se mide como zona (E10), no como acierto del tema siguiente | Vande Veire y De Bie 2018; T17 | Perfil Mantener; Comp `clap` | T17 (`clap_historia`), P8 | B | Pendiente. T17 medido con el acierto del siguiente no mejoró (p ≥ 0,72); esa no es la medida de esta idea (adenda 11-09: entra en T30, la zona del caso con el set en curso) |
| T18 | Direcciones de texto en CLAP: conceptos iniciales (energético o peak time, oscuro o hipnótico, melódico o uplifting, introspectivo o deep, vocal, con breakdowns), varias frases por concepto promediadas, dirección como diferencia de dos frases opuestas o percentil dentro de la biblioteca (la brecha entre modalidades hace que el coseno con un texto sea bajo y parejo: vale el orden, no el valor). Puntaje = w·cos(candidato, principal) + (1−w)·cos(candidato, concepto) | Gabriel 11-08 (pregunta); plan T3 y T7 | Comp `texto_clap`; perfiles de dirección | Codificador de texto (`tag_vocals.py`, `laion/clap-htsat-unfused`) y embeddings de la biblioteca: existen | A | Pendiente (auditoría 11-09, exploratorio: en un banco de referencia más concepto, la suma de audio y texto no cumple «parecido y concepto» mejor que el audio solo, 0,041 frente a 0,042; ordenar por el concepto dentro de una zona chica conserva más parecido. Queda como línea de base; propuesta T23 y T24 en `auditoria_recomendacion_2026-11-09.md`) |
| T19 | Cabezas de Essentia como coordenadas candidatas: danceability, engagement y approachability, mood_*, arousal y valencia (estas dos solo sobre MusiCNN o VGGish). Se corren sobre la biblioteca y sobre los bloques de D7 y D8; se mira saturación (como Mixed In Key) y movimiento con la noche (E11) | Gabriel 11-08 («deberían juzgar las trayectorias»); plan T3 | Tema; Comp `energia_objetivo` | V6 (pasada de EffNet o MusiCNN en WSL, prueba chica antes); bajar las cabezas (CC BY-NC-SA 4.0) | B | Pendiente (adenda 11-09: sube al tercer lugar del orden de la sección 12.5, porque la pasada da además el modelo independiente del banco E13 y las etiquetas de sus atributos; si la prueba chica dice que es pesada, va a gpuq, H1) |
| T20 | Cabeza propia sobre embeddings congelados (MAEST capa 7, estilos, CLAP): regresión del minuto normalizado o clasificación de fase (apertura, pico, después de la 2.ª hora) con etiquetas de los sets profesionales; partición por DJ. Es el «fine-tuning de Essentia» realista | Gabriel 11-08 (pregunta); precedentes `genre_model.py` y `clap_probe` | Tema; Comp `energia_objetivo`, `fase` | T11 (bloques de D7 y D8 con posición); D5 cuando exista | B | Pendiente |
| T21 | Variar la energía con tipos de transición elegidos por una máquina de estados (double drop, rolling, relajada) | Vande Veire y De Bie 2018, 4.2.3 | Perfil | — | — | No aplica por ahora: la dirección la elige Gabriel con el mouse (DECISIONS 11-07, punto 8). Queda como idea para una lista «sorpresa» |
| T22 | Representar el tema por sus tramos altos (drops) y no solo por el centro del 35-65 %, como el descriptor de estilo del paper; se relaciona con T16 (el comienzo del candidato) | Vande Veire y De Bie 2018, 4.2.2 | Tema (vector alternativo); Comp `clap` variante | Segmentación alto/bajo por bloque (energía) sobre la biblioteca; CLAP por bloques ya existe para D7 y D8 | C | Pendiente |
| M5 | Afinar el tronco de MAEST en torch (`maesthf`) con pares de sets | Gabriel 11-08 (pregunta) | Tema; Comp `aprendido` | GPU; etiquetas de sets abundantes (D5, D7) | 5 | Lo cubre M4; MAEST-HF es el tronco afinable que ya está en el repo |

Marcas en su lugar: T3 (lo concretan T18 y T19; el juez es E11) y T17 (se vuelve a medir como zona).

### 11.4 Orden de ejecución

Criterios de salida comunes: los de la sección 5, más un informe en `docs/reports/` con etiquetas por
afirmación, los números reproducidos desde los artefactos (N por fuente) y el estado de cada idea
actualizado en su lugar en las tablas de 11.2 y 11.3.

**Etapa A (ahora, CPU, sin audio nuevo).**

1. [HECHO 2026-11-08: `src/v4/evaluation/zona_sets.py`, informe `docs/reports/zona_sets_2026-11-08.md`. Además de lo listado entraron los temas de la biblioteca que aparecen juntos en sets de D3 y los originales de Bedrock, los únicos sets profesionales en el espacio de la biblioteca] **E9 y E10 sobre CLAP crudo.** Entradas: `indice_recomendacion.json` (1760 temas), los vectores por
   tema validados de D7, los bloques de los discos enteros, `d1.csv`. Sale: módulo de evaluación
   (`src/v4/evaluation/zona_sets.py` o una extensión de `banco_sets.py`) e informe con el radio de zona,
   la fracción de los 30 dentro del radio y la pureza fuera de muestra, con líneas de base al azar y BPM
   solo. Criterio: números reproducibles; una sola variable por comparación.
2. **P8 en el recomendador.** Campo `seguro` en el registro de cada lista, sin cambiar el orden ni la
   latencia; opcional cortar al radio. Sale: para unos 10 temas principales (elegidos por Gabriel o al
   azar de D1), las dos listas de E12: los 30 crudos y los «seguros». Criterio: tests del lado vivo en
   verde; Gabriel escucha y decide.
3. **T18, direcciones de texto.** Herramienta que toma frases por concepto, calcula las direcciones,
   ordena la biblioteca, exporta M3U de los 10 primeros y los 10 últimos por concepto (E11 paso 1, con
   E3) y proyecta los bloques de D7 y D8 y los sets largos de Gabriel (E11 paso 2). Sale: informe con qué
   direcciones sobreviven. Criterio: una dirección entra a un perfil solo si pasa los pasos 1 y 2.

**Etapa B (semanas; una unidad por vez).**

1. **P9**: proyección con discos fuera de muestra; E10 contra CLAP crudo; si gana, E12.
2. **V6 y T19**: pasada chica de EffNet o MusiCNN en WSL con costo por tema; cabezas sobre la biblioteca
   y los bloques de D7 y D8; E11.
3. **T20**: cabeza con posición o fase; E11; T18, T19 y T20 en la misma tabla.
4. **P10** para Mantener, con E10.

**Etapa C.** Perfiles Más, Menos e Introspectiva (T5 a T8) con la coordenada que sobreviva, un `.nml` por
perfil (I1), el registro I5, y D2 en sesiones reales. Después, las etapas 4 y 5 del plan (M3, M4) como
estaban.

### 11.5 Reglas de lectura para quien implemente

- El acierto en 30 sobre D1 descarta; no decide. «No mejora el acierto» no se traduce a «no mejora la lista».
- Nada se ajusta solo contra los sets de Gabriel (referencia con ruido, por sus palabras; E6).
- Una sola variable por comparación; partición por DJ o por disco fuera de la muestra (E5, E7).
- Las listas que escucha Gabriel van a ciegas, enteras y de a dos; unas 10 por ronda, para no gastar su
  escucha. Entre lo que sobrevive, decide él.
- Una coordenada se valida con E11 antes de entrar a un perfil, por buena que suene su etiqueta.
- CPU local; prueba chica antes de una corrida larga; `--low-priority`. Nada de GPU, Lightning ni Slurm sin
  pedido de Gabriel en el chat. [REEMPLAZADO 2026-11-09 por DECISIONS 2026-11-09 (sets de 1001Tracklists), punto 3: en volumen se lee la página con `tools/dj_sets/tracklists_1001.py`, un pedido por set a 8 s] 1001Tracklists se baja con la cuenta de Gabriel por el botón de exportar del
  sitio (DECISIONS 2026-11-08, punto 1); el `robots.txt` no se esquiva de otra forma.
- El lado en vivo no se rompe: el botón sigue por debajo de 1 s y usa solo la biblioteca estándar.

### 11.6 Preguntas para Gabriel (antes de A3)

1. [REEMPLAZADO 2026-11-08 por respuesta de Gabriel (DECISIONS 2026-11-08, punto 9): los conceptos son provisionales; 4 ortogonales para probar el método y, si funciona, se escriben desde el móvil] Los conceptos iniciales para T18, en sus palabras (los de la tabla son una propuesta).
2. [REEMPLAZADO 2026-11-08 por respuesta de Gabriel (DECISIONS 2026-11-08, punto 10): se usa todo lo disponible, curado; los tracklists de 1001Tracklists están en importación] Qué DJs y qué sets para D8, y si sus 4 sets largos valen como forma de referencia para E11.
3. Cuántas listas quiere escuchar por ronda (la propuesta es 10, de a dos).
4. [REEMPLAZADO 2026-11-08 por Gabriel (DECISIONS 2026-11-08, puntos 1 y 10): se bajan muchos con su cuenta y están en importación] Cuántos tracklists de 1001Tracklists bajar con su cuenta y de qué DJs (DECISIONS 2026-11-08, punto 1: el
   correo ya se mandó; bajar muchos quedó sin decidir).

### 11.7 Afirmaciones de la adenda

| Afirmación | Etiqueta | Evidencia |
| :--- | :--- | :--- |
| El codificador de texto de CLAP ya está en el repo y corre en Windows | EVIDENCIA OBSERVADA | `src/v4/pipeline/tag_vocals.py` (`VOCAL_PROMPTS`, `INSTRUMENTAL_PROMPTS`); `extract_representations.py` usa `laion/clap-htsat-unfused` por `transformers` |
| Un prompt de texto detecta voz con AUC 0,89 por tema y 0,81 por segundo; la cabeza lineal sobre CLAP entrenada con 54 temas sube a 0,98 y 0,88 | EVIDENCIA documentada, no re-medida el 11-08 | `docs/plans/20260929_deteccion_voz.md`, tablas de Electrobyte; los artefactos de `vocal_eval` no están en este escritorio |
| El repo ya entrena cabezas sobre embeddings congelados | EVIDENCIA OBSERVADA | `genre_model.py` (StandardScaler + LogisticRegression sobre `maesthf_full_l7` y `clap_full`, validación agrupada por artista) |
| Las cabezas de Essentia para energía no están bajadas y no hay EffNet ni MusiCNN de `musica` | EVIDENCIA OBSERVADA (auditoría 11-07; sin cambios el 11-08) | `models/essentia/` tiene solo los troncos y `genre_discogs400` |
| En D7, la compilación agrupa en CLAP (14,5 % de los 30 primeros contra 3,0 % de los candidatos) y, con pistas en archivos separados, CLAP no elige el tema siguiente mejor que el azar (17,9 % contra 15,3 %, 123 transiciones) | EVIDENCIA OBSERVADA (revalidación D7, 11-07) | `docs/reports/trayectorias_d7_2026-11-07.md` y su auditoría; es la base de leer «zona» y «orden dentro de la zona» como dos problemas |
| Todo lo medido en la etapa 1 y en líneas D1 usa el acierto en 30 de D1 (9 a 12 % en todas las variantes) y ninguna variante pasó por la escucha | EVIDENCIA OBSERVADA | Informes E1, líneas D1 y sus auditorías; DECISIONS 11-07 |
| Con esa medida no se distingue una lista buena de una regular | INFERENCIA | Unos 1700 candidatos, decenas igual de buenos para un momento; el azar exacto es 2,5 % |
| Los números del paper (α 0,4, β −0,1; 17 de 18; 3,89 frente a 2,39; 10 de 15) | EVIDENCIA OBSERVADA | PDF de acceso abierto, secciones 4.2.2, 4.2.3 y 5.6, leído el 2026-11-08 |
| Una proyección aprendida de sets mejora la pureza de Parecidos sobre CLAP crudo | HIPÓTESIS | La decide E10 con discos fuera de muestra y después E12 |
| Alguna cabeza de Essentia o dirección de texto se mueve con la noche en los sets reales | HIPÓTESIS | La decide E11 paso 2 |

## 12. Adenda 2026-11-09: zona con varias vistas, conceptos en dos casos y cómputo pesado

Escrita por la sesión auditora y aprobada por Gabriel el 2026-11-09 (DECISIONS de esa fecha, «algoritmo de
recomendación», puntos 4 a 7). Entra por los contratos de la sección 3 y no abre ningún camino propio hasta
Traktor. Nada de lo anterior se borra. Los números y sus etiquetas están en
`docs/reports/auditoria_recomendacion_2026-11-09.md`: son exploratorios y ninguna lista pasó por la escucha.
Para implementarla desde un chat nuevo: `PROMPT_FUSION_Y_CONCEPTOS.md`.

### 12.0 Qué cambió y por qué

- **El puntaje de Parecidos es una ecuación que pondera varios factores** (Gabriel; DECISIONS 2026-11-09,
  punto 8): modelos como CLAP y los demás, género, tonalidad, BPM. BPM y tonalidad son factores propios y se
  usan después: no se le piden a los modelos. Es la arquitectura de la sección 3.3 (componentes con peso). Por
  eso los modelos se miden con el tempo controlado (E14) y los pesos de todos los factores se fijan juntos
  (P12). Los filtros en vivo (BPM ±20 y tonalidad a ±2 semitonos) siguen como están (Gabriel).
- **Mirar con un solo modelo era el límite de Parecidos** (auditoría). CLAP, MAEST capa 7 y los estilos
  Discogs arman listas de 30 casi distintas y aciertan transiciones distintas; sumadas superan a CLAP crudo en
  los sets de Gabriel y en sets de otros DJs.
- **Las playlists de concepto tienen dos casos de uso** (Gabriel): sin tema de referencia, antes de empezar a
  tocar, y con el set en curso. Sección 12.2.
- **El puntaje de concepto limita más que la fórmula que lo mezcla con el audio** (auditoría). Se elige en
  un banco local construido como el del paper «Guided Music Retrieval».
- **Lo pesado en procesamiento o en disco va a gpuq y al Drive de Gabriel** (Gabriel). Sección 12.6.

### 12.1 Lo que se toma de «Guided Music Retrieval» (ideas, no copia)

Brandão, Silva, Alonso-Jiménez y Bogdanov; manuscrito enviado a IEEE TASLP el 16-07-2026, sin revisión por
pares todavía. Leído entero el 2026-11-09 (EVIDENCIA OBSERVADA del texto; tablas entre paréntesis).

| Idea del paper | Lo que dice | Dónde entra acá |
| :--- | :--- | :--- |
| La consulta es un audio de referencia más un texto que lo modifica | El resultado tiene que parecerse a la referencia y cumplir el cambio: son dos condiciones | T23, E13 |
| Hay varios blancos válidos (Tabla V) | Medir si aparece el blanco exacto subestima; se mide por separado parecido, modificación y las dos cosas | E13 |
| La supervisión se construye sola (MTIQ) | Pares muy parecidos según un modelo que el método no usa (MusiCNN, coseno ≥ 0,8) y que difieren con confianza en un atributo, según clasificadores de Essentia con umbral de alta precisión | E13 (etiquetas y parecido independientes del método), T26, T27 |
| Una fusión chica entrenada sobre CLAP congelado gana (Tabla IV) | En MTIQ, R@100: suma de audio y texto 2,08; Combiner 10,30; TG-CIR 14,75. En MTSQ la suma queda debajo del audio solo | T26, T27; T18 queda como línea de base |
| Quitar un atributo falla sin entrenar (Tabla VIII) | La suma acierta 0,8 % en quitas y 4,3 % en adiciones | T28 |
| CLAP es el mejor de siete modelos en parecido juzgado por personas (Tabla I) | Dim-Sim: CLAP 87,1; MusiCNN 85,9; MAEST 85,0; EffNet 78,6; MERT 67,1 | P1 sigue vigente; P11 le suma vistas |
| Sus límites | Conjunto cerrado de modificaciones con plantilla, música de Jamendo, sin pesos publicados | T27 es condicional; no hay un modelo listo para usar |

### 12.2 Los dos casos de uso (DECISIONS 2026-11-09, punto 5)

| | Sin tema de referencia | Con el set en curso |
| :--- | :--- | :--- |
| Cuándo | El DJ todavía no empezó: no hay tema principal | Hay tema principal y un historial de lo tocado |
| Qué pide | Una playlist por concepto («vibras africanas», «música de muertos»): los temas que lo transmiten, agrupados bajo esa etiqueta | Temas que van bien con lo que suena; si hay concepto, filtrados por él |
| Candidatos | Toda la biblioteca, con las condiciones del concepto (voces) | La zona de Parecidos alrededor del tema principal y del historial (T30), con los filtros en vivo |
| Orden | Puntaje del concepto (T29) | Parecido; con concepto, puntaje del concepto relativo al del tema principal (T23) |
| Largo | Variable, con piso de confianza | 30 en Parecidos; variable en las de concepto |

El paso de un caso al otro lo da el estado: cuando el DJ elige un tema de la lista sin referencia, ese tema pasa
a ser el principal y la iteración siguiente es el caso con el set en curso. Parecidos (D0) es siempre el
segundo caso sin concepto.

### 12.3 Medidas nuevas (se suman a la tabla de evaluación de la sección 4)

| Id | Idea | Etapa |
| :--- | :--- | :--- |
| E13 | Banco de conceptos local, con dos modos. Sin referencia: fracción de los primeros de la lista que cumple el concepto. Con referencia: fracción que es parecida a la referencia, que cumple el concepto y que cumple las dos cosas, con el número de blancos válidos al lado. En los dos: adición y quita por separado, conceptos vistos y no vistos, varios juegos de frases. Etiquetas independientes del método (género de Beatport, estilos Discogs, cabezas de Essentia; la etiqueta Vocal, marcada como no independiente); el parecido lo juzga un modelo que el método no use. Versión exploratoria medida el 11-09 | Ahora |
| E14 | Control de tempo al medir los modelos (las vistas), porque BPM y tonalidad tienen sus propios factores en la ecuación: pares al azar del mismo BPM, pool a ±8 BPM y correlación de cada vista con la diferencia de BPM. La ecuación entera se mide aparte, con todos sus factores y los pares al azar tomados dentro de los filtros en vivo. MEDIDO 2026-11-09 (`docs/reports/fusion_vistas_2026-11-09.md`): el tempo de los estilos está casi todo en su media (−0,37 crudo, −0,16 centrado); quitar sus 2 direcciones lineales de tempo no agrega nada; a ±8 BPM la ventaja de las fusiones sigue en D1 y a ±4 no se distingue | Siempre |

### 12.4 Ideas nuevas (se suman al inventario de la sección 4; los ids siguen la numeración)

| Id | Idea | Origen | Contrato | Depende de | Etapa | Estado |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| P11 | Fusión de vistas para Parecidos: CLAP centrado, MAEST capa 7 centrado y estilos Discogs, cada coseno en desvíos de su distribución, sumados; pesos iguales de partida; EffNet-Discogs como cuarta vista cuando exista para `musica` | Auditoría 11-09; G6; Tabla I del paper | Tema (vector fusionado, índice versión 4); Comp `clap` | Representaciones ya extraídas; MAEST de los originales de Bedrock | 12.5, unidad 1 | Exploratorio medido: en D1 de 9,0 % a 11,4-12,7 %; AUC en D3 × biblioteca de 0,66 a 0,74-0,77. Falta Bedrock, D5 y E12 [REEMPLAZADO 2026-11-09 por el piloto de D5 (sección 12.8): el cruce de D5 con la biblioteca casi no da pares; faltan Bedrock, E12 y, si Gabriel lo aprueba, el audio de los temas de los sets] [REEMPLAZADO 2026-11-09 por la unidad 1 (`docs/reports/fusion_vistas_2026-11-09.md`): reproducido; Bedrock con MAEST no lo confirma ni lo tumba (frente al azar de la biblioteca gana CLAP crudo, 0,695 contra 0,58-0,68; dentro de su propio audio, en temas consecutivos, ganan los estilos y las fusiones que los incluyen, 0,60-0,62 contra 0,56); falta E12] |
| P12 | La ecuación de Parecidos, aprendida: regresión logística sobre los factores (parecido por vista, género, tonalidad, BPM), entrenada con pares de un mismo set profesional frente a pares al azar dentro de los filtros en vivo, con los DJs de prueba fuera del entrenamiento. Da el peso de cada factor, una probabilidad de «van juntos» para el campo `seguro` y dónde cortar la lista. Con los pares al azar igualados en BPM se mide aparte lo que aportan los modelos (E14). Es M3 en su forma más simple, adelantado; el valor de la tonalidad por paso puede venir de P7 | Gabriel 11-09 (una ecuación pondera el peso de cada variable); auditoría 11-09 | Eval; Perfil Parecidos (pesos); registro I3 | Pares de sets profesionales en el espacio de la biblioteca (D3 × biblioteca, Bedrock, D5), con BPM y tonalidad de los dos temas [REEMPLAZADO 2026-11-09 por el piloto de D5 (sección 12.8): D5 cruzado con la biblioteca no da pares; los pesos de BPM, tonalidad y género salen de los pares consecutivos de D5 con metadatos de Beatport (P12a), y los de los modelos, de las fuentes con audio (P12b)] | Unidad 1 | Pendiente. Los pesos por defecto siguen en solo CLAP hasta que Gabriel los cambie [REEMPLAZADO 2026-11-09 por la unidad 1 (`docs/reports/fusion_vistas_2026-11-09.md`): medida con D3 × biblioteca (63 pares que pasan los filtros, 42 DJs) y Bedrock, dejando cada DJ fuera; la ecuación aprendida (AUC 0,774) no supera a los pesos iguales (todos los factores 0,788, las tres vistas 0,779; CLAP crudo 0,650) y sus pesos no se identifican; en D1 con filtros, 13,8 % frente a 16,8 % de los pesos iguales y 11,3 % de CLAP crudo. Para E12 se propone todos los factores con pesos iguales; los pesos por defecto siguen en solo CLAP] |
| P13 | Consenso entre vistas como marca de seguridad: cuántas vistas ponen al candidato entre sus K primeros | Auditoría 11-09 | Registro I3 | P11 | Unidad 1 | Exploratorio medido (11,6-11,8 % en D1) [REEMPLAZADO 2026-11-09 por la unidad 1 (`docs/reports/fusion_vistas_2026-11-09.md`): con las tres vistas de acuerdo (entre sus 30 primeros), el 99 % de los pares es de un mismo set profesional con pesos parejos, pero pasa en el 3,5 % de esos pares; con una o más, 0,86. Candidato al campo `seguro` junto con P12] |
| T23 | Con el set en curso, concepto en dos etapas: candidatos = zona de Parecidos; orden = puntaje del concepto relativo al del tema principal; largo variable con piso de confianza | Auditoría 11-09; Gabriel 11-09 (primero los parecidos, sobre esa lista el concepto) | Perfil de concepto; Comp `texto_clap` | P11; T24 o T25 | Unidad 4 | Exploratorio medido: conserva más parecido que la suma con el mismo cumplimiento |
| T24 | Puntaje de texto: varias frases por concepto, CLAP centrado en la biblioteca y contraste contra un banco fijo de descripciones | Auditoría 11-09; T18 | Comp `texto_clap` | Codificador de texto; `src/v4/evaluation/clap_texto.py` (otra sesión, reglas de redacción de frases) | Unidad 4 | Exploratorio medido: AUC contra el género de Beatport de 0,64 a 0,68 |
| T25 | Sondas supervisadas para los conceptos con etiqueta (voz, géneros, estilos, energía); son el techo del banco y sirven tal cual para las direcciones fijas | `genre_model.py`, `clap_probe`; auditoría 11-09 | Tema (puntajes precalculados); Comp `texto_clap`, `energia_objetivo` | Etiquetas; T19 para energía | Unidad 4 | Techo medido para género: AUC 0,76 |
| T26 | Mapa aprendido de texto a audio: con los vocabularios de etiquetas de la biblioteca (47 géneros, 400 estilos, cabezas de Essentia), un modelo chico lleva la descripción de una etiqueta a la región del espacio de audio donde están sus temas; se evalúa con etiquetas no vistas | Paper GMR (fusión entrenada sobre CLAP congelado) | Comp `texto_clap` | E13; T19 | Unidad 5 | Pendiente |
| T27 | Fusión tipo Combiner entrenada con MTIQ más tripletas locales construidas igual | Paper GMR | Comp `texto_clap` | E13 muestra que T24 a T26 no alcanzan; D9; H1 | Unidad 7, condicional | Pendiente |
| T28 | Quitar y «menos X» como eje de dos polos (frases positivas menos negativas) o como sonda, nunca como texto «sin X»; las `negative_prompts` del Worker se miden en E13 antes de usarse | Paper GMR, Tabla VIII; DECISIONS 2026-11-09 (conceptos, punto 1) | Comp `texto_clap` | E13 | Unidad 4 | Pendiente |
| T29 | Sin tema de referencia: playlist por concepto sobre toda la biblioteca, ordenada por el puntaje del concepto, con piso de confianza y largo variable. Es lo que hace hoy `conceptos_agente.py` con el coseno crudo | Gabriel 11-09 (caso antes de empezar a tocar) | Perfil de concepto sin referencia | T24 a T26 para el puntaje | Unidad 4 | En uso con texto crudo; exploratorio medido: de los 30 primeros cumple el 21 % (género de Beatport, base 9 %) |
| T30 | Con el set en curso, la zona usa el historial: la consulta suma el centro de lo tocado y el paso reciente (T17, P10) sobre las vistas de P11, y el estado de la trayectoria (T5) da nivel y pendiente a los perfiles. Se mide como zona (E10, E13), no como acierto del tema siguiente | Gabriel 11-09 (el historial como fuente para la trayectoria del tema siguiente) | Estado de la trayectoria (3.5); Perfil | P11; D2 con sesiones | Después de la unidad 2 | Pendiente |
| D9 | MTIQ y MTSQ: tripletas y audio de Jamendo recortado a 10 s (Hugging Face `sb-lucas/Music-Text-Queries`, Apache-2.0, 59,7 GB en total) | Paper GMR | Datos | H1 | Con T27 | Sin bajar |
| H1 | Cómputo pesado en gpuq (cola sobre Colab y Kaggle gratuitos) y disco en el Google Drive de Gabriel; lo liviano sigue en CPU local | Gabriel 11-09 | Infraestructura (sección 12.6) | gpuq y `rclone` instalados y con credenciales puestas por Gabriel | Cuando haga falta | [REEMPLAZADO 2026-11-09 por la puesta en marcha de gpuq: instalado y probado de punta a punta, docs/GPUQ.md] En este escritorio no están instalados (mirado el 11-09) |

Marcas en su lugar: G6, P2, P8, P9, P10, T17, T18 y T19.

### 12.5 Orden de ejecución

Criterios de salida comunes: los de la sección 5 y los de la 11.4.

1. **Fusión de vistas medida como código** (P11, P12, P13, E14): reproduce la tabla de la auditoría, suma los
   originales de Bedrock con MAEST y, cuando lleguen, los pares de D5 por DJ [REEMPLAZADO 2026-11-09 por el
   piloto de D5 (sección 12.8): esos pares casi no existen; la unidad sigue con Bedrock y D3 × biblioteca, y
   suma P12a] [HECHO 2026-11-09 sin P12a, que espera a D5 y a la decisión de Gabriel sobre Beatport: `docs/reports/fusion_vistas_2026-11-09.md`].
2. **Escucha E12 de Parecidos**: CLAP crudo frente a la fusión. Si Gabriel la prefiere, índice versión 4 y
   campo `seguro`; el lado vivo no cambia de arquitectura.
3. **Essentia** (V6, T19): prueba chica y después la biblioteca; EffNet como cuarta vista.
4. **Banco E13 en sus dos modos y escalera de puntajes** (T24, T25, T28); el agente de conceptos pasa a tener
   los dos casos (T29 y T23) sobre un módulo de puntaje compartido con el banco; escucha de listas de concepto.
5. **Mapa de texto a audio** (T26).
6. **Coordenada de energía** (T19, T20, juez E11) y **zona con historial** (T30).
7. **Condicional**: fusión entrenada con MTIQ (T27, D9, H1).

Después siguen las etapas 4 y 5 del plan (M3, M4) como estaban, con H1 para lo que pida GPU.

### 12.6 Cómputo pesado: gpuq y Drive (H1)

- **Regla** (Gabriel): todo lo pesado en procesamiento o en disco se procesa con las GPU de gpuq
  (github.com/gbibbo/gpuq) y con su Google Drive. Detalle y salvaguardas: DECISIONS 2026-11-09, «algoritmo
  de recomendación», punto 7, y `AGENTS.md`, «Compute»; piezas y comandos en `docs/GPUQ.md`.
- **Liviano, en local**: tests, bancos sobre embeddings ya calculados, texto de CLAP, MAEST de los 179
  originales de Bedrock, entrenar sondas y modelos lineales.
- **Candidatos a pesado**: la pasada de Essentia sobre la biblioteca si la prueba chica da más de una hora;
  extraer vistas de audio nuevo en volumen (D5 con audio, D6, D7, D8); MTIQ (D9); M3 y M4.
- **Cómo corre un trabajo** (leído de la documentación de gpuq): el código sale de este repositorio en la rama
  o commit pedido; la máquina es nueva en cada intento y puede perderse, así que el trabajo guarda puntos de
  control y los retoma; lo que tiene que sobrevivir se sube al remoto de `rclone` (por defecto `gdrive:gpuq`; [REEMPLAZADO 2026-11-09 por la puesta en marcha de gpuq: en este repo, `erclm:traktor_ml/`, docs/GPUQ.md]).
  A este escritorio vuelven resultados chicos (embeddings, tablas), que se verifican como cualquier artefacto:
  formas, N, valores finitos y alineación por `track_uid`.
- **Lo que conviene**: bajar los datos públicos (MTIQ) directo a la máquina remota y subir a Drive solo los
  embeddings. De la música privada, subir embeddings antes que audio; si hace falta audio, primero se le dice a
  Gabriel qué y cuánto.

### 12.7 Afirmaciones y preguntas

Las afirmaciones etiquetadas de esta adenda están en la tabla «Afirmaciones» del informe de auditoría, con su
revisión hostil; no se repiten acá. Lo que sigue siendo HIPÓTESIS: que la fusión dé una lista mejor (lo
deciden Bedrock, E12 y, si se consigue, el audio de los temas de los sets; D5 cruzado con la biblioteca no
alcanza, sección 12.8), que un mapa o una fusión entrenada sirva para conceptos escritos libremente (E13
con conceptos no vistos) y que las cabezas de Essentia sirvan de coordenada (E11).

Preguntas para Gabriel:
1. Quién elige los temas principales de la escucha E12 y si quiere marcar además tema por tema lo que no pondría.
2. [REEMPLAZADO 2026-11-09 por la puesta en marcha de gpuq: el despachador corre en la Raspberry `asr-rp5` de Gabriel; el Drive tiene unos 17 TiB libres
   (`rclone about erclm:`), docs/GPUQ.md] Dónde corre el despachador de gpuq y cuánto espacio libre tiene el Drive.
3. Si el umbral de «pesado» (alrededor de una hora de CPU o de 10 GB de disco nuevo) y el OK previo al primer
   envío de cada tipo de trabajo quedan así.
4. El audio de los temas de los sets (D6), que es decisión suya: sección 12.8.

### 12.8 El piloto de D5 y el juez de la ecuación (2026-11-09, noche)

La sesión de 1001Tracklists corrió el piloto de 30 sets (commit `45c0c52`). La auditoría rehízo las cuentas
desde `D:\TRAKTOR ML datos\dj_sets\1001tracklists\medidas.csv` (EVIDENCIA OBSERVADA): 30 sets, uno por DJ,
tomados de arriba de la tabla (no es una muestra al azar); 1206 temas y 1092 pares consecutivos identificados.

| Dato del piloto | Valor |
| :--- | :--- |
| Temas que están en la biblioteca (artista y título con la versión) | 11 de 1206 (0,9 %), en 7 sets |
| Pares consecutivos con los dos temas en la biblioteca | 0 |
| Pares de un mismo set con los dos temas en la biblioteca | 5, en 3 sets |
| Temas de artistas que sí están en la biblioteca, sobre los temas con artista nombrado | 33 % (media por set; mediana 29,5 %) |
| [REFUTADO 2026-11-09 por la aclaración de la sesión de 1001Tracklists y por `tracklists_1001.py`, función de medidas: es la parte de los temas identificados cuya fila tiene el botón de compra («affiliate»); que el botón lleve a Beatport se comprobó en un solo tema, y la URL real pide un pedido por tema que no se hizo (S14)] Temas con enlace a Beatport en la página | 70 % (media por set; mediana 80 %) |
| [REFUTADO 2026-11-09 por lo mismo: es la parte de los temas identificados cuya fila tiene el ícono de YouTube; ningún enlace se resolvió] Temas con enlace a YouTube en la página | 59 % (media por set) |

**Qué cambia.** La sección 12 contaba con los pares de D5 cruzados con la biblioteca como juez con otros DJs.
Ese juez no existe: los DJs de D5 casi no tocan los temas de la biblioteca. Las marcas están en P11, P12, 12.5
y 12.7. Lo que D5 sí da, sin audio, son unos 36 pares consecutivos identificados por set, con su orden.

| Id | Idea | Origen | Contrato | Depende de | Etapa | Estado |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| P12a | Los pesos de los factores que no son modelos (BPM, tonalidad, género) se aprenden de los pares consecutivos de D5 con metadatos de Beatport: cuántas veces más que al azar aparece cada paso de tonalidad (P7), cada diferencia de BPM y cada paso entre géneros (T1). Solo texto; DJs de prueba fuera del entrenamiento | Auditoría 11-09, por el piloto de D5; P7, T1 | Sets; Comp `bpm`, `clave`, `genero`, `transicion` | C1 sobre los temas de D5 (el 70 % de los identificados trae botón de compra, que se supone de Beatport; C1 busca por artista y título, no depende de ese enlace); que Gabriel decida sobre la señal `ai-train=no` de Beatport (sección 6, C1) | Unidad 1, cuando D5 esté bajado | Pendiente |
| P12b | Los pesos de los modelos (CLAP, MAEST, estilos) y su peso frente a los demás factores se aprenden donde hay audio de los dos temas: Bedrock, D3 × biblioteca, D7 validado con su control de dominio y, si Gabriel lo aprueba, D6 | Auditoría 11-09, por el piloto de D5 | Eval; Perfil Parecidos | Audio de los dos temas del par | Unidad 1 | Pendiente. Con las fuentes de hoy es un juez chico: un DJ con originales (Bedrock) y 79 pares de D3 [REEMPLAZADO 2026-11-09 por la unidad 1 (`docs/reports/fusion_vistas_2026-11-09.md`): medida con esas fuentes; falta D7 validado con su control de dominio y, si Gabriel lo aprueba, D6] |
| M6 | Vector de sets por artista: como M1, pero con el artista como unidad, porque los temas casi no se repiten entre sets y los artistas sí (el 33 % de los temas del piloto es de artistas de la biblioteca). Da un factor «artistas que los DJs ponen juntos» para la ecuación, solo con texto | Auditoría 11-09; S8 de la sección 6; M1 | Tema (vector por artista); Comp `set2vec` | D3 y D5 (texto) | Después de la unidad 1 | Pendiente. Riesgo: un artista no suena siempre igual; se mide como un factor más, con el control de popularidad de E7 |

**D6, el audio de los temas de los sets, pasa de la etapa 5 a ser la decisión que destraba el juez.** Es de
Gabriel (DECISIONS 2026-09-12, punto 3) y es trabajo pesado (H1). Propuesta de la auditoría, sin decidir:
un piloto con los 30 sets ya bajados, que mida antes de escalar (a) si el embedding de un avance de Beatport o
de un audio de YouTube cae cerca del embedding del mismo tema en la biblioteca, usando temas de la biblioteca
como verdad; (b) qué parte de los 1206 temas se consigue; (c) tiempo, disco y pedidos por tema. Con eso se
estima el total para los 1549 candidatos. Un cruce más laxo, sin la versión, no sirve de juez: versiones
distintas de un tema suenan distinto (en D3, 170 pares con el cruce laxo frente a 81 con el estricto, `zona_sets_2026-11-08.md`).
