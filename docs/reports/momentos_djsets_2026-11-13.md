# Momentos de la noche medidos en DJSets v1, 2026-11-13

**Para:** Gabriel, que aprueba la tabla, y quien implemente el ejecutor del teléfono (L10 y B1 del plan
`docs/plans/plan_recomendacion_v2.md`, § 3.4). **Pedido de Gabriel:** para cada momento (apertura, subida, pico,
cierre, after), qué rango de BPM y qué géneros de Beatport tocan los DJs, medido en el dataset y no supuesto.
Protocolo: skill `auditar`. Nada de esto está aprobado: `config/telefono/momentos.json` queda en `"estado": "propuesta"`.

- **Código:** `tools/dj_sets/momentos_noche.py` (nuevo, todavía sin commit).
- **Datos:** DJSets v1, copia local `D:\TRAKTOR ML datos\linea_a\djsets_v1\metadata`. El md5 de las cinco tablas es
  igual al de `checksums.md5` de v1 (armado el 2026-11-13 a las 09:35 con `build_djsets.py` en `4c83050`).
- **Versión:** HEAD `a6fc0f33907c9ca6d8613064e0e96ed10be4246f`. Bootstrap de 1000 remuestreos por DJ, semilla 0. Dos
  corridas dieron el mismo JSON byte a byte.

## Veredicto

1. **Los títulos no alcanzan.** En 1549 sets, ninguno dice Opening Set ni Warm Up, ninguno dice Peak Time y solo 3
   dicen Closing Set. En 1001Tracklists el título nombra el evento y el lugar, no el turno. After sale de 19 sets de
   12 DJs, y N2N pone la mitad de las filas. Queda bajo el mínimo (30 sets y 15 DJs), así que **after queda vacío** en
   la tabla. Apertura, subida, pico y cierre salen de la posición.
2. **A lo largo de una noche entera, el BPM sube, aunque poco.** Se miraron 96 sets de noche entera: los titulados
   All Night Long y similares, o los de 6 h o más.
   - El primer quinto está 2,6 BPM por debajo de la mediana del propio set (intervalo por DJ: 1,7 a 3,8 por debajo).
   - El último quinto está 1,7 por encima (1,0 a 2,7).
   - En 67 de 96 noches, el quinto más lento es el primero.
   - **El cierre no baja: sigue subiendo.**
   - Cuanto más larga la noche, más lenta la apertura: 0,9 BPM por debajo en los sets de 3 a 6 h que no son noche
     entera, 2,6 en la noche entera y 4,3 en los All Night Long titulados. Es lo que se esperaba de «una hora de
     festival no es una noche».
3. **El estilo mueve el BPM mucho más que el momento.**
   - Entre momentos, la banda se corre de 1 a 3 BPM: apertura 120-126, pico 123-128.
   - Entre sets, la mediana va de 122 (percentil 10) a 130 (percentil 90).
   - Un set de techno abre en 126-135 y llega a 126-138. Uno melódico abre en 120-124 y llega a 122-126.
   - La banda absoluta de la tabla describe la mezcla de estilos de las noches enteras: 60 de las 96 son melódicas o
     progresivas, solo 12 son de techno. No describe una noche de techno y tech house como la de Gabriel.
4. **Los géneros cambian con el momento en la dirección esperada, pero poco.** Las cifras son veces la proporción del
   género en toda la noche:
   - **Apertura:** Electronica 2,5, Organic House 2,4 (la mitad la pone un solo DJ), Deep House 1,6.
   - **Pico:** Techno (Peak Time / Driving) 1,24.
   - **Cierre:** Dance / Pop 1,8 y House 1,15, con menos Tech House (0,73).
   - Dentro de cada familia de estilo se repite lo mismo: en una noche de techno, la apertura lleva más Tech House.
5. **Propuesta:** la tabla de abajo, tal como pide el formato. Sirve tal cual como preset blando de dirección (más lento
   y más suave al abrir, más Peak Time en el pico). No sirve como banda dura para la biblioteca de Gabriel, porque
   dejaría afuera el techno de 130 o más. Las dos formas de usarla están en «Qué decide Gabriel».

## La tabla propuesta (`config/telefono/momentos.json`)

| Momento | Tramo de la noche | `bpm` (p25-p75) | Mediana (IC por DJ) | Δ set (IC por DJ) | Los 3 géneros más tocados (veces la proporción de toda la noche) | Sets | DJs | Filas con BPM | De dónde |
| :--- | :--- | :--- | :--- | :--- | :--- | --: | --: | --: | :--- |
| apertura | 0-20 % | 120-126 | 123 (122-124) | −2,6 (−3,8 a −1,7) | Melodic House & Techno 14 % (0,69); Tech House 11 % (1,05); Electronica 9 % (2,51) | 96 | 60 | 1343 | posición |
| subida | 20-60 % | 122-127 | 124 (123-125) | −0,1 (−0,3 a +0,1) | Melodic House & Techno 22 % (1,07); Progressive House 12 % (1,14); Tech House 12 % (1,13) | 96 | 60 | 2643 | posición |
| pico | 60-80 % | 123-128 | 125 (124-125) | +0,8 (+0,3 a +1,3) | Melodic House & Techno 22 % (1,07); Techno (Peak Time / Driving) 15 % (1,24); Progressive House 11 % (1,02) | 96 | 60 | 1313 | posición |
| cierre | 80-100 % | 122-128 | 124 (124-125) | +1,7 (+1,0 a +2,7) | Melodic House & Techno 23 % (1,11); Techno (Peak Time / Driving) 13 % (1,10); Progressive House 10 % (0,97) | 96 | 60 | 1227 | posición |
| after | — | null | — | — | {} | 19 | 12 | 705 | títulos (debajo del mínimo) |

- **`bpm`:** los percentiles 25 y 75 de las filas del tramo. Como el BPM de Beatport es entero, la banda junta del 58
  al 62 % de las filas, no el 50 %.
- **`generos`:** la fracción de las filas del tramo con género de Beatport; son los 8 primeros y suman de 0,77 a 0,80.
- **Δ set:** cuánto se aparta el tramo de la mediana de BPM de su propio set, en BPM. En cada set y tramo se toma la
  mediana; después, la media entre sets. Es la medida del momento sin el estilo.
- **Si cada DJ pesa igual,** en lugar de cada fila, las medianas quedan en 122, 124, 125 y 125 y las bandas casi no se
  mueven: 120-125, 122-127, 123-128 y 122-128.

## Qué se midió y sobre qué datos

- **Filas.** Son 53 650 filas de tracklist. Se sacan las 1329 que suenan encima de otra («w/») y quedan 52 321. De
  ellas, 47 576 están identificadas, 40 159 tienen BPM de Beatport y 40 178 tienen género de Beatport. Las filas «ID -
  ID» cuentan para la posición pero no tienen BPM.
- **DJ.** El DJ es la `split_key` de `split_by_dj.csv` (el primer nombre del DJ): son 356 claves para 573 nombres de
  DJ. El bootstrap remuestrea DJs enteros, así que un DJ con 7 sets no cuenta como 7 datos independientes.
- **BPM.** Es el `bp_bpm` de Beatport: el tempo del tema original, no el que sonó en la mezcla (el DJ lo acelera o lo
  frena). Mide qué tema eligió el DJ. Para un preset que elige temas de la biblioteca es la medida correcta: Traktor
  también guarda el BPM del original.
- **Posición** (`positions()`, `momentos_noche.py:120`). Se mide por la hora de entrada cuando el set la tiene en la
  mitad de las filas o más, las horas no retroceden y la última no pasa del 120 % de la duración. Las filas sin hora se
  interpolan entre sus vecinas. Si no se cumple, se mide por el orden de las filas. De los 385 sets largos, 257 se
  midieron por hora y 128 por orden. En los medidos por hora, las dos medidas difieren en 0,013 del set (mediana; p90
  0,040).
- **Noche entera.** Son los 96 sets largos que cumplen al menos una de dos condiciones:
  - el título dice All Night Long, Allnighter, Open To Close o «N Hour Set»: 40 sets;
  - duran 6 h o más: 80 sets, 24 de ellos ya contados por el título.

  Son 60 DJs y 17 sets b2b, de 254 a 954 minutos (mediana 425). Los DJs con más sets son Hernan Cattaneo (7), Vintage
  Culture (5), Tony McGuinness (5, los «Sunday Deep Set», transmisiones con varios DJs en fila), Jonathan Kaspar (4) y
  MEDUZA (3). Incluye maratones de Club Space Miami (8), Stereo Montréal (9) y Warung (4).
- **Tramos.** Apertura (primer quinto) y cierre (último quinto) se fijaron antes de medir. El corte del 60 % entre
  subida y pico se eligió mirando la curva, donde deja de subir rápido, y es exploratorio. Con el pico en 40-80 % la
  banda queda en 122-128 en lugar de 123-128.
- **Las dos fuentes casi no se tocan.** De los 96 sets de noche entera, solo uno tiene una etiqueta de momento en el
  título (Dawn Patrol de Antdot & Maz).

## Fuente 1: los títulos

Se buscaron palabras clave en el título (`KEYWORDS`, `momentos_noche.py:62`). Cada coincidencia se miró a ojo con
`--titulos`.

| Etiqueta | Palabras clave | Sets (todos / largos) |
| :--- | :--- | --: |
| apertura | Opening Set, Opener, Warm Up | 0 / 0 |
| pico | Peak Time, Prime Time | 0 / 0 |
| cierre | Closing Set, Closer Set | 3 / 3 |
| after | After Party, After Hours, After by Day | 5 / 1 |
| after | Sunrise Set, Sunrise Edition | 12 / 11 |
| after | Dawn Patrol | 2 / 2 |
| (no es momento) temporada | Opening/Closing Party, Night, Fiesta, Weekend; Reopening | 16 / 8 |
| (no es momento) marca | Afterlife, Afterwatt, Sunrise Festival | 13 / 1 |
| (no es momento) atardecer | Sunset | 7 / 5 |
| noche entera | All Night Long, Allnighter, Open To Close, N Hour Set | 41 / 40 |

**Falsos positivos que se sacaron:**

- **Fin o comienzo de la temporada de un club**, no del turno: «Closing Party», «Closing Fiesta» y «Closing Weekend»
  (Space, Amnesia, DC10, Omen), y también «Opening Night», «Opening Party» y «Reopening Night».
- **Marcas y nombres:** «Afterlife» es una marca de fiestas (10 sets, 8 de Tale Of Us), «Afterwatt» es parte del nombre de un
  DJ y «Sunrise Festival» es un festival.
- **«Speakerbox»** coincide con una búsqueda ingenua de «peak».

**Dudosos que quedaron:**

- «SUNRISE EDITION» de Sébastien Léger, que es un programa transmitido y no un turno al amanecer.
- El «Sunrise Set» de Josh Baker en el cierre de temporada de Amnesia: es el último turno de la noche, entre cierre y
  after.
- Los dos «Dawn Patrol» de Warung.

| Etiqueta del título | Sets | DJs | Filas con BPM | BPM p25-p50-p75 (IC de la mediana por DJ) | Géneros (veces la proporción de todos los sets) |
| :--- | --: | --: | --: | :--- | :--- |
| todos los sets | 1549 | 356 | 40 159 | 122-125-128 (125-125) | Techno (Peak Time / Driving) 17 %, Melodic House & Techno 15 %, Tech House 15 % |
| cierre | 3 | 3 | 148 | 118-121-124 (118-125) | Melodic House & Techno 30 % (2,0), Organic House 14 % (7,1), Electronica 11 % (3,8). Treavor Moontribe, 47 % de las filas |
| after | 19 | 12 | 705 | 122-123-125 (123-124) | House 23 % (2,6), Tech House 10 % (0,7), Melodic House & Techno 10 % (0,7). N2N, 50 % de las filas |
| after sin N2N | 12 | 11 | — | 121-123-126 | Melodic House & Techno 17 %, House 15 %, Afro House 14 %, Deep House 10 % |
| after: solo after party | 5 | 4 | 133 | 122-125-126 (123-126) | House 16 % (1,7), Deep House 15 % (2,9), Tech House 14 % (1,0) |
| after: solo sunrise y dawn | 14 | 8 | 572 | 121-123-124 (122-124) | House 25 % (2,8), Afro House 10 % (3,9), Funky House 8 % (22, casi todo N2N) |
| atardecer (informativo) | 7 | 5 | 207 | 118-122-125 (120-125) | Deep House 14 % (2,8), Electronica 9 % (3,2). John Digweed, 54 % de las filas |

**¿Coinciden las dos fuentes?** Solo se pueden comparar en cierre y after, y en ninguno de los dos alcanza para decir
nada:

- Los 3 «Closing Set» son más lentos y melódicos (mediana 121) que el último quinto de la noche entera (124, IC
  124-125). Pero son 3 sets y uno pone la mitad de las filas.
- After (123, IC 123-124) queda un BPM por debajo del cierre por posición, con más House y Afro House. Son 19 sets de 12
  DJs, la mitad de un solo DJ.

## Fuente 2: la posición dentro de los sets largos

**Noche entera** (96 sets, 60 DJs), por quintos:

| Quinto | Filas con BPM | p25-p50-p75 | Δ set (IC por DJ) | Filas de 130 BPM o más | Filas de menos de 110 | Sets cuyo quinto más lento es este | Sets cuyo quinto más rápido es este |
| :--- | --: | :--- | :--- | --: | --: | --: | --: |
| 0-20 % | 1343 | 120-123-126 | −2,62 (−3,75 a −1,73) | 10,4 % | 8,5 % | 67 | 12 |
| 20-40 % | 1317 | 122-124-126 | −0,66 (−1,11 a −0,26) | 11,8 % | 3,3 % | 9 | 12 |
| 40-60 % | 1327 | 122-124-127 | +0,18 (−0,55 a +0,67) | 13,9 % | 3,2 % | 7 | 19 |
| 60-80 % | 1312 | 123-125-128 | +0,82 (+0,25 a +1,29) | 16,5 % | 4,2 % | 8 | 29 |
| 80-100 % | 1227 | 122-124-128 | +1,70 (+0,94 a +2,74) | 19,0 % | 5,1 % | 5 | 24 |

**Todos los sets largos** (385 sets, 162 DJs), por quintos:

| Quinto | p25-p50-p75 | Δ set | 130 o más | Menos de 110 | Más lento | Más rápido |
| :--- | :--- | :--- | --: | --: | --: | --: |
| 0-20 % | 121-124-127 | −1,31 (−1,74 a −0,89) | 13,4 % | 5,9 % | 203 | 61 |
| 20-40 % | 122-124-127 | −0,45 (−0,79 a −0,19) | 14,6 % | 3,5 % | 60 | 57 |
| 40-60 % | 122-125-128 | +0,19 (−0,07 a +0,40) | 16,4 % | 3,2 % | 43 | 79 |
| 60-80 % | 123-125-128 | +0,41 (−0,02 a +0,83) | 18,6 % | 3,9 % | 31 | 92 |
| 80-100 % | 122-125-128 | +0,51 (−0,01 a +0,93) | 18,2 % | 5,3 % | 48 | 96 |

**Cuanto más larga la noche, más honda la apertura.** Es el mismo cálculo sobre cuatro poblaciones:

| Población | Sets | DJs | Δ apertura (0-20 %) | Δ cierre (80-100 %) |
| :--- | --: | --: | :--- | :--- |
| Sets largos que no son noche entera (3 a 6 h, sin título de noche entera) | 289 | 129 | −0,87 (−1,26 a −0,46) | +0,11 (−0,51 a +0,57) |
| Todos los sets largos | 385 | 162 | −1,31 (−1,71 a −0,89) | +0,51 (+0,03 a +0,94) |
| Noche entera (la tabla) | 96 | 60 | −2,62 (−3,76 a −1,67) | +1,70 (+0,97 a +2,69) |
| Solo los titulados All Night Long y similares | 40 | 35 | −4,29 (−6,15 a −2,81) | +3,02 (+1,35 a +5,23) |

**Géneros por quinto, noche entera** (fracción de las filas con género de Beatport):

| Quinto | Melodic H&T | Techno Peak Time | Tech House | Progressive | House | Deep House | Organic House | Dance / Pop |
| :--- | --: | --: | --: | --: | --: | --: | --: | --: |
| 0-20 % | 14,4 | 8,3 | 11,2 | 8,0 | 8,3 | 8,9 | 9,1 | 1,3 |
| 20-40 % | 21,6 | 10,4 | 11,9 | 12,1 | 6,5 | 6,8 | 4,4 | 2,7 |
| 40-60 % | 22,6 | 13,3 | 12,1 | 12,7 | 6,9 | 3,9 | 2,6 | 2,3 |
| 60-80 % | 22,3 | 14,9 | 10,0 | 11,1 | 8,0 | 4,9 | 1,2 | 2,3 |
| 80-100 % | 23,1 | 13,3 | 7,7 | 10,5 | 8,8 | 3,8 | 1,6 | 4,9 |

En la apertura, Electronica es el 9,2 % de las filas (2,5 veces lo de toda la noche). No está en esta tabla porque no
entra entre los 10 géneros que se siguen a lo largo del set.

**Cobertura:**

- La fracción de filas identificadas es pareja a lo largo del set: del 86 al 88 % en cada quinto. El pico no tiene más
  temas inéditos («ID») que el resto.
- Entre las identificadas, las que tienen BPM de Beatport bajan al final: del 86 % en los primeros quintos al 83 % en el
  último quinto de la noche entera, y al 79 % en el último decil de los sets largos. El cierre lleva más temas que no
  están en Beatport (clásicos, ediciones), así que es el momento peor cubierto.

## Por familia de estilo del set

La familia del set es la que tiene más filas en el set (`FAMILIAS`, `momentos_noche.py:82`):

- **techno:** Peak Time, Raw y Hard Techno;
- **house y tech house:** Tech House, House, Deep, Minimal, Afro y otros;
- **melódica y progresiva:** Melodic, Progressive, Organic, Indie, Electronica, Trance y otros.

BPM p25-p50-p75, Δ set y el género que más sube respecto de la misma familia:

| Familia (población) | Sets | DJs | Apertura | Subida | Pico | Cierre |
| :--- | --: | --: | :--- | :--- | :--- | :--- |
| techno (noche entera) | 12 | 10 | 125-128-133, Δ −2,8; Tech House ×2,7 | 127-131,5-137, Δ +0,1 | 126-134-139, Δ +1,3; Hard Techno ×1,3 | 128-136-142, Δ +6,0 (+2,7 a +11,6); Hard Techno ×1,5 |
| techno (todos los largos) | 67 | 36 | 126-129-135, Δ −1,1; Tech House ×1,8 | 126-130-137, Δ −0,1 | 126-132-138, Δ +0,9; Hard Techno ×1,3 | 125-130-138, Δ +0,2 (−2,2 a +2,3) |
| house y tech house (noche entera) | 24 | 18 | 122-124-126, Δ −2,1; Deep House ×1,5 | 124-125,5-128, Δ +0,3 | 124-126-128, Δ +0,8; Techno Peak Time ×1,4 | 123-125-128, Δ +0,2; House ×1,24, Tech House ×0,69 |
| house y tech house (todos los largos) | 132 | 75 | 122-124-126, Δ −0,7; Deep House ×1,3 | 123-125-127, Δ +0,1 | 123-125-127, Δ +0,2 | 123-125-127, Δ +0,3; House ×1,13 |
| melódica y progresiva (noche entera) | 60 | 37 | 120-122-124, Δ −2,8; Organic House ×2,5 | 121-123-125, Δ −0,3 | 122-124-126, Δ +0,7; Techno Peak Time ×1,5 | 122-123-126, Δ +1,4 |
| melódica y progresiva (todos los largos) | 181 | 75 | 120-122-124, Δ −1,9; Organic House ×2,0 | 121-123-125, Δ 0,0 | 122-124-126, Δ +0,4 | 122-124-126, Δ +0,9 |

- **Techno en noche entera:** son 12 sets de 10 DJs, debajo del mínimo; se informan, pero no entrarían a una tabla.
- **Techno en todos los sets largos:** 67 sets de 36 DJs, que sí alcanzan.

## Sensibilidad (noche entera, BPM p25-p50-p75)

| Variante | Apertura | Subida | Pico | Cierre | Sets |
| :--- | :--- | :--- | :--- | :--- | --: |
| La tabla | 120-123-126 | 122-124-127 | 123-125-128 | 122-124-128 | 96 |
| Sin b2b | 120-123-126 | 122-124-127 | 123-125-128 | 122-125-128 | 79 |
| Solo posición por hora | 120-123-125 | 122-124-126 | 123-124-127 | 122-124-127 | 63 |
| Posición por fila en todos | 120-123-126 | 122-124-127 | 123-125-128 | 122-124-128 | 96 |
| Sin las filas fuera de 95-150 BPM | 120-123-126 | 122-124-127 | 123-125-128 | 122-124-128 | 96 |
| Apertura solo 0-10 %; pico 40-80 %; cierre solo 90-100 % | 120-122-126 | — | 122-125-128 | 122-124-128 | 96 |

Ninguna variante corre una banda más de 1 BPM.

## Afirmaciones

| Afirmación | Etiqueta | Evidencia |
| :--- | :--- | :--- |
| Los títulos no dan apertura ni pico, y dan 3 sets de cierre | EVIDENCIA OBSERVADA | Salida del script, sección 1: 0, 0 y 3 coincidencias; lista revisada a ojo con `--titulos` |
| After queda debajo del mínimo: 19 sets, 12 DJs, la mitad de las filas de N2N | EVIDENCIA OBSERVADA | Sección 1: «título after», `dj_mayor` n2n 50 % |
| En la noche entera, la apertura está 2,6 BPM debajo de la mediana del set y el cierre 1,7 arriba | EVIDENCIA OBSERVADA | Sección 3: Δ set con IC por DJ que no cruza el 0 |
| En 67 de 96 noches el quinto más lento es el primero | EVIDENCIA OBSERVADA | Sección 2, «sets cuyo quinto más lento es ese» |
| El cierre de una noche entera no baja de tempo | EVIDENCIA OBSERVADA para la mediana; INFERENCIA para «no baja» | Δ cierre +1,7 (+1,0 a +2,7); la media de las filas sí baja en los sets largos (124,4 frente a 125,0 del pico) por unos pocos temas lentos: 5,3 % de menos de 110 BPM en el último quinto, frente a 3,2 % en el medio |
| La apertura es más lenta cuanto más larga la noche | EVIDENCIA OBSERVADA (cuatro poblaciones); la causa (un set de 3 h es el turno central de otro, no la noche) es INFERENCIA | Tabla de cuatro poblaciones; no hay hora del día en los datos para comprobarlo |
| El estilo del set mueve el BPM más que el momento | EVIDENCIA OBSERVADA | Mediana por set, de 122 (p10) a 130 (p90), desvío 4,6; dentro del set, la apertura y el cierre se apartan de 1 a 3 BPM; techno 126-138 frente a melódica 120-126 |
| La banda de la tabla describe sobre todo música melódica y progresiva | EVIDENCIA OBSERVADA | Sección 2b: 60 de 96 sets de noche entera son de esa familia; 12 son de techno |
| La apertura lleva más Electronica, Organic House y Deep House | EVIDENCIA OBSERVADA para Electronica y Deep House (35 y 37 DJs); para Organic House, INFERENCIA débil | Organic House en la apertura: 122 filas de 19 DJs, 47 % de Tony McGuinness |
| El pico lleva más Techno Peak Time y el cierre más House y Dance / Pop | EVIDENCIA OBSERVADA, efecto chico | Peak Time 1,24 veces en el pico (196 filas, 41 DJs); Dance / Pop 1,8 veces en el cierre (60 filas, 19 DJs, 25 % de Vintage Culture) |
| Medir la posición por hora o por fila da lo mismo | EVIDENCIA OBSERVADA | Diferencia mediana de 0,013 del set; bandas iguales en la sensibilidad |
| Las bandas no dependen de los b2b ni de los temas a medio tiempo | EVIDENCIA OBSERVADA | Tabla de sensibilidad |
| Con la tabla como banda dura, una noche de techno de Gabriel perdería sus temas de 130 o más | INFERENCIA | Banda del pico de 123-128 frente a 126-138 del techno; no se miró la biblioteca de Gabriel |
| Un preset relativo (corrimiento sobre lo que suena o sobre el estilo pedido) andaría mejor que la banda absoluta | HIPÓTESIS | Se decide con listas armadas de las dos formas y escucha de Gabriel |
| After es más lento y más house que el cierre | HIPÓTESIS | 19 sets, debajo del mínimo. Hace falta bajar más sets etiquetados after o afterhours, con su permiso |

## Revisión hostil

- **¿La tabla mide el momento o el estilo?** Sobre todo el estilo. Es la falla principal y está en el veredicto. Para
  medir el momento sin el estilo, el informe da el Δ set y las familias.
- **¿El tramo se eligió mirando los datos?** Apertura y cierre, no. El corte del 60 % entre subida y pico, sí; con el
  pico en 40-80 % la banda cambia 1 BPM en el p25. Es exploratorio.
- **¿Se cambió la medida después de mirar?** Sí, una vez. La primera versión del Δ set promediaba las filas. Con la
  media, el último quinto de los sets largos bajaba (−0,9) y el de la noche entera quedaba en +0,3. Se pasó a medianas
  porque algunos temas de techno figuran a medio tiempo en Beatport (139 filas de techno de menos de 95 BPM en los sets
  largos, contadas en la exploración). Con medianas, el cierre sube. Las dos lecturas son ciertas a la vez: el grueso
  del cierre sube, y además aparecen unos pocos temas muy lentos (de 3,2 a 5,1 % de menos de 110 BPM). El informe da las
  dos.
- **¿Un DJ arrastra la tabla?** El que más pesa es Vintage Culture, con el 7 % de las filas. Los intervalos remuestrean
  DJs, y con cada DJ pesando igual las medianas se mueven 1 BPM o menos. Organic House en la apertura sí depende de un
  DJ: Tony McGuinness pone el 47 %.
- **¿«Noche entera» es de verdad una noche?** No se comprobó set por set. Entran maratones de clubes que abren tarde
  (Space Miami, Stereo Montréal, Warung) y cinco transmisiones con varios DJs. El subconjunto de títulos All Night Long
  da la misma dirección, más fuerte.
- **¿El BPM de Beatport es lo que sonó?** No: es el del tema original. Para elegir temas de la biblioteca es lo que
  corresponde. Para el tempo de la mezcla habría que medir el audio del set, que no está en el dataset.
- **¿La muestra representa a los DJs?** No es al azar:
  - sale de las semillas de Gabriel, con un tope de 15 sets por DJ y sin los sets comerciales;
  - 1001Tracklists sobrerrepresenta a los DJs conocidos;
  - los sets largos del dataset tiran a progresivo (Hernan Cattaneo, Guy J, Nick Warren, Sasha, Digweed).
- **¿Los temas sin identificar sesgan algún tramo?** La identificación es pareja (del 86 al 88 % por quinto). Lo que
  falta al final es el BPM de Beatport (83 % y 79 %), así que el cierre es el peor cubierto.
- **¿La aritmética cierra?**
  - 40 titulados más 56 de 6 h o más sin título dan los 96 sets.
  - 52 321 filas propias más 1329 «w/» dan las 53 650.
  - Los géneros de cada momento suman de 0,774 a 0,802, menos de 1.
  - El JSON se volvió a cargar y tiene las cinco claves.
- **¿Se cerró algo para no dejarlo abierto?** After queda vacío y la comparación de las dos fuentes queda sin
  conclusión. La tabla es una propuesta.

## Qué decide Gabriel

1. **Si aprueba la tabla tal cual,** con la banda absoluta de la noche entera mezclada, que es sobre todo melódica, o si
   prefiere otra forma. Opciones:
   - (a) Bandas por familia: techno y house con tech house, de todos los sets largos, que pasan el mínimo. Pide cambiar
     el formato del JSON: una tabla por familia.
   - (b) Un corrimiento relativo por momento: apertura unos −2,5 BPM y cierre unos +1,5 sobre lo que suena o sobre el
     estilo pedido, más los multiplicadores de género. También pide cambiar el formato.
   - (c) La tabla como está, usada solo como dirección blanda.
2. **Qué hacer con after:** dejarlo vacío, llenarlo a mano o bajar más sets etiquetados (afterhours, sunrise) para
   medirlo. Lo último es una descarga nueva y pide su visto bueno.
3. **Si el pico y el cierre quedan separados.** Para el BPM, una noche entera no tiene un pico en el medio y una bajada
   al final: sigue subiendo. Lo que cambia es el género (más House y Dance / Pop al cierre).

Para la sesión que haga el commit:

- el campo `commit` del JSON es el HEAD medido (`a6fc0f3`), que todavía no contiene el script. Conviene volver a
  correrlo después del commit para que el campo apunte a un commit que lo contenga; el contenido no cambia;
- falta una línea en `docs/PROJECT_MAP.md` para `tools/dj_sets/momentos_noche.py` y `config/telefono/`;
- no hay test unitario: la verificación fue la corrida de punta a punta, repetida.

## Cómo reproducir

```
python tools/dj_sets/momentos_noche.py --json config/telefono/momentos.json
python tools/dj_sets/momentos_noche.py --titulos        # además, la lista de cada set con palabra clave
python tools/dj_sets/momentos_noche.py --metadata "<ruta a djsets_v1/metadata>"
```

Tarda unos 6 s en CPU y solo lee las cuatro tablas, sin tocarlas. Con la misma semilla, la salida y el JSON son
idénticos. Un JSON reescrito vuelve a `"estado": "propuesta"`: es una medición nueva. Las secciones de la salida son 1
(títulos), 2 (posición), 2b (familias), 3 (momentos y sensibilidad) y 4 (las dos fuentes); el informe cita esos números.
