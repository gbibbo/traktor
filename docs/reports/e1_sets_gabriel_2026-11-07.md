# Banco E1 sobre los sets de Gabriel (D1), 2026-11-07

Plan: `docs/plans/recomendacion_playlists_plan.md`, etapa 1, puntos 1 y 2 (almacén de sets con D1 y E1
con líneas de base). Por la regla E6, lo medido sobre los sets de Gabriel orienta y no fija pesos.
Auditado el 2026-11-07 (skill `auditar`, la sesión que auditó el plan):
`docs/reports/auditoria_e1_sets_gabriel_2026-11-07.md`. Las marcas «(auditoría E1)» son suyas.

## Veredicto

- **CLAP es, de lejos, el mejor componente solo.** El tema que Gabriel tocó después queda entre los 30
  primeros en el 9,3 % de las transiciones. Al azar es el 2,0 %, con solo BPM el 4,3 % y con solo
  tonalidad el 2,6 % [REEMPLAZADO 2026-11-07 por la auditoría E1: esos perfiles empatan a cientos de
  candidatos (el grupo del puesto 30 tiene de 102 a 149 temas) y el banco desempata por el orden del
  índice; con desempate al azar, 5 semillas: BPM 4,6 %, tonalidad 4,2 %, género 5,0 %; CLAP y
  CLAP + BPM no cambian]. Sumar BPM a CLAP lo lleva a 10,2 % (auditoría E1: 29 frente a 34
  transiciones discordantes, p = 0,61; no se distingue de CLAP solo).
- **El filtro de tonalidad deja fuera el 22 % de las transiciones reales de Gabriel.** En 119 de 535
  transiciones, el tema siguiente tiene una tonalidad que la regla armónica considera incompatible
  (puntaje menor a 0,60), aunque el mejor ajuste siempre queda a ±2 semitonos. El filtro de BPM ±20
  deja fuera solo el 1,7 %.
- **NO CONCLUYENTE para fijar pesos:** son los sets de un solo DJ, con ruido (E6). Falta E1 sobre sets
  de otros DJs (D7).
- **(auditoría E1)** CLAP también supera a la popularidad (los temas que Gabriel repite entre sets:
  4,8 %; 46 frente a 23 discordantes, p = 0,008) y sus aciertos no son duplicados: 48 de 50 son entre
  artistas distintos. La regla de tonalidad deja pasar al azar el 69 % de los pares y el 76 % de las
  transiciones reales: saca el 31 % del pool al precio del 23,5 % de los movimientos reales.

## D1: los sets

`tools/dj_sets/sets_traktor.py` lee los 37 archivos de `Traktor 4.4.1/History`. Escribe
`<root>/sets/d1.csv`, en `D:\TRAKTOR ML datos\dj_sets`, fuera de Git.

| | Este lector | Auditoría 11-07 | Por qué cambia |
| :--- | --: | --: | :--- |
| Archivos | 37 | 36 | El de hoy no tiene ningún tema al aire |
| Sets | 22 | 21 | Ver la nota debajo |
| Temas al aire (2 min o más) | 602 | 592 | |
| Transiciones | 580 | 571 | |
| Temas en el índice | 571 | 562 | |
| Transiciones con los dos temas | 543 | 535 | |

Por qué hay un set más. Contando un set por archivo, el lector reproduce exactamente los 21 sets y 592
temas de la auditoría. La regla de 20 minutos sin nada al aire, que pidió la auditoría, hace dos
cambios:
- junta dos archivos del 2026-02-04 que estaban a 30 minutos (5 y 6 temas): un reinicio de Traktor en
  medio de un set;
- en el set de 130 temas separa un tema suelto del principio y lo descarta.

Sets de 3 h o más: 3,5, 7,2, 3,9 y 3,5 h. El de 8,5 h de la auditoría queda en 7,2 sin el tema suelto.

## E1

`src/v4/evaluation/banco_sets.py`. En cada transición a → b:
- el tema principal es a, y lo tocado son los temas anteriores del set;
- los candidatos son los del recomendador en vivo: los 1717 temas del paquete que están en la colección
  de Traktor, con BPM ±20 y tonalidad compatible, menos lo tocado;
- se supone tempo 1,0, porque la historia de Traktor no guarda el pitch.

Medidas, como pidió la auditoría:

| Medida | Qué es |
| :--- | :--- |
| Acierto en 30 | b entre los 30 primeros, sobre las b que están en el pool |
| Rango recíproco | Medio; vale 0 si los filtros dejaron fuera a b |
| Percentil | Mediano, del puesto de b; 0 es el primero |
| Recuperación | Fracción de las b del pool que pasan los filtros |

543 transiciones; 539 tienen a b entre los temas sugeribles. Las 4 restantes no son sugeribles: no
están en la colección o no tienen tonalidad.

| Perfil (pesos) | Recuperación | Acierto en 30 | Acierto en 30, de las que pasan | Rango recíproco | Percentil mediano |
| :--- | --: | --: | --: | --: | --: |
| Azar | 0,738 | 0,020 | 0,028 | 0,0044 | 0,522 |
| Tonalidad | 0,738 | 0,026 | 0,035 | 0,0110 | 0,462 |
| Género (taxonomía) | 0,738 | 0,039 | 0,053 | 0,0070 | 0,355 |
| BPM | 0,738 | 0,043 | 0,058 | 0,0094 | 0,366 |
| CLAP | 0,738 | 0,093 | 0,126 | 0,0208 | 0,232 |
| CLAP + BPM | 0,738 | 0,102 | 0,138 | 0,0206 | 0,249 |

## Qué deja fuera el filtro

Sobre las 535 transiciones con los dos temas en el pool:

| Caso | Transiciones |
| :--- | --: |
| Pasan los dos filtros | 400 |
| Tonalidad incompatible (puntaje < 0,60) | 117, y 2 más que además están a más de 20 BPM |
| A más de 20 BPM, con tonalidad compatible | 7 |
| El tema siguiente no tiene tonalidad | 9 |

Diferencia de BPM entre temas consecutivos (tags o Traktor, sin pitch): mediana 2, percentil 90 de 8.

En todas las transiciones con las dos tonalidades, el mejor ajuste está entre −2 y +2 semitonos:

| Semitonos | 0 | +1 | −1 | +2 | −2 |
| :--- | --: | --: | --: | --: | --: |
| Transiciones | 263 | 64 | 101 | 44 | 33 |

## Lectura

| Afirmación | Etiqueta | Evidencia |
| :--- | :--- | :--- |
| El lector de D1 reproduce a la auditoría por archivo, y la regla de 20 min da 22 sets | EVIDENCIA OBSERVADA | Conteos de esta sesión; diferencias explicadas arriba |
| CLAP solo supera a las líneas de base | EVIDENCIA OBSERVADA | Tabla de E1 (539 transiciones) |
| Sumar BPM a CLAP ayuda poco | EVIDENCIA OBSERVADA en D1; NO CONCLUYENTE como regla | 0,093 frente a 0,102; la diferencia es chica para un solo DJ |
| El filtro de tonalidad deja fuera el 22 % de las transiciones reales | EVIDENCIA OBSERVADA | 119 de 535 |
| Parte de ese 22 % son tonalidades mal estimadas o temas tocados con el pitch movido | HIPÓTESIS | Se decide con la tonalidad de Traktor analizada para todos (hoy 231 de 1717 sugeribles tienen la de Traktor) y con el pitch del registro en vivo (D2) |
| La regla de tonalidad hay que cambiarla | Decisión de Gabriel, no del agente | El dato está; la regla (P3, DECISIONS 2026-11-06, punto 7) es suya. Kell y Tzanetakis 2013 no encontraron efecto de la tonalidad en el orden de los DJs |
| CLAP supera a la popularidad y sus aciertos no son el mismo artista ni el mismo título | EVIDENCIA OBSERVADA (auditoría E1) | 46 frente a 23 discordantes, p = 0,008; 48 de 50 aciertos entre artistas distintos, 0 con el mismo título |
| El filtro de tonalidad pasa el 69 % de los pares al azar y el 76 % de las transiciones reales; los ±1 semitono ocurren a la tasa del azar y la misma tonalidad al doble | EVIDENCIA OBSERVADA (auditoría E1) | 387 de 506 frente a 349,5 esperadas; 165 frente a 185; 86 frente a 42 |

## Adenda: desempate al azar y tonalidad como componente (2026-11-07, noche)

Después de la auditoría (`docs/reports/auditoria_e1_sets_gabriel_2026-11-07.md`), el banco ya no depende
del orden del índice.
- **Desempate al azar.** Antes de cada consulta se mezcla el orden de los candidatos con una semilla, y se
  repite con 5 semillas.
- **Intervalo por sets.** Además, un intervalo del 95 % remuestreando sets enteros.
- **Aciertos por transición.** Se guarda el acierto de cada transición, para las comparaciones apareadas.

Los resultados están en `D:\TRAKTOR ML datos\dj_sets\sets\d1_e1.json` y `d1_e1_clave.json`.

**Líneas de base con desempate al azar** (acierto en 30; media y rango de 5 semillas; intervalo por sets
con la semilla 0):

| Perfil | Media | Rango | Intervalo por sets |
| :--- | --: | --: | --: |
| Azar | 2,2 % | 1,3-3,3 % | 0,6-2,3 % |
| BPM | 4,1 % | 3,5-4,6 % | 2,2-5,5 % |
| Tonalidad | 4,3 % | 3,7-4,6 % | 2,9-5,8 % |
| Género | 4,5 % | 4,1-5,2 % | 2,5-5,9 % |
| CLAP | 9,3 % | sin empates | 6,3-11,5 % |
| CLAP + BPM | 10,2 % | sin empates | 6,9-13,3 % |

**La tonalidad como componente.** Gabriel, 2026-11-07: transpone a menudo, a veces secuencias de 3 o 4
temas un semitono. Pide que la tonalidad puntúe en vez de filtrar: más alto lo compatible sin
transponer, apenas menos lo que pide 1 o 2 semitonos.
- El recomendador tiene ahora las opciones `filtro_clave`, `clave_penalizacion` y `clap_escala`. Por
  defecto queda lo de antes.
- Se probó una penalización de 0,05 por semitono: compatible sin transponer vale 1, a un semitono 0,95 y
  a dos 0,90. [REFUTADO 2026-11-07 por la auditoría (adenda de
  `auditoria_e1_sets_gabriel_2026-11-07.md`): con transposiciones de ±2 casi toda tonalidad encuentra
  alguna relación, y con 0,05 por semitono todo tema con tonalidad vale entre 0,60 y 1,00, en ocho
  valores; nada vale 0] Lo incompatible vale 0.
- CLAP va en percentil entre los candidatos (`clap%`), para pesarlo junto a la tonalidad.
- En todos los perfiles hay 539 transiciones en el pool.

| Perfil | Acierto en 30 | Recuperación | Percentil mediano | Rango recíproco | Frente a CLAP con filtro (solo uno / solo otro, p) |
| :--- | --: | --: | --: | --: | :--- |
| CLAP con filtro (lo actual) | 9,3 % | 0,738 | 0,232 | 0,0208 | — |
| CLAP sin filtro | 8,9 % | 0,970 | 0,207 | 0,0219 | 17 / 15, p = 0,86 |
| CLAP% + tonalidad 0,25, sin filtro | 7,4 % | 0,970 | 0,211 | 0,0119 | 23 / 13, p = 0,13 |
| CLAP% + tonalidad 0,5, sin filtro | 6,7 % | 0,970 | 0,212 | 0,0126 | 30 / 16, p = 0,054 |
| CLAP% + tonalidad 1, sin filtro | 7,3 % | 0,970 | 0,214 | 0,0130 | 33 / 23, p = 0,23 |
| CLAP% + tonalidad 0,5, con filtro | 9,1 % | 0,738 | 0,237 | 0,0200 | 22 / 21, p = 1,0 |

| Afirmación | Etiqueta | Evidencia |
| :--- | :--- | :--- |
| Con desempate al azar, BPM, tonalidad y género solos rinden parecido (4,1-4,5 %) y CLAP los supera también por sets | EVIDENCIA OBSERVADA | Primera tabla; los intervalos por sets de CLAP y de las líneas de base no se tocan |
| Sacar el filtro de tonalidad no cambia el acierto en 30 y deja disponible el tema siguiente en el 97 % de las transiciones | EVIDENCIA OBSERVADA en D1 | 8,9 % frente a 9,3 %, p = 0,86; recuperación 0,970 frente a 0,738 |
| Sumar la tonalidad como componente empeora el orden en D1 | EVIDENCIA OBSERVADA en D1; NO CONCLUYENTE como regla | 6,7-7,4 % frente a 9,3 %; el rango recíproco cae casi a la mitad |
| Eso pasa porque D1 compara tonalidades originales y Gabriel transpone | [REEMPLAZADO 2026-11-07 por la auditoría: la causa son los valores de la regla armónica. En D1 la vecina sin transponer aparece 1,5 veces más que al azar y recibe 0,70; la misma tonalidad transpuesta aparece menos que al azar (0,76 veces) y recibe 0,95. Los aciertos de CLAP que no piden transponer caen (32 → 11) y los transpuestos se mantienen (16 → 14). La hipótesis queda debilitada por D1, y D2 la decide] HIPÓTESIS | La historia de Traktor no guarda el pitch. Desde 2026-11-07, el registro en vivo (D2) anota en «principal» y «tocado» la tonalidad original, la resultante y los semitonos; con eso se repite la prueba con la tonalidad que sonó |
| Sin filtro, los temas sin tonalidad se cuelan entre los primeros | EVIDENCIA OBSERVADA (auditoría) | Con el componente en None, `weighted` los puntúa solo por CLAP: ocupan 8,5 de los 30 puestos (son el 13 % de los candidatos; Gabriel va a uno de ellos el 1,7 % de las veces). Necesitan un valor explícito: opción `clave_sin_tonalidad` (2026-11-07) |
| Qué hacer con el filtro | Decisión de Gabriel, escuchando | Las opciones existen; en D1 apagar el filtro no cuesta aciertos y recupera sus elecciones |
