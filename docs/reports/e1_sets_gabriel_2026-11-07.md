# Banco E1 sobre los sets de Gabriel (D1), 2026-11-07

Plan: `docs/plans/recomendacion_playlists_plan.md`, etapa 1, puntos 1 y 2 (almacén de sets con D1 y E1
con líneas de base). Por la regla E6, lo medido sobre los sets de Gabriel orienta y no fija pesos.

## Veredicto

- **CLAP es, de lejos, el mejor componente solo.** El tema que Gabriel tocó después queda entre los 30
  primeros en el 9,3 % de las transiciones. Al azar es el 2,0 %, con solo BPM el 4,3 % y con solo
  tonalidad el 2,6 %. Sumar BPM a CLAP lo lleva a 10,2 %.
- **El filtro de tonalidad deja fuera el 22 % de las transiciones reales de Gabriel.** En 119 de 535
  transiciones, el tema siguiente tiene una tonalidad que la regla armónica considera incompatible
  (puntaje menor a 0,60), aunque el mejor ajuste siempre queda a ±2 semitonos. El filtro de BPM ±20
  deja fuera solo el 1,7 %.
- **NO CONCLUYENTE para fijar pesos:** son los sets de un solo DJ, con ruido (E6). Falta E1 sobre sets
  de otros DJs (D7).

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
