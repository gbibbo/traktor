# Auditoría del informe de líneas D1 (2026-11-08)

Informe auditado: `docs/reports/lineas_d1_2026-11-07.md` (traktor-github-6e, commit
`08dc5150e91c48d82a8bbf935ac8bf4b9189b141`), con `src/v4/evaluation/lineas_d1.py`, `tools/dj_sets/clap_extremos.py`,
`harmonic.wheel_step` y las opciones nuevas del recomendador (`clave_tabla`, `clap_historia`, `extremos`, apagadas).
Artefactos leídos: `D:\TRAKTOR ML datos\dj_sets\sets\d1_lineas.json`, `d1_extremos.json`, `d1.csv`,
`biblioteca\clap_extremos.npz`, el índice de recomendación (1760 temas, 1717 sugeribles) y la colección de Traktor de
este escritorio con sus copias (`Traktor 4.4.1\Backup\Collection`, 4.2.0). Tests: pasan 70 de los que tocan
`harmonic`, el recomendador y `dj_sets`; falla `test_block3_clustering::test_phase2_cluster_on_test20` (artefactos de
`test_20` de esta máquina, sin relación con lo cambiado; `lineas_d1.py` y `clap_extremos.py` no tienen tests). Los
scripts de esta auditoría están fuera del repo (`pasos_d1.py`, `anacronismo_d1.py`, `extremos_control.py`).

## Veredicto

**Los números se reproducen y los tres veredictos dicen lo que los datos dicen.** La tabla de pasos se reprodujo con
código propio (coincide fila por fila, 515 transiciones, la aritmética cierra: 515 observadas, 515,0 esperadas) y
`wheel_step` coincide con una implementación propia en los 576 pares de tonalidades. Los 539 pares, los aciertos
de los 36 perfiles y las 39 pruebas apareadas salen de los JSON. Dos lecturas cambian y una se precisa:

1. **Lo de los temas sin tonalidad no es tonalidad: es «temas que Gabriel no toca».** Bajarlos mejora el banco
   (10,2-10,4 frente a 8,9 %), pero esos temas están donde Gabriel casi no va: son el 3-10 % de las tres carpetas
   que más toca (C, N, D) y el 23-60 % de A, F, I, K y O. Con el pool limitado a la carpeta del tema elegido, el
   lift de «?» sube de 0,15 a 0,30. Vienen en racha (12 de los 24 saltos desde un tema sin tonalidad van a otro
   sin tonalidad; 8 de los 13 temas del set del 2026-01-17 y 5 de los 35 del 2025-09-25, casi todos de
   `Downtempo`). Varios llevan la tonalidad en el nombre del archivo («(10A) Solomun - Friends.mp3») y no en
   los tags. Y Traktor pone tonalidad a lo que analiza al cargarlo: de los 593 candidatos de hoy que estaban en
   la colección de este escritorio antes del paquete del 2026-10-30, 37 no tienen tonalidad (6 %), frente al 13 %
   del pool. No tener tonalidad es en parte **consecuencia** de no haber sonado nunca. La fila «Eso es por la
   tonalidad» se queda en HIPÓTESIS con evidencia en contra; la mejora del banco es real pero no dice nada de
   armonía y en vivo esconde lo que Gabriel no conoce.
2. **Extremos: lo que predice es el comienzo del candidato, no el final del actual.** El par importa (con el
   final de otro tema al azar o el final medio de la biblioteca como consulta, 1,5-3,5 %, el azar es 2,5 %), pero
   del lado del tema actual da lo mismo qué se use: su comienzo frente al comienzo del candidato acierta 8,2 %,
   su final 7,1 %, su centro 6,5 % (p 0,43 y 0,77 frente al final). Lo que marca la diferencia de sentido es el
   lado del candidato: todo lo que se compara con su comienzo da 6,5-8,2 %, con su final 3,9-5,2 %, y el final
   del actual contra el centro del candidato da 4,8 %. La inferencia «hay algo propio de la transición» se
   reemplaza: hay algo propio del **comienzo del candidato**. Sumado a CLAP sigue NO CONCLUYENTE (comienzo del
   actual 11,1 %, final 10,6 %, centro 10,0 % frente a 8,9 %; 33-21, p 0,13 la mejor, y ya son siete variantes).
   Para el vivo alcanza con el comienzo de cada tema en el índice: el final del actual no hace falta.
3. **Precisiones.** Transiciones con los dos temas en el índice: 543; con el siguiente sugerible: 539 (las que
   cuenta el banco). Temas sin tonalidad: 288 de 1760 en el índice, pero en el banco son 227 de 1717 (Traktor
   da la tonalidad a 37); tocados: 48 de 571 filas con la clave del índice, 27 con la que usa el banco. Azar
   exacto del top-30: 2,52 % (mediana de 1668 candidatos; el BPM ±20 casi no filtra).

## Lo reproducido

| Afirmación del informe | Etiqueta | Evidencia |
|---|---|---|
| 22 sets, 539 transiciones, 3 semillas idénticas | EVIDENCIA OBSERVADA | `d1_lineas.json` y `d1_extremos.json`: `transiciones` 539 en los 36 perfiles, `min` = `max` en todos |
| Tabla de pasos (86/37,0 → 2,29 … «?» 9/68,2 → 0,15) | EVIDENCIA OBSERVADA | Reproducida con código propio desde `d1.csv`, el índice y la colección (pool BPM ±20 sin el principal): las 25 filas iguales; «−5» y «+6» dan las dos 30/25,3 por coincidencia (comprobado) |
| Regla con sin tonalidad 0 empata con CLAP | EVIDENCIA OBSERVADA | 8,91 / 9,28 / 9,28 % frente a 8,91 %; 28-28, 32-30, 34-32; p 1,00, 0,90, 0,90 |
| Aprendida 0,25: 11,7 frente a 8,9 %, 42-27, p 0,09; frente a la regla 39-24, p 0,08 | EVIDENCIA OBSERVADA | `apareadas`; el informe elige la comparación justa (las dos con sin tonalidad 0); la otra (42-19, p 0,004) incluye el efecto de los sin tonalidad |
| Pesos 0,5 y 1: 9,5 %, no se distinguen | EVIDENCIA OBSERVADA | 9,46 %, 37-34 y 39-36, p 0,81 y 0,82 |
| Solo sin tonalidad: 10,2-10,4 %, 8-9 frente a 1, p 0,02-0,04 | EVIDENCIA OBSERVADA | 10,20 / 10,39 / 10,39 %; 8-1 p 0,039, 9-1 p 0,022 (dos veces) |
| Zona de la noche 8,4-9,5 %, p ≥ 0,72 | EVIDENCIA OBSERVADA | 9,28 / 8,35 / 9,46 / 9,46 %; p 0,84, 0,76, 0,72, 0,77 |
| Tabla de extremos (7,1/3,9; 9,3/7,8; 10,6/8,2; 6,7/5,9; 9,8/9,3) y sus p | EVIDENCIA OBSERVADA | `d1_extremos.json`, las diez cifras y las diez pruebas |
| 1760 temas con comienzo y final, 0 errores | EVIDENCIA OBSERVADA | `clap_extremos.npz`: `comienzo` y `final` (1760, 2, 512) finitos, uids únicos, duración 90-889 s (3 temas de menos de 120 s, donde comienzo y final se pisan) |
| Mismo modelo que la biblioteca | INFERENCIA | Coseno del centro con el comienzo 0,83 y con el final 0,86 (mediana), entre temas distintos 0,69-0,76: es el mismo espacio; no se repitió la verificación de `clap_blocks --verificar` |
| Filas P7, T16 y T17 del plan | EVIDENCIA OBSERVADA | Dicen lo mismo que el informe |

## Lo nuevo: los temas sin tonalidad

| Medida | Resultado |
|---|---|
| Sin tonalidad por carpeta (candidatos; % tocados sin tonalidad) | C Melodic 3,3 % (101 veces tocada, 2 %); N Tech House 10,1 % (75, 2,7 %); D Techno 6,5 % (66, 0 %); K Tech House 27,4 % (55, 10,9 %); F Tech House 25,6 % (47, 6,4 %); I House 22,9 % (34, 14,7 %); O Deep House 24,4 % (23, 26,1 %); A Peak Time 59,5 % (7, 0 %) |
| Lift de «?» con el pool limitado a la carpeta del tema elegido (539 transiciones) | 21 observadas frente a 69,5 esperadas: 0,30 (con todo el pool, 0,15) |
| Rachas | Desde un tema sin tonalidad, 12 de 24 saltos van a otro sin tonalidad; desde uno con tonalidad, 9 de 515 |
| Dónde caen los 27 tocados | 2026-01-17: 8 de 13 (4 de `Downtempo`); 2025-09-25: 5 de 35 (3 de `Downtempo`); el resto 1-3 por set; 3 archivos con la tonalidad en el nombre («(9A)», «(10A)», «(11A)») |
| Candidatos de hoy en la colección de este escritorio antes del paquete (copia 2026-10-30 21:07) | 593 por nombre de archivo; 37 sin tonalidad (6,2 %) frente a 227 de 1717 (13,2 %) |
| ¿Estaba el pool en la colección cuando sonó cada set? | No sirve para acotarlo: Gabriel carga desde el explorador y Traktor importa al cargar (85 importados el 2025-07-29, 194 el 2026-01-08…); del tema siguiente solo 17 de 44 y 44 de 119 estaban en la copia anterior al set. NO CONCLUYENTE para el anacronismo del pool |

## Lo nuevo: qué lado del par predice en «extremos»

Mismo banco, mismos 539 pares, semilla 0, `extremos` en percentil; se cambia solo qué vector va de cada lado
(`extremos_control.py`, fuera del repo). Azar exacto del top-30: 2,52 %.

| Lado del actual → lado del candidato | Solo | Frente a «final → comienzo» | Frente a CLAP solo (8,9 %) |
|---|--:|---|---|
| final → comienzo (el del informe) | 7,1 % | — | 31-41, p 0,29 |
| comienzo → comienzo | 8,2 % | 23-17 a favor de este, p 0,43 | 37-41, p 0,73 |
| centro → comienzo | 6,5 % | 21-24, p 0,77 | 30-43, p 0,16 |
| final → final | 5,2 % | 13-23, p 0,13 | 24-44, p 0,02 |
| final → centro | 4,8 % | 18-30, p 0,11 | 15-37, p 0,003 |
| comienzo → final (el control del informe) | 3,9 % | 16-33, p 0,02 | 18-45, p 0,0009 |
| final de **otro** tema al azar → comienzo (3 permutaciones) | 3,5 / 1,7 / 2,0 % | 13-32, 6-35, 7-34; p ≤ 0,007 | 19-48 y peores, p ≤ 0,0005 |
| final medio de la biblioteca → comienzo | 1,5 % | 4-34, p < 0,001 | 7-47, p < 0,001 |

| Con CLAP en percentil y peso 0,5 | Acierto | Frente a «final → comienzo» 0,5 | Frente a CLAP solo |
|---|--:|---|---|
| final → comienzo (el del informe) | 10,6 % | — | 27-18, p 0,23 |
| comienzo → comienzo | 11,1 % | 16-13 a favor de este, p 0,71 | 33-21, p 0,13 |
| centro → comienzo | 10,0 % | 21-24, p 0,77 | 25-19, p 0,45 |
| comienzo → final | 8,2 % | 12-25, p 0,047 | 24-28, p 0,68 |
| final de otro tema → comienzo | 6,5 % | 15-37, p 0,003 | 24-37, p 0,12 |

Lectura: la consulta tiene que ser el tema actual (barajado y media caen al azar o debajo), pero no importa qué
parte de él; lo que importa es comparar con el **comienzo** del candidato y no con su final ni con su centro.
El informe comparó «final → comienzo» con «comienzo → final» y cambió los dos lados a la vez; con un solo lado
cambiado, el final del actual no es mejor que su comienzo.

## Afirmaciones

| Afirmación | Etiqueta | Evidencia |
|---|---|---|
| Los números del informe salen de los artefactos y la tabla de pasos es correcta | EVIDENCIA OBSERVADA | Reproducción propia (arriba) |
| Bajar los temas sin tonalidad mejora el acierto del banco [REEMPLAZADO 2026-11-08 por la adenda del par (commit d123afc): con la tonalidad del nombre del archivo en 41 temas, 9,5-9,7 % frente a 8,9 %, 4-5 frente a 1, p 0,22-0,38: ya no se distingue de CLAP] | EVIDENCIA OBSERVADA | 8-9 pares ganados frente a 1 perdido |
| Esa mejora es «temas que Gabriel no toca», no armonía | INFERENCIA | Carpetas, rachas, nombres con tonalidad, 6 frente a 13 % en la colección previa; Traktor da tonalidad a lo que suena. Lo que la decidiría: en D2, cuando el analizador de Traktor haya pasado por toda la biblioteca, ver si Gabriel sigue evitando los que hoy no tienen tonalidad |
| El pool de hoy incluye temas que no estaban cuando sonaron los sets viejos | HIPÓTESIS | Las copias de la colección no lo acotan (ver arriba); afectaría por igual a todos los perfiles salvo a los que bajan a los sin tonalidad |
| Con 60 s, «final → comienzo» predice mejor que «comienzo → final» | EVIDENCIA OBSERVADA | Reproducido: 7,1 frente a 3,9 %, 33-16, p 0,02 |
| Eso es «algo propio de la transición» (el final del actual con el comienzo del candidato) | REEMPLAZADO | Con el mismo comienzo del candidato, el comienzo del actual da 8,2 % y su centro 6,5 % (p 0,43 y 0,77 frente al final). Lo propio es el comienzo del candidato: contra su final o su centro, 3,9-5,2 % |
| El par importa (no es que algunos comienzos «gusten» siempre) | EVIDENCIA OBSERVADA | Final de otro tema al azar → comienzo: 1,7-3,5 %; final medio → comienzo: 1,5 %; azar 2,5 % |
| El comienzo del candidato sumado a CLAP mejora el siguiente | NO CONCLUYENTE | 10,0-11,1 % frente a 8,9 %, la mejor 33-21, p 0,13; siete variantes miradas sobre los mismos datos |
| Por qué el comienzo del candidato y no su centro ni su final | HIPÓTESIS | Gabriel elige escuchando el comienzo del candidato (cue), o los comienzos describen el estilo con menos ruido que los finales. Lo decidiría D2 con el punto de cue y, con por_tema validado de D7, repetir la tabla con otros DJs |
| La zona de la noche no mejora el siguiente | EVIDENCIA OBSERVADA | Reproducido; una variable por comparación (CLAP crudo en los dos) |
| Los valores aprendidos con peso 0,25 mejoran a CLAP | NO CONCLUYENTE | Como dice el informe; además, 539 transiciones de 22 sets no son independientes (un set tiene 129 temas) y McNemar las trata como tales |

## Revisión hostil de esta auditoría

- **¿El lift por carpeta es circular?** La carpeta del tema elegido se usa para elegir el pool: eso mide «dado
  que fue a esa carpeta, ¿evitó los sin tonalidad?». Sube a 0,30, no a 1: dentro de la carpeta también los evita,
  pero la mitad del efecto es la carpeta.
- **¿La colección previa dice algo del pool?** Solo de lo que Gabriel cargó alguna vez en este escritorio, no de
  lo que tenía disponible. Por eso el anacronismo queda en HIPÓTESIS y lo de los 6 frente a 13 % es evidencia de
  «no los carga», no de «no los tenía».
- **¿«Comienzo → comienzo» gana por elegir la mejor de varias?** Sí, entre tres lados del actual; por eso no se
  afirma que el comienzo sea mejor que el final (p 0,43), solo que el final no es mejor. Lo firme es lo otro: todo
  lo comparado con el final del candidato pierde contra CLAP (p ≤ 0,02) y lo barajado cae al azar (p ≤ 0,007).
- **¿El control barajado es justo?** Usa el final de otro tema de la biblioteca como consulta, con los mismos
  candidatos y el mismo percentil; cambia solo la consulta. El final medio es el caso límite (una consulta igual
  para todas las transiciones). Los dos caen al azar: el acierto del 7,1 % necesita al tema actual.
- **¿Los extremos de 60 s se pisan con el centro?** El centro de la biblioteca son 3 tramos del 35-65 %; en un
  tema de 90 s (el más corto) los 60 s de cada punta cubren todo el tema, pero son 3 temas de 1760.
- **¿Los números dependen de la semilla?** No: `min` = `max` en los 36 perfiles; mi corrida con semilla 0 da los
  mismos aciertos.

## Propuestas

- **P1.** En el informe, la fila «Bajar los temas sin tonalidad mejora un poco» lleva al lado que la mejora es
  sobre temas que Gabriel no toca, y la opción `clave_sin_tonalidad` en vivo se decide sabiendo que esconde temas
  que él no conoce (lo decide Gabriel).
- **P2.** Antes de D2, pasar el analizador de Traktor por la biblioteca (o leer la tonalidad del nombre del
  archivo cuando la lleva) para que «sin tonalidad» deje de ser «nunca cargado».
- **P3.** La línea T16 pasa de «comienzos y finales» a «comienzos»: el componente compara el vector del tema
  actual que ya está en el índice (centro, o su comienzo) con el comienzo de cada candidato. Para el índice de
  recomendación alcanza con `comienzo` de `clap_extremos.npz`; el final del actual y los `-sseof` no hacen falta.
  Cuando se repita en D2 o en D7 por_tema validado, la comparación lleva las dos filas de control (otro tema al
  azar como consulta; el mismo lado del actual contra el final del candidato), no solo el sentido inverso.
- **P4.** Cuando se repita en D2, mirar la mejora por set y no solo el McNemar global: con un set de 129 temas,
  una noche puede mover el resultado.

## Confirmación de la adenda del par (2026-11-08, commit `d123afcba036ce357ce66268b064b1a53c33aa7e`)

Desde `aef033e` el recomendador lee la tonalidad del nombre del archivo («(9A) …») cuando no la hay en Traktor
ni en los tags: 41 de los 227 sugeribles sin tonalidad. El banco repetido está en `d1_lineas_clave_nombre.json`.

| Afirmación | Etiqueta | Evidencia |
|---|---|---|
| Todos los números de la adenda salen del JSON | EVIDENCIA OBSERVADA | 519 transiciones con paso (515,0 → 519,0 esperadas, cierra); «solo sin tonalidad» 9,46 / 9,65 / 9,65 %, 4-1 p 0,375 y 5-1 p 0,219; aprendida 0,25 11,69 %, 42-27 p 0,091; frente a regla 0,25 41-22 p 0,023; regla 0,25 sin valor para «?» 8,16 %; tabla +0 2,239, +0x 1,877, −1 1,830, +1 1,498, −5 1,121, +5 0,971, +2 0,726, −2 0,604, «?» 0,174 (9 frente a 56,4) |
| La mitad de la ventaja de bajar los sin tonalidad eran esos 41 temas | EVIDENCIA OBSERVADA | De 8-1 a 4-1 pares discordantes con la única diferencia de 41 tonalidades leídas; las 9 veces que Gabriel fue a un tema sin tonalidad siguen siendo 9, lo que bajó es lo esperado del pool (68,2 → 56,4) |
| Lo que queda («?» 0,17 con 186 temas) sigue siendo «temas que Gabriel no toca» | INFERENCIA | Mismos argumentos de arriba (carpetas, rachas, Traktor pone tonalidad a lo que se carga); se decide analizando esos 186 en Traktor (lista `TRAKTOR ML sin tonalidad.m3u8`, propuesta del par) y repitiendo el banco |
| Los valores aprendidos no dependían de los sin tonalidad | EVIDENCIA OBSERVADA | Aprendida 0,25 sin cambio (11,69 %, 42-27); «solo pasos» frente a regla 40-24, p 0,06 (antes 39-24) |
