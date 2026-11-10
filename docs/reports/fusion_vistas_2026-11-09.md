# Fusión de vistas, ecuación de factores y consenso para Parecidos (unidad 1), 2026-11-09

Unidad 1 de `docs/plans/PROMPT_FUSION_Y_CONCEPTOS.md` (plan de playlists en vivo, sección 12: P11, P12, P13 y E14).
Protocolo: skill `auditar`. Código: `src/v4/evaluation/fusion_vistas.py` (reutiliza `zona_sets.py`) y
`tools/dj_sets/maest_originales.py`; tests: `tests/v4/test_fusion_vistas.py` (14) y `tests/v4/test_zona_sets.py`.
Resultado, fuera de Git: `D:\TRAKTOR ML datos\dj_sets\sets\fusion_vistas.json` (semilla 0, 223 s, código del
commit de este informe sobre `b7c9f87`). MAEST de Bedrock: `D:\TRAKTOR ML datos\dj_sets\maest\<carpeta>\*.npz`.

Datos (EVIDENCIA OBSERVADA en esta sesión): índice `indice_recomendacion.json` versión 3, 1760 temas, creado el
2026-11-06; `clap_full`, `maesthf_full_l7` y `maesthf_full_styles` con 1770 filas finitas cada una, los 1760
`track_uid` del índice presentes en las tres; el CLAP del índice coincide con `clap_full`. Fuentes: D1, 22 sets
de Gabriel, 542 transiciones con los dos temas en el índice; D3 × biblioteca, 79 pares de 67 sets y 48 DJs;
Bedrock, 179 originales de 3 discos de John Digweed, 5243 pares.

Todo lo de este informe es exploratorio: ninguna lista pasó por el oído de Gabriel, y «mismo set» mide
coherencia de estilo dentro de un set, no «suena bien mezclado».

## Veredicto

1. **Los números de la auditoría se reproducen.** Con el módulo nuevo salen iguales el acierto en 30 de D1 de
   todas las filas, los discordantes y sus p, los AUC de D3 × biblioteca y de D1, la correlación de cada vista con
   la diferencia de BPM y la complementariedad (53, 52 y 45 aciertos; unión 110; 6,76, 2,94 y 5,05 temas en
   común). Los intervalos por remuestreo difieren en el tercer decimal (son Monte Carlo).
2. **Bedrock no confirma la fusión, y tampoco la tumba.** Con MAEST extraído de sus 179 originales:
   - Frente a pares al azar de la biblioteca con el mismo BPM (la medida de la auditoría), **CLAP crudo es lo mejor
     de todas las filas**: AUC 0,695, contra 0,58-0,68 de las vistas y las fusiones; y la pureza de CLAP crudo
     (5,7 % de los 30) no la supera ninguna (3,0-5,6 %).
   - Pero esa medida compara pares de audio de YouTube con pares de la biblioteca. Dentro de Bedrock, con todo el
     audio del mismo origen, las tres noches de Digweed no se distinguen entre sí en ninguna vista (AUC 0,51-0,53),
     así que lo que esa medida separa es «Bedrock frente a la biblioteca», no «el mismo set».
   - En el control sin cambio de dominio que mira lo que importa para D0, temas **consecutivos** frente a temas de
     otra noche del mismo DJ, **los estilos y las fusiones que los incluyen superan a CLAP crudo**: 0,60-0,62
     contra 0,56 (MAEST solo y CLAP con MAEST, 0,58).
   - Lectura: en Bedrock, NO CONCLUYENTE. Es un solo DJ y tres sets.
3. **El control de tempo (E14) no deshace la ventaja en D3 ni en D1.** Los estilos llevan tempo (−0,37 con la
   diferencia de BPM), pero casi todo está en su media: centrarlos lo baja a −0,16, lo mismo que CLAP crudo. La
   variante fijada de antemano para quitar el tempo (sacar las 2 direcciones que lo predicen) no agrega nada
   (−0,15) y da los mismos números que centrar. Con el pool a ±8 BPM la ventaja de las fusiones sigue en D1; a ±4
   BPM no se distingue de CLAP (p ≥ 0,23).
4. **La ecuación aprendida (P12) no supera a los pesos iguales.** Fuera de muestra, dejando cada DJ afuera:
   CLAP crudo 0,650; las tres vistas con pesos iguales 0,779; todos los factores (vistas, BPM, tonalidad, género)
   con pesos iguales 0,788; la ecuación aprendida 0,774. El juez son 63 pares de D3 de 42 DJs que pasan los
   filtros en vivo, más Bedrock (un DJ). Con eso los pesos no se identifican: sus intervalos se tocan entre sí, y
   el de tonalidad y el de CLAP centrado tocan el 0. La calibración no es confiable con este N.
5. **En los sets de Gabriel (aparte; no ajustan nada), con los filtros en vivo,** el tema siguiente cae entre los
   30 primeros en el 11,3 % de las transiciones con CLAP crudo, 15,1 % con las tres vistas (p 0,04) y 16,8 % con
   todos los factores con pesos iguales (p 0,010). La ecuación aprendida da 13,8 % (p 0,24).
6. **El consenso entre vistas (P13) marca seguridad, pero rara vez.** Cuando las tres vistas ponen al candidato
   entre sus 30 primeros, el 99 % de esos pares es de un mismo set profesional (con pesos parejos entre set y
   azar); pasa en el 3,5 % de los pares profesionales. Con una vista o más, 0,86.
7. **No hay juez de 1001Tracklists para los modelos.** En el piloto de 30 sets, el cruce estricto con la
   biblioteca da 5 pares de un mismo set (coincide con la sección 12.8 del plan). Para los modelos, el juez con
   otros DJs es lo de este informe: Bedrock y 79 pares de D3.

**Qué sobrevive para la escucha E12**: la fusión no se descarta (gana en D3 × biblioteca con intervalo que no toca
0, en D1 y en el control de consecutivos de Bedrock; pierde en la medida de Bedrock con cambio de dominio) y la
ecuación aprendida no se distingue de los pesos iguales. Propuesta del agente para E12: la lista de hoy (solo CLAP
crudo) frente a **todos los factores con pesos iguales** (CLAP centrado, MAEST centrado y estilos en z, más BPM,
tonalidad y género con las fórmulas del recomendador, cada uno en desvíos de los pares al azar dentro de los
filtros), con los filtros en vivo. Es una regla fija, sin pesos aprendidos de estos datos. Decide Gabriel.

## Parecidos: vistas y fusión, con el tempo controlado (E14)

Acierto = el tema siguiente de Gabriel entre los 30 primeros; pool = biblioteca menos lo tocado (542 transiciones;
a ±8 BPM, 495; a ±4, 412). Entre paréntesis: transiciones que acierta solo la variante / solo CLAP crudo y p
binomial exacta. AUC D3 = mismo set frente a pares de la biblioteca al azar con cada tema a ±4 BPM; entre
corchetes, la diferencia con CLAP crudo, intervalo del 95 % remuestreando sets. Bedrock, «biblioteca» = la misma
medida; «consec.» = consecutivos frente a otra noche, dentro de Bedrock. Pureza = fracción de los 30 que son temas
posteriores del mismo set (zona_sets, E10b). ρ = Spearman entre el parecido y la diferencia de BPM en 200 000 pares.

| Variante | Acierto D1 | ±8 BPM | ±4 BPM | AUC D3 × biblioteca | AUC Bedrock, biblioteca | Bedrock, consec. | Pureza Bedrock | AUC D1 | ρ |
| :--- | :--- | :--- | :--- | :--- | --: | --: | --: | --: | --: |
| CLAP crudo (hoy) | 9,04 % | 10,51 % | 14,56 % | 0,664 | 0,695 | 0,558 | 5,74 % | 0,574 | −0,16 |
| CLAP centrado | 9,78 % (14/10, 0,54) | 11,31 % (13/9, 0,52) | 14,08 % (10/12, 0,83) | 0,676 [−0,030, +0,055] | 0,650 | 0,571 | 5,49 % | 0,605 | −0,05 |
| MAEST capa 7 centrado | 9,59 % (32/29, 0,80) | 10,30 % (29/30, 1,0) | 13,11 % (25/31, 0,50) | 0,743 [+0,028, +0,131] | 0,620 | 0,580 | 3,73 % | 0,576 | −0,12 |
| Estilos Discogs | 8,30 % (38/42, 0,74) | 9,29 % (36/42, 0,57) | 11,17 % (32/46, 0,14) | 0,756 [+0,024, +0,156] | 0,633 | 0,617 | 3,30 % | 0,546 | −0,37 |
| Estilos centrados | 7,93 % (36/42, 0,57) | 9,09 % (34/41, 0,49) | 11,65 % (34/46, 0,22) | 0,767 [+0,030, +0,170] | 0,582 | 0,623 | 3,02 % | 0,560 | −0,16 |
| Estilos sin tempo | 7,93 % (36/42, 0,57) | 9,09 % (34/41, 0,49) | 11,41 % (33/46, 0,18) | 0,767 [+0,030, +0,170] | 0,581 | 0,622 | 2,99 % | 0,560 | −0,15 |
| z: CLAP c. + MAEST c. | 11,44 % (31/18, 0,085) | 13,33 % (31/17, 0,060) | 16,75 % (27/18, 0,23) | 0,738 [+0,032, +0,116] | 0,650 | 0,580 | 5,00 % | 0,600 | −0,10 |
| z: CLAP c. + estilos | 12,73 % (34/14, 0,006) | 13,94 % (30/13, 0,014) | 16,02 % (22/16, 0,42) | 0,750 [+0,045, +0,129] | 0,680 | 0,612 | 5,58 % | 0,598 | −0,28 |
| z: las tres (P11) | 11,62 % (35/21, 0,081) | 12,93 % (32/20, 0,13) | 16,50 % (29/21, 0,32) | 0,774 [+0,064, +0,154] | 0,674 | 0,605 | 5,21 % | 0,599 | −0,24 |
| z: las tres, estilos centrados | 12,36 % (41/23, 0,033) | 13,74 % (38/22, 0,052) | 16,02 % (30/24, 0,50) | 0,780 [+0,069, +0,162] | 0,647 | 0,613 | 4,58 % | 0,597 | −0,14 |
| z: las tres, estilos sin tempo | 12,36 % (41/23, 0,033) | 13,74 % (38/22, 0,052) | 16,02 % (30/24, 0,50) | 0,780 [+0,068, +0,161] | 0,647 | 0,613 | 4,62 % | 0,597 | −0,14 |
| Consenso: mínimo de percentiles, CLAP c. y MAEST c. | 11,62 % (37/23, 0,093) | 13,13 % (34/21, 0,10) | — | 0,718 | — | — | — | 0,601 | — |
| Consenso, las tres | 11,81 % (43/28, 0,096) | 13,13 % (40/27, 0,14) | — | 0,735 | — | — | — | 0,594 | — |

- Diferencia de acierto en D1 remuestreando sets: CLAP c. + estilos +3,7 puntos [+1,4, +6,6]; las tres +2,6
  [+0,7, +5,1]; CLAP c. + MAEST c. +2,4 [0,0, +4,6]. La auditoría daba [+1,4, +6,3], [+0,8, +5,2] y [−0,2, +4,4].
- Pureza en D1, todas entre 1,2 y 2,0 % con intervalos que se pisan (CLAP crudo 1,8 % [1,2, 2,4]).
- **Control de dominio**: los 8 temas de la biblioteca que son originales de Bedrock (coseno CLAP ≥ 0,97) dan, entre
  su versión de la biblioteca y la de YouTube, coseno 0,97-0,99 en CLAP, 0,95-0,99 en MAEST (uno, 0,43) y 0,91-1,00
  en estilos. El cambio de origen por tema es chico, pero se suma en todos los pares de Bedrock a la vez.
- **Bedrock dentro de su propio audio**: mismo disco frente a otro disco, todas las vistas 0,51-0,53. Consecutivos
  frente a otro disco (171 pares contra 10 632): CLAP crudo 0,557; la diferencia con estilos [+0,003, +0,112], con
  CLAP c. + estilos [+0,022, +0,088], con las tres [+0,013, +0,080]. Estos intervalos se calcularon aparte en esta
  sesión, con las funciones del módulo y sin pesos por set (400 remuestreos de pares, porque con 3 sets no se puede
  por sets): son optimistas.
- **Escala z** (media y desvío del coseno entre pares de la biblioteca): CLAP crudo 0,745 y 0,104; CLAP centrado
  0,001 y 0,251; MAEST centrado 0,000 y 0,155; estilos 0,557 y 0,249.

## La ecuación de factores (P12, aquí P12b)

Pares: cada par de un mismo set con artistas distintos, el que sonó antes como principal. Solo los que pasan los
filtros en vivo respecto del principal (BPM ±20; tonalidad conocida con `best_key_shift` ≥ 0,60). Por cada uno,
20 candidatos al azar de la biblioteca que también pasan los filtros (sin el principal, sin temas de ese set, sin
el mismo artista y, para Bedrock, sin los 8 duplicados). Cada set pesa 1 en positivos y 1 en negativos. Factores
con las fórmulas de `recomendador.py`: vistas en z, `bpm` = 1 − |ΔBPM|/20, `clave` = la regla armónica, `genero`
= la taxonomía; BPM y tonalidad de Traktor cuando los analizó (como en vivo). Un factor desconocido vale la media
de los negativos (no cuenta). Regresión logística, C = 1 fijado antes, factores en desvíos de los negativos.
Bedrock no tiene tonalidad ni género: esos factores no cuentan en sus pares, tampoco en sus negativos.

| Fuente | Pares de un mismo set | Pasan los filtros | Sets | DJs |
| :--- | --: | --: | --: | --: |
| D3 × biblioteca | 79 | 63 | 55 | 42 |
| Bedrock | 5243 | 5041 | 3 | 1 |
| D1 (aparte) | 12 966 | 8824 | 22 | 1 |

AUC fuera de muestra (cada DJ predicho por un modelo entrenado sin él; intervalo por remuestreo de DJs):

| Puntaje | Todas | D3 × biblioteca | Bedrock (entrenado sin él) | Diferencia con CLAP crudo |
| :--- | --: | --: | --: | :--- |
| CLAP crudo (hoy) | 0,650 | 0,649 | 0,665 | — |
| Las tres vistas, pesos iguales (P11) | 0,779 | 0,785 | 0,660 | [+0,072, +0,191] |
| Todos los factores, pesos iguales | 0,788 | 0,795 | 0,659 | [+0,062, +0,228] |
| Ecuación aprendida (P12) | 0,774 | 0,780 | 0,652 | [+0,047, +0,209] |
| Ecuación aprendida, solo vistas | 0,774 | 0,780 | 0,649 | [+0,056, +0,198] |

Pesos de la ecuación ajustada con todos los pares profesionales (coeficiente por desvío; intervalo remuestreando
DJs, 250 veces): MAEST centrado 0,50 [0,18, 1,04]; estilos 0,37 [0,01, 0,98]; BPM 0,36 [0,14, 1,18]; género 0,36
[−0,02, 0,73]; CLAP centrado 0,27 [−0,14, 0,64]; tonalidad 0,08 [−0,30, 0,41]; intercepto −0,72.

Calibración fuera de muestra: Brier 0,194; por deciles de la probabilidad, la fracción de pares del mismo set no
crece pareja (el decil más bajo, p 0,06, tiene 0,24; el de p 0,24, 0,04; el más alto, p 0,79, 0,80).

**D1, aparte** (el modelo se entrena sin D1): AUC 0,587 CLAP crudo, 0,619 las tres vistas, 0,622 todos los
factores con pesos iguales, 0,623 la ecuación aprendida. Acierto en 30 del tema siguiente **con los filtros en
vivo** (398 de las 542 transiciones dejan pasar el tema siguiente; mediana de 1167 candidatos):

| Puntaje | Acierto | Solo variante / solo CLAP crudo, p | Diferencia, intervalo por sets |
| :--- | --: | :--- | :--- |
| CLAP crudo (hoy) | 11,3 % | — | — |
| Las tres vistas, pesos iguales | 15,1 % | 31 / 16, 0,040 | [+0,8, +6,2] |
| Todos los factores, pesos iguales | 16,8 % | 45 / 23, 0,010 | [+2,4, +10,2] |
| Ecuación aprendida | 13,8 % | 34 / 24, 0,24 | [−0,6, +5,4] |

## Consenso entre vistas (P13)

Consenso = cuántas de las tres vistas (CLAP centrado, MAEST centrado, estilos) ponen al candidato entre sus 30
primeros de la biblioteca. Con pesos parejos entre pares del mismo set y pares al azar dentro de los filtros:

| Consenso | Profesionales: del mismo set / del azar | Fracción del mismo set | D1: del mismo set / del azar | Fracción |
| :--- | :--- | --: | :--- | --: |
| 0 | 72,1 % / 96,0 % | 0,43 | 89,4 % / 95,6 % | 0,48 |
| 1 | 20,9 % / 3,6 % | 0,86 | 8,2 % / 3,8 % | 0,69 |
| 2 | 3,6 % / 0,4 % | 0,89 | 2,0 % / 0,6 % | 0,76 |
| 3 | 3,5 % / 0,03 % | 0,99 | 0,4 % / 0,05 % | 0,90 |

## Afirmaciones

| Afirmación | Etiqueta | Evidencia |
| :--- | :--- | :--- |
| Los números de la tabla de vistas de la auditoría se reproducen con el código nuevo | EVIDENCIA OBSERVADA | `fusion_vistas.json`, `vistas`; además, los scripts de la auditoría corridos hoy dan lo mismo |
| La reproducción no es una validación independiente | INFERENCIA | El módulo implementa el mismo método que los scripts de la auditoría, con las funciones de `zona_sets.py` |
| MAEST de los originales de Bedrock está en el mismo espacio que la biblioteca | EVIDENCIA OBSERVADA | `maest_originales.py --verificar 4`: coseno 1,0000 en capa 7 y estilos; 179 de 179, 0 fallos, 11 s por tema |
| En Bedrock, frente al azar de la biblioteca, CLAP crudo es lo mejor | EVIDENCIA OBSERVADA | AUC 0,695 frente a 0,58-0,68; pureza 5,74 % [5,45, 5,99] frente a 2,99-5,58 % |
| Esa medida de Bedrock separa «Bedrock frente a la biblioteca» más que «el mismo set» | INFERENCIA | Dentro de Bedrock, mismo disco frente a otro disco da 0,51-0,53 en todas las vistas |
| Dentro de Bedrock, para temas consecutivos, los estilos y las fusiones que los incluyen superan a CLAP crudo | EVIDENCIA OBSERVADA, exploratoria | 0,60-0,62 frente a 0,557; intervalos remuestreando pares, que con 3 sets son optimistas |
| La fusión es mejor que CLAP crudo en Bedrock | NO CONCLUYENTE | Las dos medidas de Bedrock dicen cosas opuestas; un solo DJ |
| En D3 × biblioteca, MAEST, estilos y las fusiones separan mejor el mismo set que CLAP crudo, con el BPM igualado | EVIDENCIA OBSERVADA | Intervalos por sets que no tocan 0; 79 pares, 48 DJs |
| El tempo de los estilos está casi todo en su media | EVIDENCIA OBSERVADA | ρ −0,37 crudo, −0,16 centrado, −0,15 sin las 2 direcciones de tempo |
| Quitar las 2 direcciones lineales del tempo no cambia nada respecto de centrar | EVIDENCIA OBSERVADA | Mismas cifras al tercer decimal en todas las medidas |
| A ±4 BPM ninguna variante se distingue de CLAP crudo en D1 | EVIDENCIA OBSERVADA | p 0,23 a 0,83 |
| La ecuación aprendida no supera a los pesos iguales | EVIDENCIA OBSERVADA | 0,774 frente a 0,779 y 0,788 fuera de muestra; 13,8 % frente a 15,1 y 16,8 % en D1 en vivo |
| Con estos datos los pesos de la ecuación no se pueden fijar | INFERENCIA | 63 pares de D3 y un DJ con originales; intervalos anchos que se pisan; tonalidad y CLAP centrado tocan 0 |
| Los pesos iguales de todos los factores mejoran la lista en vivo de D1 | EVIDENCIA OBSERVADA en D1, que es referencia con ruido | 16,8 % frente a 11,3 %, 45/23, p 0,010; nada se ajustó con D1 |
| El consenso de las tres vistas marca pares muy probablemente del mismo set | EVIDENCIA OBSERVADA | 0,99 con pesos parejos; 3,5 % de los pares profesionales |
| La fusión o los pesos iguales dan una lista de Parecidos mejor | HIPÓTESIS | La decide E12, escuchando |
| 1001Tracklists no sirve de juez de los modelos | EVIDENCIA OBSERVADA en el piloto | 5 pares de un mismo set en 30 sets (también en la sección 12.8 del plan) |

## Revisión hostil

- **¿Elegí la fila ganadora mirando?** La tabla tiene 13 filas sobre los mismos datos. Las filas de la auditoría
  vienen fijadas; «estilos centrados» y «estilos sin tempo» se fijaron antes de correr, y la proyección de tempo
  (ridge a BPM y BPM², 2 direcciones, α = 1) también. La propuesta para E12 (todos los factores con pesos iguales)
  no tiene parámetros que salgan de estos datos, salvo la escala de cada factor, que sale de pares al azar.
- **¿Bedrock contradice o no?** Contradice en la medida de la auditoría y apoya en el control limpio. La medida de
  la auditoría compara audio de YouTube con audio de la biblioteca; el control limpio compara temas de un mismo DJ
  y un mismo origen. Ninguna de las dos es un juez fuerte con un solo DJ: por eso queda NO CONCLUYENTE.
- **¿El juez de la ecuación es D3?** Sí. Bedrock pesa 3 sets frente a 55 de D3, y es un solo grupo en la
  validación. Con 63 pares positivos, una regresión de 6 factores apenas se sostiene.
- **¿Los filtros cambian algo?** Sacan el 20 % de los pares de D3 (16 de 79), el 4 % de Bedrock y el 32 % de D1
  (son todos los pares de sets largos, no solo los consecutivos). Lo que no pasa los filtros nunca aparece en vivo,
  así que la ecuación se mide donde actúa.
- **¿«Mismo set» es «suena bien mezclado»?** No. Es coherencia de estilo dentro de un set. Lo decide la escucha.
- **¿La aritmética cierra?** Bedrock: 65 + 56 + 58 = 179 originales; los 16 archivos salteados son las 16 mezclas
  continuas (6 + 5 + 5). D3: 63 + 16 = 79. D1: 8824 + 4142 = 12 966. En vivo: 398 de 542.
- **Lo que no miré**: Bedrock con tonalidad y género estimados (sus pares no aportan a esos factores); D7 con su
  control de dominio como tercera fuente de P12b; la calibración con más pares.

## Qué sigue

1. **Escucha E12 (unidad 2)**: unos 10 temas principales; para cada uno, dos listas enteras de 30, a ciegas y en
   orden al azar: CLAP crudo (hoy) frente a todos los factores con pesos iguales, con los filtros en vivo. Hace
   falta que Gabriel diga quién elige los temas principales y si quiere marcar tema por tema lo que no pondría.
2. **P12a** (sección 12.8): los pesos de BPM, tonalidad y género desde los pares consecutivos de 1001Tracklists
   con metadatos de Beatport. Antes, Gabriel decide sobre la señal `ai-train=no` de Beatport.
3. **P12b con más jueces**: D7 validado con su control de dominio, y el audio de los temas de los sets (D6) si
   Gabriel lo aprueba (trabajo pesado: gpuq).
4. Los pesos por defecto del recomendador siguen en solo CLAP hasta que Gabriel diga otra cosa.
