# Auditoría del informe de trayectorias D7 (2026-11-07)

> **Aviso (2026-11-07, noche).** Parte del audio «continua» de D7 es otra mezcla (traktor-github-6e); los
> números de este informe y del auditado están contaminados y se recalculan con lo validado. La adenda del
> final muestra además que el «siguiente» dentro del disco mide continuidad de la mezcla, no elección del
> DJ. Las marcas de abajo son de esa adenda.

Informe auditado: `docs/reports/trayectorias_d7_2026-11-07.md` (commit
`c493ca8005e8437862a4c5c8cabd4c94aebf137b`), con el protocolo `.claude/skills/auditar/SKILL.md`.
Evidencia de esta sesión: `tools/dj_sets/trayectorias_d7.py` leído entero; los artefactos de
`D:\TRAKTOR ML datos\dj_sets\` (`sets/d7_temas.csv`, `sets/d7_clap.npz`, `sets/d7_trayectorias.json` con
`temas_con_vector` 1119, más `audio/d7_*/manifest.json`, `musicbrainz/*.json` y `clap/d7_*/*.npz`);
cinco scripts de auditoría fuera del repo que reproducen la lógica de `e1()` devolviendo el puesto por
transición, miden la alineación de los límites y rehacen los vectores con otros bordes (lo que calculan
está descrito en cada fila). Los tests: 11 de 11 en `tests/v4/test_dj_sets.py`. La otra sesión reintenta
24 archivos en segundo plano; si entra audio nuevo, estos números son los de los 1119 temas de hoy.

## Veredicto

**El informe se sostiene** [REFUTADO 2026-11-07 noche por la adenda: lo reproducido sigue reproducido, pero
el audio de varios discos es otra mezcla y el «siguiente» dentro del disco mide continuidad, no elección]. Todos los números del veredicto y de la tabla salen del JSON y del código:
vecinos, a dos puestos y azar del mismo disco con los tres bordes; los centroides por compilación, DJ y
época con sus n; el banco en los dos pools con sus intervalos; los quintos, el puente y los bordes
reducidos. Además se verificó algo que el informe daba por hecho: **los límites de MusicBrainz caen donde
cambia el audio** (la suma de duraciones y el archivo difieren 1,3 s de mediana, 8,8 s como máximo; la
novedad CLAP en el límite está en el percentil 0,77 de la del disco) [REFUTADO 2026-11-07 noche: la suma
es circular, el archivo se eligió por esa duración; el 0,77 es una media, y discos con otra mezcla dan
0,78-0,83]. Tres cambios de lectura:

1. **CLAP dentro del disco es más firme de lo que dice el informe.** [REFUTADO 2026-11-07 noche por la
   adenda: sobre el mismo audio, límites desplazados media pista dan 41,5 %, N segmentos iguales 34,6 % y
   cortes al azar 37-39 %, frente a 27,1 % con los reales; el apareado mide la continuidad de la mezcla] Con prueba apareada: 241 aciertos
   frente a 159,8 esperados (z = 7,5); supera a su propio azar en 52 de 74 discos (p = 0,0006) y en los
   cuatro DJs. El azar exacto (media de 1 sobre los candidatos) es 17,4 %, no el 16,8 % simulado. El
   exceso va entero al tema inmediato: el más parecido es el siguiente el 25,9 % (azar 17,4 %) y el de dos
   puestos el 16,7 % (azar 17,2 %); después la distancia crece despacio con la separación (0,124 a dos
   puestos, 0,144 a diez o más). «Deriva lenta» vale, con el paso inmediato como el único especial.
2. **La inferencia «la ventaja no es audio de la transición» necesitaba otra prueba, y en parte se
   corrige.** [REEMPLAZADO 2026-11-07 noche: la señal junto al límite es parte del mismo efecto de
   continuidad; con audio mezclado ningún borde mide la elección del siguiente] La diferencia de medianas entre vecinos y a dos puestos no decide: se conserva (0,012)
   con vectores del tercio central aunque el ranking empeore. Con el top-1 dentro del disco: borde 0
   (mezclas incluidas) 31,0 %, borde 30 27,2 %, borde 60 26,2 %, tercio central 22,6 % (apareado contra
   borde 60: 85 frente a 52, p = 0,006). Control de una variable: borde 60 con la misma cantidad de
   bloques que el tercio da 22,9-24,8 % (3 semillas). Lectura: los 60 s junto a cada límite llevan señal
   de la mezcla (31 → 26 %) y el borde de 60 s la saca; más allá, la caída es sobre todo por tener menos
   bloques, y quedan 1-2 puntos sin resolver. El 26,2 % con borde 90 está sostenido por los 245 temas
   que vuelven al borde reducido. Con cualquier borde, CLAP supera al azar (el intervalo del tercio,
   20,0-25,7 %, no toca 17,4 %).
3. **«Momento» empeora solo con todo D7 como pool.** [REEMPLAZADO 2026-11-07 noche: comparación sobre
   una medida contaminada; se repite sobre lo validado] Dentro del disco no se distingue de CLAP: 69 frente
   a 50 transiciones discordantes, p = 0,10; con borde 30 empata (27,2 %). «Historia» tampoco se
   distingue (56 frente a 44, p = 0,27). La vara para T13 sigue siendo CLAP dentro del disco, pero
   «extrapolar el último paso empeora» vale para el pool con la señal de la compilación, no dentro del
   disco.

Precisiones: 0,237 es la distancia entre dos temas de la biblioteca (`azar_biblioteca`), no de un tema
de D7 a la biblioteca; los «4 DJs» son en la práctica tres, porque Nick Warren aporta 3 discos y 30
transiciones (2 de sus 15 compilaciones tienen audio); los discos de una sola pista son 20 en
MusicBrainz, 16 con audio.

## Lo reproducido (borde 60 salvo indicación)

| Dato del informe | Reproducido | Fuente |
|---|---|---|
| 1572 temas, 130 discos, 83 compilaciones; 1119 con vector en 94 discos; 14 con `track_uid` | Sí | `d7_temas.csv`, JSON |
| Vecinos 0,114, a dos 0,124, azar disco 0,131; borde 30: 0,105, 0,114, 0,123; borde 90: 0,121, 0,129, 0,136 | Sí (0,1143, 0,1239, 0,1313…) | JSON `trayectoria`, n = 1010, 931, 2449 |
| Diferencia vecinos − a dos: 0,0093, 0,0096, 0,0088 | Sí | Idem |
| Misma compilación otro disco 0,160; mismo DJ 0,219; otro DJ 0,227; biblioteca 0,237 | Sí; el último es biblioteca-biblioteca | n = 145, 6594, 12 980, 4997 |
| Centroides: misma 0,034; mismo DJ 0,100, 0,091, 0,114; otro DJ 0,098, 0,101, 0,118 | Sí, con n = 24, 91, 237, 488, 231, 493, 1137 | JSON `entre_discos`, 74 discos |
| Pool «disco»: CLAP 26,2 % (23,1-29,6), azar 16,8 % (13,6-18,9), percentil 0,33 frente a 0,50; 920 transiciones, 74 discos; borde 30 27,2 %, borde 90 26,2 % | Sí; azar exacto 17,4 % | JSON `e1` y recálculo propio (241 de 920) |
| Pool «d7»: 41,9 % (37,1-46,5), azar 3,0 %; borde 30 47,4 %; misma compilación 11,8 % en los 30 frente a 1,5 %; «historia» 47,3 % y 13,6 % | Sí | JSON `e1` |
| «Historia» disco 24,9 %; «momento» d7 30,1 % (25,6-34,6), disco 24,1 % | Sí; «momento» disco con borde 30 es 27,2 % y con 90 25,7 % | JSON |
| Quintos: paso 0,123 frente a 0,110-0,114; al primero 0,134, 0,143, 0,151, 0,141, 0,141 | Sí | JSON |
| Puente: 12 temas, 2 primeros, mediana 66 | Sí | JSON `puente_biblioteca` |
| Borde reducido 21, 108, 245; 16 transiciones saltadas | Sí | JSON |
| 16 discos de una sola pista | 20 en MusicBrainz, 16 con audio | `d7_temas.csv` |
| Tests | 11 de 11 | `pytest tests/v4/test_dj_sets.py` |

## Lo nuevo

**Apareado dentro del disco (borde 60, 920 transiciones).**

| Comparación | Resultado |
|---|---|
| CLAP frente al azar exacto | 241 aciertos frente a 159,8 esperados; z = 7,5 |
| Por disco | CLAP por encima de su azar en 52, por debajo en 22, de 74; signo p = 0,0006 |
| CLAP frente a «historia» (top-1) | 56 frente a 44 discordantes; p = 0,27 |
| CLAP frente a «momento» (top-1) | 69 frente a 50; p = 0,10 |
| Por DJ (CLAP / azar) | Carl Cox 23,2 / 15,3 % (470, 30 discos); Cattáneo 28,1 / 18,4 % (221, 19); Sasha 30,7 / 20,7 % (199, 22); Warren 30,0 / 20,2 % (30, 3) |
| Según cuántos quedan (CLAP / azar) | 2-3: 46 / 42 %; 4-6: 30 / 21 %; 7-10: 22 / 12 %; 11 o más: 16 / 7 % |

**A qué puesto apunta el más parecido** (separación respecto del actual; esperado al azar entre paréntesis):
1: 25,9 % (17,4); 2: 16,7 % (17,2); 3: 12,7 % (13,2); 4: 8,8 % (10,6); 5: 6,3 % (8,7); 6 o más: 29,6 %
(33,0). **Distancia mediana por separación** (7478 pares): 1: 0,114; 2: 0,124; 3: 0,126; 4: 0,133; 5:
0,129; 6: 0,134; 7: 0,130; 8: 0,136; 9: 0,132; 10 o más: 0,144.

**Límites de MusicBrainz frente al audio** (76 discos «continua» con audio): |archivo − suma de
duraciones| mediana 1,3 s, p90 6,9 s, máximo 8,8 s; ninguno pasa de 30 s. Novedad CLAP (coseno entre la
media de 2 bloques antes y 2 después) en el bloque del límite, como percentil de la del disco: mediana
0,77, media 0,73; 45 % de los 819 límites por encima de 0,8 y 16 % por debajo de 0,5. El peor disco es
«Communicate» 2 de Sasha (0,44).

**Gradiente con el borde** (top-1 dentro del disco, mismas 920 transiciones; intervalo por discos):

| Vector | Bloques por tema (mediana) | Top-1 |
|---|---|---|
| Borde 0 (todo el tema) | 8 | 31,0 % (28,1-34,1) |
| Borde 30 / 60 / 90 (JSON) | 6 / 5 / 4 | 27,2 / 26,2 / 26,2 % |
| Tercio central (borde = duración/3, mediana 99 s; 72 reducidos) | 3 | 22,6 % (20,0-25,7) |
| Borde 60 con la cantidad de bloques del tercio, al azar (3 semillas) | 3 | 22,9, 24,1, 24,8 % |
| Borde 120 sin borde reducido (316 transiciones) / borde 60 en los mismos temas | — | 24,7 / 29,1 % |
| Borde 150 sin borde reducido (158) / borde 60 en los mismos temas | — | 26,6 / 31,6 % |

Apareado borde 60 frente a tercio: 85 frente a 52, p = 0,006. Las dos últimas filas cambian el borde y
la cantidad de bloques a la vez: no deciden.

**Identidades en el pool «d7»:** 1066 con vector; 42 tienen más de una copia (95 temas), así que tomar
la mejor copia casi no mueve nada.

## Afirmaciones

| Afirmación | Etiqueta | Evidencia |
|---|---|---|
| Los números del informe salen de los artefactos y del código | EVIDENCIA OBSERVADA | Tabla «Lo reproducido» |
| Los límites de MusicBrainz están alineados con el audio de la mezcla [REFUTADO 2026-11-07 noche: suma circular; novedad media; ver adenda] | EVIDENCIA OBSERVADA | Suma frente a archivo ≤ 8,8 s en 76 discos; novedad en el límite, percentil 0,77 |
| Dentro del disco, CLAP elige el siguiente más que el azar, en los 4 DJs y en 52 de 74 discos [REFUTADO 2026-11-07 noche: con límites falsos gana más; es continuidad de la mezcla, no elección del DJ] | EVIDENCIA OBSERVADA | z = 7,5; signo p = 0,0006; tabla por DJ |
| Parte de la ventaja del vecino es audio de los 60 s junto al límite; el borde de 60 s la saca [REEMPLAZADO 2026-11-07 noche: mismo efecto de continuidad] | INFERENCIA | 31,0 % con borde 0 frente a 26,2 % con 60; los bordes 30 y 60 cambian una variable (21 y 108 reducidos) |
| Más allá de 60 s, la caída del top-1 es sobre todo por tener menos bloques | INFERENCIA | Tercio 22,6 % frente a 22,9-24,8 % con los mismos bloques en borde 60 |
| Quedan 1-2 puntos que podrían ser mezcla de más de 60 s [REEMPLAZADO 2026-11-07 noche: la pregunta pierde sentido con audio mezclado] | NO CONCLUYENTE | Diferencia entre el tercio y el control, dentro del ruido de 3 semillas; se decide con un borde fijo y la misma cantidad de bloques en todos los temas, o con los originales (propuesta P2) |
| «Momento» y «historia» no se distinguen de CLAP dentro del disco | EVIDENCIA OBSERVADA | p = 0,10 y 0,27 |
| «Momento» empeora con todo D7 como pool | EVIDENCIA OBSERVADA | Intervalos separados (JSON): 25,6-34,6 frente a 37,1-46,5 % |
| Lo que agrupa es la compilación y no el DJ | INFERENCIA, con tres DJs con peso | Centroides del informe; Warren aporta 3 discos |
| La mezcla junta a los temas de un disco (frente a los originales del piloto) | HIPÓTESIS | Comparación indirecta, como dice el informe; decide una compilación con originales sueltos |
| El 26 % es «poco» | INFERENCIA que depende del pool | Con 11 o más candidatos CLAP dobla al azar (16 frente a 7 %); el margen absoluto es de 9-10 puntos en todos los tamaños |

## Revisión hostil de esta auditoría

- **¿Razoné lo que se podía medir?** El control de bloques se midió (3 semillas); la mezcla de más de
  60 s no se midió directamente: queda NO CONCLUYENTE.
- **¿El control es limpio?** Elige al azar bloques del núcleo de borde 60, que pueden caer junto al
  borde; el tercio toma los del centro. Si la mezcla pasara de 60 s, el control conservaría algo de
  señal y el tercio no, y la diferencia (22,6 frente a 23,9 % de media) mediría eso. Es lo que no se
  resuelve con estos datos.
- **¿La prueba apareada infla la certeza?** Las transiciones de un disco no son independientes; por eso
  se informa también el signo por disco (52 frente a 22), que no depende de eso.
- **¿Comparé la misma cosa?** El azar exacto (17,4 %) y el simulado (16,8 %) difieren por el muestreo
  de 5 semillas: el error típico de una semilla es 1,2 puntos.
- **¿Las anclas de las marcas son únicas?** Comprobado con `grep -c` antes de aplicar.

## Propuestas (no cambian el código de la otra sesión)

- **P1.** En el pool «disco», informar el azar exacto (media de 1 sobre los candidatos) y la prueba
  apareada por disco además del intervalo por remuestreo.
- **P2.** Para la pregunta de la mezcla, comparar bordes con la misma cantidad de bloques por tema (por
  ejemplo, los 3 bloques más centrales en todos los casos) y sin el borde reducido; o medir con una
  compilación que traiga los originales sueltos (lo mismo que pide la HIPÓTESIS del informe).
- **P3.** La vara de T13 es CLAP del tema actual dentro del disco, con discos fuera de la muestra y
  prueba apareada; «momento» no está por debajo de esa vara dentro del disco.
- **P4.** Decir que la lectura DJ frente a compilación se apoya en Cox, Sasha y Cattáneo; reintentar las
  13 compilaciones de Warren sin audio antes de generalizar.
- **P5.** Corregir la frase de la distancia a la biblioteca (es entre dos temas de la biblioteca).

## Adenda de la auditoría (2026-11-07, noche): el 26 % dentro del disco es continuidad de la mezcla

**Disparador.** traktor-github-6e avisó que muchos discos «continua» son otra mezcla: `youtube_buscar.validate`
marca «continua» si la mitad de los discos tiene un archivo de la misma duración (±8 s), y en playlists de
colección con decenas de sets de 75 min eso pasa por azar (Sasha GU013 d2 es GU019 de Digweed; Nick Warren
GU035 Lima d1 es GU018 Amsterdam; Carl Cox Roundhouse es un set de Mind Against; etc.). Con una regla de
título pasan unos 35 de 96 archivos continua; de los por_tema, 5 de 12 tienen títulos que no coinciden. Está
armando `tools/dj_sets/validar_d7.py` y vuelve a correr solo con lo validado.

**Lo que eso tumba de esta auditoría.** La verificación de los límites era circular: la suma de duraciones
de MusicBrainz coincide con el archivo porque el archivo se eligió por esa misma duración. Y el percentil de
novedad 0,77 es una media: discos que el par identificó como otra mezcla dan 0,78 (Communicate d1), 0,82
(GU013 d1) y 0,83 (Roundhouse). La novedad por disco no alcanza para validar.

**La prueba que decide lo que mide el 26 %.** Sobre el mismo audio de los 76 discos «continua» (3 o más
pistas; vectores con borde 60; top-1 del segmento siguiente entre los que quedan):

| Límites | Transiciones | Top-1 | Azar | Discos por encima / por debajo de su azar |
|---|---|---|---|---|
| (a) reales de MusicBrainz | 939 | 27,1 % | 17,6 % | 57 / 19 |
| (b) desplazados media pista (cada segmento es mitad de un tema y mitad del siguiente) | 937 | 41,5 % | 17,6 % | 71 / 5 |
| (c) N segmentos iguales, sin relación con las pistas | 939 | 34,6 % | 17,6 % | 68 / 8 |
| (d) N − 1 cortes al azar, 3 semillas | 844-867 | 37,1-39,3 % | 18,0-18,3 % | 68-71 / 5-8 |

Cualquier segmentación del audio mezclado da un «siguiente» más parecido que el azar, y las falsas lo dan
**más** que la real: dos tramos contiguos de una mezcla se parecen porque son la misma mezcla en el mismo
minuto, no porque el DJ haya elegido ese tema. Los límites reales son justamente donde el audio más cambia,
por eso dan el valor más bajo. Dos controles independientes dicen lo mismo:

| Subconjunto (borde 60) | Discos | Transiciones | CLAP top-1 | Azar |
|---|---|---|---|---|
| Continua con novedad en los límites < 0,6 (audio ajeno probable) | 4 | 44 | 34,1 % | 18,6 % |
| Continua con novedad 0,6-0,7 | 6 | 66 | 24,2 % | 19,0 % |
| Continua con novedad 0,7-0,8 | 27 | 354 | 26,6 % | 16,9 % |
| Continua con novedad ≥ 0,8 | 22 | 293 | 29,4 % | 17,1 % |
| Por_tema (un archivo por pista, sin audio compartido entre vecinos) | 15 | 163 | 18,4 % | 17,9 % |

Donde el audio es ajeno, el «siguiente» gana igual o más; donde cada pista es un archivo aparte, CLAP no
supera al azar (y 5 de esas 12 compilaciones tienen títulos que no coinciden, así que ni ese 18,4 % es
limpio). Coincide con el piloto con los originales de Digweed: consecutivos 0,16 frente a 0,18 al azar.

| Afirmación | Etiqueta | Evidencia |
|---|---|---|
| El top-1 del «siguiente» dentro del disco sobre audio mezclado mide la continuidad de la mezcla, no la elección del DJ | EVIDENCIA OBSERVADA | Límites falsos (b), (c) y (d) dan 34,6-41,5 % frente a 27,1 % con los reales, sobre el mismo audio |
| Los números de D7 (informe y auditoría) sobre compilación, DJ, época, identidades y puente están contaminados por audio ajeno | EVIDENCIA OBSERVADA (del par) | Ejemplos verificados por traktor-github-6e; regla de título: ~35 de 96 archivos continua |
| Con pistas como archivos separados, CLAP del tema actual no elige el siguiente mejor que el azar | NO CONCLUYENTE | 18,4 frente a 17,9 % en 15 discos, 5 de 12 compilaciones con títulos que no coinciden; se decide con los por_tema validados y con compilaciones con originales |
| La novedad CLAP en los límites no sirve sola para validar el audio | EVIDENCIA OBSERVADA | Discos ajenos con percentil 0,78-0,83 |
| «Momento» e «historia» frente a CLAP dentro del disco | REEMPLAZADO | La comparación corría sobre una medida que no mide la elección del DJ; se repite sobre lo validado y, para el «siguiente», sobre vectores por tema de fuentes separadas |

**Propuestas nuevas.**

- **P6.** Toda medida de orden sobre audio mezclado lleva el control de límites falsos (desplazados, iguales,
  al azar) al lado del valor real. Si las falsas empatan o ganan, la medida es continuidad.
- **P7.** Lo que D7 puede medir después de validar: la estructura entre discos y compilaciones (centroides),
  la deriva por hora del set (T11 a nivel de bloque, como D8) y el pool «d7» con la compilación controlada.
  La elección del siguiente tema (T13, E1 por tema) necesita vectores por tema de fuentes separadas:
  por_tema validados, originales en la biblioteca (14 hoy) o compilaciones con los originales sueltos.
- **P8.** Repetir esta auditoría sobre el `d7_trayectorias.json` validado cuando llegue el SHA; los
  scripts están listos.

## Confirmación sobre lo validado (2026-11-07, noche; commit `149754bf8772d81abddcf568627948337b339e1f`)

La revalidación de traktor-github-6e (`tools/dj_sets/validar_d7.py`, `trayectorias_d7.py` con `--validacion`)
se verificó contra `sets/d7_validacion.json`, `d7_trayectorias.json` (límites usables) y
`d7_trayectorias_titulo.json`. Los tests: 15 de 15 en `tests/v4/test_dj_sets.py`.

| Afirmación | Etiqueta | Evidencia |
|---|---|---|
| Todos los números de la sección «Revalidación» del informe D7 salen de los JSON | EVIDENCIA OBSERVADA | 531 temas, 440 transiciones, 32 discos (título: 634, 526, 40); control 24,0 / 42,4 / 35,6 / 40,3-46,4 %; por_tema 22 frente a 18,9 (z 0,8); bordes 30,2 → 20,7 % con azar 20,0 %; McNemar 22/6 p 0,004; compilación 14,5 frente a 3,05 %; centroides y perfiles; Cox 299 temas en 17 discos, Warren 100 en 9, Sasha 81 en 15, Cattáneo 51 en 4 |
| El control de límites falsos se reproduce con una implementación independiente | EVIDENCIA OBSERVADA | Script propio sobre los mismos 24 discos: reales 24,0 %, desplazados 43,0 %, iguales 36,0 %, al azar 40,6-46,0 % (azar exacto 16,7-17,5 %); con todo lo que pasa el título (32 discos): 25,7 / 44,9 / 37,4 / 41,3-45,7 %. Las diferencias con el banco (hasta 0,6 puntos) son de cómo se desplazan y recortan los tramos |
| El validador por título rechaza todos los discos ajenos que motivaron la revisión | EVIDENCIA OBSERVADA | GU013 d1 y d2, GU035 d1 y d2, GU008 d2, Non Stop 2000 d1 y d2, Non Stop 98/01, Roundhouse, Pure Intec 3 y Communicate d1: `valido` falso con el motivo correcto. Communicate d2 (título correcto) pasa, con p de límites 0,99: entra solo en el modo «titulo» |
| Los falsos rechazos del título son pocos y conservadores | INFERENCIA | 21 de los 53 rechazados nombran al DJ; leídos uno a uno, casi todos son otro lanzamiento (GU 010 por GU013, GU 018 por GU035, Masters 19 por 18, Crobar por Space). Dudosos: «F.A.C.T.» CD1 y CD2 frente a «F.A.C.T. (Silver Edition)», y «Masters Series Part 6 Volume 2» frente a «Sequential, Volume 2»: 2-3 discos que podrían ser la misma mezcla con otro nombre |
| La prueba de los límites separa poco | EVIDENCIA OBSERVADA | p mediano 0,31 en los válidos (24 de 32 con p ≤ 0,5) y 0,52 en los rechazados (21 de 44 con p ≤ 0,5). Ocho discos con título correcto tienen p > 0,5 y quedan fuera del modo «limites» |
| La tabla de bordes (0 → 3 bloques centrales) mezcla dos variables: borde y cantidad de bloques | INFERENCIA | Como en la adenda: con la misma cantidad de bloques, borde 60 ya baja a 22,9-24,8 % sobre los datos sin validar. La fila «esa ventaja está en el audio cerca de los cambios» vale como INFERENCIA, no como EVIDENCIA OBSERVADA. No cambia el veredicto: el control de límites falsos decide qué mide la medida |
| Lo que D7 validado sigue diciendo | EVIDENCIA OBSERVADA | La compilación agrupa (14,5 frente a 3,05 % en los 30; centroides 0,049 frente a 0,07-0,11); DJ frente a época NO CONCLUYENTE con 32 discos; «historia» y «momento» no se distinguen de CLAP dentro del disco; «momento» empeora con todo D7 como pool (31,8 frente a 44,7 %) |

Lo que queda abierto está en el informe D7: la elección del siguiente tema se mide con vectores por tema
de archivos separados (por_tema validadas, originales de la biblioteca, compilaciones con originales), y la
hipótesis «comienzos que combinan con finales» se decide en D1 con los originales.
