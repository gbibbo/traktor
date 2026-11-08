# Trayectorias y banco E1 en las compilaciones mezcladas (D7), 2026-11-07

Plan de playlists en vivo, etapa 2, punto 5: T11 y E1 sobre sets de otros DJs
(`docs/plans/recomendacion_playlists_plan.md`, adenda 10). Script `tools/dj_sets/trayectorias_d7.py`.
Datos fuera de Git, en `D:\TRAKTOR ML datos\dj_sets\sets\` (`d7_temas.csv`, `d7_clap.npz`,
`d7_trayectorias.json`).
Auditado el 2026-11-07 (skill `auditar`): `docs/reports/auditoria_trayectorias_d7_2026-11-07.md`. Las marcas
«(auditoría D7)» son suyas.
Adenda de la auditoría (noche): las marcas «(auditoría D7)» sobre el top-1 dentro del disco y sobre los
límites quedan refutadas en `auditoria_trayectorias_d7_2026-11-07.md`:
- sobre el mismo audio, límites falsos (desplazados, iguales, al azar) dan un «siguiente» más parecido
  (34,6-41,5 %) que los reales (27,1 %);
- el 26 % es continuidad de la mezcla, no elección del DJ;
- la verificación de los límites era circular;
- con pistas como archivos separados (por_tema), CLAP no supera al azar: 18,4 frente a 17,9 %.

## Revalidación: solo los archivos que son la compilación (2026-11-07, después de la auditoría)

**Qué pasó.** Muchos discos enteros («continua») eran otra mezcla. `youtube_buscar.py` daba por buena una
playlist si la mitad de los discos tenía un archivo de la misma duración (±8 s). En una playlist de
colección, con decenas de sets de unos 75 minutos, eso pasa por azar. Algunos ejemplos:
- el disco 2 de Sasha «GU013 Ibiza» era «John Digweed - GU 019 Los Angeles CD1»;
- el disco 1 de Nick Warren «GU035 Lima» era «GU 018 Amsterdam (CD2)»;
- Carl Cox «at the Roundhouse» era «EXIT 2023 Mind Against»;
- «Pure Intec 3» era «Mad Mickey 2 After Workout»;
- en «GU008 Brazil», los dos discos duran lo mismo y quedaron cruzados.

`tools/dj_sets/validar_d7.py` revisa cada archivo con dos pruebas:
- **el título**, que decide si el archivo es la compilación;
- **los límites con el audio**, que dicen si se pueden usar para separar los temas.

Resultado:
- de los 96 archivos de discos enteros, 43 son la compilación y 35 tienen límites usables;
- de las 12 compilaciones bajadas tema por tema, 7 pasan.

`trayectorias_d7.py` usa ahora solo lo validado. Con límites usables quedan 531 temas con vector y 440
transiciones en 32 discos. Aceptando todo lo que pasa el título (`--validacion titulo`), 634 temas y 526
transiciones en 40 discos.

### Veredicto nuevo

**Con audio mezclado, D7 no mide qué tema elige el DJ después.** Sobre el mismo audio de los discos
enteros validados (24 discos con límites usables, unas 317 transiciones), el más parecido es el tramo
siguiente:

| Límites | Acierto | Azar exacto | z |
| :--- | --: | --: | --: |
| Reales (MusicBrainz) | 24,0 % | 16,7 % | 3,7 |
| Desplazados media pista | 42,4 % | 16,7 % | 13,0 |
| En partes iguales | 35,6 % | 16,7 % | 9,6 |
| Al azar (3 semillas) | 40,3-46,4 % | 17,5 % | 11,0-13,4 |

- Los límites falsos ganan más que los reales: lo que se mide es la continuidad de la mezcla. Con límites
  falsos, dos tramos vecinos suelen ser partes del mismo tema.
- Lo mismo con todo lo que pasa el título: reales 25,7 %; falsos 36,9-45,5 %.
- Coincide con la adenda de la auditoría, que lo midió sobre los 76 discos sin validar.

**Con las pistas en archivos separados, CLAP no supera al azar.** Solo las «por_tema» validadas: 17,9 %
frente a 15,3 % (22 aciertos frente a 18,9 esperados, z = 0,8; 8 discos, 123 transiciones). Es poco, pero
coincide con el piloto con originales: 0,16 frente a 0,18 entre vecinos y al azar del mismo disco.

[REEMPLAZADO 2026-11-07 por el control de límites falsos, arriba: estos números son ciertos, pero miden la
continuidad de la mezcla y no la elección del DJ] **CLAP dentro del disco sigue superando al azar, pero
menos de lo que se informó.**
- Con borde 60: 22,5 % de aciertos frente a 16,5 % de azar exacto.
- Prueba apareada: 99 aciertos frente a 72,7 esperados (z = 3,6). Supera al azar en 22 de 32 discos
  (p = 0,05).
- Aceptando todo lo que pasa el título: 24,0 % frente a 17,0 % (z = 4,6; 29 de 40 discos, p = 0,006).

**La ventaja sale casi toda del audio cerca de las transiciones** (con los límites reales; el control de
arriba dice que ni eso es la elección del DJ). Comparación sobre los mismos 388 temas
y las mismas 242 transiciones (azar 20,0 %):

| Bloques usados | Acierto | z contra el azar |
| :--- | --: | --: |
| Borde 0 | 30,2 % | 4,2 |
| Borde 30 | 27,3 % | 3,0 |
| Borde 60 | 23,6 % | 1,5 |
| Borde 90 | 23,1 % | 1,3 |
| Los 3 bloques centrales | 20,7 % | 0,3 |

- Bordes 0 y 30 superan a borde 60 en la prueba apareada (p = 0,004 y 0,02).
- Aceptando todo lo que pasa el título (483 temas, 316 transiciones), los 3 bloques centrales dan 23,7 %
  (z = 1,7; 22 de 40 discos por encima, p = 0,64).
- Lectura (INFERENCIA): el centro de un tema mezclado casi no dice cuál viene después; lo que lo dice es
  lo que suena junto al cambio.
- Con audio mezclado no se puede separar la mezcla en sí (las dos pistas suenan a la vez) de que el DJ
  elija un tema cuyo comienzo combina con el final del anterior. Eso se decide con originales, en los
  sets de Gabriel (D1): el CLAP del tema entero contra el del final del actual frente al comienzo del
  candidato.

**La compilación sigue agrupando.**
- Entre los 30 primeros de todo D7, el 14,5 % es de la misma compilación, contra el 3,0 % entre los
  candidatos (4,8 veces).
- Los dos discos de una misma compilación quedan a 0,049 (8 pares).
- **DJ frente a época: NO CONCLUYENTE.** Con 32 discos no hay un orden:
  - mismo DJ: 0,108, 0,071 y 0,101 con 0-1, 2-5 y 6 años o más de diferencia (n = 15, 32 y 94 pares);
  - otro DJ: 0,080, 0,101 y 0,108 (n = 37, 97 y 213).

**Los perfiles siguen sin distinguirse de CLAP dentro del disco.**
- «Historia»: 21 frente a 18 discordantes, p = 0,75.
- «Momento»: 28 frente a 29, p = 1,0.
- Con todo D7 como pool, «momento» empeora: 31,8 % frente a 44,7 % entre los 30 primeros.

**Composición.** De los 531 temas, Carl Cox aporta 299 (17 discos), Nick Warren 100, Sasha 81 y Cattáneo
51. Lo de D7 habla sobre todo de Carl Cox.

**Puente con la biblioteca.** Quedan 6 temas en común con vector: NO CONCLUYENTE.

| Afirmación | Etiqueta | Evidencia |
|---|---|---|
| 53 de los 96 archivos de discos enteros no son la compilación, y 5 de las 12 compilaciones tema por tema tienen títulos que no coinciden con las pistas | EVIDENCIA OBSERVADA | `sets/d7_validacion.json`. Ejemplos en el texto. En las tema por tema: «Mind Over Machine» 0/19, «Never Say Never» 0/12, «Invol2ver» 2/9 |
| La prueba de los límites separa poco | EVIDENCIA OBSERVADA | p mediano 0,31 en los que pasan el título y 0,52 en los rechazados. Hay discos con el título correcto y p de 0,95 a 1,0 («Communicate» disco 2, «Funky Deep ’n’ Tribal»). Por eso decide el título y los límites solo se usan cuando p ≤ 0,5 |
| Dentro del disco, CLAP supera al azar [REEMPLAZADO 2026-11-07 por el control de límites falsos: es continuidad de la mezcla, no elección del DJ] | EVIDENCIA OBSERVADA | `d7_trayectorias.json`, `e1/disco/clap`, borde 60: 99 frente a 72,7 esperados, z = 3,6. `d7_trayectorias_titulo.json`: 126 frente a 89,2, z = 4,6 |
| Esa ventaja está en el audio cerca de los cambios | EVIDENCIA OBSERVADA | `comparacion_bordes`, sobre los mismos temas y candidatos: borde 0 30,2 %, 3 bloques centrales 20,7 %, azar 20,0 %. McNemar borde 0 frente a 60: 22 contra 6, p = 0,004 |
| Que eso venga de la mezcla o de que el DJ elige comienzos que combinan con el final | HIPÓTESIS | Se decide en D1 con los originales: CLAP del final del tema actual frente al comienzo del candidato, contra el tema entero, en E1. Con audio mezclado no se decide |
| Sobre audio mezclado, límites falsos dan un «siguiente» más parecido que los reales | EVIDENCIA OBSERVADA | `control_limites_falsos`, borde 60, discos validados: reales 24,0 %, desplazados 42,4 %, iguales 35,6 %, al azar 40,3-46,4 % |
| Con pistas en archivos separados, CLAP no supera al azar dentro del disco | NO CONCLUYENTE | `solo_por_tema`: 17,9 % frente a 15,3 %, z = 0,8, con 8 discos y 123 transiciones. No alcanza para descartar un efecto chico |
| Vecinos 0,133, a dos puestos 0,142, al azar del mismo disco 0,143 (borde 60) | EVIDENCIA OBSERVADA | `trayectoria`; n = 483, 448 y 1162. Solo el vecino inmediato queda más cerca |
| La misma compilación agrupa: 4,8 veces más temas suyos entre los 30 primeros que entre los candidatos | EVIDENCIA OBSERVADA | `e1/d7/clap`: 14,5 % contra 3,05 % |

### Qué cambia (además de «Qué cambia en el plan», abajo)

1. **El banco D7 se corre solo con lo validado.** Es el valor por defecto de `trayectorias_d7.py`.
   `youtube_buscar.py` (versión 4) ya exige el título además de la duración.
2. **Toda medida de orden sobre audio mezclado va con el control de límites falsos** (auditoría,
   adenda P6). Si los falsos empatan o ganan, lo medido es la continuidad de la mezcla.
3. **El «siguiente» (T13 y E1 tema a tema) se mide con vectores por tema de archivos separados:**
   - las «por_tema» validadas;
   - los originales de la biblioteca;
   - compilaciones que traigan los originales sueltos.

   Los discos enteros validados sirven para otras cosas: centroides de compilación, deriva por hora a
   nivel de bloque (como D8) y el pool «d7» con la compilación controlada.
4. **Línea nueva: comienzos y finales.** Medir en D1, con originales, si el final del tema actual y el
   comienzo del candidato predicen mejor que el tema entero.

## Veredicto de la primera corrida (sin validar los archivos)

[REEMPLAZADO 2026-11-07 por la revalidación de arriba: este veredicto y la tabla «Datos» se midieron con
los 96 archivos de discos enteros, y 53 no eran la compilación. Los números válidos son los de arriba.]

**Dentro de un disco, CLAP ayuda a elegir el tema siguiente, pero poco.** Entre los temas que quedan
del disco, el más parecido al actual es el que el DJ puso después el 26-27 % de las veces. Al azar es
el 17 %, y los intervalos por remuestreo de discos no se tocan. El tema siguiente queda apenas más cerca
que uno al azar del mismo disco: 0,114 contra 0,131 de distancia coseno, con el de dos puestos más allá
en el medio (0,124). [REFUTADO 2026-11-07 por la auditoría D7 y la revalidación: con los mismos temas, la
ventaja cae al agrandar el borde y casi desaparece con los 3 bloques centrales] Esa ventaja no cambia al
sacar 30, 60 o 90 s de cada borde. La trayectoria es
entonces una deriva lenta dentro de una zona, no una sucesión de saltos.

**La señal más grande de D7 es la compilación misma, no el DJ ni la época.** [REEMPLAZADO 2026-11-07
por la revalidación: la compilación sigue agrupando, pero DJ frente a época queda NO CONCLUYENTE con los 32
discos validados]
- Distancia entre centroides de discos: los dos discos de una misma compilación quedan a 0,034. Dos
  compilaciones del mismo DJ con 0-1 años de diferencia quedan a 0,100, y dos de distinto DJ con 0-1
  años, a 0,098.
- La época suma un poco: con 6 años o más de diferencia, 0,114 y 0,118.
- Por eso el banco con toda D7 como pool (el tema siguiente entre los 30 primeros el 42-47 % de las
  veces) está inflado: entre los 30 primeros hay 8 veces más temas de la misma compilación que entre los
  candidatos.
- El resultado que vale para las playlists es el del pool «disco». Ese pool se parece al caso de
  Gabriel: elegir dentro del propio cajón.

**Dos perfiles probados no mejoran a «el más parecido al tema actual».**
- **«Momento»**, el tema actual más el último paso (un T13 ingenuo), empeora en los dos pools
  [REEMPLAZADO 2026-11-07 por la auditoría D7: empeora con todo D7 como pool (intervalos separados);
  dentro del disco no se distingue de CLAP: 69 frente a 50 transiciones discordantes, p = 0,10, y con
  borde 30 empata, 27,2 %]: los DJs no siguen la dirección del último paso.
- **«Historia»**, la media de los últimos 3 temas, no cambia nada dentro del disco. Mejora con toda D7,
  pero sube a la vez la parte de la misma compilación entre los 30 primeros, así que esa mejora es
  probablemente la señal de la compilación.

**D7 no puede decir nada sobre la parte introspectiva.** Los discos duran 75 min de media. Para lo que
pasa después de la segunda hora hacen falta sets largos (D8, los de Gabriel). Tampoco sirve como puente
con la biblioteca: solo 14 de los 1572 temas coinciden por artista y título.

## Datos

| Afirmación | Etiqueta | Evidencia |
|---|---|---|
| Se bajaron 300 archivos (118 h, 9,1 GB) de 70 de las 83 compilaciones validadas | EVIDENCIA OBSERVADA | `bajar_d7.py`: 67 completas, 3 incompletas y 13 sin audio. Faltan 24 archivos (26 h): YouTube pidió confirmar que no es un bot (22), un 403 y un video no disponible |
| Los 300 tienen CLAP: 117,9 h, 14 197 bloques, todos finitos, duración igual a la del manifest (±10 s) | EVIDENCIA OBSERVADA | Verificación de esta sesión sobre `clap/d7_*/*.npz`. La primera corrida de `clap_blocks.py` murió sin mensaje después de 238 archivos; la segunda completó los 62 restantes (exit 0) |
| CLAP tardó 0,6-0,9 min por hora de audio en los discos enteros | EVIDENCIA OBSERVADA | Log de `clap_blocks.py`. Las pistas sueltas de 2-6 min, 1,1-2,3 min por hora (el costo fijo por archivo) |
| Almacén D7: 1572 temas en 130 discos; 1119 temas con vector, en 94 discos | EVIDENCIA OBSERVADA | `d7_temas.csv`. Sin vector: los discos sin audio (13 compilaciones sin bajar, discos sueltos que fallaron) y 28 pistas sueltas que faltan |
| 16 discos son un solo tema en MusicBrainz (la mezcla sin lista de temas) (auditoría D7: 20 en MusicBrainz, 16 con audio) | EVIDENCIA OBSERVADA | `d7_temas.csv`, discos «continua» con una pista. Sirven solo como set crudo (D8) |
| Los vectores salen del audio mezclado, no de los originales | EVIDENCIA OBSERVADA | `trayectorias_d7.py`: media de los bloques de 30 s enteros dentro del tema, lejos de los cambios (borde de 30, 60 o 90 s). Los temas cortos usan los bloques con el centro dentro (21, 108 y 245 temas según el borde). A diferencia del piloto (Bedrock), estas compilaciones no traen los originales sueltos |
| Vecinos 0,114, a dos puestos 0,124, al azar del mismo disco 0,131 (borde 60; mediana de la distancia coseno) | EVIDENCIA OBSERVADA | `d7_trayectorias.json`, `trayectoria`. n = 1010, 931 y 2449. Con borde 30: 0,105, 0,114 y 0,123; con borde 90: 0,121, 0,129 y 0,136 |
| La ventaja de los vecinos no es audio de la transición filtrado en el vector | INFERENCIA | La diferencia entre vecinos y a dos puestos es la misma con los tres bordes (0,0093, 0,0096 y 0,0088). Si fuera filtración, se achicaría al agrandar el borde. No descarta transiciones de más de 90 s. (auditoría D7: la diferencia de medianas no decide, porque se conserva aun cuando el ranking empeora; con el top-1 dentro del disco: borde 0 31,0 %, borde 60 26,2 %, tercio central 22,6 %; borde 60 con la misma cantidad de bloques que el tercio, 22,9-24,8 %: los 60 s junto al límite llevan señal de la mezcla y el borde de 60 s la saca; más allá, la caída es sobre todo por tener menos bloques; quedan 1-2 puntos sin resolver. El 26,2 % con borde 90 está sostenido por los 245 temas con borde reducido) |
| La mezcla junta a los temas de un disco | INFERENCIA | Con los originales, en el piloto, dos temas al azar del mismo disco quedaban a 0,168-0,185. Con el audio mezclado de estas compilaciones quedan a 0,131. Son discos y DJs distintos (Digweed en el piloto), así que la comparación es indirecta |
| Otro disco de la misma compilación queda a 0,160; otra compilación del mismo DJ, a 0,219; otro DJ, a 0,227; la biblioteca de Gabriel, a 0,237 (auditoría D7: 0,237 es entre dos temas de la biblioteca, `azar_biblioteca`, no de un tema de D7 a la biblioteca) | EVIDENCIA OBSERVADA | `trayectoria`, borde 60, 20 000 pares al azar |
| Centroides de discos: misma compilación 0,034; mismo DJ 0,100, 0,091 y 0,114 (0-1, 2-5 y 6+ años); otro DJ 0,098, 0,101 y 0,118 | EVIDENCIA OBSERVADA | `entre_discos`, borde 60; 74 discos con 3 temas o más; n = 24, 91, 237, 488, 231, 493 y 1137 pares |
| Lo que agrupa en CLAP es la compilación, no el DJ; la época suma poco | INFERENCIA | Con la misma diferencia de años, el mismo DJ y otro DJ quedan iguales (0,100 contra 0,098). Pasar de 0-1 a 6+ años suma 0,013-0,020, y la misma compilación queda tres veces más cerca |
| Lo de la compilación es la cadena de mezcla y masterización y no la selección | HIPÓTESIS | Se decide comparando los originales de los temas de una misma compilación con su audio mezclado (como en el piloto), o con compilaciones que traigan los originales sueltos |
| Pool «disco» (lo que queda del disco): CLAP acierta el siguiente en primer lugar el 26,2 %, contra el 16,8 % al azar | EVIDENCIA OBSERVADA | `e1`, borde 60. 920 transiciones en 74 discos; intervalos del 95 % por discos: 23,1-29,6 % y 13,6-18,9 %. Percentil mediano 0,33 frente a 0,50. Con borde 30, 27,2 %; con borde 90, 26,2 %. (auditoría D7: el azar exacto, media de 1 sobre los candidatos, es 17,4 %; apareado, 241 aciertos frente a 159,8 esperados, z = 7,5; CLAP supera a su azar en 52 de 74 discos, p = 0,0006, y en los 4 DJs) |
| Pool «d7» (todo D7 menos lo tocado, unas 1000 identidades): CLAP pone el siguiente entre los 30 primeros el 41,9 % (azar 3,0 %) | EVIDENCIA OBSERVADA | `e1`, borde 60; 994 transiciones, intervalo por discos 37,1-46,5 %. Con borde 30, 47,4 % |
| Ese 42 % está inflado por la compilación | EVIDENCIA OBSERVADA | Entre los 30 primeros, el 11,8 % es de la misma compilación (con «historia», el 13,6 %), contra el 1,5 % entre los candidatos. No se compara con el 9 % de los sets de Gabriel (`e1_sets_gabriel_2026-11-07.md`), donde todo el pool es su biblioteca |
| «Historia» (los últimos 3, pesos 1, 0,5 y 0,25) no mejora dentro del disco | EVIDENCIA OBSERVADA | Pool «disco», borde 60: 24,9 % contra 26,2 % de CLAP; los intervalos se superponen |
| En todo D7, «historia» mejora el acierto en los 30 (47,3 % contra 41,9 %) por la señal de la compilación | INFERENCIA | Sube a la vez la parte de la misma compilación entre los 30 (13,6 % contra 11,8 %), y dentro del disco no mejora. No se hizo la prueba apareada |
| «Momento» empeora en los dos pools [REEMPLAZADO 2026-11-07 por la auditoría D7: solo con todo D7 como pool; dentro del disco no se distingue de CLAP, 69 frente a 50 discordantes, p = 0,10] | EVIDENCIA OBSERVADA | «d7»: 30,1 % (intervalo 25,6-34,6 %) contra 41,9 % (37,1-46,5 %). «disco»: 24,1 % contra 26,2 % |
| Según la posición en el disco, el paso es algo mayor en el primer quinto (0,123 contra 0,110-0,114) y la distancia al primer tema crece hasta la mitad y después se estabiliza (0,134, 0,143, 0,151, 0,141, 0,141) | EVIDENCIA OBSERVADA | `vecinos_por_quinto_del_disco` y `distancia_al_primero_por_quinto`, borde 60, unos 200 pasos por quinto. Sin intervalos: no concluyente como forma del set |
| D7 no informa sobre la parte después de la segunda hora | NO CONCLUYENTE | Los discos duran 75 min de media y cada uno es una mezcla aparte. Hace falta D8 (sets crudos de 3 h o más) o los sets largos de Gabriel |
| Puente con la biblioteca: 12 temas en común con vector, original de la biblioteca primero en 2 (mediana del puesto 66 de 1760) | NO CONCLUYENTE | `puente_biblioteca`, borde 60. n = 12 es muy poco. El piloto, con más temas en común, encontró el original en el 79-86 % por tema dentro del disco |
| 16 transiciones se saltaron porque el siguiente ya había sonado en el disco con el mismo nombre (otra versión) | EVIDENCIA OBSERVADA | `e1`, `repetidas_saltadas`. Los temas repetidos entre compilaciones cuentan una vez (artista principal y título sin paréntesis) |
| (auditoría D7) Los límites de MusicBrainz caen donde cambia el audio [REEMPLAZADO 2026-11-07 por `validar_d7.py`: el percentil 0,77 promedia discos de la compilación con discos ajenos; disco por disco, 53 de 96 archivos no son la compilación] | EVIDENCIA OBSERVADA | En los 76 discos «continua» con audio, la suma de las duraciones de MusicBrainz y la del archivo difieren 1,3 s de mediana y 8,8 s como máximo; la novedad CLAP en el bloque del límite está en el percentil 0,77 de la del disco (819 límites) |
| (auditoría D7) Dentro del disco, el exceso de aciertos va entero al tema inmediato: el más parecido es el siguiente el 25,9 % (azar 17,4 %) y el de dos puestos el 16,7 % (azar 17,2 %); la distancia crece despacio con la separación (0,124 a dos puestos, 0,144 a diez o más) | EVIDENCIA OBSERVADA | `d7_clap.npz` borde 60; 920 transiciones y 7478 pares |
| (auditoría D7) «Historia» tampoco se distingue de CLAP dentro del disco | EVIDENCIA OBSERVADA | 56 frente a 44 discordantes, p = 0,27 |

## Revisión hostil

- **Pool «disco».** ¿Sale el 26 % de que los discos son cortos? No: el azar se mide sobre el mismo pool
  y da el 17 %, que es la media de 1/(temas que quedan). El percentil mediano (0,33 frente a 0,50) dice lo
  mismo sin depender del tamaño.
- **¿Se eligió el borde mirando el resultado?** Se informan los tres. El veredicto vale con cualquiera
  (dentro del disco, 26-27 % frente a 17 %).
- **La señal de la compilación pudo comprarse con ceguera.** En el pool «d7», un tema de la misma
  compilación que no es el siguiente cuenta como error. Eso baja el acierto: no lo sube. Lo que sube el
  acierto es que los candidatos de otras compilaciones quedan lejos, y eso es justamente la señal de la
  compilación.
- **¿Son independientes las transiciones?** No: los intervalos remuestrean discos enteros. Los 74 discos
  vienen de 4 DJs. Lo del DJ frente a la compilación está medido con esos 4. (auditoría D7: Nick Warren
  aporta 3 discos y 30 transiciones, porque solo 2 de sus 15 compilaciones tienen audio; en la práctica
  son Cox, Sasha y Cattáneo.)

## Qué cambia en el plan

1. **E1 sobre D7 se mide en el pool «disco»**, o en uno que controle la compilación. El pool de toda D7
   mide sobre todo de qué compilación es el tema.
2. **T13 (`siguiente`) tiene que superar a CLAP del tema actual en el pool «disco»**, con discos fuera
   de la muestra. Extrapolar el último paso ya empeora (auditoría D7: con todo D7 como pool; dentro del disco no se
   distingue de CLAP). Lo próximo a probar es un modelo aprendido (lineal
   o de vecinos sobre los pasos de otros discos).
3. **T7 (introspectiva) y la parte tardía de los sets salen de D8 y de los sets largos de Gabriel**, no
   de D7.
4. **Más datos.**
   - Reintentar los 24 archivos que faltan (26 h) más despacio. Si YouTube sigue pidiendo la
     confirmación, hacen falta las cookies del navegador de Gabriel: lo decide él.
   - Sumar las compilaciones de los otros 33 artistas de su lista.
   - Para el puente con la biblioteca, emparejar con más tolerancia (sin «Original Mix», remixes)
     antes de usarlo.
5. **Para probar si la mezcla junta los temas** (HIPÓTESIS de arriba): compilaciones que traigan los
   originales sueltos, como las de Bedrock del piloto.
