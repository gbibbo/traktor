# Trayectorias y banco E1 en las compilaciones mezcladas (D7), 2026-11-07

Plan de playlists en vivo, etapa 2, punto 5: T11 y E1 sobre sets de otros DJs
(`docs/plans/recomendacion_playlists_plan.md`, adenda 10). Script `tools/dj_sets/trayectorias_d7.py`.
Datos fuera de Git, en `D:\TRAKTOR ML datos\dj_sets\sets\` (`d7_temas.csv`, `d7_clap.npz`,
`d7_trayectorias.json`).

## Veredicto

**Dentro de un disco, CLAP ayuda a elegir el tema siguiente, pero poco.** Entre los temas que quedan
del disco, el más parecido al actual es el que el DJ puso después el 26-27 % de las veces. Al azar es
el 17 %, y los intervalos por remuestreo de discos no se tocan. El tema siguiente queda apenas más cerca
que uno al azar del mismo disco: 0,114 contra 0,131 de distancia coseno, con el de dos puestos más allá
en el medio (0,124). Esa ventaja no cambia al sacar 30, 60 o 90 s de cada borde. La trayectoria es
entonces una deriva lenta dentro de una zona, no una sucesión de saltos.

**La señal más grande de D7 es la compilación misma, no el DJ ni la época.**
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
- **«Momento»**, el tema actual más el último paso (un T13 ingenuo), empeora en los dos pools: los DJs
  no siguen la dirección del último paso.
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
| 16 discos son un solo tema en MusicBrainz (la mezcla sin lista de temas) | EVIDENCIA OBSERVADA | `d7_temas.csv`, discos «continua» con una pista. Sirven solo como set crudo (D8) |
| Los vectores salen del audio mezclado, no de los originales | EVIDENCIA OBSERVADA | `trayectorias_d7.py`: media de los bloques de 30 s enteros dentro del tema, lejos de los cambios (borde de 30, 60 o 90 s). Los temas cortos usan los bloques con el centro dentro (21, 108 y 245 temas según el borde). A diferencia del piloto (Bedrock), estas compilaciones no traen los originales sueltos |
| Vecinos 0,114, a dos puestos 0,124, al azar del mismo disco 0,131 (borde 60; mediana de la distancia coseno) | EVIDENCIA OBSERVADA | `d7_trayectorias.json`, `trayectoria`. n = 1010, 931 y 2449. Con borde 30: 0,105, 0,114 y 0,123; con borde 90: 0,121, 0,129 y 0,136 |
| La ventaja de los vecinos no es audio de la transición filtrado en el vector | INFERENCIA | La diferencia entre vecinos y a dos puestos es la misma con los tres bordes (0,0093, 0,0096 y 0,0088). Si fuera filtración, se achicaría al agrandar el borde. No descarta transiciones de más de 90 s |
| La mezcla junta a los temas de un disco | INFERENCIA | Con los originales, en el piloto, dos temas al azar del mismo disco quedaban a 0,168-0,185. Con el audio mezclado de estas compilaciones quedan a 0,131. Son discos y DJs distintos (Digweed en el piloto), así que la comparación es indirecta |
| Otro disco de la misma compilación queda a 0,160; otra compilación del mismo DJ, a 0,219; otro DJ, a 0,227; la biblioteca de Gabriel, a 0,237 | EVIDENCIA OBSERVADA | `trayectoria`, borde 60, 20 000 pares al azar |
| Centroides de discos: misma compilación 0,034; mismo DJ 0,100, 0,091 y 0,114 (0-1, 2-5 y 6+ años); otro DJ 0,098, 0,101 y 0,118 | EVIDENCIA OBSERVADA | `entre_discos`, borde 60; 74 discos con 3 temas o más; n = 24, 91, 237, 488, 231, 493 y 1137 pares |
| Lo que agrupa en CLAP es la compilación, no el DJ; la época suma poco | INFERENCIA | Con la misma diferencia de años, el mismo DJ y otro DJ quedan iguales (0,100 contra 0,098). Pasar de 0-1 a 6+ años suma 0,013-0,020, y la misma compilación queda tres veces más cerca |
| Lo de la compilación es la cadena de mezcla y masterización y no la selección | HIPÓTESIS | Se decide comparando los originales de los temas de una misma compilación con su audio mezclado (como en el piloto), o con compilaciones que traigan los originales sueltos |
| Pool «disco» (lo que queda del disco): CLAP acierta el siguiente en primer lugar el 26,2 %, contra el 16,8 % al azar | EVIDENCIA OBSERVADA | `e1`, borde 60. 920 transiciones en 74 discos; intervalos del 95 % por discos: 23,1-29,6 % y 13,6-18,9 %. Percentil mediano 0,33 frente a 0,50. Con borde 30, 27,2 %; con borde 90, 26,2 % |
| Pool «d7» (todo D7 menos lo tocado, unas 1000 identidades): CLAP pone el siguiente entre los 30 primeros el 41,9 % (azar 3,0 %) | EVIDENCIA OBSERVADA | `e1`, borde 60; 994 transiciones, intervalo por discos 37,1-46,5 %. Con borde 30, 47,4 % |
| Ese 42 % está inflado por la compilación | EVIDENCIA OBSERVADA | Entre los 30 primeros, el 11,8 % es de la misma compilación (con «historia», el 13,6 %), contra el 1,5 % entre los candidatos. No se compara con el 9 % de los sets de Gabriel (`e1_sets_gabriel_2026-11-07.md`), donde todo el pool es su biblioteca |
| «Historia» (los últimos 3, pesos 1, 0,5 y 0,25) no mejora dentro del disco | EVIDENCIA OBSERVADA | Pool «disco», borde 60: 24,9 % contra 26,2 % de CLAP; los intervalos se superponen |
| En todo D7, «historia» mejora el acierto en los 30 (47,3 % contra 41,9 %) por la señal de la compilación | INFERENCIA | Sube a la vez la parte de la misma compilación entre los 30 (13,6 % contra 11,8 %), y dentro del disco no mejora. No se hizo la prueba apareada |
| «Momento» empeora en los dos pools | EVIDENCIA OBSERVADA | «d7»: 30,1 % (intervalo 25,6-34,6 %) contra 41,9 % (37,1-46,5 %). «disco»: 24,1 % contra 26,2 % |
| Según la posición en el disco, el paso es algo mayor en el primer quinto (0,123 contra 0,110-0,114) y la distancia al primer tema crece hasta la mitad y después se estabiliza (0,134, 0,143, 0,151, 0,141, 0,141) | EVIDENCIA OBSERVADA | `vecinos_por_quinto_del_disco` y `distancia_al_primero_por_quinto`, borde 60, unos 200 pasos por quinto. Sin intervalos: no concluyente como forma del set |
| D7 no informa sobre la parte después de la segunda hora | NO CONCLUYENTE | Los discos duran 75 min de media y cada uno es una mezcla aparte. Hace falta D8 (sets crudos de 3 h o más) o los sets largos de Gabriel |
| Puente con la biblioteca: 12 temas en común con vector, original de la biblioteca primero en 2 (mediana del puesto 66 de 1760) | NO CONCLUYENTE | `puente_biblioteca`, borde 60. n = 12 es muy poco. El piloto, con más temas en común, encontró el original en el 79-86 % por tema dentro del disco |
| 16 transiciones se saltaron porque el siguiente ya había sonado en el disco con el mismo nombre (otra versión) | EVIDENCIA OBSERVADA | `e1`, `repetidas_saltadas`. Los temas repetidos entre compilaciones cuentan una vez (artista principal y título sin paréntesis) |

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
  vienen de 4 DJs. Lo del DJ frente a la compilación está medido con esos 4.

## Qué cambia en el plan

1. **E1 sobre D7 se mide en el pool «disco»**, o en uno que controle la compilación. El pool de toda D7
   mide sobre todo de qué compilación es el tema.
2. **T13 (`siguiente`) tiene que superar a CLAP del tema actual en el pool «disco»**, con discos fuera
   de la muestra. Extrapolar el último paso ya empeora. Lo próximo a probar es un modelo aprendido (lineal
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
