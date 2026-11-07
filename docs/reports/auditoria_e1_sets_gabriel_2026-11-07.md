# Auditoría del banco E1 sobre los sets de Gabriel, 2026-11-07

Objeto: `docs/reports/e1_sets_gabriel_2026-11-07.md` (otra sesión), con su código
(`tools/dj_sets/sets_traktor.py`, `src/v4/evaluation/banco_sets.py`) y sus datos
(`D:\TRAKTOR ML datos\dj_sets\sets\d1.csv`, `d1_e1.json`). Protocolo: skill `auditar`. Todo lo de abajo
se corrió en esta sesión sobre el índice `indice_recomendacion.json` (1717 sugeribles en la colección de
Traktor de esta PC) y los 22 sets de `d1.csv`; los scripts quedaron en el scratchpad de la sesión, no
en el repo. El informe del piloto de Digweed y la adenda D7/D8 no se auditaron acá.

## Veredicto

El informe se sostiene. El titular (CLAP es, de lejos, el mejor componente solo) es EVIDENCIA OBSERVADA
reproducida: 50 de 539 transiciones en los 30 primeros, y la diferencia con cada línea de base pasa una
prueba apareada, incluida una que el informe no tenía, la popularidad (los temas que Gabriel repite
entre sets). Los aciertos no son duplicados: 48 de los 50 son entre artistas distintos y ninguno es el
mismo título. Tampoco los explica el set más largo: sin él, 36 de 420 (8,6 %).

Tres cosas cambian la lectura, ninguna el titular:

1. **Las líneas de base de un solo componente estaban mal desempatadas.** `bpm`, `clave` y `genero` toman
   pocos valores distintos (mediana de 36, 6 y 12 puntajes entre ~1000 candidatos), así que el grupo de
   empate que contiene el puesto 30 tiene 102, 146 y 149 temas. El recomendador ordena con un `sort`
   estable: dentro del empate manda el orden del índice. Con desempate al azar (5 semillas), tonalidad
   sube de 2,6 % a 4,2 %, género de 3,9 % a 5,0 %, BPM queda en 4,6 %. CLAP (833 puntajes distintos) y
   CLAP + BPM no cambian. La tonalidad sola no es «casi azar»: es como el BPM solo.
2. **«Sumar BPM a CLAP lo lleva a 10,2 %» no se distingue de CLAP solo**: 29 transiciones acierta solo
   CLAP, 34 solo CLAP + BPM, p = 0,61. Y con pesos 1 y 1 el BPM pesa más de lo que parece: entre los
   candidatos de una consulta, `bpm` se dispersa casi el doble que `clap` (desvío mediano 0,15 frente a
   0,08). Los pesos de los perfiles (plan, 3.4) no son comparables entre componentes sin normalizar.
3. **La regla de tonalidad filtra poco y cuesta bastante.** Con las tonalidades del pool, un par al azar
   pasa el corte 0,60 el 69 % de las veces; las transiciones reales de Gabriel lo pasan el 76 %. El
   filtro saca el 31 % de los candidatos y deja afuera el 23,5 % de sus transiciones reales. Hay señal
   de tonalidad, pero chica: misma tonalidad exacta en 86 transiciones contra 42 esperadas al azar;
   los movimientos de ±1 semitono (165) ocurren a la tasa del azar (185 esperados): Gabriel ni los
   busca ni los evita. La frase del informe «el mejor ajuste siempre queda a ±2 semitonos» no informa
   nada: `best_key_shift` solo busca en ±2.

La decisión sobre la regla (P3) sigue siendo de Gabriel. Lo que la auditoría agrega es el precio y el
beneficio medidos, y una cota del ruido de las tonalidades (abajo).

## Lo reproducido

| Dato del informe | Esta sesión | Nota |
| :--- | :--- | :--- |
| 22 sets, 602 temas, 580 transiciones, 571 en el índice, 543 con los dos | Igual (`d1_resumen.json`, y recontado desde `d1.csv`) | 592 + 11 (los dos archivos del 2026-02-04 unidos: 5 + 6 temas, ahora un set) − 1 (tema suelto del set de 130) = 602 |
| Los dos archivos «estaban a 30 minutos» | Los nombres de archivo distan 31 min; al aire, el último tema del primero terminó 19:50:57 y el primero del segundo empezó 20:02:23: 11,4 min sin nada al aire | La regla de 20 min los une bien; la frase habla de los archivos, no del hueco |
| Al aire = `PLAYEDPUBLIC` 1 y 120 s o más | 710 entradas `PLAYEDPUBLIC` 1 en 37 archivos; 94 de menos de 120 s (20 en un solo archivo, 2026-09-11); 0 sin `STARTDATE` | El criterio de 2 min saca el 13 % de las entradas; es una elección, no un dato (ver hipótesis) |
| 539 en el pool, 4 fuera, recuperación 0,738 | Igual: 398 de 539 pasan los filtros | 24 transiciones con el principal sin tonalidad (no se filtra por tonalidad); 9 con el siguiente sin tonalidad |
| Azar 2,0 %, BPM 4,3 %, tonalidad 2,6 %, género 3,9 %, CLAP 9,3 %, CLAP + BPM 10,2 % | 2,3 % (5 semillas), 4,6 %, 2,6 %, 3,9 %, 9,3 %, 10,2 % con el orden del índice | BPM 25 frente a 23 aciertos: los componentes vienen redondeados a 4 decimales en `info["componentes"]` |
| BPM entre consecutivos: mediana 2, p90 8; 9 a más de 20 | Igual | |
| 117 + 2 + 7 + 9 fuera por los filtros | 119 por tonalidad (506 − 387), 9 sin tonalidad; el resto por BPM | |
| 7 tests de `tests/v4/test_dj_sets.py` | 7 en verde | |

## Lo nuevo

**Popularidad.** En D1 hay 445 temas distintos en 602 pasadas; 97 están en 2 o más sets y 20 en 3 o
más. Ordenando por cuántos *otros* sets tienen al candidato (empates al azar), acierto en 30 de 4,8 %
(4,5 a 5,4 en 5 semillas). Popularidad primero y CLAP para desempatar: 6,9 %, peor que CLAP solo.

**Pruebas apareadas** (aciertos en 30, orden del índice como el banco; discordantes y p binomial exacta):

| CLAP frente a | Solo CLAP | Solo el otro | p |
| :--- | --: | --: | --: |
| Azar | 49 | 10 | < 0,0001 |
| Popularidad | 46 | 23 | 0,008 |
| BPM | 48 | 23 | 0,004 |
| Género | 49 | 20 | 0,0006 |
| CLAP + BPM | 29 | 34 | 0,61 |

Las transiciones no son independientes: están anidadas en 22 sets, y el acierto por set va de 0 de 41
a 5 de 20. Las diferencias con azar, BPM y género sobreviven a eso con margen; la de popularidad es la
menos firme.

**De qué están hechos los aciertos.** De las 539 transiciones, 18 son entre el mismo artista (2 de ellas
aciertan) y 1 repite el título (no acierta). Con el mismo género de Beatport (124) CLAP acierta el
14,5 %; con distinto, el 7,7 %. Parecido CLAP mediano en los aciertos: 0,91.

**Tonalidad: de dónde sale y cuánto ruido tiene.** De los 1717 sugeribles, 231 tienen la tonalidad de
Traktor, 1259 la de `bpm_key.parquet` (Essentia, fase 1; el recomendador la llama «tags») y 227 no
tienen ninguna. Cuando el tema principal tiene tonalidad, esos 227 (13 % de la colección) nunca se
sugieren (`compute`: `it.key == "?"` → fuera). En 194 temas con las dos estimaciones, coinciden en 160
(82 %) y son compatibles en 190 (98 %).

Distribución del mejor ajuste en las 506 transiciones con las dos tonalidades, frente a lo esperado
si el siguiente saliera al azar del pool (con la tonalidad real del principal):

| Semitonos | −2 | −1 | 0 | +1 | +2 | Pasan 0,60 | Misma tonalidad |
| :--- | --: | --: | --: | --: | --: | --: | --: |
| Real | 33 | 101 | 264 | 64 | 44 | 387 (76,5 %) | 86 |
| Esperado al azar | 48 | 110 | 188 | 75 | 84 | 350 (69,1 %) | 42 |

## Hipótesis e inferencias

- **Cuánto del 23,5 % filtrado es ruido de tonalidad.** INFERENCIA con supuestos: si Traktor y Essentia
  se equivocan de forma independiente con la misma tasa, el 82 % de coincidencia da unos 9-10 % de
  error cada uno; entonces en un 18 % de los pares hay al menos una tonalidad mal, y una tonalidad mal
  falla el corte un 31 % de las veces: un 5-6 % de las transiciones reales quedan fuera por ruido, un
  cuarto de las filtradas. Los otros tres cuartos son movimientos que la regla considera incompatibles.
  Salvedades: las dos son estimaciones (coincidir no es acertar), y los 194 temas con tonalidad de
  Traktor son los que Gabriel analizó, no una muestra al azar. Se decide con el pitch de D2 y la
  tonalidad de Traktor para todos, como dice el informe.
- **Los 94 temas al aire de menos de 2 min.** HIPÓTESIS del lector: son pruebas o arranques fallidos.
  Si alguno es una mezcla rápida real, el banco mide una transición a → c en vez de a → b → c. Se
  decide repitiendo E1 con el corte en 60 s y mirando si cambia el orden de los perfiles.
- **Los candidatos son la colección de hoy.** Los sets van desde 2025-07; temas comprados después no
  podían sonar entonces. Castiga igual a todos los perfiles (la comparación vale) y hace pesimista el
  número absoluto. HIPÓTESIS de que cambia poco; se mide con la fecha de alta del catálogo.
- **Normalizar los componentes por consulta** (percentil o rango entre los candidatos) antes de pesar.
  HIPÓTESIS de que ordena mejor que la escala cruda; se decide en E1, un componente por vez, cuando se
  fijen pesos (etapa 2).

## Afirmaciones

| Afirmación | Etiqueta | Evidencia |
| :--- | :--- | :--- |
| El lector de D1 reproduce los conteos del informe y la unión de los dos archivos del 2026-02-04 es por un hueco de 11,4 min al aire | EVIDENCIA OBSERVADA | `d1.csv`, `d1_resumen.json`, recuento de esta sesión |
| CLAP solo: 50 de 539 en los 30 primeros; supera a azar, BPM, género y popularidad con prueba apareada | EVIDENCIA OBSERVADA | Tabla de pruebas apareadas |
| CLAP + BPM no se distingue de CLAP solo | EVIDENCIA OBSERVADA en D1 | 29 frente a 34, p = 0,61 |
| Tonalidad 2,6 %, género 3,9 % y BPM 4,3 % dependen del orden del índice; con desempate al azar son 4,2 %, 5,0 % y 4,6 % | EVIDENCIA OBSERVADA | `compute`: `rows.sort` estable sobre `self.candidates`; grupos de empate de 102 a 149 en el puesto 30 |
| Los aciertos de CLAP no son duplicados ni el mismo artista | EVIDENCIA OBSERVADA | 48 de 50 entre artistas distintos; 0 con el mismo título |
| El filtro de tonalidad pasa el 69 % de los pares al azar y el 76 % de las transiciones reales | EVIDENCIA OBSERVADA | 387 de 506 frente a 349,5 esperadas con las tonalidades del pool |
| Los movimientos de ±1 semitono ocurren a la tasa del azar; la misma tonalidad, al doble | EVIDENCIA OBSERVADA | 165 frente a 185; 86 frente a 42 |
| Un cuarto de lo filtrado por tonalidad es ruido de estimación | INFERENCIA | Coincidencia Traktor/Essentia 160 de 194, con independencia y tasas iguales supuestas |
| Los 227 temas sin tonalidad no se sugieren cuando el principal tiene tonalidad | EVIDENCIA OBSERVADA | `recomendador.py`, `compute`; en D1 Gabriel fue a uno de ellos 9 veces de 535 |
| Con pesos iguales, `clap+bpm` pesa más el BPM | INFERENCIA | Desvío entre candidatos 0,15 frente a 0,08; `weighted` promedia la escala cruda |
| Sumar más sets de Gabriel cambiaría estos números | NO CONCLUYENTE | 22 sets; acierto por set de 0/41 a 5/20 |

## Revisión hostil de esta auditoría

- Reordené con los componentes redondeados a 4 decimales que devuelve `compute`; el banco ordena con
  el valor exacto. Diferencia vista: 2 aciertos en BPM. No cambia nada.
- La expectativa «al azar» de la tonalidad condiciona por la tonalidad del principal y la distribución
  del pool, no por el filtro de BPM ni por lo ya tocado. Si tonalidad y BPM estuvieran correlacionados
  en la colección, la expectativa se movería unos puntos; no cambiaría el signo de la diferencia.
- Las pruebas apareadas tratan las transiciones como independientes y no lo son. Por eso informo la
  dispersión por set y señalo la comparación con la popularidad como la menos firme.
- Cinco semillas para los desempates: informo el rango, no solo la media.
- No reproduje la tabla «Qué deja fuera el filtro» celda por celda (el informe no deja su script); sus
  totales cierran con los míos (119 por tonalidad, 9 sin tonalidad, 9 a más de 20 BPM).

## Qué haría ahora (propuestas, no tareas)

1. Banco: desempate al azar en `evaluate` (mezclar los índices y ordenar estable por `info["puntaje"]`);
   informar el acierto por set o un remuestreo por sets. Tres líneas y un test.
2. Plan 3.4: antes de fijar pesos, normalizar los componentes por consulta o declarar la escala cruda
   (queda escrito en el plan).
3. Para Gabriel, sobre P3, con los números: dejar el corte 0,60 (saca el 31 % del pool y el 23,5 % de
   sus movimientos reales); o bajarlo a 0,55, que deja pasar 22 de 24 tonalidades, casi sin filtro; o
   pasar la tonalidad de filtro a componente con peso, con lo que los 227 temas sin tonalidad vuelven a
   ser sugeribles. Lo decide escuchando; el banco solo dice qué cuesta cada opción en D1.
