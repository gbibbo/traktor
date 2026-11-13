# Auditoría del protocolo de A1, antes de medir

**Fecha:** 2026-11-13. **Para:** la sesión de la línea A; Gabriel no tiene que decidir nada acá. **Qué se auditó:**
`docs/plans/linea_a_protocolo_a0_a1.md` en el commit `6bdbd5b`. El código (`tablero_a1.py`, `vistas_djsets.py`) no
se auditó. **Con qué:** las tablas de DJSets v1 y las del tablero en `D:\TRAKTOR ML datos\linea_a\` (`tracks.csv`,
`set_tracks.csv`, `particion.csv`, `pares_test.csv.gz`). Solo metadatos: no se miró ninguna vista ni el puntaje de
ningún modelo. Medidas exploratorias, con un script fuera de Git.

## Veredicto

El protocolo está bien armado y sus conteos se reproducen. Tiene un hueco que conviene cerrar antes de medir: **la
fecha de lanzamiento.** Tal como está el tablero, ordenar por «fecha cercana» da un AUC de 0,737 en el resultado
principal (P1 frente a N3), sin oír nada. Un modelo puede ganar aprendiendo de qué año es un tema, que no es lo que se
busca. Recomiendo un nivel de negativos igualado en fecha y que el resultado principal sea ese. Hay cinco agregados
menores, todos baratos.

## Lo verificado

| Afirmación | Etiqueta | Evidencia |
| :--- | :--- | :--- |
| Los conteos de partida son correctos: 33 321 temas con audio (30 255 y 3066), 33 226 grabaciones | EVIDENCIA OBSERVADA | `tracks.csv` |
| El preview empieza en el 40 % del tema: mediana 0,4000, p5 0,3998, p95 0,4002 | EVIDENCIA OBSERVADA | 30 252 previews con los dos campos |
| La cuarentena deja 994, 191 y 175 sets, y 189 afuera | EVIDENCIA OBSERVADA | `particion.csv`, columna `parte_tablero` |
| P1 principal de test tiene 2076 positivos, 1948 con negativos N3 | EVIDENCIA OBSERVADA | `pares_test.csv.gz` |
| Ningún tema del tablero principal está en un set de train como fila normal. 6 aparecen en train solo encima de otro tema («w/») | EVIDENCIA OBSERVADA | `set_tracks.csv` cruzada con `particion.csv` |
| **La fecha separa sola.** AUC de «fecha cercana» (menos la diferencia de año de lanzamiento entre a y el candidato): 0,769 en N1, 0,753 en N2 y 0,737 en N3 | EVIDENCIA OBSERVADA | P1 principal de test, AUC pesado como en el protocolo |
| Entre a y b la diferencia de año tiene mediana 1; entre a y un negativo N3, mediana 4 | EVIDENCIA OBSERVADA | Ídem |
| Los DJs ponen música del momento: el año del set menos el año de lanzamiento de b tiene mediana 0 (p90, 11) | EVIDENCIA OBSERVADA | `sets.csv` y `tracks.csv`, positivos de P1 |
| La ventaja de la fecha sale sobre todo de sortear negativos de sets de otros años | INFERENCIA | Los sets de test van de 2016 (p10) a 2025 (p90), y el negativo se sortea de todo el pool |
| Mismo sello: 11,1 % de los positivos y 0,4 % de los negativos N3. Artista en común: 13,7 % y 0,1 % | EVIDENCIA OBSERVADA | `bp_label` y `artist` |
| La popularidad del candidato (`n_sets`) da AUC 0,58 | EVIDENCIA OBSERVADA | Ídem |
| 133 de los 2185 temas «no vistos» tienen otra versión del mismo tema en train (mismo artista y título sin la mezcla); 7 comparten ISRC. Tocan a 258 de los 2076 positivos | EVIDENCIA OBSERVADA | Ejemplos: «Add Arp» en test y «Add Arp (Catz 'N Dogz Remix)» en train |
| Un nivel igualado en fecha tiene tamaño suficiente: entre los N3 ya sorteados, el 26 % tiene el año del candidato a ±1 del de b, y 1589 positivos de 33 DJs conservan al menos uno | EVIDENCIA OBSERVADA | Es una cota inferior: sorteando del pool entero con esa condición quedan más |
| Un modelo de audio aprendería la época si no se la iguala | HIPÓTESIS | La decide el propio tablero: medir cada modelo en N3 y en el nivel igualado en fecha |

## Recomendaciones, para aplicar antes de medir

1. **Nivel N4, igualado en fecha, como resultado principal.** N4 es N3 y además el año de lanzamiento del candidato a
   ±1 del de b. Control de construcción, como C3: en N4 el AUC de «fecha cercana» queda entre 0,47 y 0,53. N3 se sigue
   informando. «Fecha cercana» entra como línea de base en todos los niveles. Lo mismo en validation y en train: si los
   negativos de entrenamiento no igualan la fecha, la proyección de A3 aprende la época.
2. **El resultado principal, además, sin tres grupos de pares:** los 258 con otra versión del tema en train, los que
   comparten artista y los que comparten sello. Se informa aparte, sin cambiar el tablero.
3. **Dos líneas de base más:** la popularidad del candidato, y CLAP crudo como tercera referencia de las diferencias
   pareadas. CLAP crudo es el producto de hoy, y en la escucha del 11-13 Gabriel no prefirió los pesos iguales.
4. **Una medida que mire el comienzo de la lista:** cuántas veces b queda primero entre sus 21 candidatos. Es lo que
   juzga la escucha corta. Y una regla escrita para pedirle una escucha a Gabriel: solo cuando un candidato le gana a
   CLAP crudo y a los pesos iguales en el resultado principal, con el intervalo sobre 0, y no pierde en esa medida.
   La escucha del 11-13 mostró que 0,08 de AUC a favor no se oyó.
5. **En A0, un control de fuente:** para cada vista, si los vecinos de un tema de YouTube son de YouTube más de lo
   esperado. Recodificar iguala el códec; el nivel y el máster pueden seguir distinguiendo las dos fuentes. Se informa,
   sin umbral.
6. **Los 6 temas «w/»:** contarlos como vistos, o dejar escrito que ningún entrenamiento usa esas filas.

## Control del cambio aplicado (commit `d598fcb`)

La línea A aplicó las seis recomendaciones antes de medir. Verificado sobre la tabla nueva de test (md5 `540adcce`),
otra vez solo con metadatos:

| Afirmación | Etiqueta | Evidencia |
| :--- | :--- | :--- |
| P1 principal tiene 2067 positivos; 1660 con negativos N4, de 33 DJs y 166 sets | EVIDENCIA OBSERVADA | `pares_test.csv.gz` |
| En N4 ningún candidato está a más de 1 año de b | EVIDENCIA OBSERVADA | Diferencia máxima: 1 |
| En N4 el BPM y el género ya no separan: AUC 0,503 y 0,500 | EVIDENCIA OBSERVADA | Mismo AUC pesado |
| **En N4 la fecha todavía separa: AUC 0,571.** El control C5 del protocolo (entre 0,47 y 0,53) va a fallar | EVIDENCIA OBSERVADA | El candidato queda más lejos de a que b en el 46 % de los negativos, igual en el 35 % y más cerca en el 19 % |
| La causa es el margen de ±1 que recomendó esta auditoría | INFERENCIA | En el 43 % de los positivos a y b son del mismo año; un candidato a ±1 de b queda entonces a 0 o a 1 año de a, nunca más cerca |

**La recomendación 1 estaba incompleta: fue un error de esta auditoría.** Medí que ±1 dejaba tamaño suficiente y no
medí si alcanzaba para igualar la fecha. Corrección:

- **Igualar la distancia al ancla, como hace N2 con el BPM:** el candidato cumple
  |año(c) − año(a)| = |año(b) − año(a)|. Con eso «fecha cercana» empata siempre y su AUC es 0,5 por construcción.
- **Tamaño.** Dentro de lo ya sorteado, 1381 de los 1660 positivos conservan algún negativo así (33 DJs, 161 sets). Es
  una cota inferior: sorteando del pool entero entran también los candidatos del otro lado de a.
- **Si quedan menos de 1500 positivos,** recomiendo mantenerlo igual como resultado principal y decirlo. El intervalo
  se calcula remuestreando DJs, y los DJs siguen siendo 33.

## Revisión hostil

- **Un error mío, corregido.** La primera corrida usó la columna `split` y no `parte_tablero`, y dio 106 temas
  «vistos» dentro del tablero principal. Con la columna correcta son los 6 de arriba. Los números de fecha, sello,
  artista y popularidad no dependen de esa columna.
- **¿La fecha es un atajo o una señal?** Las dos cosas. Los DJs mezclan música de una misma época, y eso es real. Pero
  en la biblioteca de Gabriel la fecha no limita qué puede poner, y casi toda la separación viene de cómo se sortean
  los negativos. Por eso van los dos niveles.
- **Mi cruce de versiones es aproximado:** junta por artista y título sin lo que va entre paréntesis. Puede unir de más
  o de menos.
- **La cota de N4** se midió sobre los negativos ya sorteados, no sobre el pool.
- **No audité el código.** Queda para otra ronda.
