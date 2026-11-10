# E17 y V8: el códec en CLAP y CLAP sin la dirección del códec, 2026-11-10

Plan de playlists en vivo, sección 12.9 («Después de E15»); pedido de Gabriel del 2026-11-10 (DECISIONS, «después de
E15», punto 4). Protocolo: skill `auditar`; reglas y umbrales fijados en el commit `f863a9e` **antes de recodificar
ni medir** (encabezado de `src/v4/evaluation/codec_clap.py`). Tests: `tests/v4/test_codec_clap.py`. Resultados, fuera
de Git: `D:\TRAKTOR ML datos\dj_sets\e17\` (`e17_codec.json`, `v8_clap_sin_codec.json`, `direcciones.npz`,
`muestra.json` y los embeddings por segmento en `emb\`).

**Con qué se midió**: índice del 2026-11-06 (1760 temas); `clap_full`, `maesthf_full_l7` y `maesthf_full_styles`. E17:
20 archivos de la biblioteca de 320 a 324 kbps (17 a 44,1 kHz y 3 a 48 kHz), elegidos al azar con semilla 0 fuera de
los 166 temas de E15; 55 s de CPU por tema, 0 fallos. Para la dirección de E15 se usaron los 146 previews con
`same_cut` y los 40 del control (muestra congelada de E15). Para el juez, los pares de la unidad 1: D3 × biblioteca,
79 pares de un mismo set de 55 sets y 42 DJs, más Bedrock; D1 va aparte. **No incluye D5**: la parte de E16 del
umbral 3 se mide en la unidad de E16.

## Veredicto

1. **E17: la causa es la compresión**, según la regla fijada de antemano. Sobre la dirección preview − biblioteca de
   E15, el mismo audio se corre +0,43 desvíos a 192 kbps, +2,32 a 128 y +3,20 a 96, frente a +2,53 del códec del
   preview. Con 6 dB menos se corre solo +0,22.
   - La dirección que da recodificar la biblioteca a 96 kbps, **estimada sin previews**, tiene coseno 0,991 con la de
     E15. El 83 % de cada diferencia de 96 kbps cae sobre ella.
   - **El nivel también mueve a CLAP, pero hacia otro lado.** Con −6 dB, el coseno con el original baja a 0,965;
     esa diferencia queda casi ortogonal a la del preview (coseno 0,13; un 1,8 % cae sobre ella). V8 no la quita.
2. **V8 pasa los cuatro umbrales con D3, con una sola dirección.** Decide v8c, la vista que reemplazaría a CLAP
   centrado en la lista nueva:
   - (1) en los 146 previews, sin centrarlos aparte, el tema verdadero queda primero en el 96,6 % y entre los 5
     primeros en el 99,3 %. Con CLAP centrado era 31,5 % y 45,9 %; los umbrales son 90 % y 97 %. En los 166 da
     96,4 % y 98,8 %.
   - (2) el exceso de vecinos de bajo bitrate sobre lo esperado por género baja de 0,277 (CLAP centrado) a 0,071,
     por debajo de MAEST centrado (0,103).
   - (3) en D3, frente al azar de mismo BPM, v8c da 0,700 y CLAP centrado 0,676: +0,024, con intervalo
     [+0,000, +0,048]. Con todos los factores con pesos iguales: 0,798 frente a 0,795. Ninguno pierde más de 0,02.
   - (4) cambian 6 de los 30 temas de la lista de hoy por tema principal (mediana; p10 3, p90 12) y 2 de los 30 de
     la lista de todos los factores (p90 5).
   - **El veredicto es provisorio**: falta la misma prueba en E16. Si pasa allí, v8c es el CLAP de la lista nueva de
     E12. Los valores por defecto del recomendador no cambian hasta que Gabriel lo diga.
3. **En los sets de Gabriel (aparte; nada se ajustó con ellos) V8 pierde un poco, sin que sea concluyente.**
   - Lista de hoy con v8 crudo en lugar de CLAP crudo: acierto en 30 del tema siguiente de 9,8 % frente a 11,3 %.
     Son 2 transiciones que acierta solo v8 contra 8 que acierta solo CLAP (p 0,11).
   - Todos los factores con v8c: 16,1 % frente a 16,8 % (p 0,45).
4. **MAEST y estilos (solo informado).** A MAEST la compresión también lo corre (+1,60 a 96 kbps, «las dos» por la
   regla, porque −6 dB lo corre +0,53), pero su puesto ya era 100 %. Quitarle su dirección no cambia nada medible
   (vecinos 0,103 → 0,094; D3 0,743 → 0,745). Los estilos casi no se mueven (+0,07).

## Tablas

**E17**: corrimiento sobre la dirección de E15, en desvíos de la proyección de la biblioteca. Coseno = con el
original, tema entero (media de los 3 segmentos).

| Variante | CLAP: corrimiento | CLAP: coseno | MAEST: corrimiento | MAEST: coseno | Estilos: corrimiento |
| :--- | --: | --: | --: | --: | --: |
| MP3 192 kbps | +0,43 | 0,997 | +0,13 | 1,000 | 0,00 |
| MP3 128 kbps | +2,32 | 0,921 | +0,57 | 0,999 | +0,01 |
| MP3 96 kbps | +3,20 | 0,881 | +1,60 | 0,993 | +0,07 |
| 6 dB menos | +0,22 | 0,965 | +0,53 | 0,989 | 0,00 |
| Códec del preview (E15, control de 40) | +2,53 | — | +1,51 | — | +0,12 |

- Coseno entre la diferencia media de cada variante y la dirección de E15, en CLAP: 192 kbps 0,89; 128 kbps 0,96;
  96 kbps 0,99; −6 dB 0,13.
- **Verificación de la auditoría** (proyección de la biblioteca sobre esa dirección, fuera de los 166 de E15): de
  200 a 300 kbps +0,51 y bajo 200 kbps +1,49 (60 y 381 temas), contra 300 kbps o más; los previews, +2,63. La
  auditoría daba +0,50, +1,49 y +2,68 (su dirección salía de los 166, la mía de los 146).
- Control: el original recodificado por este código frente al embedding guardado de la biblioteca da coseno mediano
  1,000. El mínimo es 0,9968 en CLAP y 0,9974 en MAEST: algún archivo cambió un poco desde que se extrajo. No afecta
  la comparación, que es dentro del mismo código.

**V8, una dirección** (decide v8c; v8 crudo se informa):

| Medida | CLAP crudo | v8 crudo | CLAP centrado | v8c | MAEST centrado | Umbral |
| :--- | --: | --: | --: | --: | --: | :--- |
| Previews, primero / entre 5 (146) | 32,2 % / 46,6 % | 95,2 % / 98,6 % | 31,5 % / 45,9 % | **96,6 % / 99,3 %** | 100 % / 100 % | 90 % / 97 % |
| Vecinos bajo 200 kbps: observado / esperado / exceso (420 anclas) | 52,5 / 27,5 / 0,250 | 34,6 / 25,8 / 0,088 | 56,6 / 28,9 / 0,277 | 33,4 / 26,4 / **0,071** | 38,0 / 27,7 / 0,103 | ≤ MAEST c. |
| AUC D3 × biblioteca, nulo de mismo BPM | 0,664 | 0,672 [−0,014, +0,032] | 0,676 | **0,700** [+0,000, +0,048] | 0,743 | ≥ −0,02 |
| AUC D1 × biblioteca (aparte) | 0,574 | 0,551 | 0,605 | 0,591 | 0,576 | — |

| Ecuación (pares de la unidad 1) | D3 | Bedrock | Diferencia en D3, intervalo por DJs |
| :--- | --: | --: | :--- |
| CLAP crudo | 0,649 | 0,665 | — |
| v8 crudo | 0,657 | 0,656 | +0,008 [−0,009, +0,027] |
| Todos los factores, pesos iguales (unidad 1) | 0,795 | 0,659 | — |
| Todos los factores con v8c | **0,798** | 0,661 | +0,003 [−0,003, +0,009] |
| Todos con v8c, MAEST y estilos sin su dirección | 0,797 | 0,662 | +0,003 [−0,004, +0,009] |

- Más direcciones (umbral 1 en los 146): con 2, 97,3 %; con 3 a 5, entre 95,2 % y 95,9 %. Una alcanza; con 2 el
  exceso de vecinos baja a 0,054.
- Los anclas de 300 kbps o más tenían, en CLAP crudo, menos vecinos de bajo bitrate que lo esperado (11,3 % frente a
  21,3 %); con v8c, 19,2 % frente a 22,7 %.
- Los números de la unidad 1 se reproducen: AUC D3 × biblioteca 0,664 (CLAP crudo) y 0,676 (CLAP centrado);
  ecuación en D3, 0,649 y 0,795; acierto en vivo en D1, 11,3 % y 16,8 %.

## Afirmaciones

| Afirmación | Etiqueta | Evidencia |
| :--- | :--- | :--- |
| El corrimiento de CLAP entre preview y biblioteca es de la compresión | EVIDENCIA OBSERVADA (20 archivos; contraste de una variable por vez) | `e17_codec.json`: 96 kbps +3,20 y coseno 0,991 con la dirección de E15; −6 dB +0,22 y 0,13; dosis y respuesta 192 < 128 < 96 |
| El nivel también mueve a CLAP, en otra dirección | EVIDENCIA OBSERVADA | −6 dB: coseno con el original 0,965, 1,8 % de la diferencia sobre la dirección del preview |
| La dirección del códec se puede estimar sin previews | EVIDENCIA OBSERVADA | u1 (biblioteca a 96 kbps) frente a la dirección de E15: coseno 0,991 |
| V8 pone al preview sobre su tema sin centrarlo aparte | EVIDENCIA OBSERVADA | v8c primero en 141 de 146 (96,6 %); CLAP centrado, 46 de 146 |
| V8 saca la mayor parte del agrupamiento por bitrate de CLAP | EVIDENCIA OBSERVADA | Exceso de vecinos 0,277 → 0,071 (MAEST centrado 0,103) |
| V8 no le cuesta a CLAP en el juez de D3 | EVIDENCIA OBSERVADA, n chico | +0,024 [+0,000, +0,048] solo; +0,003 [−0,003, +0,009] en la ecuación; 79 pares |
| V8 pasa con E16 | NO CONCLUYENTE | Falta medirlo; va en la unidad de E16 |
| En D1 V8 pierde un poco | NO CONCLUYENTE | 9,8 % frente a 11,3 % (2 contra 8, p 0,11) |
| En D1 CLAP acierta en parte porque Gabriel toca seguidos archivos de calidad parecida | HIPÓTESIS | La auditoría vio que los pares de un mismo set comparten clase de bitrate el 71 % de las veces frente a 64 %; se decide midiendo esa fracción en las transiciones de D1 |
| La lista nueva con v8c cambia poco | EVIDENCIA OBSERVADA | Mediana de 2 de 30 temas por tema principal (p90 5) |

## Revisión hostil

- **¿20 archivos alcanzan?** Para la dirección, sí: el efecto es grande y parejo (coseno 0,991 con la de E15, que
  salió de otros 146 temas y de otro codificador). Para afirmar que el nivel no importa en general, no: 6 dB es un
  solo valor, y no sé a qué nivel están los previews respecto de los archivos.
- **¿El codificador es el mismo que el de Beatport?** No: LAME por ffmpeg. Que la dirección coincida igual sugiere que
  lo que CLAP registra es la pérdida de agudos y el ruido del MP3 de baja tasa en general, no un codificador. Es una
  INFERENCIA.
- **¿El umbral 1 se compró mirando?** La dirección no usa previews, y la regla de sumar direcciones estaba fijada.
  Ganó la primera.
- **¿La mejora en D3 es real?** El intervalo toca el 0, y son 79 pares. Lo que el umbral pedía era no perder más de
  0,02, y eso se cumple.
- **¿Qué falta?** E16 (D5 × biblioteca) para cerrar el umbral 3. Volver a bajar los 420 temas de menos de 200 kbps
  sigue siendo el arreglo de fondo, independiente de cualquier modelo (auditoría de E15).
