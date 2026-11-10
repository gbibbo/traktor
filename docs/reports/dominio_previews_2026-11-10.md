# E15: control de dominio de los previews de Beatport, 2026-11-10

Plan de playlists en vivo, sección 12.9 (E15). Protocolo: skill `auditar` y el de la auditoría en 12.9, con las
decisiones operativas fijadas en el commit `269af4c` **antes de medir** (encabezado de
`src/v4/evaluation/dominio_previews.py`). Tests: `tests/v4/test_dominio_previews.py`. Resultado, fuera de Git:
`D:\TRAKTOR ML datos\dj_sets\e15\e15_previews.json`; la muestra congelada en `e15\muestra.json` y los embeddings
por segmento en `e15\emb\`. Los MP3 se leyeron en streaming de los `.tar` del Drive (35 tandas de 6 corridas) y no
quedó ninguno en disco. No se le pidió nada a Beatport.

**Con qué se midió**: 166 temas del índice con id de Beatport (niveles A y B) y preview bajado, congelados el
2026-11-10 (la bajada sigue). 146 con `same_cut` (la muestra principal), 20 sin `same_cut` (aparte); submuestra de
control de 40 temas con `same_cut` (semilla 0). Todos los previews duran 120 s y son el MP3 `LOFI` de Beatport a
96 kbps (mediana sobre los 13 202 previews bajados). Extracción: 55 min de CPU, 20 s por tema, 0 fallos.

## Veredicto

1. **Con la regla fijada de antemano, los previews no pasan**: el Spearman del puntaje de todos los factores es
   0,78 (umbral 0,90), el códec de CLAP da 0,92 (umbral 0,97) y CLAP pone primero al tema verdadero solo en el 32 %
   (umbral 90 %). La regla dice: no sirven de juez de los modelos y hay que decírselo a Gabriel antes de gastar GPU
   en V7. Eso hago.
2. **El problema no es el mismo en cada pieza**, y eso cambia qué se puede hacer:
   - **MAEST pasa todo**: el tema verdadero queda primero en el 100 % de los 146, el coseno con el mismo tema en el
     recorte es 0,988 y el Spearman de pares es 0,92. Los **estilos** casi no se mueven con el códec (0,993).
   - **CLAP sufre el códec**: el mismo tramo de audio da coseno 0,92 entre el preview y el archivo de la biblioteca,
     y el preview queda corrido como bloque (la diferencia de medias es el 69 % de la dispersión de la biblioteca,
     contra 13 % en MAEST). Centrar los previews por su propia media devuelve el puesto (primero en el 92 %, entre
     los 5 primeros en el 95 %), pero el Spearman de CLAP centrado queda en 0,80.
   - **El tramo no es el problema**: el archivo de la biblioteca cortado en la ventana del preview queda primero en
     el 97,5 % en CLAP y el 100 % en MAEST.
   - **Lo que hunde el puntaje de todos los factores es la tonalidad de Beatport** (diagnóstico agregado después de
     ver el resultado; no cambia el veredicto): coincide con la de Traktor o los tags en el 47 % de los temas, y el
     factor clave da Spearman 0,22 entre los dos lados. El BPM coincide (a ±1) en el 94 % y el género en el 100 %.
     Con los metadatos de la biblioteca y el audio de los previews, el mismo puntaje da 0,947.
     **Un metadato por vez** (pedido de la auditoría; `medir` corrido de nuevo el 2026-11-10 con
     `dominio_previews.py` del commit del informe `codec_clap_2026-11-10.md`, mismos embeddings): con todo de la
     biblioteca salvo la tonalidad de Beatport, 0,815; salvo el BPM, 0,909; salvo el género, 0,947 (el género es el
     mismo). La tonalidad explica casi toda la caída; el BPM, el resto. **Tónica y modo**: la tónica coincide en el
     73 % de los 146 y el modo en el 64 %; Beatport dice «menor» en el 64 % y la biblioteca en el 91 %. De la
     biblioteca a Beatport: misma 69, paralela 38, vecina 18, relativa 6, otras 15.
3. **La ventana**: la regla elige **V4** (el preview entero en 4 segmentos de 30 s); V4 y el tramo 35-65 % empatan
   (0,7825 frente a 0,7804). La regla del extractor tal cual es mucho peor (Spearman de las tres vistas 0,63 frente a
   0,85 en la misma submuestra): confirma la inferencia de la auditoría y no se usa.
4. **YouTube**: ningún tema tiene a la vez audio de YouTube y preview (0 de 188 audios de YouTube): las tandas de
   YouTube son de temas que Beatport no resolvió. Queda anotado; no se bajó nada.

**Qué decide Gabriel antes de V7** (propuesta del agente):
- V7 sigue teniendo sentido para **MAEST y estilos**, que pasan. **CLAP** de los previews solo se compara con otros
  previews, centrado aparte, y se informa como vista más débil.
- La **tonalidad** de un par de previews es la de Beatport, y en vivo manda la de Traktor: coinciden en menos de la
  mitad de los temas. Un peso de tonalidad aprendido con tonalidades de Beatport (P12a o los pares de V7) puede no
  valer en vivo. Opciones: aprender ese peso solo con la tonalidad de Traktor (pares de la biblioteca), o estimar la
  tonalidad de todos los temas con un mismo método.

## Tablas

Puesto = cuántos de los 1760 temas de la biblioteca quedan más parecidos al preview que el tema verdadero.
Spearman = sobre los pares de temas de la muestra (10 585 en la principal, 780 en el control), parecido preview con
preview frente a parecido biblioteca con biblioteca. «Todos los factores» = las tres vistas en z más bpm, clave y
genero con las fórmulas del recomendador, en desvíos de 200 000 pares al azar de la biblioteca dentro de los
filtros en vivo; del lado del preview, BPM, tonalidad y género de Beatport.

**Muestra principal** (146 con `same_cut`):

| Medida | V4 | Vtramo |
| :--- | --: | --: |
| CLAP crudo: primero / entre 5 | 32,2 % / 46,6 % | 33,6 % / 48,6 % |
| CLAP centrado: primero / entre 5 | 31,5 % / 45,9 % | 30,8 % / 47,3 % |
| CLAP centrado aparte: primero / entre 5 | 92,5 % / 95,2 % | 92,5 % / 95,9 % |
| MAEST centrado: primero / entre 5 | 100 % / 100 % | 99,3 % / 100 % |
| Estilos: primero / entre 5 | 66,4 % / 81,5 % | 68,5 % / 85,6 % |
| Spearman CLAP crudo / CLAP centrado / MAEST / estilos | 0,83 / 0,69 / 0,92 / 0,87 | 0,82 / 0,69 / 0,91 / 0,88 |
| Spearman, tres vistas (P11) | 0,889 | 0,886 |
| Spearman, tres vistas centradas aparte | 0,903 | 0,899 |
| **Spearman, todos los factores** | **0,783** | 0,780 |
| Spearman, todos los factores con los metadatos de la biblioteca | 0,947 | 0,946 |
| Spearman de cada metadato: BPM / tonalidad / género | 0,85 / 0,22 / 1,00 | igual |
| Todos los factores, metadatos de la biblioteca salvo uno de Beatport: tonalidad / BPM / género | 0,815 / 0,909 / 0,947 | — |
| Coseno con el mismo tema: CLAP crudo / MAEST centrado / estilos | 0,907 / 0,919 / 0,987 | 0,909 / 0,917 / 0,986 |
| Diferencia de medias / dispersión: CLAP / MAEST / estilos | 0,69 / 0,13 / 0,09 | 0,69 / 0,13 / 0,08 |

**Control** (40 temas): el mismo preview en sus tres ventanas, y el archivo de la biblioteca cortado en la ventana
del preview.

| Medida | Preview V4 | Preview Vtramo | Preview, regla del extractor | Recorte V4 | Recorte Vtramo |
| :--- | --: | --: | --: | --: | --: |
| CLAP centrado: primero | 25,0 % | 25,0 % | 22,5 % | 97,5 % | 97,5 % |
| MAEST centrado: primero | 100 % | 97,5 % | 92,5 % | 100 % | 97,5 % |
| Estilos: primero | 72,5 % | 67,5 % | 32,5 % | 80,0 % | 77,5 % |
| Spearman, tres vistas | 0,848 | 0,833 | 0,628 | 0,914 | 0,895 |
| Spearman, todos los factores | 0,777 | 0,769 | 0,665 | 0,810 | 0,799 |
| Spearman, todos los factores con metadatos de la biblioteca | 0,924 | — | — | 0,954 | — |
| Ídem salvo la tonalidad de Beatport / salvo el BPM | 0,777 / 0,921 | — | — | 0,810 / 0,953 | — |
| Coseno con el mismo tema, CLAP crudo | 0,901 | 0,902 | 0,870 | 0,988 | 0,988 |
| Diferencia de medias / dispersión, CLAP | 0,78 | 0,77 | 0,79 | 0,06 | 0,06 |

Códec (preview frente al recorte, segmento a segmento, 160 segmentos, coseno sin centrar): CLAP mediana 0,920
(p10 0,847); MAEST capa 7 0,988 (p10 0,959); estilos 0,993 (p10 0,968).

Metadatos de Beatport frente a los de la biblioteca (146 temas): BPM a ±1 en el 93,8 % (el doble o la mitad, 0,7 %);
tonalidad igual en el 47,3 %; género igual en el 100 % (el género del índice es el de Beatport en estos temas).
Tónica igual en el 73,3 % y modo igual en el 63,7 %; «menor» en el 64,4 % para Beatport y en el 91,1 % para la
biblioteca. En el control de 40: tonalidad igual 60 %, tónica 82,5 %, modo 75 %.

Sin `same_cut` (20 temas, V4): primero en CLAP centrado 60 %, en MAEST 100 %; Spearman de todos los factores 0,73.

## Afirmaciones

| Afirmación | Etiqueta | Evidencia |
| :--- | :--- | :--- |
| Con los umbrales fijados de antemano, los previews no pasan E15 | EVIDENCIA OBSERVADA | `e15_previews.json`, `veredicto`: Spearman 0,783, códec CLAP 0,920, CLAP primero 31,5 % |
| MAEST de un preview cae sobre el mismo tema de la biblioteca | EVIDENCIA OBSERVADA | Primero en 146 de 146; códec 0,988 |
| El problema de CLAP es el códec, no el tramo | EVIDENCIA OBSERVADA | Recorte frente a biblioteca: 0,988 y 97,5 % primero; preview frente a recorte: 0,920 |
| El códec corre los previews de CLAP como un bloque | EVIDENCIA OBSERVADA | Diferencia de medias 0,69 de la dispersión; centrado aparte, primero en 92,5 % |
| La causa del corrimiento es la compresión a 96 kbps | HIPÓTESIS | Todos los previews son `LOFI` a 96 kbps; se decidiría con el mismo tema de la biblioteca recodificado a 96 kbps |
| Lo que baja el puntaje de todos los factores es la tonalidad de Beatport | EVIDENCIA OBSERVADA (diagnóstico posterior al veredicto) | Factor clave: Spearman 0,22, 47 % igual; con los metadatos de la biblioteca, 0,947 |
| Beatport y Traktor estiman distinto la tonalidad en más de la mitad de los temas | EVIDENCIA OBSERVADA en estos 146 | 47,3 % igual; la de la biblioteca es la de Traktor si la analizó, si no la de los tags |
| Un peso de tonalidad aprendido con tonalidades de Beatport puede no valer en vivo | INFERENCIA | En vivo manda Traktor; las dos fuentes coinciden en menos de la mitad |
| La regla del extractor tal cual no sirve para los previews | EVIDENCIA OBSERVADA | Tres vistas: 0,63 frente a 0,85 en la misma submuestra |
| Los 20 temas sin `same_cut` se comportan parecido | EVIDENCIA OBSERVADA, n chico | MAEST 100 % primero; todos los factores 0,73 |

## Revisión hostil

- **¿Cambié un umbral después de mirar?** No: el veredicto es el del código fijado en `269af4c`. Lo que agregué
  después de ver el resultado son dos diagnósticos (el Spearman de cada metadato y el puntaje con los metadatos de la
  biblioteca) y está dicho así. Sin ellos el veredicto es el mismo; con ellos se sabe por qué falla.
- **¿La submuestra de control alcanza?** Son 40 temas y 160 segmentos; el efecto del códec en CLAP (0,92 contra
  0,988 en MAEST) es grande frente a su dispersión (p10 0,85 contra 0,96).
- **¿El puesto de los estilos es justo?** Los estilos son probabilidades de género: versiones del mismo tema y temas
  muy parecidos dan estilos casi iguales. Por eso el protocolo no les pone umbral.
- **¿La tonalidad de la biblioteca es la buena?** No se sabe cuál de las dos acierta: el 47 % mide que difieren, no
  quién se equivoca. Traktor es la que usa Gabriel en vivo.
- **¿Los temas de la muestra representan a los de los sets?** Son temas de la biblioteca de Gabriel que también
  aparecen en sets de D5; la mezcla de géneros puede ser distinta a la del total de previews.

## Qué sigue

Paro aquí, como pide el protocolo. V7 espera la decisión de Gabriel con lo de arriba. G13, P12a y E16 no dependen
de E15: G13 y P12a son texto y E16 usa los embeddings de la biblioteca. P12a tiene que tener en cuenta lo de la
tonalidad.
