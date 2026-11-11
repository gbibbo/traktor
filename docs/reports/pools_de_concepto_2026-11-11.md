# Pools de concepto con temas ajenos, y filtros por campo: opinión de la auditoría

**Fecha:** 2026-11-11. **Para:** Gabriel y quien implemente la búsqueda por concepto (`buscar()` en
`tools/traktor_live/conceptos_agente.py`, plan § 12, unidad 4). **Pedido de Gabriel:** opinión sobre su propuesta
del 11-11. Nada de esto está decidido: las ideas E18 y T31 del plan (§ 12.10) esperan a Gabriel.

## La propuesta de Gabriel

Armar grupos de temas que se sabe que van juntos (por ejemplo, 10 de bossa nova de una playlist de Spotify), mezclarlos
con temas ajenos (uno de techno, después más, de a poco) y mirar qué frase o técnica de búsqueda sigue rescatando los 10
y los separa del resto. El grupo queda como pool de evaluación, nunca de entrenamiento. Además: la búsqueda del teléfono
debería poder filtrar por género y por otros campos.

## Veredicto

La idea sirve y conviene hacerla, sobre todo por una razón: hoy los conceptos se juzgan contra estilos de MAEST y géneros
de Beatport, que son modelos y metadatos; un grupo armado por personas es un juez que no depende de ningún modelo del
proyecto. Necesita cuatro ajustes para que mida lo que importa:

1. **Varios escalones de dificultad.** Bossa nova frente a techno es el escalón fácil y hoy no existe en la biblioteca
   (ningún tema de bossa nova). Lo difícil, y lo que pasa en el 100 % de la biblioteca, es separar dentro de la
   electrónica. Ahí la misma prueba, hecha hoy con los géneros de Beatport, da casi azar con texto (tabla de abajo).
2. **Los ajenos, en tamaños fijos y siempre con la biblioteca entera.** Con un puntaje que no depende del pool (el coseno
   de hoy), ordenar una vez da la curva completa: agregar de a uno no cambia nada. Con técnicas que sí miran el pool
   (centrar, el juez LLM que lee lo que trajo la búsqueda) hay que repetir en cada tamaño. El uso real es el grupo
   escondido entre los 1760 temas.
3. **Medir el corte, no solo el orden.** `buscar()` devuelve siempre 30 temas. Si hay 10, sobran 20; si no hay ninguno,
   devuelve 30 igual. La prueba tiene que decir cuántos temas entrega el sistema y qué hace cuando el grupo no está.
4. **Varios grupos, y una parte sin tocar.** Elegir «la frase que funciona» mirando un solo grupo es ajustar contra un
   ejemplo. Con 8 a 12 grupos, unos sirven para elegir la técnica y los otros se miran una sola vez, al final.

Filtros por campo: sí. Lo que existe como dato (género, BPM, tonalidad, voz) se filtra por el dato, que es exacto; el
texto queda para lo que no tiene campo («oscuro», «con trompetas»). El género con texto solo, dentro de la electrónica,
casi no supera al azar.

## Lo medido (exploratorio)

Qué lo produjo: `clap_full` (`laion/clap-htsat-unfused`), texto con `Texto` de `src/v4/evaluation/clap_texto.py`, coseno
crudo, que es lo que hace hoy `buscar()`. Sobre qué datos: índice `indice_recomendacion.json`, 1760 temas; 1429 con género
confirmado de Beatport; los 10 géneros con 40 temas o más. Versión: HEAD `3134ba9`, semilla 20261111. El script quedó
fuera de Git. Nada tiene umbral fijado antes: son medidas para opinar, no para decidir.

La prueba de Gabriel, simulada: 10 temas al azar de un género, más k temas con otro género de Beatport; consulta = el
nombre del género; cuántos de los 10 quedan entre los 10 primeros. Media de 400 sorteos por género y de los 10 géneros.
La sonda es una regresión logística sobre CLAP de audio, entrenada con las etiquetas de Beatport y medida fuera de muestra
(5 particiones): es el techo de lo que CLAP sabe del género, no una técnica de texto.

| De los 10, quedan entre los 10 primeros | +1 ajeno | +10 | +100 | +todos (1178 a 1383) | AUC |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Azar | 9,1 | 5,0 | 0,9 | 0,1 | 0,50 |
| Texto: nombre del género | 9,2 | 5,7 | 1,5 | 0,2 | 0,59 (0,46 a 0,78) |
| Sonda sobre CLAP de audio | 9,3 | 7,2 | 3,5 | 0,6 | 0,79 (0,68 a 0,88) |
| Filtro por el campo de Beatport | 10 | 10 | 10 | 10 | 1 por definición: el campo es la etiqueta |

Con un solo ajeno la medida casi no puede fallar: entre 11 temas, los 10 primeros tienen 9 del grupo como mínimo.

| Afirmación | Etiqueta | Evidencia |
| :--- | :--- | :--- |
| La biblioteca no tiene bossa nova: activación máxima de los estilos de MAEST 0,025 (`Jazz---Bossa Nova`) y 0,011 (`Latin---Bossanova`); samba 0,060, MPB 0,088 | EVIDENCIA OBSERVADA | `maesthf_full_styles`, 1760 temas del índice |
| Pedir «bossa nova» hoy devuelve 30 temas igual; los primeros son house y tech house (Ican, «Caminos Del Nino»; Michael Bibi, «Hanging Tree»); el primero queda a 3,0 desvíos de la media (0,542; media 0,351; desvío 0,063), como un concepto que sí existe | EVIDENCIA OBSERVADA | Coseno crudo sobre el índice; `buscar()` corta en `MAX_TRACKS = 30` (`conceptos_agente.py:68`, `:267`) |
| Con texto, el género dentro de la electrónica casi no se separa: AUC medio 0,59; mejor «minimal deep tech» 0,78 y «melodic house and techno» 0,76; peor «progressive house» 0,46 y «electronica» 0,49 | EVIDENCIA OBSERVADA, exploratoria | Tabla de arriba; el género lo pone el sello en Beatport, no un oído |
| La mejor técnica de texto medida antes da 0,64 a 0,68 y la sonda 0,76 | Transcripto del plan (T24, T25), no vuelto a medir | Plan § 12.4 |
| El género es un campo en 1602 de los 1760 temas: 1429 de Beatport, 173 del clasificador; 158 sin género; 28 géneros distintos | EVIDENCIA OBSERVADA | `genero` y `genero_fuente` del índice |
| Bossa nova frente a techno se va a separar bien con texto | HIPÓTESIS | No hay audio de bossa nova en esta máquina. La decide la prueba de Gabriel; con pocos ajenos lo espero, con los 1760 no lo sé: hay house con aire latino que hoy ya puntúa alto para esa frase |
| Para un género que Beatport no tiene, los 400 estilos Discogs de MAEST son un detector ya calculado para toda la biblioteca | INFERENCIA | Las etiquetas existen (`Latin---Bossanova`, `Jazz---Bossa Nova`) y la matriz está extraída; no se midió cuánto aciertan fuera de la electrónica |

## Revisión hostil

- **El género de Beatport no es el oído.** Un tema etiquetado House puede sonar a tech house. La tabla mide acuerdo con
  la etiqueta, y por eso mismo un grupo armado por personas es mejor juez que el que usé.
- **Usé el nombre del género tal cual.** El circuito real promedia tres textos escritos por el LLM y debería dar algo
  más; el número del plan para la mejor técnica (0,64 a 0,68) sigue lejos de la sonda.
- **El filtro «acierta 10 de 10» por construcción.** No prueba que el campo esté bien, solo que filtrar por un campo
  devuelve ese campo. Sirve si lo que Gabriel pide con una palabra de género es ese campo.
- **Los 158 temas sin género** quedan fuera de cualquier filtro duro por género.
- **Calidad de archivo.** CLAP registra el bitrate (auditoría de E15). Un pool bajado por otra vía que la biblioteca
  puede separarse por el archivo y no por la música. Control: mismo camino de bajada, y medir también con CLAP sin la
  dirección del MP3 (v8c).
- **Un pool chico.** Con 10 temas, un tema mal puesto en la playlist de Spotify mueve el resultado un 10 %.

## Ideas (esperan a Gabriel; plan § 12.10)

- **E18, pools de concepto con ajenos** (Gabriel, 11-11). Grupos de 10 a 20 temas armados por personas, que elige
  Gabriel; viven en un dataset aparte, fuera de la biblioteca y de Traktor; solo evaluación. Cuatro escalones: otro mundo
  (bossa nova, rock, reggaetón), vecinos de la electrónica (drum and bass, trance, disco, ambient), subestilos de su
  música (acid, dub techno, tribal, afro), y conceptos que no son género (trompetas, voz de mujer, oscuro). Medidas: de
  los temas del grupo, cuántos quedan entre los primeros con +10, +100 y la biblioteca entera; cuántos ajenos aguanta
  antes de que entre el primero; cuántos temas entrega; y qué hace cuando el grupo no está. Técnicas a comparar sobre los
  mismos pools: texto crudo, los tres textos del LLM, T24, estilos Discogs, sonda y filtro por campo.
- **T31, consulta con campos.** El LLM devuelve, además de los textos, campos tomados de listas cerradas (los géneros
  que existen en la biblioteca, rango de BPM, voz). El ordenador filtra por esos campos y ordena lo que queda con el
  puntaje de texto. Si quedan menos de 30, la lista es más corta; si no queda nada, el teléfono lo dice.
