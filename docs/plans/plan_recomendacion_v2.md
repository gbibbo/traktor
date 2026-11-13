# Plan de recomendación, versión 2

**Vigente desde el 2026-11-13.** Lo pidió Gabriel ese día: el dataset de sets terminó de bajar, quiere usarlo entero en
la GPU, y quiere que el teléfono sea una herramienta completa para elegir la música. Lo escribió el agente auditor.

Reemplaza el orden de trabajo de `docs/plans/recomendacion_playlists_plan.md` (secciones 5, 11.4, 12.5 y 12.9). Ese
documento queda como historia y como referencia de sus secciones. Ninguna idea se pierde: las 124 del plan viejo y las
nuevas están en `docs/plans/inventario_ideas.md`, cada una con su estado y su lugar en este plan.

Para empezar a trabajar desde un chat nuevo: `PROMPT_LINEA_A_DATASET.md` y `PROMPT_LINEA_B_TELEFONO.md`, en esta carpeta.

## 0. En una página

**Qué hay hoy.** El recomendador en vivo arma «Parecidos» con un solo modelo (CLAP). Se midió una ecuación con varios
factores (tres modelos de audio, género, tonalidad, BPM) y hay una lista nueva hecha con ella que espera la escucha de
Gabriel (E12). El teléfono ya crea una playlist a partir de un texto, pero busca solo por el parecido entre el texto y
el audio: no filtra, no corta la lista y no separa géneros dentro de la electrónica.

**Qué cambió.** El dataset DJSets v1 está completo: 1549 sets de 573 DJs, 36 611 temas, 33 321 con audio.

**Dos líneas de trabajo, en paralelo:**

- **Línea A, el dataset.** Aprender de los DJs profesionales qué temas van juntos y en qué momento de la noche, y
  llevarlo a la biblioteca de Gabriel. Sección 2.
- **Línea B, el teléfono.** Una consulta con casillas (filtros, tema de referencia, pesos, momento, texto) que se
  llena con presets o con texto libre. El LLM solo llena casillas; el ordenador filtra y ordena. Sección 3.

La línea A le entrega a la B datos nuevos por tema (momento de la noche, género por el sonido, «como tal DJ») y los
pesos de la ecuación. La B no espera a la A: empieza con los campos que ya existen.

**Lo que espera a Gabriel** está junto en la sección 6.

## 1. Dónde estamos

Verificado el 2026-11-13 salvo donde dice «según».

| Pieza | Estado | De dónde sale |
| :--- | :--- | :--- |
| Biblioteca | 1760 temas en el índice; género en 1602 (1429 de Beatport, 173 del clasificador), 158 sin género | `indice_recomendacion.json`, medido el 11-11 |
| Parecidos en vivo | Por defecto, solo CLAP, con los filtros de BPM ±20 y tonalidad ±2 semitonos | `recomendador.py`; DECISIONS 11-10 |
| Ecuación de factores | Medida con 270 pares de sets cruzados con la biblioteca: los pesos aprendidos no se distinguen de los pesos iguales | Según `juez_d5_2026-11-10.md` |
| Escucha E12 | Lista desde el 11-10: 10 temas principales, listas A y B a ciegas. Sin respuestas [REEMPLAZADO 2026-11-13 por la escucha corta: Gabriel eligió la lista nueva en 5 temas y la de hoy en 5; no decide, Parecidos sigue con CLAP] | Carpeta `e12\` (no hay archivo de respuestas) |
| CLAP y la calidad del archivo | CLAP registra la compresión MP3. La variante v8c, sin esa dirección, pasó los umbrales | Según `codec_clap_2026-11-10.md` |
| Género y tonalidad en los sets | La coaparición de géneros aprendida le gana a la taxonomía (AUC 0,72 frente a 0,68). Los DJs repiten la tonalidad 2,0 veces el azar | Según `pares_texto_2026-11-10.md` |
| Dataset DJSets v1 | 1549 sets, 573 DJs, 36 611 temas; 33 321 con audio (30 255 previews de Beatport de 2 min, 3066 temas enteros de YouTube); 30 655 con metadatos de Beatport; 42 493 pares consecutivos, 36 327 con audio en los dos | `datasets/djsets/README.md`; carpeta listada en el Drive |
| Vistas de los previews (V7) | 60 archivos de vistas en `erclm:traktor_ml/v7_vistas/` (472 MB). [REEMPLAZADO 2026-11-13 por el conteo de la línea A: cubren 19 923 de los 30 255 previews (19 847 grabaciones) y ningún audio de YouTube; faltan 10 322 grabaciones de Beatport en 38 shards y las 3057 de YouTube; `linea_a_protocolo_a0_a1.md`, sección 0] No se contó cuántos de los 33 321 audios cubren | Listado del Drive; `cola.py status` |
| GPU | Kaggle terminó dos trabajos largos; el último hizo 10 266 previews a 2,04 s cada uno, 5 h 50 min sin cortes. En Colab el mismo trabajo fue más rápido (1,10 s por preview) y retomó bien desde su parcial, pero la sesión se perdió a los 13 minutos y gpuq lo dio por fallido en el intento 11 | `cola.py status`; `logs traktor-v7-5` y `traktor-v7-4` |
| Teléfono | Worker v5 con un LLM chico (Gemma en Workers AI): escribe 3 textos, el ordenador los sondea y arma una lista de 30 por coseno. Probado en local; publicar es paso de Gabriel | `cloudflare/README.md`, `conceptos_agente.py` |
| Búsqueda por texto | Dentro de la electrónica casi no separa géneros (AUC 0,59; un clasificador sobre CLAP, 0,79). Pedir algo que no hay devuelve 30 temas igual | `pools_de_concepto_2026-11-11.md` |
| Repositorio | El árbol de trabajo está en la rama `dataset-djsets`, 2 commits delante de `main` | `git status` |

## 2. Línea A: qué sacar del dataset

### 2.1 Qué tiene el dataset y qué no

- **Tiene la práctica de 573 DJs:** qué temas pusieron en una misma noche y en qué orden. En 788 sets, además, casi
  todas las filas dicen a qué minuto entró el tema.
- **Tiene metadatos de Beatport** para 30 655 temas: BPM, tonalidad, género, sello, fecha.
- **Casi ningún tema se repite:** 30 261 de los 36 611 aparecen en un solo set, y solo 544 en cinco o más. Por eso no
  sirve aprender un vector por tema a partir de las listas solas (ideas M1 y M2). Lo que se aprende tiene que pasar
  por el audio, que es además lo único que llega a la biblioteca de Gabriel.
- **El audio es un preview:** unos 2 minutos del medio del tema, en MP3 de 96 kbps (según E15: empieza cerca del
  40 %). No tiene el comienzo ni el final. La biblioteca tiene temas enteros con calidades mezcladas.
- **No es una muestra al azar:** sale de semillas que eligió Gabriel, con tope de 15 sets por DJ. El 9 % de los temas
  no tiene audio, y es probable que sean los inéditos.

### 2.2 Los cuatro productos

| Producto | Qué se aprende | Con qué | Dónde se usa |
| :--- | :--- | :--- | :--- |
| **1. Compatibilidad** | Una medida de «estos dos temas van juntos» y la probabilidad de acertar | Pares de un mismo set frente a pares de sets distintos | Parecidos: el orden, el campo `seguro` y dónde cortar la lista |
| **2. Momento de la noche** | Para cada tema, en qué parte de un set largo lo pondría un profesional | Posición y minuto de entrada en los sets de 3 horas o más | Presets y texto del teléfono; perfiles de más y menos energía |
| **3. Campos nuevos por tema** | Género por el sonido, escena (grupos de DJs y sellos), «como tal DJ» | Etiquetas de Beatport, sellos y DJs del dataset | Filtros del teléfono; completa los 158 temas sin género |
| **4. Reglas de transición** | Cuánto cambian el BPM, la tonalidad y el género de un tema al siguiente | Pares consecutivos con metadatos | Pesos y valores por defecto de la ecuación |

### 2.3 Experimentos, en orden

Cada uno fija sus umbrales en un commit antes de medir. «GPU» es tiempo de una T4 de Kaggle; las cifras que no dicen
«medido» son estimaciones que el primer trabajo corto corrige.

| Paso | Qué se hace | Cómputo | Sale cuando |
| :--- | :--- | :--- | :--- |
| **A0. Vistas de todo el dataset** (V9) | CLAP, MAEST capa 7 y estilos para los 33 321 audios, por segmento, guardados por `track_id`. Primero se cuenta qué cubre V7 y se completa el resto. Para los temas enteros de YouTube, la misma ventana que un preview | GPU: unas 19 h si se hiciera todo de nuevo (2,04 s por preview, medido); menos lo que V7 ya cubre | Todo audio tiene sus vistas, sin valores no finitos, alineadas por `track_id`; los 166 temas de E15 dan lo mismo que antes |
| **A1. Tablero fijo** (E19) | La partición por DJ del dataset, con pares de prueba cuyos dos temas no aparecen en el entrenamiento. Negativos en tres niveles: al azar, mismo BPM, mismo BPM y género. Líneas de base: BPM solo, género, cada vista, pesos iguales | CPU | Tabla con intervalos por DJ. Todo modelo posterior se mide acá, sin cambiar el tablero |
| **A2. Ecuación con pesos aprendidos y calibrada** (P12b, P15) | Regresión logística sobre los factores con los pares de previews. Probabilidad calibrada de «van juntos» | CPU | Va a escucha solo si le gana a los pesos iguales con el intervalo sobre 0 en los DJs de prueba |
| **A3. Espacio aprendido** (P9, M3) | Una proyección chica sobre las tres vistas congeladas, entrenada para acercar temas de un mismo set. Se comparan tres definiciones de «van juntos»: consecutivos, a 3 temas o menos, mismo set | GPU: 5 a 10 h en total; cada corrida lleva minutos | Le gana a la mejor ecuación en el tablero; en los pares de la biblioteca no pierde más de 0,02 de AUC; pasa el control de dominio (E20) |
| **A5. Momento de la noche** (T20, T2) | Una cabeza que predice la posición del tema en el set, entrenada con los sets de 3 h o más con minuto de entrada | GPU: 1 a 2 h | Le gana a «solo BPM» en los DJs de prueba, y Gabriel escucha los 10 primeros y los 10 últimos de su biblioteca (E11) |
| **A6. Campos del dataset** (G3, G14, T32) | Clasificador de género con las 30 655 etiquetas; escenas a partir de DJs y sellos; el centro de lo que toca cada DJ. Y las tablas de transición vueltas a correr con v1 | CPU | Género: acierto sobre temas de DJs de prueba, frente al clasificador de hoy. «Como tal DJ»: recupera temas de ese DJ que no vio |
| **A7. Otros codificadores** (V10, V11) | Un CLAP entrenado con música para la búsqueda por texto; otra vista de audio solo si A3 muestra que falta | GPU: 6 a 10 h por modelo | Reemplaza al actual solo si gana en los pools de concepto que no se miraron (E18) |
| **A4. Afinar un codificador** (M4, M5) | Entrenar las últimas capas de MAEST o de CLAP con el audio de los previews | GPU: 30 h o más por corrida, una semana de cuota | Solo si A3 gana 0,03 de AUC o más y sigue mejorando con más capacidad. Necesita el OK de Gabriel |
| **A8. Después** | Predictor del tema siguiente (T13), trayectorias (T5, T6, T30), temas basura (P14) | — | Cuando A3 y A5 estén cerrados |

A4 va al final a propósito: casi todo lo que se puede aprender sale de vectores ya calculados, en minutos. Lo caro es
calcular esos vectores (A0) y afinar un modelo entero (A4), y lo segundo solo se justifica si lo primero se queda corto.

### 2.4 Reglas de método

- **Nada se ajusta con los sets de Gabriel.** Sus sets y su biblioteca son solo para medir.
- **El DJ de prueba nunca se ve al entrenar,** y los resultados se informan aparte para pares con temas no vistos.
- **Los negativos difíciles mandan.** Separar un tema de techno de uno de house no prueba nada: el resultado principal
  es frente a negativos del mismo BPM y género.
- **Control de dominio antes de tocar la biblioteca (E20).** Un modelo aprendido con previews de 96 kbps se prueba
  con archivos de la biblioteca recodificados a 96 kbps y recortados como un preview: cada archivo tiene que quedar
  primero frente a su propia copia, y los vecinos no tienen que agruparse por calidad de archivo.
- **Una variable por comparación,** mismos pares y misma semilla.
- **Un mejor número no cambia el recomendador.** Lo que pasa el tablero va a una escucha a ciegas; el valor por
  defecto cambia cuando Gabriel lo dice.
- **Cada informe dice** la versión del dataset, cuántos sets y temas midió, y el commit.
- **Presupuesto humano** (Gabriel, 11-13, tras rechazar una escucha de 600 temas). Toda tarea que se le pida a
  Gabriel dice antes cuánto lleva, entra en 15 minutos y tiene una decisión por pantalla. Lo que un agente puede
  decidir con un valor por defecto o revisar por su cuenta, no se le pregunta. Las escuchas usan el formato corto de
  `tools/playlist_review/pagina_escucha.py`. Regla completa: `docs/reports/escucha_corta_e12_2026-11-13.md`.

### 2.5 Presupuesto de GPU

[REEMPLAZADO 2026-11-13 por la respuesta de Gabriel: la GPU es gpuq, que reparte entre Kaggle y Colab según la
demanda; no se fija una sola plataforma] La cuota confiable es la de Kaggle, unas 30 horas por semana en sesiones de
12. A0, A3 y A5 entran en la primera semana. A7 lleva otra. A4 consume una semana por corrida. Si Gabriel quiere otra
máquina (cómputo pago o el HPC de Surrey), lo decide él; el plan no cambia, cambian los tiempos.

Lo que rige: gpuq usa las dos plataformas. gpuq detecta el corte, relanza y el trabajo sigue desde lo guardado; lo
que no puede es evitar que Colab corte. Por eso todo trabajo se escribe para sesiones cortas:

- guarda en el Drive cada pocos minutos, nunca más de 10 sin guardar (`docs/LESSONS_LEARNED.md`);
- se parte en trabajos chicos, de una hora de trabajo útil o menos, para no agotar los intentos de gpuq;
- lo que necesita horas seguidas, como A4, se pide con `--platform kaggle`.

Kaggle da unas 30 horas por semana en sesiones de 12. Colab suma lo que dure cada sesión.

## 3. Línea B: el teléfono y el LLM

### 3.1 El principio

**El LLM no elige temas ni hace cuentas: llena un formulario.** El formulario es una consulta con casillas de listas
cerradas. El ordenador la ejecuta siempre igual: filtra, puntúa, corta y explica. Un preset del teléfono es el mismo
formulario ya lleno, y no pasa por el LLM.

Tres consecuencias:

- Lo que existe como dato se resuelve con el dato: un filtro por género o por BPM es exacto.
- El LLM chico hace una tarea chica por vez, con las opciones a la vista.
- Todo cambio de reglas se puede medir, porque la salida es un formulario que se compara con el esperado.

### 3.2 El formulario de consulta (L1)

| Casilla | Valores | Quién la llena |
| :--- | :--- | :--- |
| `filtros.bpm` | Rango exacto, o «más lento, igual o más rápido que lo que suena» | Preset, reglas o LLM |
| `filtros.generos` | Lista cerrada: los géneros que existen en la biblioteca. Duro si viene de un preset; blando si viene del texto, porque nadie escribe las etiquetas de Beatport tal cual (Gabriel 11-13; puede cambiar) | Preset, reglas o LLM |
| `filtros.tonalidad` | Compatible con lo que suena (la regla del vivo) o una tonalidad dada | Preset o reglas |
| `filtros.voz` | Con voz, sin voz, da igual | Preset o LLM |
| `filtros.otros` | Año, sello, artista; excluir lo ya tocado esta noche | Preset o LLM |
| `referencia` | Ninguna, lo que suena ahora, un tema nombrado, o un DJ (cuando exista A6) | LLM extrae el nombre; el ordenador lo resuelve |
| `pesos` | Un perfil con nombre (equilibrado, sonido, género, armónico) y hasta dos ajustes de tres niveles | Preset o LLM. Nunca números |
| `momento` | Apertura, subida, pico, cierre, after | Preset o LLM |
| `texto` | Hasta 3 descripciones del sonido en inglés, con las reglas ya medidas | LLM |
| `orden` y `largo` | Por puntaje, por BPM o por momento; cuántos temas | Preset o por defecto |

El ordenador devuelve la lista y además qué hizo: cuántos temas pasaron cada filtro y qué casillas usó. Si quedan
menos de 30, la lista es más corta. Si no queda ninguno, el teléfono lo dice.

### 3.3 El camino de un pedido

1. **Preset.** Va directo al ordenador como formulario.
2. **Texto libre.** Primero un intérprete por reglas, sin LLM (L6): números de BPM, nombres exactos de género,
   tonalidades, «como <tema>». Lo que entiende, lo llena.
3. **Lo que queda va al LLM en dos pasos cortos (L7).** El primero clasifica el pedido en tipos de una lista cerrada.
   El segundo llena cada casilla con un prompt chico que trae solo las opciones de esa casilla.
4. **Validadores.** Lo que no está en una lista cerrada se descarta. Si todo falla, el pedido sigue por el camino de
   hoy (tres textos), así el teléfono nunca queda vacío.
5. **El ordenador ejecuta** (L2) y el teléfono muestra qué se entendió, en palabras, con un toque para corregir (L12).

Un tema o archivo nombrado lo resuelve el ordenador por coincidencia aproximada sobre su índice (L8). Si hay dudas,
el teléfono muestra los candidatos. La lista de temas no sale del ordenador.

### 3.4 Los deberes hechos de antemano

| Pieza | Qué es |
| :--- | :--- |
| **Catálogo** (L4) | Lo genera el código a partir de la biblioteca: géneros con su conteo, rangos de BPM, campos disponibles. El LLM elige de ahí. Se regenera cuando cambia la biblioteca |
| **Reglamento** (L5) | Una tabla de decisión: para cada tipo de pedido, qué casillas se llenan y cuáles no. Un solo archivo en el repo; de él salen los prompts cortos de cada paso |
| **Ejemplos** | Pedidos reales con su formulario, varios por tipo, incluidos los que no se deben llenar |
| **Perfiles de pesos** (L9) | Los nombres y lo que mueve cada uno, definidos con los pesos de la línea A |
| **Tabla de momentos** (L10) | Hasta que exista A5, una tabla fija de BPM y géneros por momento, que aprueba Gabriel |
| **Banco de regresión** (L11) | De 150 a 200 pedidos con su formulario esperado. Se corre ante cada cambio de reglas o de modelo |

El reglamento es exhaustivo en el repo y corto en cada llamada: el LLM ve solo la parte que necesita para ese paso.

### 3.5 Unidades

| Paso | Qué se hace | Sale cuando |
| :--- | :--- | :--- |
| **B0. Contrato** | El formulario (L1) escrito como esquema, con tests; el catálogo (L4) | El esquema valida los ejemplos y rechaza lo que está fuera de lista |
| **B1. Ejecutor** (L2) | `buscar()` recibe un formulario: filtra por los campos de hoy (género, BPM, tonalidad, voz), puntúa y corta. Los dos casos de uso: sin referencia y con el set en curso | Tests con la biblioteca real; un formulario vacío da la lista de hoy |
| **B2. Presets** (L3) | Pantalla del teléfono con los presets y el texto libre | Un preset llega al ordenador sin pasar por el LLM |
| **B3. Reglas y LLM** (L5 a L8) | Intérprete por reglas, clasificador de tipo, llenado de casillas, validadores | El banco (L11) pasa el umbral fijado antes; ninguna casilla empeora frente a la versión anterior |
| **B4. Explicación** (L12) | El teléfono muestra qué entendió y cuántos temas pasaron cada filtro | Probado de punta a punta en local |
| **B5. Las listas** (E18, E13) | Pools de concepto con ajenos, y escucha de Gabriel | Las técnicas se comparan sobre los mismos pools; una parte no se mira hasta el final |
| **B6. Lo que llega de la línea A** | Momento (A5), género por el sonido y «como tal DJ» (A6), corte por confianza (A2) | Cada campo entra al catálogo cuando su experimento cierra |

Publicar en producción sigue siendo paso de Gabriel (`python cloudflare/publicar.py`).

**Estado de la línea B (2026-11-13):**

| Paso | Estado |
| :--- | :--- |
| B0 | Hecho: `config/telefono/formulario.json`, `formulario.py` y el catálogo |
| B1 | Hecho: `ejecutor.py`. Con solo texto o con solo «lo que suena» da la lista de hoy, comprobado con la biblioteca real |
| B2 y B4 | Hechos en el Worker v6, probado en local: presets sin LLM y pasos con conteos |
| B3 | Espera el banco de pedidos |
| L10 | Tabla medida en DJSets (`momentos_djsets_2026-11-13.md`). Forma elegida por el agente: el momento ordena y no filtra (DECISIONS 11-13, punto 10). Falta implementarla |

## 4. Orden

```
Línea A:  A0 ──> A1 ──> A2 ──> A3 ──> (escucha) ──> A7 ──> A4 si hace falta
                  └────> A5 ──┐
                  └────> A6 ──┤
Línea B:  B0 ──> B1 ──> B2 ──>├──> B6
                  └──> B3 ──> B4 ──> B5
```

A0 es lo primero de la línea A y lo único que bloquea al resto. B0 y B1 no dependen de la línea A.

## 5. Quién hace qué

| Sesión | Modelo | Qué hace | Cómo empieza |
| :--- | :--- | :--- | :--- |
| Línea A | Opus 5.5 | Experimentos con el dataset y trabajos de gpuq | Chat nuevo con `PROMPT_LINEA_A_DATASET.md` |
| Línea B | Sonnet 5.5 | Formulario, ejecutor, reglas del LLM, banco | Chat nuevo con `PROMPT_LINEA_B_TELEFONO.md` |
| Auditoría | El modelo más capaz | Revisa protocolos antes de medir y resultados después | Chat nuevo por ronda, con este plan |

Las dos líneas tocan archivos distintos. Los documentos compartidos (`STATUS.md`, `DECISIONS.md`, este plan) se editan
agregando solo las líneas propias y con `git add` de rutas explícitas.

## 6. Lo que espera a Gabriel

| # | Qué | Para qué |
| :--- | :--- | :--- |
| 1 | Confirmar que «la GPU del cluster» es gpuq (Kaggle), o decir si quiere otra máquina [REEMPLAZADO 2026-11-13 por Gabriel: es gpuq, con Kaggle y Colab según la demanda; sección 2.5] | Presupuesto de la línea A |
| 2 | Confirmar que se entrenan modelos con el dataset, de uso privado. Beatport declara `ai-train=no`, que es una señal y no una licencia. El agente lee el pedido del 11-13 como un sí [REEMPLAZADO 2026-11-13 por Gabriel: sí, se usa el dataset completo, de uso privado] | A2 en adelante |
| 3 | Unir la rama `dataset-djsets` a `main` [REEMPLAZADO 2026-11-13: Gabriel lo pidió y está hecho; `main` quedó en `dcd93a3`] | Que todas las sesiones trabajen sobre lo mismo |
| 4 | La escucha E12: 10 temas, A o B. Página con las indicaciones: `D:\TRAKTOR ML datos\dj_sets\e12\escucha.html` (la arma `tools/playlist_review/pagina_escucha.py`) [REEMPLAZADO 2026-11-13 por Gabriel: oír 10 × 60 temas es inviable. Versión corta, unos 12 minutos: 3 temas por lista y una decisión por pantalla, en `artifacts\v4\escucha\e12\escucha.html`; protocolo en `docs/reports/escucha_corta_e12_2026-11-13.md`. Es lo único de esta tabla que hace falta de Gabriel ahora] [REEMPLAZADO 2026-11-13: hecha; 5 a 5, no decide, Parecidos sigue con CLAP. La próxima escucha es para lo que salga de A2 o A3] | Es lo que decide si Parecidos deja de usar solo CLAP |
| 5 | Teléfono: de 30 a 50 pedidos escritos como los escribiría de verdad [REEMPLAZADO 2026-11-13 por Gabriel: no puede escribir 50; los produce el agente y él supervisa. Borrador de 82 en `pedidos_telefono_borrador.md`: revisarlo y agregar los suyos] [REEMPLAZADO 2026-11-13 por Gabriel: tampoco revisa 82 filas. Los revisa la auditoría, y sus pedidos reales se juntan del registro del teléfono] | Ejemplos y banco de la línea B |
| 6 | Teléfono: filtro de género duro o blando por defecto [REEMPLAZADO 2026-11-13 por Gabriel: blando desde el texto, duro desde un preset]; la tabla de momentos; de 8 a 12 conceptos con una playlist de Spotify cada uno; y las cuatro preguntas del borrador de pedidos [REEMPLAZADO 2026-11-13 por el presupuesto humano (2.4): las playlists las proponen los agentes y Gabriel dice sí o no; las cuatro preguntas pasaron a valores por defecto del agente] | B1, L10 y E18 |
| 7 | Si el factor de género pasa a ser la coaparición aprendida de los sets | Ecuación (pregunta abierta desde el 11-10) |
| 8 | Vetos en `candidatos.csv` | Si veta sets, sale DJSets v2; los experimentos arrancan con v1 |
| 9 | `python cloudflare/publicar.py` | El teléfono en producción |
| 10 | A4, afinar un codificador | Solo cuando A3 lo pida |

## 7. Reglas que siguen

Las de `AGENTS.md`. En particular: el primer trabajo de cada tipo en gpuq necesita el OK de Gabriel con los números
de la prueba chica; nada del dataset vive solo en el pendrive; el repositorio es público y los datos van solo al
Drive; ningún pedido nuevo a Beatport, 1001Tracklists o YouTube sin que Gabriel lo pida.
