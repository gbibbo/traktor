# Auditoría del algoritmo de recomendación y estrategia propuesta, 2026-11-09

Pedido de Gabriel: revisar todo lo hecho sobre el algoritmo de sugerencia (sin la interfaz), entender el
problema y las opciones, y proponer la estrategia más prometedora para (1) la playlist de D0, temas
parecidos al que suena y que con seguridad van a sonar bien mezclados con él, sin contar BPM ni tonalidad
porque se ajustan después, y (2) las playlists que redirigen el set hacia un concepto escrito en palabras.
Inspiración pedida: «Guided Music Retrieval: Music Search with Multimodal Queries» (Brandão, Silva,
Alonso-Jiménez y Bogdanov; manuscrito enviado a IEEE TASLP, 16-07-2026, todavía sin revisión por pares).

Protocolo: skill `auditar`. Leído en esta sesión: el plan `docs/plans/recomendacion_playlists_plan.md`
entero, `PROMPT_SEGURIDAD_Y_DIRECCION.md`, DECISIONS 11-06 a 11-08, los informes E1, líneas D1, D7 y zona
con los veredictos de sus auditorías, el informe de Cloudflare, el código del recomendador, de `zona_sets.py`
y del agente de conceptos, y el PDF completo. Medido en esta sesión: lo de las tablas de abajo, con cinco
scripts exploratorios que quedaron fuera de Git en `D:\TRAKTOR ML datos\dj_sets\auditoria_20261109\`. No
escriben nada en el repo ni en los artefactos.

Los números nuevos son **exploratorios**: se miraron varias variantes sobre los mismos datos y ninguna lista
pasó por el oído de Gabriel. Sirven para elegir qué construir primero, no para fijar pesos.

## Veredicto

1. **El cuello de botella de Parecidos no es CLAP, es mirar con un solo modelo.** CLAP, MAEST capa 7 y los
   400 estilos Discogs de MAEST, que ya están extraídos para toda la biblioteca, arman listas de 30 casi
   distintas: comparten 6,8 temas (CLAP y MAEST) y 2,9 (CLAP y estilos). Cada una acierta sola unas 50 de las
   542 transiciones de Gabriel, pero no las mismas: entre las tres, 110. Sumadas con pesos iguales, el
   acierto en 30 pasa de 9,0 % a 11,4-12,7 %, y los temas que DJs profesionales pusieron en un mismo set
   quedan mejor separados de pares al azar del mismo BPM (AUC 0,66 → 0,74-0,77; 79 pares de 48 DJs). Es la
   primera variante que se distingue de CLAP crudo en dos fuentes a la vez, con la salvedad de que la medida
   es exploratoria. Blanquear CLAP o quitarle componentes principales empeora.
2. **«Seguro» se puede definir como acuerdo entre vistas independientes**, y calibrar con los pares de
   sets profesionales. El radio de zona (P8) solo corta una lista ya ordenada; el acuerdo cambia el orden.
3. **Las playlists de concepto hoy ignoran lo que suena.** El agente de conceptos
   (`tools/traktor_live/conceptos_agente.py`, commit `73d042d` de hoy) ordena por el coseno entre el texto y el
   audio. En un banco al estilo del paper, esa lista se parece al tema de referencia lo mismo que una lista
   al azar (10 %). La suma de audio y texto (lo que el plan llamaba T18) tampoco supera a «solo audio» en
   cumplir las dos cosas, que es lo que el paper encuentra con su línea de base «Sum».
4. **El texto de CLAP entiende descripciones de sonido, no nombres de género.** Contra los estilos
   acústicos de MAEST (ambient, acid, hard techno, dub techno, minimal) ordena bien (AUC medio 0,79). Contra
   el género de Beatport, con el nombre solo casi no ordena (0,58); con tres frases descriptivas, 0,64; con
   contraste entre conceptos, 0,68; una sonda supervisada sobre el mismo embedding llega a 0,76. Lo que
   limita la lista no es la fórmula de mezcla, es la calidad del puntaje de concepto.
5. **Del paper sirven el protocolo y la receta, no un modelo listo.** No publican pesos; sus modelos
   entrenados solo conocen un conjunto cerrado de modificaciones con plantilla (18 en MTIQ), sobre música de
   Jamendo. Lo que se toma: medir
   por separado parecido, concepto y las dos cosas con varios blancos válidos; construir la supervisión sola
   (un modelo independiente de parecido más clasificadores con umbral de alta precisión); una fusión chica
   entrenada sobre CLAP congelado; y que «quitar» algo no sale de la aritmética de embeddings.

**Estrategia: una zona segura con varias vistas, y dentro de ella un paso medido en la dirección pedida.**
Parecidos pasa a ser la zona (fusión de vistas, calibrada con sets profesionales). Cada playlist de concepto
parte de esa zona y ordena por el puntaje del concepto relativo al tema que suena. El puntaje de concepto se
elige con un banco local construido como el del paper, subiendo una escalera de métodos de menor a mayor
costo. Entre lo que sobrevive, decide Gabriel escuchando listas a ciegas. Detalle en «Estrategia».

## Lo medido en esta sesión

Biblioteca: índice `indice_recomendacion.json` (1760 temas), representaciones `clap_full`,
`maesthf_full_l7` y `maesthf_full_styles` de `musica` (1770 filas cada una, alineadas por `track_uid`; el
CLAP del índice coincide con `clap_full`, coseno 1,0). Sets: `d1.csv` (22 sets de Gabriel, 542 transiciones
con los dos temas en el índice) y D3 × biblioteca (79 pares de 67 sets, 48 DJs), con las funciones de
`zona_sets.py`. Los números de CLAP crudo reproducen el informe de zona (AUC 0,664, 0,695 y 0,574; pureza
0,10, 5,74 y 1,79 %).

### Parecidos: vistas y fusión

«Centrado» = restar la media de la biblioteca y normalizar. «z» = cada coseno en desvíos de su propia
distribución entre pares de la biblioteca, sumados con pesos iguales. Acierto = el tema siguiente de Gabriel
entre los 30 primeros, pool = biblioteca menos lo ya tocado, sin filtros. Discordantes = transiciones que
acierta solo la variante / solo CLAP crudo, con p binomial exacta. AUC = mismo set frente a pares al azar con
cada tema a ±4 BPM.

| Variante | Acierto D1 | Discordantes, p | Acierto, pool a ±8 BPM | AUC D3 × biblioteca (diferencia, IC 95 % por sets) | AUC D1 |
| :--- | --: | :--- | --: | :--- | --: |
| CLAP crudo | 9,04 % | — | 10,51 % | 0,664 | 0,574 |
| CLAP centrado | 9,78 % | 14 / 10, 0,54 | 11,31 % | 0,676 (+0,012; −0,029 a +0,054) | 0,605 |
| MAEST capa 7 centrado | 9,59 % | 32 / 29, 0,80 | 10,30 % | 0,743 (+0,080; +0,030 a +0,130) | 0,576 |
| Estilos Discogs | 8,30 % | 38 / 42, 0,74 | 9,29 % | 0,756 (+0,092; +0,033 a +0,159) | 0,546 |
| z: CLAP + MAEST | 11,44 % | 31 / 18, 0,085 | 13,33 % (31 / 17, 0,059) | 0,738 (+0,074; +0,030 a +0,120) | 0,600 |
| z: CLAP + estilos | 12,73 % | 34 / 14, 0,006 | 13,94 % (30 / 13, 0,014) | 0,750 (+0,086; +0,046 a +0,128) | 0,598 |
| z: las tres | 11,62 % | 35 / 21, 0,081 | 12,93 % (32 / 20, 0,126) | 0,774 (+0,110; +0,066 a +0,150) | 0,599 |
| Consenso: mínimo de percentiles, CLAP y MAEST | 11,62 % | 37 / 23, 0,092 | — | 0,718 | 0,601 |
| Consenso, las tres | 11,81 % | 43 / 28, 0,096 | — | 0,735 | 0,594 |
| CLAP blanqueado, 64 dimensiones | 6,27 % | 17 / 32, 0,04 | — | 0,600 | 0,548 |
| CLAP sin los 3 primeros componentes | 5,72 % | 14 / 32, 0,01 | — | 0,586 | 0,549 |

- Diferencia de acierto en D1 remuestreando sets: CLAP + estilos +3,7 puntos (+1,4 a +6,3); las tres +2,6
  (+0,8 a +5,2); CLAP + MAEST +2,4 (−0,2 a +4,4).
- Con el peso elegido dejando un set fuera, CLAP + estilos da lo mismo (12,73 %; el peso 1 sale en los 22
  sets). No depende de haber elegido el peso mirando el mismo set.
- Con el pool a ±4 BPM (412 transiciones) la ventaja se achica y no se distingue: 14,6 % frente a
  16,0-16,8 %, p 0,23 a 0,42.
- Cuánto refleja cada vista el tempo (Spearman entre parecido y diferencia de BPM, 200 000 pares): CLAP
  −0,16, CLAP centrado −0,05, MAEST −0,12, estilos −0,37. Los estilos llevan tempo; por eso el control.
- Juicios explícitos de Gabriel (`test_20`, tripletas uniformes, n = 55): CLAP 0,673, EffNet-Discogs 0,709,
  CLAP + EffNet 0,691; no se distinguen. MAEST de `test_20` no está extraído en este escritorio (2 temas).
  La coherencia con las carpetas de v1 no se usa: esas carpetas salieron de agrupar EffNet + MAEST.

### Conceptos: qué ordena el texto de CLAP

Frases en inglés escritas por el auditor, del tipo que arma el Worker. «Contraste» = puntaje del concepto
menos el promedio de los otros conceptos. «Sonda» = regresión logística sobre CLAP de audio, 5 pliegues por
artista (techo supervisado). P@30 = fracción de los 30 primeros con la etiqueta.

| Contra | Texto crudo | Con contraste | Sonda | Nota |
| :--- | --: | --: | --: | :--- |
| Género de Beatport, 10 géneros, 1429 temas: AUC medio | 0,58 con el nombre; 0,64 con 3 frases | 0,68 | 0,76 | House 0,47 y Progressive House 0,53 con texto crudo |
| Lo mismo, P@30 medio (base 0,09) | 0,21 | 0,25 | 0,38 | Tech House: 0,20 crudo, 0,57 con contraste |
| Estilos Discogs de MAEST, 11 estilos: AUC medio (activación > 0,3 frente a < 0,05) | 0,79 | 0,79 | — | Ambient 0,95; acid 0,83-0,91; hard techno 0,84; electro 0,54 |
| Lo mismo, P@30 medio | 0,14 | 0,15 | — | Estilos con 26 a 184 temas sobre 1760 |

### Conceptos: referencia más concepto (protocolo de la Tabla V del paper)

3000 consultas: un tema de referencia y un género de Beatport distinto del suyo como concepto; pool de 1266
temas. «Parecido» = el candidato está entre los 127 vecinos más cercanos de la referencia por MAEST (10 %),
criterio independiente de CLAP. «Concepto» = tiene ese género. Fracción de los 30 que cumple cada cosa. Hay
pocos blancos válidos (mediana 7 por consulta, 3 a 14 entre cuartiles): en la consulta mediana una lista
perfecta tendría 0,23 en «las dos».

| Método | Parecido | Concepto | Las dos |
| :--- | --: | --: | --: |
| Al azar | 0,101 | 0,097 | 0,009 |
| Solo audio (Parecidos) | 0,556 | 0,080 | 0,042 |
| Solo texto (centrado; el agente de conceptos de hoy usa el coseno crudo, misma familia) | 0,097 | 0,216 | 0,020 |
| Suma cruda de audio y texto («Sum» del paper, T18 del plan) | 0,296 | 0,167 | 0,041 |
| Suma de z con contraste | 0,324 | 0,197 | 0,051 |
| Dos etapas: zona de 80 por audio, orden por contraste | 0,445 | 0,122 | 0,052 |
| Dos etapas, zona de 150 | 0,359 | 0,152 | 0,051 |
| Dos etapas, zona de 300 | 0,253 | 0,193 | 0,046 |
| Suma de z con la sonda (techo) | 0,251 | 0,279 | 0,067 |
| Dos etapas, zona de 150, con la sonda (techo) | 0,350 | 0,180 | 0,061 |

## Afirmaciones

| Afirmación | Etiqueta | Evidencia |
| :--- | :--- | :--- |
| Las listas de 30 de CLAP, MAEST y estilos comparten 2,9 a 6,8 temas, y aciertan transiciones distintas (53, 52 y 45; unión 110 de 542) | EVIDENCIA OBSERVADA | `explora_fusion.py`, D1 |
| La suma con pesos iguales supera a CLAP crudo en el acierto en 30 de D1 | EVIDENCIA OBSERVADA, exploratoria | 11,4-12,7 % frente a 9,0 %; una de tres pasa p < 0,05 (0,006); más de diez variantes miradas |
| En sets de otros DJs, MAEST, estilos y las fusiones separan mejor «mismo set» de azar del mismo BPM | EVIDENCIA OBSERVADA | D3 × biblioteca, 79 pares, intervalos por sets que no tocan 0 |
| Parte de la ventaja en D1 es tempo | EVIDENCIA OBSERVADA | Estilos −0,37 con la diferencia de BPM; a ±8 BPM la ventaja sigue (p 0,014 y 0,059), a ±4 no se distingue |
| La fusión da una lista de Parecidos mejor | HIPÓTESIS | La deciden los pares de 1001Tracklists × biblioteca por DJ, Bedrock con MAEST y la escucha E12 |
| Blanquear CLAP o quitarle los 3 primeros componentes empeora | EVIDENCIA OBSERVADA | 5,7-6,6 % frente a 9,0 %, p 0,01 a 0,09; quitar solo el primero no se distingue (8,1 %, p 0,42) |
| El agente de conceptos ordena solo por texto; del tema que suena usa el filtro de BPM ±20 cuando el concepto lo pide, y no aplica la energía ni las frases negativas | EVIDENCIA OBSERVADA | `conceptos_agente.py` en el commit `73d042d`: `resolver` (`bib.matrix @ query`) y `filtrar`; no se corrió |
| Una lista «solo texto» se parece a la referencia como una al azar | EVIDENCIA OBSERVADA en el banco exploratorio | 0,097 frente a 0,101 |
| La suma cruda no supera a solo audio en cumplir las dos cosas | EVIDENCIA OBSERVADA en el banco exploratorio | 0,041 frente a 0,042; coincide con el paper, donde «Sum» queda debajo de «Audio-Only» en MTSQ |
| Ordenar una zona chica por el concepto conserva más parecido con el mismo cumplimiento | EVIDENCIA OBSERVADA en el banco exploratorio | Zona de 80: parecido 0,445 y las dos 0,052; suma de z: 0,324 y 0,051 |
| El puntaje de concepto limita más que la fórmula | INFERENCIA | Las fórmulas con texto van de 0,041 a 0,052; con la sonda, 0,061-0,067; todas lejos del techo |
| El texto de CLAP ordena descripciones acústicas y casi no ordena por nombre de género | EVIDENCIA OBSERVADA, con frases del auditor | AUC 0,79 frente a 0,58-0,64 |
| El paper no publica pesos; MTIQ y MTSQ están en Hugging Face (Apache-2.0, 59,7 GB en total) | EVIDENCIA OBSERVADA | github.com/sb-lucas/Guided-Music-Retrieval y huggingface.co/datasets/sb-lucas/Music-Text-Queries, leídos hoy |
| En el paper, la fusión entrenada sobre CLAP congelado supera a las líneas de base sin entrenar | EVIDENCIA OBSERVADA del texto | Tabla IV: MTIQ R@100 2,08 (Sum) frente a 10,30 (Combiner) y 14,75 (TG-CIR). Tabla V, «las dos»: 30,5 frente a 61,8 y 69,4 |
| En el paper, quitar un atributo falla sin entrenar | EVIDENCIA OBSERVADA del texto | Tabla VIII: Sum 0,8 en quitas frente a 4,3 en adiciones |
| En el paper, CLAP es el mejor de siete modelos en parecido juzgado por personas | EVIDENCIA OBSERVADA del texto | Tabla I, Dim-Sim: CLAP 87,1; MusiCNN 85,9; MAEST 85,0; EffNet 78,6; MERT 67,1 |
| Una fusión entrenada como la del paper sirve para conceptos escritos libremente | HIPÓTESIS | El paper no lo mide: entrena y evalúa con el mismo conjunto cerrado de modificaciones |
| Las cabezas de Essentia sirven de coordenada de energía y de etiquetas del banco | HIPÓTESIS | Falta la pasada en WSL (V6, T19); el paper las usa de etiqueta con umbrales de alta precisión |
| El audio de Bedrock y de D7 sigue en disco (11 GB) | EVIDENCIA OBSERVADA | `D:\TRAKTOR ML datos\dj_sets\audio`, 88 carpetas; D: tiene 50 GB libres |
| La historia de Traktor no da negativos útiles | EVIDENCIA OBSERVADA | 68 entradas con `PLAYEDPUBLIC` 0 frente a 718 con 1, en 40 archivos |

## Revisión hostil de esta auditoría

- **¿Elegí la fusión mirando el resultado?** Los pesos iguales se fijaron antes de correr, pero se miraron
  más de diez variantes en los mismos 22 sets. Lo que sostiene la lectura es que las seis combinaciones
  probadas suben (+1,9 a +3,7 puntos; la sexta, CLAP con peso doble más MAEST, no está en la tabla), que
  D3 × biblioteca, de otros DJs y con el BPM controlado, va en la misma dirección, y que las vistas sueltas
  no suben. Aun así es exploratorio y D1 es referencia con ruido (E6).
- **¿Es tempo?** En parte. Gabriel pidió dejar BPM y tonalidad fuera. Toda medida de la fusión tiene que
  llevar el control de tempo, y hay que probar los estilos con el tempo descontado.
- **¿«Mismo set» es «suena bien mezclado»?** No. Es coherencia de estilo dentro de un set. Un par al azar
  del mismo BPM cruza géneros, y los estilos son predicciones de género: el AUC premia la coherencia de
  género. Si eso es lo que Gabriel quiere en Parecidos lo dice la escucha.
- **¿Ya se había decidido esto?** Gabriel eligió CLAP frente a MAEST capa 7 escuchando organizaciones
  enteras (DECISIONS 2026-09-29, no del todo a ciegas). Eso comparó dos modelos solos para armar carpetas.
  La fusión y la lista en vivo no pasaron por su oído.
- **¿El banco de conceptos mide lo que importa?** Mide género editorial, con frases mías y un criterio de
  parecido exigente. Sirve para ordenar métodos, no para predecir cuánto le va a gustar una lista. Los
  conceptos reales de Gabriel (energía, oscuro, hipnótico) no tienen etiqueta todavía.
- **¿Leí el código o el comentario?** Del agente de conceptos leí el encabezado y las líneas de `resolver` y
  `filtrar` que arman el puntaje y las notas; no lo corrí. Es de otra sesión.
- **Lo que no miré:** el volumen y el cruce de 1001Tracklists con la biblioteca (en importación); las
  cabezas de Essentia; la fusión sobre audio mezclado de D7; las frases que escribe Gemma de verdad; la
  medición de los 4 conceptos del 11-08 (no la reproduje).

## Estrategia

### 1. Parecidos = zona segura con varias vistas

- **P11, fusión de vistas.** Parecido = suma de z de CLAP centrado, MAEST capa 7 centrado y estilos
  Discogs. Pesos iguales de partida. EffNet-Discogs entra como cuarta vista cuando exista para `musica`.
- **P12, probabilidad de «van juntos».** Una regresión logística sobre los parecidos por vista, entrenada
  con pares de un mismo set profesional frente a pares al azar del mismo BPM, con los DJs de prueba fuera
  del entrenamiento. Da los pesos de P11 y una probabilidad calibrada para el campo `seguro` y para cortar
  la lista. Son 3 o 4 parámetros: alcanzan los cientos de pares que hay. La capacidad sube con los datos:
  proyección por vista (P9) con miles de pares, ranker (M3) después, tronco afinado (M4) al final.
- **P13, consenso.** Cuántas vistas ponen al candidato entre sus K primeros. Es la marca de seguridad que no
  necesita entrenamiento.
- **E14, control de tempo en toda medida de parecido**: azar del mismo BPM, pool a ±8 BPM y la correlación
  de cada vista con la diferencia de BPM. BPM y tonalidad no entran al puntaje.
- **Datos para decidir**: Bedrock con MAEST extraído de sus 179 originales (el audio está en disco), D3 ×
  biblioteca, los pares de 1001Tracklists × biblioteca por DJ cuando lleguen, y D1 informado aparte.
- **Juez**: E12, diez temas principales, CLAP crudo frente a la fusión, listas enteras a ciegas.
- **En vivo no cambia la arquitectura**: la suma de cosenos por vista es el producto punto de los vectores
  concatenados y escalados. El índice lleva un vector más largo y el lado vivo sigue con la biblioteca
  estándar.

### 2. Conceptos = zona más dirección, con el puntaje elegido en un banco local

- **T23, dos etapas.** Candidatos = la zona de Parecidos alrededor del tema que suena. Orden = puntaje del
  concepto relativo al del tema que suena. Lista de largo variable, con un piso de confianza: menos temas
  pero seguros.
- **E13, banco de composición local**, como la Tabla V: parecido, concepto y las dos cosas, como fracción de
  los 30; adición y quita por separado; conceptos vistos y no vistos. Atributos con etiqueta independiente
  del método: género de Beatport, estilos Discogs, Vocal, cabezas de Essentia y fase del set cuando
  existan. El parecido se juzga con un modelo que el método no use (EffNet o MusiCNN, como hace el paper).
- **Escalera de puntajes de concepto**, de menor a mayor costo; se sube un peldaño solo si el banco muestra
  la brecha:
  1. T24: texto de CLAP con varias frases, centrado en la biblioteca y contraste contra un banco fijo de
     descripciones (medido: 0,64 → 0,68).
  2. T25: sondas supervisadas para los conceptos con etiqueta (voz, géneros, estilos, energía). Son el
     techo y sirven tal cual para las direcciones fijas.
  3. T26: mapa aprendido de texto a audio. Con los vocabularios de etiquetas de la biblioteca (47 géneros,
     400 estilos, cabezas de Essentia) se aprende a llevar la descripción de una etiqueta al lugar del
     espacio de audio donde están sus temas, y se evalúa con etiquetas que no vio. Es la lección del paper
     (fusión aprendida sobre CLAP congelado) con supervisión propia.
  4. T27, condicional: fusión tipo Combiner entrenada con MTIQ más tripletas locales construidas igual.
     Pide bajar el audio de MTIQ y unas horas de CLAP en CPU: antes, estimación medida y el OK de Gabriel.
- **T28, quitar y «menos».** Un eje con dos polos (frases positivas menos negativas) o una sonda. Nunca un
  texto «sin X». Las `negative_prompts` del Worker se miden en el banco antes de usarse.
- **Energía** (`constraints.energy`): la coordenada sale de T19 (cabezas de Essentia) y T20 (posición en el
  set), validada con E11, como ya estaba decidido. Sube de prioridad porque la pasada de Essentia da tres
  cosas a la vez: el modelo independiente del banco, las etiquetas de los atributos y los candidatos a
  coordenada.

### 3. Orden

1. Fusión de vistas medida como código con tests y con Bedrock; ronda de escucha E12.
2. Prueba chica de Essentia en WSL y, si el tiempo por tema lo permite, la biblioteca entera.
3. Banco E13 y los peldaños 1 y 2 de la escalera; el agente de conceptos pasa a dos etapas.
4. Peldaño 3; coordenada de energía con E11; escucha de listas de concepto.
5. Solo si hace falta: peldaño 4.

Prompt para el agente implementador: `docs/plans/PROMPT_FUSION_Y_CONCEPTOS.md`.

## Inventario: ideas nuevas y cambios de estado

Para sumar al plan (sección 4 y sección 11) cuando Gabriel apruebe; nada de lo anterior se borra.

| Id | Idea | Origen | Etapa | Estado |
| :--- | :--- | :--- | :--- | :--- |
| P11 | Fusión de vistas para Parecidos: CLAP, MAEST capa 7 y estilos Discogs centrados, suma de z | Auditoría 11-09; G6 del plan; Tabla I del paper | Ahora | Exploratorio medido; falta Bedrock, 1001Tracklists y E12 |
| P12 | Probabilidad calibrada de «van juntos» sobre los parecidos por vista; campo `seguro` y corte de lista | Auditoría 11-09; reemplaza al radio de P8 como definición de seguro | Ahora | Pendiente |
| P13 | Consenso entre vistas como marca de seguridad | Auditoría 11-09 | Ahora | Exploratorio medido (11,6-11,8 %) |
| E13 | Banco de composición local: parecido, concepto y las dos; adición y quita; conceptos vistos y no vistos | Paper GMR, Tabla V y sección VI | Ahora | Versión exploratoria medida |
| E14 | Control de tempo en toda medida de parecido | Gabriel 11-09 (BPM y tonalidad se ajustan después) | Siempre | Vigente |
| T23 | Conceptos en dos etapas: zona de Parecidos y orden por el concepto relativo al tema que suena; largo variable | Auditoría 11-09; paper (parecido y modificación son dos condiciones) | Ahora | Exploratorio medido |
| T24 | Puntaje de texto con varias frases, centrado y contraste contra un banco fijo | Auditoría 11-09; T18 | Ahora | Exploratorio medido |
| T25 | Sondas supervisadas por concepto con etiqueta; techo del banco | `genre_model.py`, `clap_probe`; auditoría 11-09 | Ahora | Techo medido para género |
| T26 | Mapa aprendido de texto a audio con los vocabularios de etiquetas de la biblioteca, evaluado con etiquetas no vistas | Paper GMR (fusión entrenada sobre CLAP congelado) | Después de E13 | Pendiente |
| T27 | Fusión tipo Combiner entrenada con MTIQ y tripletas locales | Paper GMR | Condicional | Pendiente; los autores no publican pesos |
| T28 | Quitar y «menos X» como eje de dos polos o sonda; `negative_prompts` medidas antes de usarse | Paper GMR, Tabla VIII; nota del agente de conceptos | Con E13 | Pendiente |
| D9 | MTIQ y MTSQ (Hugging Face `sb-lucas/Music-Text-Queries`, Apache-2.0; audio de Jamendo recortado a 10 s y tripletas) | Paper GMR | Condicional | Sin bajar |

Cambios de estado propuestos:
- **G6** (estilos Discogs como vector): medido; entra en P11.
- **P8** (radio de zona): `seguro` pasa a definirse con P12 y P13; el radio queda como referencia.
- **P9** (proyección de CLAP aprendida de sets): es el paso siguiente de P12 cuando haya miles de pares.
- **T18** (suma de audio y texto): medido en el banco exploratorio; no supera a solo audio. Sigue como línea
  de base de E13. Lo reemplazan T23 y T24.
- **V6 y T19** (Essentia en WSL): suben de la etapa B al segundo lugar.
- **P2** (CLAP en escala comparable): lo cubre la suma de z.

## Para las otras sesiones

- **Importación de 1001Tracklists**: lo que esta estrategia necesita es, por set, DJ, fecha, posición de cada
  tema y su cruce con la biblioteca por artista y título con la versión (`zona_sets.estricto`). Sirven los
  pares consecutivos y los del mismo set.
- **Agente de conceptos**: el cambio a dos etapas toca `resolver`. Conviene que entre por una función de
  puntaje separada, para que el banco E13 y el agente usen la misma. La candidata a energía que anotó esa
  sesión (Energy de Mixed In Key; DECISIONS 2026-11-09, conceptos, punto 2) choca con la decisión de Gabriel
  del 2026-11-07, punto 3: no se usa.
