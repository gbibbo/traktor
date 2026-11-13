# Línea A: protocolo de A0 y A1

Lo escribió la sesión de la línea A el 2026-11-13, antes de calcular ninguna vista nueva y antes de medir nada en el
tablero. Plan: `docs/plans/plan_recomendacion_v2.md`, sección 2.3. Dataset: DJSets v1 (`erclm:traktor_ml/djsets_v1/`,
armado en el commit `4c83050`), que no se modifica.

**Regla de este documento.** Los umbrales de abajo valen desde el commit que lo agrega. Si cambian antes de medir (por
la auditoría o por Gabriel), el cambio va en otro commit y el informe cita el último anterior a la medición. Lo que se
agregue después de ver datos va marcado como exploratorio.

## 0. Conteo de partida (medido el 2026-11-13)

Copia local de las tablas y de las 60 salidas de V7 en `D:\TRAKTOR ML datos\linea_a\`; el Drive manda.

| Qué | Cuántos |
| :--- | ---: |
| Temas con audio en `tracks.csv` | 33 321 (30 255 de Beatport, 3066 de YouTube) |
| Grabaciones distintas (`recording_id`) | 33 226 (30 169 `bp_`, 3057 `yt_`) |
| Filas en las 60 salidas de V7 | 19 887; 19 847 ids distintos (los 40 repetidos son idénticos), 0 fallidos, 0 valores no finitos |
| Grabaciones de Beatport cubiertas por V7 | **19 847 de 30 169** (19 923 temas); ningún id de V7 queda fuera del dataset |
| Faltan de Beatport | **10 322 grabaciones**, todas en 38 shards sin salida de V7: 25 de `lote_05`, 7 de `res_lote_04`, 6 de `res_lote_05` (14,8 GB). Dentro de los 60 shards con salida no falta ninguna |
| Faltan de YouTube | **las 3057 grabaciones** (34 shards, 19,0 GB; M4A AAC de 129 kbps a 44,1 kHz; duración mediana 377 s) |
| Los 166 previews de E15 | los 166 están en V7 |
| Tiempo de V7 en Kaggle | 1,95 s por preview (`traktor-v7-2`, 6343 previews) y 2,02 s (`traktor-v7-5`, 10 266), con la bajada incluida (0,04 a 0,07 s), una GPU |

## A0. Vistas de todo el dataset

### Qué se calcula

- CLAP, MAEST capa 7 y los 400 estilos Discogs, por segmento, con el mismo código de modelos que V7
  (`tools/dj_sets/vistas_previews.py`, clase `Modelos`, que usa `ClapBackend` y `MaestHfBackend`). Se reutilizan las
  salidas de V7 para las 19 847 grabaciones que cubren, si pasa el umbral 4 de abajo.
- Se calculan las 10 322 grabaciones de Beatport que faltan y las 3057 de YouTube, leídas de los shards de
  `djsets_v1/audio/`, no de las carpetas de la descarga.

### Ventana

- **Beatport:** la de V7 y E15 (V4). El preview entero, en 4 segmentos seguidos de 30 s desde el comienzo, con ceros si
  es más corto.
- **YouTube:** se decodifica el tema entero y se toma la ventana que tendría su preview. Beatport empieza el preview en
  el 40 % del tema (`preview_start_s / bp_length_s` = 0,400, p5 a p95 de 0,3998 a 0,4002, en 30 254 previews) y la
  mayoría dura 120 s (28 765 de 30 255). Entonces:
  - comienzo = 0,40 × D, donde D es la duración decodificada;
  - largo = mín(120 s, D − comienzo);
  - los mismos 4 segmentos de 30 s, con ceros si falta.
- **Códec de YouTube.** Los previews son MP3 de 96 kbps. E17 midió que comprimir a 96 kbps corre a CLAP +3,20 desvíos.
  Además, 453 de los 2525 pares consecutivos de prueba sin temas vistos (A1) tienen un tema de YouTube. Por eso la ventana
  de YouTube se recodifica a MP3 CBR de 96 kbps con LAME, en estéreo, como `codec_clap.mp3_ida_y_vuelta` (E17), antes de
  los modelos. Esa es la vista principal. Como control, también se calcula la vista con el códec original para el 10 %
  de las grabaciones de YouTube, elegido por hash (`md5(recording_id) % 10 == 0`, unas 306).

### Salidas, en `erclm:traktor_ml/djsets_v1_vistas/`

- `a0/beatport/beatport_NNN.npz` y `a0/youtube/youtube_NNN.npz`, uno por shard del dataset: `recording_id`,
  `segundos` (audio real en la ventana), `clap` (n, 4, 512), `l7` (n, 4, 768) y `styles` (n, 4, 400) en float32.
  - En YouTube, además: `comienzo_s` y `duracion_s`, y para el control `control_id` con `clap_orig`, `l7_orig` y
    `styles_orig`.
  - `meta` con el commit, los modelos, el dispositivo, el trabajo de gpuq y los fallidos con su motivo.
- `a0_prueba/` y `a0_cpu/`: lo mismo, para el trabajo corto y la prueba en CPU.
- `vistas/`, que arma esta PC juntando V7 y A0:
  - `indice.csv`: una fila por tema con audio, en el orden de `tracks.csv`, con `track_id`, `recording_id`, `fuente`,
    `origen` (V7 o A0, con el trabajo), `segundos` y `n_seg_validos`;
  - `clap.npy`, `l7.npy` y `styles.npy`, de forma (33 321, 4, d), en float32;
  - `youtube_control.npz`;
  - `LEEME.json`, con la ventana, los commits, los trabajos y los md5.

### Umbrales para cerrar A0

1. **Cobertura:** cada uno de los 33 321 temas con audio tiene vistas, o figura como fallido con su motivo. Los
   fallidos son 0,5 % o menos (167 temas como máximo).
2. **Finitos:** el 100 % de los valores.
3. **Alineación:** el `recording_id` de cada fila coincide con el de `tracks.csv` para su `track_id`. Los temas que
   comparten grabación tienen filas idénticas. Cada salida por shard cubre exactamente los miembros de ese shard que
   están en `tracks.csv`.
4. **Mismo código que V7:** el trabajo corto vuelve a calcular un shard que ya cubre V7 (el que más previews de E15
   tiene). Por vista y por segmento, el coseno con V7 tiene que dar una mediana de 0,9999 o más y un mínimo de 0,999 o
   más. Si no pasa, V7 no se reutiliza: se recalcula todo, unas 11 h más de GPU, y vuelvo a pedir el OK.
5. **E15:** los 166 previews de E15 en `vistas/`, frente a `e15/emb/<id>.npz` (`prev_*`), por vista y segmento: la
   mediana del coseno es 0,999 o más y el mínimo, 0,99 o más.
6. **Ventana de YouTube:** comienzo / D = 0,40 y `segundos` = mín(120, 0,6 D) en todos.
7. *Se informa, sin umbral:* cuánto corre a cada vista la recodificación de YouTube en las grabaciones de control,
   sobre la dirección `u1` de E17 y en coseno con el original.

### Cómo se calcula

- Código: `tools/dj_sets/vistas_djsets.py`, con los subcomandos `correr`, `juntar` y `verificar`, y sus tests en
  `tests/v4/test_vistas_djsets.py`.
- **Se retoma, en sesiones cortas** (plan, 2.5; Gabriel, 11-13). Las unidades de trabajo son los shards. Se salta todo
  shard cuya salida ya está en el remoto, y se sube un parcial cada 96 temas, como en V7: son unos 3 minutos en Kaggle y
  menos de 2 en Colab, nunca más de 10 sin guardar. Cada trabajo de gpuq lleva una lista fija de shards que suma una
  hora de trabajo útil o menos, y va con `--platform auto`.
- **Prueba chica en CPU**, con el mismo código y con `TRAKTOR_REMOTE` apuntando a una copia local con las rutas del
  Drive. Usa un shard de `lote_05`, el shard de V7 con más previews de E15 y `youtube_000`, 6, 6 y 4 temas. Comprueba:
  - formas, valores finitos y alineación;
  - la ventana de YouTube;
  - el coseno de los previews de E15 con V7 y con E15;
  - el tiempo por tema en CPU de cada tipo;
  - `juntar` sobre esa salida chica.
- **Resultado de la prueba en CPU** (2026-11-13). Los 8 previews de E15 de `beatport_002`, frente a V7 y frente a
  E15, dan coseno 1,000 en las tres vistas y en los 32 segmentos. En YouTube, comienzo / D = 0,40 y 120 s, con D igual
  a `audio_duration_s`.
  - El control mueve a CLAP (coseno de 0,91 a 0,93 entre la ventana recodificada y la original) y casi nada a MAEST
    (0,997) ni a los estilos (0,999).
  - Preparar el audio de YouTube cuesta de 2,5 a 3 s por tema en CPU: decodificar el tema entero, codificar a MP3,
    decodificar y alinear. Por eso se prepara en 4 hilos mientras la GPU procesa el lote anterior. Con hilos, el
    resultado es idéntico bit a bit, y en esta PC baja a 1,35 s por tema.
- **Trabajo corto** (`traktor-a0-prueba-1`, `--platform auto`), sobre esos mismos tres shards enteros, con salida en
  `a0_prueba/`. Se corta a propósito una vez y se comprueba que retoma desde el parcial.
  - Si YouTube pasa de 4,5 s por tema, que es 1,5 veces lo estimado, no se mandan sus trabajos completos sin preguntar
    a Gabriel.
- **Trabajos completos** (`traktor-a0-1` en adelante), con salida en `a0/`. Son los 72 shards que faltan, repartidos en
  unos 10 trabajos de una hora o menos:
  - de Beatport, 6 shards por trabajo: unas 272 grabaciones por shard, a 2 s cada una, dan unos 9 min por shard;
  - de YouTube, 12 shards por trabajo: unas 90 grabaciones por shard, a unos 3 s cada una.

## A1. Tablero fijo

El tablero se arma con los metadatos, así que no espera a A0. Las líneas de base de las vistas sí esperan a A0.

### Partición

- **Base:** `split_by_dj.csv`, sin cambiar la etiqueta de ningún set. Asigna 1083 sets a train, 234 a validation y 232
  a test.
- **Su fuga, medida:** el hash sale del primer DJ del nombre, pero los DJs aparecen también como segundo nombre de sets
  ajenos.
  - 136 de los 232 sets de test tienen un nombre que figura en algún set de train: MEDUZA, Eli Brown, Darius Syrossian,
    John Digweed...
  - 301 sets de train tienen un nombre que figura en test.
- **Regla: cuarentena.** Se parte el nombre del set: lo que va antes de « - », « @ » o «(», cortado en «&», «b2b»,
  «vs», «x» y comas, en minúsculas. A cada nombre se le aplica el hash del armador (`build_djsets.split_de`). Si los
  nombres de un set caen en partes distintas, el set queda en cuarentena: no entra en ninguna parte del tablero ni en
  ningún entrenamiento.
  - Medido: quedan en cuarentena 89 sets de train, 43 de validation y 57 de test. Después, ningún nombre está en dos
    partes.
  - Quedan train con 994 sets y 251 DJs, validation con 191 y 32, y test con 175 y 34.
  - Limitación: dos grafías distintas del mismo DJ no se detectan.
- **Temas vistos:** un tema de validation o test está «visto» si su `track_id` o su `recording_id` aparece en un set de
  train fuera de la cuarentena.

### Positivos (pares de un set de test, temas con audio)

- **P1, el principal:** consecutivos, tal como los arma `ds.pairs` (salta las filas «w/» y corta en las filas sin tema).
- **P2:** a 2 o 3 posiciones del tracklist.
- **P3:** a 4 posiciones o más.
- En los tres casos, el ancla es el tema que sonó antes. Se excluyen los pares cuyos dos temas comparten grabación.
- **Tablero principal:** los dos temas no vistos y los dos con BPM y género de Beatport.
  - Medido para P1: 2525 pares no vistos, de los que 2068 tienen metadatos, de 170 sets y 34 DJs.
  - Los pares con temas vistos se informan aparte. También aparte, solo con N1, los pares sin metadatos, que casi siempre
    tienen un tema de YouTube.

### Negativos

- Por cada positivo (a, b) se toman 20 negativos (a, c) en cada nivel, con la semilla 0.
- c sale del pool de test: los temas de los sets de test del tablero, no vistos, con audio y con metadatos. Además:
  - c es de un set de otro DJ (otra `split_key`) que el de a;
  - c no aparece nunca en un mismo set que a, en todo el dataset;
  - la grabación de c es distinta de las de a y b.
- Los tres niveles:
  - **N1:** al azar dentro del pool.
  - **N2, mismo BPM respecto del ancla:** | |BPM(c) − BPM(a)| − |BPM(b) − BPM(a)| | ≤ 1, con el BPM de Beatport. Así
    el BPM solo no separa: es la versión de «mismo BPM» que corresponde a negativos con ancla.
  - **N3:** N2, y además el mismo género de Beatport que b.
- Si un nivel tiene menos de 20 candidatos se usan todos. Si no tiene ninguno, el positivo sale de ese nivel. Las dos
  cosas se cuentan.
- **Pesos:** cada set suma 1 en los positivos y 1 en los negativos de cada nivel y familia.
- **Validation y train** se arman igual, cada una con su propio pool. Validation sirve para elegir modelos en A2 y A3.
  Train, para ajustar. Test no elige nada.
- *Agregado al programar, antes de medir (2026-11-13):*
  - train lleva 5 negativos por nivel en lugar de 20, porque solo sirve para ajustar y así la tabla entra en memoria;
  - P3 tiene un tope de 50 pares por set, sorteados con la semilla 0, para que las tablas no crezcan con el cuadrado
    del largo del set.
- **Conteos del armado** (2026-11-13, todavía sin medir ninguna línea de base):
  - C1 y C2 pasan;
  - test: 175 sets y 34 DJs. P1 principal: 2076 positivos; 1948 tienen negativos N3 (168 sets, 34 DJs) y 575 de
    ellos tienen menos de 20;
  - validation: 191 sets y 32 DJs. P1 principal: 2580 positivos; 2504 tienen negativos N3;
  - train: 994 sets y 251 DJs. P1: 20 067 positivos.
- Las tablas de pares se escriben una vez en `erclm:traktor_ml/linea_a/a1_tablero/`, con su md5. Todo modelo posterior
  lee esas tablas y no vuelve a sortear.

### Líneas de base

Ninguna se entrena con pares. Solo se ajustan medias y desvíos, con train.

- **El vector de un tema**, por vista, es la media de sus segmentos válidos, normalizada. Un segmento k es válido si
  `segundos` ≥ 30 k + 15; el segmento 0 siempre lo es. En YouTube se usa la vista recodificada a 96 kbps.
- **bpm:** máx(0, 1 − |ΔBPM| / 20), la fórmula del recomendador, con el BPM de Beatport.
- **tonalidad:** `harmonic.best_key_shift` sobre el Camelot de `bp_key`.
- **genero:** la taxonomía del recomendador (`genre_taxonomy`), con el género de Beatport.
- **Las vistas:**
  - `clap` y `estilos`: coseno crudo;
  - `clap_c` y `maest_c`: centradas con la media de los temas de train;
  - `v8c`: `clap_c` sin la dirección `u1` de E17 (`dj_sets/e17/direcciones.npz`, k = 1). Esa dirección sale de la
    biblioteca, así que no usa nada del dataset.
- **pesos iguales:** la suma de los factores (`v8c`, `maest_c`, `estilos`, bpm, tonalidad y género), cada uno en
  desvíos de los negativos N1 de train. Un factor que falta vale 0. Se informa también la misma suma con `clap_c`, que es
  la de la unidad 1.
- **azar:** un puntaje aleatorio, como control.

### Métrica e intervalos

- AUC pesado de los positivos frente a los negativos, por familia (P1, P2, P3) y por nivel (N1, N2, N3).
- [REEMPLAZADO 2026-11-13 por la auditoría `auditoria_protocolo_a1_2026-11-13.md`: el resultado principal es P1 frente
  a N4; ver «Cambios por la auditoría»] **Resultado principal:** P1 frente a N3.
- **Intervalos del 95 %:** se remuestrean los DJs de test (`split_key`), 1000 veces con la semilla 0, los mismos
  remuestreos para todos los puntajes. Se dan para el AUC de cada uno y para su diferencia, pareada, con «pesos iguales
  con v8c» y con «bpm» (y, desde la auditoría, también con «clap»).

### Cambios por la auditoría (2026-11-13, antes de medir)

Vienen de `docs/reports/auditoria_protocolo_a1_2026-11-13.md` (commit `7e2fa76`). Con el tablero de antes, «fecha
cercana» sola daba AUC 0,737 en P1 frente a N3, sin oír nada.

1. **Nivel N4, igualado en fecha.** Es N3 y además el año de lanzamiento del candidato (`bp_release_date`) a ±1 del de
   b. Un positivo cuyo b no tiene fecha no entra en N4. Se sortea del pool entero, con 20 negativos por positivo (5 en
   train), en las tres partes.
   - **Resultado principal: P1 principal frente a N4.** N3 se sigue informando.
   - El tamaño mínimo pasa a medirse en P1 frente a N4: 1500 positivos o más, de 25 DJs o más.
2. **Subconjunto limpio, informado aparte sin cambiar el tablero.** El resultado principal sin tres grupos de pares:
   - los que tienen un tema con otra versión en train: mismo artista y mismo título sin lo que va entre paréntesis o
     corchetes, o el mismo ISRC;
   - los que comparten un artista;
   - los que comparten sello de Beatport.
3. **Líneas de base nuevas:**
   - **fecha:** menos la diferencia de año entre a y el candidato;
   - **popularidad:** `n_sets` del candidato.
   - «clap» (CLAP crudo, el producto de hoy) se suma a las referencias de las diferencias pareadas.
4. **Acierto en el primer puesto:** la fracción pesada de positivos en los que b queda por encima de todos sus
   negativos del nivel (en un empate cuenta la parte que le toca). Lleva el mismo intervalo por DJs.
   - **Regla para pedirle una escucha a Gabriel.** Solo cuando un candidato cumple dos condiciones en P1 frente a N4:
     le gana a «clap» y a «pesos iguales (v8c)» con el intervalo de la diferencia entero sobre 0, y su acierto en el
     primer puesto no es menor que el de los dos.
   - Motivo: en la escucha corta del 11-13, 0,08 de AUC a favor no se oyó.
5. **En A0, control de fuente.** Para cada vista se mide si los 10 vecinos de un tema de YouTube son de YouTube más de
   lo esperado por su proporción. Se informa, sin umbral.
6. **Los temas que en train solo aparecen encima de otro («w/») cuentan como vistos.** Son 6.

Controles nuevos:
- C3 se extiende a N4: bpm entre 0,47 y 0,53, y género 0,500 ± 0,005.
- C5: en N4, el AUC de fecha está entre 0,47 y 0,53.

**N4 corregido por el control C5 (2026-11-13, antes de medir ningún modelo).** Con la definición de arriba (año del
candidato a ±1 del de b), C3 y C4 pasaron pero C5 falló: la fecha sola daba 0,571 [0,540; 0,606] en N4. Lo midió
`medir` solo con metadatos (`D:\TRAKTOR ML datos\linea_a\a1_medir_metadatos.json`, commit `d598fcb`), y la auditoría
lo encontró por su lado (commit `cd049e7`).
- La causa es la misma que con el BPM: igualar el año a b deja que un b del mismo año que a le gane a un candidato un
  año más lejos de a.
- [REEMPLAZADO 2026-11-13 por el fallo de C5] N4 = N3 y el año del candidato a ±1 del de b.
- **N4 vigente: N3 y |año(c) − año(a)| = |año(b) − año(a)|, en años enteros.** Iguala la distancia al ancla, como N2
  con el BPM, así la fecha empata por construcción. Necesita el año de a y el de b.
- Con esa regla, P1 principal frente a N4 en test tiene **1480 positivos de 33 DJs**: 20 menos que el mínimo de 1500.
  Por la regla de arriba, se informa y se pregunta a Gabriel antes de cambiar nada; mientras tanto no se mide ningún
  modelo con el tablero (A0 todavía no terminó).
- En validation, 2133 positivos de 32 DJs; en train, 18 907.

Conteos del primer armado con estos cambios (N4 con ±1; 2026-11-13, todavía sin medir ninguna línea de base):
- C1 y C2 pasan.
- **test:** P1 principal tiene 2067 positivos; eran 2076 y 9 pasan a vistos por las filas «w/».
  - Con negativos N4: **1660 positivos de 33 DJs**, así que el tamaño mínimo se cumple.
  - Marcados: 249 con otra versión en train, 259 con artista en común y 230 con el mismo sello. Limpios: 1442.
- **validation:** 2286 positivos con negativos N4, de 32 DJs.
- **train:** 19 339 positivos con negativos N4.

### Controles de construcción

Si alguno falla, el tablero tiene un error y no se informa nada medido con él.

- C1: después de la cuarentena, ningún nombre de DJ está en dos partes.
- C2: ningún tema del pool de test, por `track_id` o por `recording_id`, está en train.
- C3: en N2 y en N3, el AUC de bpm está entre 0,47 y 0,53. En N3, el AUC de género es 0,500 ± 0,005.
- C4: el AUC del azar contiene 0,5 en su intervalo.
- **Tamaño mínimo:** P1 frente a N3 necesita 1500 positivos o más, de 25 DJs o más. Si no llega, lo informo y pregunto
  antes de aflojar cualquier regla.

### Cómo se calcula

- Código: `src/v4/evaluation/tablero_a1.py`, con los subcomandos `armar` y `medir`, y sus tests en
  `tests/v4/test_tablero_a1.py`, sobre un dataset en miniatura.
- Corre en CPU.
- Informe: `docs/reports/a1_tablero_<fecha>.md`, con el veredicto primero y las etiquetas de `auditar`. Dice la versión
  del dataset, cuántos sets, DJs y temas midió, el commit de este protocolo y el md5 de las tablas de pares.
