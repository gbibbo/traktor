# Piloto de trayectorias de DJs: John Digweed, 2026-11-07

Plan: `docs/plans/recomendacion_playlists_plan.md`, etapa 2, punto 4 (fuentes D7 y D8, ideas T11 y
T12). Decisiones: DECISIONS 2026-11-07, puntos 12 y 13.

## Veredicto

- **CLAP es el mismo espacio aquí que en tu biblioteca.** Recalculé 6 temas de la biblioteca en esta PC
  y dan coseno 0,997 a 1,000 contra `clap_full`.
- **El audio de un set mezclado cae en ese espacio, con ruido.**
  - Bloque a bloque, el tema original sale primero entre los de su disco en el 57-62 % de los bloques,
    pero solo en el 26-34 % si compite también con los 1786 temas de la biblioteca.
  - Por tema, promediando sus bloques, el original sale primero en el 79-86 % de los temas.
- **Las compilaciones de Bedrock son la mejor fuente que tenemos.** Traen el set continuo, el orden y
  el límite exacto de cada tema (MusicBrainz) y cada original suelto. Dan trayectorias por tema, limpias
  y en el espacio de la biblioteca, sin tener que separar el set en temas.
- **Separar en temas un set crudo no funciona con el método simple.** Los picos de novedad aciertan lo
  mismo que cortar a intervalos fijos (F 0,38-0,43 frente a 0,35-0,56 a ±60 s). Para los sets crudos
  hace falta alinear contra los originales, como Kim et al. 2020, o quedarse en el nivel de bloque.
- **En CLAP, el orden dentro de la noche pesa poco frente a la elección del repertorio.** Los temas
  consecutivos están a una distancia mediana de 0,16-0,17. Dos temas al azar del mismo disco, a 0,17-0,19;
  dos temas al azar de la biblioteca, a 0,24.
- **La fase introspectiva: NO CONCLUYENTE.** Por tema, los pasos son más chicos en la primera hora y
  después de la cuarta, pero con 4 a 10 transiciones por hora. Por bloque, no hay tendencia.
- **Costo.** CLAP tarda 1,3 a 1,9 minutos de CPU por hora de audio. Cientos de sets entran en CPU, en
  prioridad baja; no hace falta GPU para esto.

## Qué se usó

| Fuente | Contenido | Duración | De MusicBrainz |
| :--- | :--- | --: | :--- |
| «Last Night at Output» (Bedrock, playlist oficial de YouTube) | 6 mezclas continuas y 65 originales | 6,3 h de set | CD `4cb11ccd`: 71 pistas en 6 discos |
| «Live in Tokyo» | 5 mezclas y 56 originales | 5,5 h | CD `1893e875`: 60 pistas |
| «Live in Brooklyn» | 5 mezclas y 58 originales | 5,9 h | CD `6810b907`: 67 pistas |
| Set crudo de Ámsterdam (29/11/2025, canal oficial) | 1 video | 4,1 h | — |
| Set crudo de Alta Gracia, Córdoba (20/01/2024, canal oficial) | 1 video | 3,1 h | — |

- **Duraciones.** La duración de cada mezcla continua coincide al décimo de minuto con la suma de las
  pistas de su CD.
- **Emparejamiento.** 179 de las 198 pistas de los CD tienen su original en las playlists, y todos los
  originales subidos tienen pareja. Las 19 pistas restantes no tienen original subido.
- **Audio.** 197 archivos, 2,6 GB, en `D:\TRAKTOR ML datos\dj_sets`, fuera de Git. Herramientas en
  `tools/dj_sets/`.
- **CLAP.** El mismo modelo y el mismo preprocesado que la biblioteca: tramos de 30 s en 3 ventanas de
  10 s y remuestreo con `resample_poly`. Para los originales, el embedding sale como en la biblioteca: 3
  tramos del 35 al 65 % del tema.

## Medidas

### 1. Identificación: ¿un bloque de la mezcla cae cerca de su tema original?

Bloques de 30 s a más de 30 s de un cambio de tema («núcleo») y en la zona de cambio («transición»).
Puesto del original correcto por coseno, contra los originales del disco (56 a 65) y contra esos más la
biblioteca (~1850).

| Disco | Bloques núcleo | 1.º entre los del disco | Top 5 | 1.º con biblioteca | Top 5 | Puesto mediano con biblioteca | 1.º en transición | Tema por mayoría de bloques |
| :--- | --: | --: | --: | --: | --: | --: | --: | --: |
| Last Night at Output | 585 | 0,568 | 0,785 | 0,328 | 0,463 | 8 | 0,269 | 0,862 |
| Live in Tokyo | 502 | 0,622 | 0,837 | 0,343 | 0,506 | 5 | 0,268 | 0,786 |
| Live in Brooklyn | 489 | 0,575 | 0,804 | 0,262 | 0,399 | 12 | 0,267 | 0,810 |

Usar la media de todos los bloques del original en lugar de sus 3 tramos centrales cambia poco (1.º
entre los del disco: 0,55-0,59). 6 de los 179 originales también están en la biblioteca. Esa copia
compite con el original y baja un poco la columna «con biblioteca».

### 2. Separación en temas (T12)

Picos de novedad: la distancia entre la media de los k bloques anteriores y la de los k siguientes, con
tantos picos como cambios reales. Se comparan con cortes a intervalos fijos, cada duración media de
tema. Los valores son F.

| Disco | Novedad k=3, ±30 s | k=3, ±60 s | Cortes fijos, ±30 s | Cortes fijos, ±60 s |
| :--- | --: | --: | --: | --: |
| Last Night at Output | 0,164 | 0,378 | 0,150 | 0,345 |
| Live in Tokyo | 0,199 | 0,433 | 0,166 | 0,435 |
| Live in Brooklyn | 0,260 | 0,422 | 0,275 | 0,563 |

Con k=2 da peor. En los sets crudos, el mismo método con un umbral estima 25 temas en 4,1 h y 18 en
3,1 h (mediana de 6,5 y 8,75 min por tema). Es poco creíble para Digweed: en las compilaciones, sus
temas duran de 5,2 a 5,5 min en promedio.

### 3. Trayectoria por tema (T11)

Distancia coseno entre los originales, como en la biblioteca, solo entre pistas vecinas del set (sin
saltear las que no tienen original).

| Disco | Pasos | Consecutivos (mediana) | Al azar del disco | Al azar de la biblioteca |
| :--- | --: | --: | --: | --: |
| Last Night at Output | 59 | 0,170 | 0,185 | 0,240 |
| Live in Tokyo | 51 | 0,162 | 0,173 | 0,237 |
| Live in Brooklyn | 50 | 0,162 | 0,168 | 0,239 |

Pasos por hora del set (mediana y número de pasos):

| Hora | Last Night at Output | Live in Tokyo | Live in Brooklyn |
| --: | :--- | :--- | :--- |
| 1.ª | 0,145 (10) | 0,113 (10) | 0,126 (11) |
| 2.ª | 0,212 (8) | 0,186 (8) | 0,169 (12) |
| 3.ª | 0,194 (11) | 0,195 (9) | 0,163 (13) |
| 4.ª | 0,172 (11) | 0,181 (9) | 0,208 (3) |
| 5.ª | 0,181 (9) | 0,146 (10) | 0,171 (7) |
| 6.ª | 0,128 (6) | 0,156 (5) | 0,125 (4) |
| 7.ª | 0,141 (4) | — | — |

La variación dentro de cada tema (bloques contra la media del tema) va de 0,028 a 0,060, sin una
tendencia que se repita en los tres discos.

### 4. Paso entre bloques de 30 s, por hora

| Hora | Last Night at Output | Live in Tokyo | Live in Brooklyn | Ámsterdam (crudo) | Alta Gracia (crudo) |
| --: | --: | --: | --: | --: | --: |
| 1.ª | 0,062 | 0,038 | 0,054 | 0,052 | 0,063 |
| 2.ª | 0,063 | 0,062 | 0,056 | 0,043 | 0,075 |
| 3.ª | 0,054 | 0,056 | 0,055 | 0,067 | 0,067 |
| 4.ª | 0,055 | 0,048 | 0,052 | 0,071 | — |
| 5.ª | 0,063 | 0,057 | 0,057 | — | — |
| 6.ª | 0,062 | 0,074 | 0,053 | — | — |
| 7.ª | 0,049 | — | — | — | — |

Las horas con pocos bloques al final (menos de 15) no se muestran.

## Qué cambia en el plan

- **D7 pasa a ser la fuente principal de trayectorias por tema.** Las compilaciones con original suelto
  dan una secuencia de embeddings limpios en el espacio de la biblioteca. Con eso, E1 puede medir los
  componentes de audio sobre sets de otros DJs desde la etapa 2. Hay 345 compilaciones `DJ-mix` de 37
  de los artistas de Gabriel (Sasha 69, Digweed 49, Carl Cox 43, Nick Warren 20, Hernán Cattáneo 13).
  Falta ver cuántas tienen en YouTube la mezcla continua y los originales.
- **D8, los sets crudos, queda en el nivel de bloque** hasta tener un separador que funcione. Pistas:
  la alineación contra los originales (Kim et al. 2020), o un modelo entrenado con los límites reales
  de D7, que da miles de cambios de tema etiquetados.
- **Para las playlists de dirección, CLAP solo no alcanza para ordenar.** Separa bien el repertorio de
  una noche del resto de la biblioteca (0,17 frente a 0,24), pero dentro de la noche el orden casi no
  se ve (0,16 frente a 0,18). La dirección tiene que venir de otras coordenadas (T3: energía, texto de
  CLAP, estilos) o de un modelo de secuencia entrenado con muchos sets (T13), no de la cercanía en CLAP.
- **Escala.** Una compilación con sus originales sueltos es unas 2,3 veces la duración del set: en el
  piloto, 41 h de audio para 17,7 h de sets.
  - Si las 345 compilaciones duraran unas 2,5 h y todas trajeran originales, serían unas 2000 h de
    audio.
  - Eso da unas 55 h de CPU a 1,6 min por hora. En prioridad baja son días; con GPU (`gpuq`), mucho
    menos.
  - En disco, unos 120 GB a 60 MB por hora: no entra en D: (62 GB libres). A esa escala, el audio va a
    otro disco (el Drive que ofreció Gabriel) o se borra después de sacar los embeddings.

## Afirmaciones

| Afirmación | Etiqueta | Evidencia |
| :--- | :--- | :--- |
| CLAP de esta PC da lo mismo que `clap_full` | EVIDENCIA OBSERVADA | `clap_blocks.py --verificar 6`: coseno 1,0000 (5) y 0,9973 (1) |
| La mezcla continua dura lo mismo que la suma de las pistas de su CD | EVIDENCIA OBSERVADA | `resultados.json`, `discos`: los 16 discos coinciden al décimo de minuto |
| Por tema, el original se identifica en el 79-86 % | EVIDENCIA OBSERVADA | `tema_por_mayoria_de_bloques` |
| Bloque a bloque y contra toda la biblioteca, solo en el 26-34 % | EVIDENCIA OBSERVADA | `rango_con_biblioteca` |
| La novedad simple no supera a los cortes fijos | EVIDENCIA OBSERVADA | Tabla de la sección 2, 3 discos y 16 mezclas |
| En CLAP, el orden pesa poco frente a la elección del repertorio | INFERENCIA | Consecutivos 0,16-0,17 frente a 0,17-0,19 (al azar del disco) y 0,24 (al azar de la biblioteca). Vale solo para CLAP: otras coordenadas pueden ordenar |
| Después de la segunda hora los pasos son más chicos | NO CONCLUYENTE | 4 a 10 pasos por hora y por disco; por bloque no aparece. Hacen falta muchos más sets largos |
| Todo esto vale para Digweed | HIPÓTESIS para otros DJs | Un solo DJ, progressive y techno melódico; se mide con los otros artistas |

## Cómo repetirlo

```
R="D:/TRAKTOR ML datos/dj_sets"
python tools/dj_sets/musicbrainz_mixes.py bajar --root "$R" --release <mbid del CD>
python tools/dj_sets/youtube_audio.py --playlist <id de la playlist oficial> --root "$R"
python tools/dj_sets/clap_blocks.py --root "$R" --hf-cache "D:/TRAKTOR ML datos/hf_cache" --carpeta <id> --low-priority
python tools/dj_sets/piloto.py --root "$R" --album <carpeta>:<mbid del CD> --set <carpeta de un set crudo>
```
