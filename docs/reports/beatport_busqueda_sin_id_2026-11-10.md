# Previews de Beatport: por qué la búsqueda deja temas sin id (2026-11-10)

Auditoría del agente, sin red y sin cambiar código, con el protocolo de `.claude/skills/auditar/`.

**Veredicto.** En los 899 temas sin id (de 4604) de `lote_01` a `lote_03`, la búsqueda siempre trajo
candidatos. En una muestra al azar de 30, el tema correcto estaba entre los resultados en 10 casos, y
`match()` lo descartó por la forma de escribir la versión o el artista. En los otros 20 la versión no
aparece en los resultados. En toda la población, una comparación de la mezcla que ignore la «s» del
posesivo y la palabra «remix» rescataría 66 de los 338 de nivel C. El resto de nivel C y casi todo el
nivel D necesitan otra vía: el enlace de 1001Tracklists, que solo tienen 211 de los 899, u otra
búsqueda.

## Datos

- Qué lo produjo: `tools/dj_sets/beatport_previews.py`, función `buscar_en_beatport`, sobre
  `src/v4/common/beatport.py`, funciones `TrackQuery.search_queries` y `match`. Es el código de
  `5e87fdd`, igual en `d95f06e` y en `e6c566a`, con los que corrieron los lotes.
- Sobre qué datos: manifiestos `tanda_*.json` y `temas.csv` de `lote_01` a `lote_03` (11 tandas completas
  hasta las 03:30; 4604 temas: 3703 ok, 899 sin id, 2 sin preview), y la caché de búsquedas
  `beatport_previews/cache/` (D:).
- Reproducción: sin red, con la caché, salen los mismos niveles que esa noche en los 899 (338 C y 561 D).
  Las 1326 consultas estaban en la caché.
- Scripts de la auditoría (fuera del repo): `auditar_sin_id.py`, que toma la muestra con semilla
  20261110, y `auditar_sin_id_poblacion.py`.

## Muestra de 30, por causa

| Causa | Casos | Ejemplo (1001Tracklists → Beatport) |
| :--- | ---: | :--- |
| Posesivo: la mezcla de Beatport deja una «s» suelta | 3 | «Sleepwalker (Iorie Lost Dream Remix)» → «(Iorie's Lost Dream Remix)» |
| «Remix» en un lado, «Mix» en el otro | 2 | «Drunken Galleon (John Tejada Remix)» → «(John Tejada Mix)» |
| La versión no se separa del título | 2 | «The Space Dance (Main Room Dubstrumental)»; «So Hot (S.A.M. Reshape)» |
| Alias de artista | 1 | «Ignacio aka Steve Rachmad - Organa» → «Ignacio - Organa» |
| Erratas o nombres raros | 2 | «On The T (Orginal)»; Edd «Any Kind Of Music (I Love)» con mezcla «I Love» |
| La versión no está entre los resultados | 20 | Remixes sin publicar en Beatport, bootlegs, temas de sellos que no están |

## Afirmaciones

| Afirmación | Etiqueta | Evidencia |
| :--- | :--- | :--- |
| Ninguno de los 899 se quedó sin candidatos; la reproducción sin red da los mismos niveles | EVIDENCIA OBSERVADA | `auditar_sin_id.py`: `{('C','C'): 338, ('D','D'): 561}`, 0 sin candidatos |
| En 10 de 30 el tema correcto estaba entre los resultados | EVIDENCIA OBSERVADA (el juicio de que es el mismo tema es del agente, leyendo artista, título y mezcla) | salida de `auditar_sin_id.py`, muestra de 30 |
| Con la mezcla sin la «s» suelta y sin «remix», y con el resguardo de que a los dos les quede un nombre, salen 66 de 338 de nivel C | EVIDENCIA OBSERVADA (el conteo) | `auditar_sin_id_poblacion.py`: 31 por el posesivo y 35 por «remix»/«mix» |
| Esos 66 son la misma versión | INFERENCIA: se leyeron 12 de 66 y los 12 lo son; el resguardo existe porque sin él «Just Close Your Eyes» coincidía con un «Remix» sin nombre | lista de ejemplos de la misma salida |
| Los 20 que no aparecen no están en Beatport | HIPÓTESIS: solo se vieron los resultados de hasta 3 consultas; se decide con el enlace de 1001Tracklists de los que tienen botón de compra, o con otra consulta | — |
| Ablandar `match()` no cambia los 3703 que ya salieron | HIPÓTESIS: no se midió; se garantiza si la comparación floja solo se usa cuando la estricta no encuentra nada | — |

## Qué haría falta, sin hacerlo esta noche

`match()` también lo usan el etiquetado de la biblioteca (`src/v4/pipeline/beatport_lookup.py`) y
«Agregar música nueva». Por eso el cambio va con Gabriel presente y con sus tests:

1. Comparación estricta primero, igual que hoy. Si no da A ni B, una comparación floja de la mezcla: sin la
   «s» suelta y sin «remix», solo cuando a los dos les queda un nombre. Se registra como un nivel aparte o
   con una marca, para poder contarlo.
2. Separar del título las versiones que `_VERSION_HINT` no conoce («Dubstrumental», «Reshape»), y partir
   el artista en «aka».
3. Volver a pasar los sin id de los lotes ya hechos con la caché: no cuesta pedidos de búsqueda, solo
   la página y el MP3 de los rescatados.
