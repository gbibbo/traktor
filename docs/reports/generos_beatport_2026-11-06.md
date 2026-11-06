# Género de Beatport en la biblioteca y taxonomía (2026-11-06)

Auditoría con el protocolo `auditar`, a pedido de Gabriel, para el componente de género de la
recomendación en vivo (DECISIONS 2026-11-06, puntos 9 y 10). Datos: dataset `musica`, organización
`biblioteca` (1760 temas, `C:\TRAKTOR ML\_organizacion.csv`); `features/beatport.parquet`,
`features/genre_pred.parquet`, `representations/clap_full` y `maesthf_full_styles` (1786 track_uids),
colección de Traktor del escritorio.

## Veredicto

Las etiquetas están al día; la curación es automática y no es completa. Los 1429 temas con género de
Beatport usan las etiquetas que Beatport sigue usando hoy. 1322 de ellos (75 % de la biblioteca)
vienen de una coincidencia confirmada. 331 (19 %) no tienen género de Beatport y dependen de un
clasificador. Beatport no tiene subgénero cargado en nuestros datos: el cliente lo descartaba.

Beatport sí publica su taxonomía en la web, en tres niveles: categoría, género y subgénero. No
publica familias por encima del género ni similitudes entre géneros: eso hay que construirlo.

## Evidencia

| Afirmación | Etiqueta | Evidencia |
| --- | --- | --- |
| Niveles en la biblioteca: A (ISRC) 115; B confirmados 1207; B sin confirmar 107; C 99; D 232 | EVIDENCIA OBSERVADA | `beatport.parquet` filtrado por los track_uid de `_organizacion.csv` |
| Los 1429 `bp_genre` de la biblioteca están entre las 47 etiquetas que devolvió Beatport en 2305 búsquedas del 2026-10-30 al 11-02 | EVIDENCIA OBSERVADA | `artifacts/v4/beatport_cache/*.json` contra `beatport.parquet` |
| 331 sin género de Beatport: clasificador 320 (confianza mediana 0.63, mínima 0.23) y gate de Discogs 11 (4 «Non-Music», 1 «Jazz») | EVIDENCIA OBSERVADA | `genre_pred.parquet`, `final_source` y `model_conf` |
| 267 temas tienen versiones a las que Beatport les da géneros distintos | EVIDENCIA OBSERVADA | `beatport.parquet`, `genre_conflict` |
| En Traktor (tag de los 1717 archivos del paquete): 90 sin género, 40 con etiquetas viejas o de otra fuente (14 «Techno»), 71 distintos del género que usa el recomendador | EVIDENCIA OBSERVADA | `INFO GENRE` de `collection.nml` |
| La búsqueda de Beatport trae `sub_genre` en algunos temas; `summarize()` lo descartaba | EVIDENCIA OBSERVADA | página de búsqueda del 2026-11-06; `src/v4/common/beatport.py` antes de este cambio |
| La página de cada tema trae `genre` con `id`, `name`, `sub_genre` y `category` («Electronic») | EVIDENCIA OBSERVADA | `beatport.com/track/take-control/29631862` |
| La página de cada género trae sus subgéneros; se leyeron 39 géneros y 97 subgéneros | EVIDENCIA OBSERVADA | `artifacts/v4/beatport_taxonomia.json` (`beatport.com/genre/<slug>/<id>`) |
| El registro de un género no tiene padre, familia ni similitud | EVIDENCIA OBSERVADA | claves de `genre-96`: `enabled`, `id`, `is_included_in_hype`, `name`, `slug`, `sub_genres` |
| Beatport no publica una similitud entre géneros en ningún lado | INFERENCIA | no aparece en las páginas leídas ni en las fuentes de abajo; no se revisó la API de socios |
| La API oficial de Beatport es solo para socios | NO CONCLUYENTE | fuentes de terceros; la documentación se arma con JavaScript y no se pudo leer |
| El subgénero es opcional y poco frecuente | HIPÓTESIS | 3 de 20 temas del top de la portada; lo decide `beatport_subgenres.py` sobre la biblioteca |
| Los estilos de Discogs que predice MAEST encajan con los géneros de Beatport (Psy-Trance → Psy-Trance 0.91; Raw / Deep / Hypnotic → Techno 0.79) | EVIDENCIA OBSERVADA | `features/genre_evidence/generos.csv` |
| CLAP distingue el género: los vecinos de un tema son de su género entre 1.8 y 4.8 veces más que por azar (Psy-Trance 45); Mainstage (10 temas) no | EVIDENCIA OBSERVADA | `clap_vecinos_lift.csv`, k = 10, géneros con al menos 10 temas |

## Lo que se usa desde hoy en el recomendador

- Género de Beatport en 1429 temas (también los 107 sin confirmar: misma versión, mismo audio).
- Clasificador con confianza de 0.6 o más en 173 temas (el umbral de DECISIONS 2026-09-29, punto 5).
- Sin género en 158 temas: el componente de género no cuenta para ellos.
- El componente vale 1 si el género es el mismo y 0 si no, hasta que exista la similitud.

## Borrador de similitud (para revisar, no es la final)

`features/genre_evidence/borrador_similitud.csv`: promedio de los percentiles de tres señales para
los 16 géneros con al menos 10 temas: coseno entre centroides CLAP, lift de los vecinos CLAP y
géneros que Beatport mezcla entre versiones. Los más cercanos de cada uno:

| Género | Los tres más cercanos |
| --- | --- |
| Tech House | House, Indie Tech, Dance / Pop |
| Techno (Peak Time / Driving) | Techno (Raw / Deep / Hypnotic), Progressive House, Melodic House & Techno |
| Melodic House & Techno | Progressive House, Indie Dance, Deep House |
| Deep House | Organic House, Electronica, Melodic House & Techno |
| House | Tech House, Dance / Pop, Afro House |
| Afro House | House, Dance / Pop, Deep House |
| Minimal / Deep Tech | Tech House, Deep House, Techno (Peak Time / Driving) |
| Psy-Trance | Techno (Peak Time / Driving), Indie Tech, Dance / Pop (todos lejos) |

## Propuesta: la taxonomía con una red de agentes

- **Entradas:** la taxonomía de Beatport, que se baja de sus páginas; taxonomías publicadas (la que
  cita la tesis de Gabriel, MusicBrainz con «subgénero de», «fusión de» e «influido por», Wikidata,
  los estilos de Discogs); la evidencia de `features/genre_evidence/`.
- **Rondas:** un agente arma la primera versión: familias, géneros y subgéneros de Beatport, con una
  regla de similitud. En cada ronda, un agente propone cambios y dos los evalúan. Uno mira el texto de
  las etiquetas y su coherencia con las taxonomías publicadas. El otro mira la coherencia con la
  evidencia de audio. Un juez acepta o rechaza cada cambio. Se termina cuando los dos evaluadores la
  aprueban sin objeciones mayores, o a las tres rondas.
- **Salida:** la taxonomía con la similitud entre todos los géneros de Beatport, el mapeo de los
  estilos de Discogs a géneros de Beatport (para los temas sin Beatport) y la justificación de cada
  relación.
- **Control final:** las tripletas de Gabriel (como mucho 20), no un puntaje.

## Fuentes

- [Letter from the CEO: Embracing New Genres (Beatportal, 2025-09-09)](https://www.beatportal.com/articles/1055726-letter-from-the-ceo-embracing-new-genres)
- [Beatport launches new sub-genre Psy-Techno (Beatportal)](https://www.beatportal.com/articles/1257068-beatport-launches-new-sub-genre-psy-techno)
- [Acoustic Overspecification in Electronic Dance Music Taxonomy (arXiv 2509.11474)](https://arxiv.org/abs/2509.11474)
- [MusicBrainz: relaciones entre géneros](https://musicbrainz.org/relationships/genre-genre) y [Genre](https://musicbrainz.org/doc/Genre)
- [API Evangelist: Beatport](https://providers.apievangelist.com/providers/beatport/)
