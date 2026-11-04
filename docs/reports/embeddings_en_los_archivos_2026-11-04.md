# ¿Embeddings dentro de los archivos de música? (2026-11-04)

Pregunta de Gabriel: guardar los embeddings dentro de la metadata de cada archivo, para que la
música y sus datos viajen siempre juntos. Pregunta comprobable: ¿se puede escribir el embedding de
cada tema en sus tags sin cambiar su `track_uid`, sin dañar el audio y sin que lo borren los programas
de DJ? ¿Alcanza con eso para que la app funcione solo con los archivos?

## Veredicto

Se puede para MP3 y FLAC, que son 1752 de los 1760 temas del paquete. En copias (un archivo por
formato), el embedding se escribe y se lee idéntico, el `track_uid` no cambia, el audio decodifica y
el archivo no creció.
Traktor, Mixed In Key y Serato ya guardan sus análisis de la misma forma en estos archivos.

Pero no alcanza para que la música sea autosuficiente:

- La organización no es solo un embedding por tema. Las versiones, el mapa, los modelos PCA y UMAP,
  las fusiones y los nombres de las playlists son datos de toda la colección, y seguirían en una
  carpeta aparte.
- Leer los embeddings de 1760 archivos desde el pendrive es lento la primera vez (unos 55 s), así
  que la app igual guardaría una copia junta.
- En WAV y AIFF (8 temas), escribir cualquier etiqueta cambia el `track_uid`.

Para el problema real («la música en un lugar y los datos en otro»), lo más simple es guardar la
carpeta de datos junto a la música, en el mismo paquete. La app ya encuentra cada tema por su
`track_uid`, esté donde esté. [REEMPLAZADO 2026-11-04 por las pruebas con Traktor 4.4.1 y
Rekordbox 7.2.19: los dos los conservan; escribirlos en los 1752 MP3 y FLAC lo decide Gabriel.]
Los embeddings en los archivos pueden sumarse después como respaldo, cuando se compruebe que
Traktor y Rekordbox los conservan.

## Evidencia

| Afirmación | Etiqueta | Evidencia |
|---|---|---|
| El embedding usado (`clap_full`) son 512 números float32: 2 KB por tema; 1786 temas | EVIDENCIA OBSERVADA | `representations/clap_full/embeddings.npy` (1786, 512) float32 |
| La organización además usa datos de toda la colección: 11 versiones de `assignments.parquet` (l1, l2, posición, x, y), `pca.joblib` 103 KB, `umap.joblib` 3,9 MB, `org.json` (historia, parámetros), fusiones y nombres | EVIDENCIA OBSERVADA | inventario de `orgs/biblioteca/` |
| El paquete tiene 1746 MP3, 6 FLAC, 7 WAV y 1 AIFF | EVIDENCIA OBSERVADA | `_organizacion.csv` |
| MP3 (marco ID3 GEOB) y FLAC (comentario Vorbis en base64): el embedding vuelve idéntico, el `track_uid` no cambia, ffmpeg decodifica sin errores y el archivo crece 0 bytes (entra en el relleno del bloque de tags) | EVIDENCIA OBSERVADA | `embed_in_tags.py` sobre copias, 1 archivo por formato (n = 1) |
| WAV y AIFF: el `track_uid` cambia al escribir, porque su hash cubre el archivo entero | EVIDENCIA OBSERVADA | misma prueba; `payload_range`, `src/v4/common/tags.py:242` |
| En el paquete, el `track_uid` de los WAV/AIFF ya no coincide con el de sus archivos | EVIDENCIA OBSERVADA | misma prueba («uid del paquete» falso antes de escribir); la app no se afecta porque usa el `track_uid` del CSV |
| Los programas de DJ ya guardan datos binarios en estos MP3: PRIV `TRAKTOR4` en 172 de 172, GEOB `Key`/`Energy` (Mixed In Key) en 144, GEOB `Serato Markers` en 83 | EVIDENCIA OBSERVADA | muestra de 1 de cada 10 MP3 del paquete |
| Traktor conserva marcos GEOB ajenos cuando escribe el suyo | INFERENCIA | el PRIV de Traktor convive con los GEOB de Mixed In Key y Serato; no se sabe en qué orden se escribieron |
| [REEMPLAZADO 2026-11-04 por las pruebas con Traktor y Rekordbox: los dos lo conservan (filas de abajo)] Traktor o Rekordbox conservan un GEOB nuevo («TRAKTOR_ML») después de editar el tema | NO CONCLUYENTE | falta probarlo: en cada programa, importar una copia, cambiar un dato y releer el archivo |
| [REEMPLAZADO 2026-11-04 por la prueba con Traktor: Traktor sí; Rekordbox sigue sin probar] Traktor 4.4.1 conserva el embedding: Gabriel importó dos copias (MP3 con GEOB, FLAC con el comentario Vorbis), Traktor las analizó y él les cambió el comentario. Traktor escribió los dos archivos (el comentario pasó a «prueba» y renovó su PRIV `TRAKTOR4`), y el embedding quedó idéntico (mismo SHA256), con el mismo `track_uid` | EVIDENCIA OBSERVADA | `traktor_test.py` antes/después, en el escritorio |
| [REEMPLAZADO 2026-11-04 por la prueba con Rekordbox 7.2.19 en la laptop: sí lo conserva (fila siguiente)] Rekordbox conserva el embedding | NO CONCLUYENTE | Rekordbox no está en el escritorio: misma prueba donde esté instalado |
| Rekordbox 7.2.19 (7.2.19.0342, laptop) conserva el embedding: Gabriel importó dos copias de prueba (MP3 con GEOB, FLAC con el comentario Vorbis), Rekordbox las analizó y él les cambió el comentario. Rekordbox escribió los dos archivos (comentario «9A - Energy 4/7 - Vocal» → «prueba rekordbox»; mismo tamaño; ningún marco o campo nuevo ni quitado; en el MP3, ID3v2.4, el segundo comentario «ID3v1 Comment» quedó con el texto viejo). El embedding quedó idéntico (mismo SHA256: 512 float32 finitos, iguales a su fila de `clap_full/embeddings.npy`), con el mismo `track_uid` | EVIDENCIA OBSERVADA | `revisar.py revisar` contra `antes.json` (`D:\PRUEBA REKORDBOX\`, del escritorio), corrido en la laptop: «el embedding se conservó en los dos»; n = 1 por formato |
| Leer los embeddings de los archivos en frío desde el pendrive tardaría ~55 s; un `.npy` de 3,6 MB, menos de 1 s | INFERENCIA | la primera página leyendo los tags de 1760 archivos del pendrive tardó 55 s y la segunda 0,46 s; el pendrive lee unos 28 MB/s |
| Cambiar de modelo obliga a reescribir los tags de todos los archivos | INFERENCIA | el embedding depende del modelo; la etiqueta lleva la versión (`clap_full v1`) |

## ¿Recalcular la organización en vez de guardarla?

Pregunta de Gabriel: lo caro son los embeddings; el resto (carpetas, playlists, mapa, nombres) se
podría recalcular con un script. Se probó en una copia de los datos, en el escritorio.

| Afirmación | Etiqueta | Evidencia |
|---|---|---|
| Recalcular toda la organización desde los embeddings y el BPM tarda 20 s | EVIDENCIA OBSERVADA | `organize.build` con los parámetros de `biblioteca` v11 sobre sus 1760 temas |
| Es determinista: dos cálculos dan exactamente lo mismo | EVIDENCIA OBSERVADA | ARI 1,000 en carpetas y playlists, 54 de 54 playlists iguales |
| Con los mismos temas, reproduce la organización original | EVIDENCIA OBSERVADA | con los 1786 temas de v1: carpetas con ARI 1,000, playlists 0,978, 48 de 51 playlists juntas en un 90 % o más |
| Las 11 versiones cambiaron poco: v1 frente a v11 da ARI 0,998; hubo dos fusiones deshechas, el orden a mano de B3 y 26 repetidos quitados | EVIDENCIA OBSERVADA | `org.json` (historia) y la comparación v1/v11 |
| Pero desde cero es muy sensible a qué temas entran: sin los 26 repetidos (1,5 %), solo 7 de 51 playlists siguen juntas (en promedio, 62 % de sus temas); carpetas con ARI 0,317 | EVIDENCIA OBSERVADA | contraste de una sola variable: mismo código, misma máquina y mismos parámetros; solo cambia el conjunto de temas (1786 frente a 1760) |
| Por eso la app agrega música sin mover lo que ya está: recalcular desde cero cada vez que se agrega o se quita música desharía la organización que Gabriel eligió escuchando | INFERENCIA | de la fila anterior |
| Lo que hay que guardar no es caro: el lugar de cada tema (carpeta, playlist, posición y punto del mapa), 5 números por tema. También podría ir en los tags de cada archivo | INFERENCIA | `assignments.parquet` tiene esas columnas (l1, l2, position, x, y) por tema |
| Para ubicar música nueva sin mover el resto, hoy se usan los modelos PCA y UMAP guardados (4 MB). Sin ellos habría que resolver de otra forma dónde caen los temas nuevos en el mapa | INFERENCIA | `models_v1/` (pca.joblib, umap.joblib); no se leyó cómo los usa `organize.add` |

## Opciones

1. **Los datos dentro del paquete** (`D:\TRAKTOR ML\` + una subcarpeta de datos): la música, las playlists
   y la organización viajan como una sola carpeta. Antes, comprobar que `usb_package.py --refresh` y
   `--prune` no borren esa subcarpeta.
2. **Además, el embedding en cada MP3 y FLAC**, como respaldo: si se pierde la carpeta de datos, no hay
   que recalcular los embeddings. Va después de la prueba con Traktor y Rekordbox, y con un respaldo de
   los archivos antes de escribir en los 1752. [REEMPLAZADO 2026-11-04 por las pruebas: Traktor sí los
   conserva y el respaldo está hecho (`D:\TRAKTOR ML respaldo 2026-11-04`: los 1760 archivos y
   `_organizacion.csv`, mismos tamaños, 40 al azar con SHA256 igual). Falta Rekordbox.]
   [REEMPLAZADO 2026-11-04 por la prueba con Rekordbox 7.2.19: Rekordbox también los conserva.
   Escribir en los 1752 lo decide Gabriel.]
3. **Cada archivo con su embedding y su lugar** (carpeta, playlist, posición y punto del mapa): la app
   puede rearmar la organización leyendo los archivos, sin recalcularla, y fuera de los archivos solo
   quedan los modelos para ubicar música nueva y la historia de versiones. Mismas condiciones que la 2,
   más resolver los 8 WAV/AIFF.
