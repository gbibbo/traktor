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
`track_uid`, esté donde esté. Los embeddings en los archivos pueden sumarse después como respaldo,
cuando se compruebe que Traktor y Rekordbox los conservan.

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
| Traktor o Rekordbox conservan un GEOB nuevo («TRAKTOR_ML») después de editar el tema | NO CONCLUYENTE | falta probarlo: en cada programa, importar una copia, cambiar un dato y releer el archivo |
| Leer los embeddings de los archivos en frío desde el pendrive tardaría ~55 s; un `.npy` de 3,6 MB, menos de 1 s | INFERENCIA | la primera página leyendo los tags de 1760 archivos del pendrive tardó 55 s y la segunda 0,46 s; el pendrive lee unos 28 MB/s |
| Cambiar de modelo obliga a reescribir los tags de todos los archivos | INFERENCIA | el embedding depende del modelo; la etiqueta lleva la versión (`clap_full v1`) |

## Opciones

1. **Los datos dentro del paquete** (`D:\TRAKTOR ML\` + una subcarpeta de datos): la música, las playlists
   y la organización viajan como una sola carpeta. Antes, comprobar que `usb_package.py --refresh` y
   `--prune` no borren esa subcarpeta.
2. **Además, el embedding en cada MP3 y FLAC**, como respaldo: si se pierde la carpeta de datos, no hay
   que recalcular los embeddings. Va después de la prueba con Traktor y Rekordbox, y con un respaldo de
   los archivos antes de escribir en los 1752.
