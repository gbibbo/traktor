# DJSets v1: sets de DJs con el audio de sus temas

Dataset privado de TRAKTOR ML. Junta 1549 sets de 1001Tracklists con el audio de los temas que suenan en ellos. Sirve
para entrenar y evaluar modelos que aprendan qué temas van juntos y cómo cambia la música a lo largo de una noche.

Este README es la ficha del dataset (*datasheet*) y la guía del código. El armador lo copia a la raíz del dataset, así
que cada copia del dataset lleva su propia explicación y su propio lector (`djsets.py`).

| | v1 (2026-11-13) |
| :--- | ---: |
| Sets | 1549 (573 DJs; fechas 1987-10-24 a 2026-11-07; mediana 120 min) |
| Filas de tracklist | 53 650 (48 793 identificadas) |
| Temas distintos | 36 611 |
| Temas con audio | **33 321 (91 %)**: 30 255 previews de Beatport, 3066 audios completos de YouTube |
| Audios distintos | 33 226 (algunos temas de 1001Tracklists son el mismo audio) |
| Temas con metadatos de Beatport (BPM, tonalidad, género, sello, ISRC) | 30 655 |
| Pares de temas consecutivos | 42 493 (36 327 con audio en los dos) |
| Tamaño | 62,1 GB de audio en 132 shards `.tar`, más 22 MB de tablas |

## Dónde está

- **Copia principal:** `erclm:traktor_ml/djsets_v1/`, en el Drive de trabajo de Gabriel (rclone).
- **Otras copias:** el NAS y el pendrive de Gabriel, copiadas desde esa carpeta.

Todo lo que hace falta para usar el dataset está dentro de esa carpeta. Las carpetas originales de la descarga
(`erclm:traktor_ml/beatport_previews/`, `youtube_temas/`, `dj_sets/`) quedan como estaban. El dataset no depende de
ellas.

## Estructura

```
djsets_v1/
  README.md              esta ficha
  djsets.py              lector (Python 3.8+, solo biblioteca estándar)
  dataset_info.json      versión, fecha, commit del armador, conteos
  checksums.md5          md5 de cada tabla y de cada shard (formato md5sum)
  metadata/
    sets.csv             un set por fila
    set_tracks.csv       una fila de tracklist por fila, en orden
    tracks.csv           un tema por fila, con su audio y sus metadatos
    shards.csv           qué hay en cada .tar y de qué tanda de la descarga salió
    split_by_dj.csv      partición train/validation/test propuesta, por DJ
  audio/
    beatport/beatport_000.tar ... beatport_097.tar    previews MP3
    youtube/youtube_000.tar  ... youtube_033.tar      audio completo M4A
  raw/                   lo original, para auditar o rearmar
    1001tracklists/      sets/<id>.json y .html.gz, búsquedas, candidatos.csv, descartados.csv
    beatport_previews/<corrida>/   temas.csv y los manifiestos de cada tanda
    youtube_temas/<corrida>/       ídem
  audio_files/           solo si se desempaqueta: audio/<fuente>/<NN>/<archivo>
```

Cada tabla une con las otras por `set_id` y `track_id`. Las tablas son CSV en UTF-8. El vacío significa «no hay
dato». Los valores lógicos son `1` y `0`.

### Por qué el audio va en shards `.tar`

Es como se distribuyen MTG-Jamendo y los datasets en formato WebDataset. Tiene tres ventajas:
- Son 132 archivos en lugar de 33 000, así que copiarlos al Drive, al NAS o a un pendrive es rápido y no choca con el
  tope de archivos por carpeta de FAT32.
- Ningún shard pasa de 4 GB.
- Para entrenar se leen de corrido.

El lector saca cualquier audio directo del `.tar`, sin desempaquetar. Para tener archivos sueltos, ver
«Desempaquetar».

## Uso

```python
import sys; sys.path.insert(0, "ruta/a/djsets_v1")    # o copiar djsets.py
from djsets import DJSets

ds = DJSets("ruta/a/djsets_v1")
ds.sets()                         # lista de dicts (filas de sets.csv)
ds.set_tracks("14y24gtk")         # el tracklist de un set, en orden
ds.tracks(with_audio=True)        # temas con audio
t = ds.track("25p3fc35")          # un tema por track_id
mp3 = ds.audio_bytes("25p3fc35")  # bytes del MP3/M4A (del .tar, o de audio_files/ si está desempaquetado)
ds.pairs(with_audio=True)         # (set_id, tema, siguiente) consecutivos
for fila, datos in ds.iter_audio("beatport"):   # recorre todo el audio leyendo cada shard de corrido
    ...
```

Con pandas: `pd.read_csv("djsets_v1/metadata/tracks.csv", dtype=str, keep_default_na=False)`.

Para decodificar a forma de onda, se pasan los bytes a cualquier decodificador, por ejemplo
`librosa.load(io.BytesIO(mp3), sr=None)`. Con M4A puede hacer falta ffmpeg. Otra opción es usar el archivo de
`audio_files/` después de desempaquetar.

### Línea de comandos

```
python djsets.py info     <raíz>
python djsets.py validate <raíz>            # md5 de tablas y shards (unos minutos)
python djsets.py validate <raíz> --full     # además el md5 de cada audio (lee los 62 GB)
python djsets.py unpack   <raíz> [--source beatport|youtube] [--skip-missing]
```

### Desempaquetar

`unpack` escribe cada audio en la ruta que dice su columna `audio_file`:
- `audio_files/beatport/NN/bp_<beatport_id>.mp3`
- `audio_files/youtube/NN/yt_<video_id>.m4a`

`NN` va de `00` a `99` y sale del md5 del nombre, para que ninguna carpeta pase de unos 400 archivos.

Si se corta, se vuelve a correr: salta lo que ya está con el tamaño correcto. Ocupa otros 62 GB. Medido en el pendrive
FAT32: 1,4 GB en 2 min 40 s. Los nombres no chocan aunque el disco no distinga mayúsculas de minúsculas (FAT32, NTFS,
SMB); el armador lo verifica.

## Tablas

### `sets.csv`

| Columna | Qué es |
| :--- | :--- |
| `set_id` | id de 1001Tracklists (la URL es `url`) |
| `title`, `dj`, `date`, `minutes` | lo que dice la página y el listado |
| `use` | `trayectoria` (3 h o más, evento real), `pares` (1 h o más), o los dos; reglas en el plan del repo, § 6 |
| `kind`, `b2b`, `night_group` | tipo (evento, radio...), si es back to back, noche publicada por partes |
| `n_rows`, `n_identified` | filas del tracklist y cuántas tienen tema identificado |
| `n_tracks`, `n_tracks_with_audio` | temas distintos del set que están en `tracks.csv`, y cuántos con audio |
| `share_with_cue` | fracción de filas con hora de entrada (788 sets con 0,8 o más) |
| `recordings` | grabaciones del set enlazadas en la página (SoundCloud, YouTube, Mixcloud); el audio del set no está |
| `listing_genres`, `listing_pct_identified`, `seed_queries` | del listado de búsqueda: géneros, % identificado, qué búsqueda lo trajo |

### `set_tracks.csv`

Una fila por fila del tracklist, en el orden del set (`position` desde 1).
- `cue_s`: segundos desde el comienzo; vacío si la página no da la hora.
- `with_previous=1`: la fila se tocó encima de la anterior («w/»).
- `identified=0`: la fila es «ID - ID».
- `track_id`: el tema en `tracks.csv`; vacío si la fila no está identificada o si el tema no entró (ver abajo).
- Lo demás es lo que dice 1001Tracklists: `id_1001`, `name`, `artist`, `label`, `genre_1001`, `duration_1001_s`,
  `edition` y `buy_button` (si la fila tiene botón de compra).

`ds.pairs()` arma los pares consecutivos. Salta las filas «w/» y corta en una fila sin `track_id`, porque no se sabe
qué sonó entre los dos temas.

### `tracks.csv`

| Grupo | Columnas |
| :--- | :--- |
| Identidad | `track_id` (el id de tema de 1001Tracklists, o `nombre:<artista - título normalizado>` si no tiene), `id_1001`, `name`, `artist`, `title`, `label_1001`, `genre_1001`, `duration_1001_s` |
| Uso | `n_sets` (en cuántos sets aparece: 30 261 en uno solo, 6350 en dos o más, 544 en cinco o más), `n_rows`, `buy_button` |
| Audio | `has_audio`, `audio_status` (`ok` o el último motivo: `youtube:sin_version`, `youtube:sin_resultados`, `youtube:error_bajada`), `audio_attempts` (todos los intentos, en orden), `audio_source` (`beatport` o `youtube`), `recording_id` (`bp_<id>` o `yt_<id>`), `audio_shard`, `audio_member` (nombre dentro del .tar), `audio_file` (ruta desempaquetada), `audio_ext`, `audio_bytes`, `audio_md5`, `audio_duration_s`, `audio_is_full_track`, `preview_start_s` (dónde empieza el preview dentro del tema) |
| Beatport | `beatport_id`, `beatport_via` (`busqueda` o `1001`, el botón de compra; vacío en el piloto), `beatport_match_level` (A ISRC, B misma versión; C y D, la búsqueda no lo encontró y vino por el botón), `beatport_soft_mix` (la mezcla se aceptó escrita de otra forma), `bp_track_name`, `bp_mix_name`, `bp_artists`, `bp_bpm`, `bp_key`, `bp_genre`, `bp_label`, `bp_isrc`, `bp_release`, `bp_release_date`, `bp_length_s` |
| YouTube | `youtube_id`, `youtube_title`, `youtube_channel`, `youtube_topic` (canal oficial «- Topic»), `youtube_codec`, `youtube_abr_kbps`, `youtube_sample_rate` |

400 temas tienen metadatos de Beatport pero no el preview: su página existe, pero el preview no estaba o no se pudo
bajar. De ellos, 287 tienen audio de YouTube.

**Qué temas entran:** los identificados de los 1549 sets que tienen artista y título separables y no aparecen
solamente encima de otro tema («w/»). Es la misma población que se intentó bajar (`beatport_previews.elegir`), de modo
que no se perdió ningún tema intentado.

### `shards.csv` y `split_by_dj.csv`

`shards.csv` da, por cada shard:
- `n_audio`: los audios que tiene;
- `bytes` y `md5`;
- `created`: cuándo se armó;
- `origin`: la tanda original de la descarga.

Cada `.tar` contiene, además del audio, el `manifest.json` de su tanda.

`split_by_dj.csv` es una partición propuesta: 1083 sets en train, 234 en validation y 232 en test. Sale de un hash del
primer DJ del nombre, así que un mismo DJ queda en una sola parte (idea E7 del plan). No es una partición oficial
validada:
- Un tema puede aparecer en sets de dos partes.
- En un B2B, el segundo DJ puede caer en otra parte.

Quien entrene decide si la usa.

## Cómo se armó (procedencia)

1. **Sets (D5):** `tools/dj_sets/tracklists_1001.py`, del 2026-11-09 al 11-12.
   - Las páginas se leyeron con la sesión de Gabriel, a 8 s entre páginas.
   - Se eligieron 1549 sets por DJ, término y evento. Se descartaron los comerciales (DECISIONS 2026-11-09).
   - `candidatos.csv` tiene una columna `veto`: los sets vetados no entran al rearmar.
2. **Audio (D6):** `tools/dj_sets/beatport_previews.py`, `youtube_temas.py` y `descarga_continua.py`, del 11-09 al
   11-13.
   - **Beatport:** primero la búsqueda por nombre (`match()`: misma versión). Si no aparece, el botón de compra de
     1001Tracklists. Se baja el preview MP3 de unos 2 min: 28 765 de 120 s, 1397 de 90 s, 93 más cortos.
   - **YouTube:** lo que Beatport no dio. Se busca la misma versión y se prefiere el canal oficial «- Topic». Es el tema
     completo en M4A AAC a 44,1 kHz, sin recodificar, con una duración mediana de 6 min 17 s.
   - Cada tanda se subió al Drive como `.tar` con `manifest.json`, verificada por tamaño y MD5.
3. **Este armado:** `datasets/djsets/build_djsets.py` del repositorio.
   - `build` lee los sets y los manifiestos (los del Drive mandan; las copias locales tienen que decir lo mismo) y
     escribe las tablas.
   - `publish` copia cada tanda al dataset con su nombre nuevo, dentro del Drive, sin pasar por esta PC. Después copia
     `raw/`, sube las tablas y compara el md5 de cada archivo del destino.
   - Para un tema resuelto en más de una tanda, manda Beatport y después la primera tanda.
   - 90 audios repetidos en otra tanda quedan en su shard sin usar. Los 7 temas que también tienen audio de YouTube
     usan el preview de Beatport, aunque los dos audios estén en los shards.

Rearmar (por ejemplo, después de vetar sets):

```
python datasets/djsets/build_djsets.py build --root "D:\TRAKTOR ML datos\dj_sets" --out <carpeta local>
python datasets/djsets/build_djsets.py publish --out <carpeta local> --dest erclm:traktor_ml/djsets_v2
```

Una versión nueva va a una carpeta nueva (`djsets_v2`). Una versión publicada no se modifica.

## Limitaciones

- **El audio de Beatport es un preview de 1,5 a 2 min**, no el tema entero. `preview_start_s` dice dónde empieza. Los
  3066 de YouTube son temas enteros. No se compararon los dos tipos de audio para el mismo tema.
- **3290 temas (9 %) no tienen audio.** No están en Beatport ni en YouTube en la misma versión. Es probable, aunque no
  se midió, que sean sobre todo temas inéditos o ediciones propias. Si es así, el subconjunto con audio está sesgado
  hacia lo publicado.
- **La identificación es de la comunidad de 1001Tracklists** y puede tener errores.
  - Se verificaron 161 coincidencias de la búsqueda que aceptó la mezcla escrita de otra forma. Entre las 158 que
    tenían el enlace del botón de compra, no apareció ninguna versión equivocada; 2 eran cortes más cortos
    (`docs/reports/control_filtro_suave_2026-11-10.md`).
  - Los temas con nivel C o D vinieron por el botón de compra.
- **Los sets no son una muestra al azar:** salen de semillas (Spotify de Gabriel, artistas de su biblioteca, DJs y
  eventos que él nombró), con un tope de 15 sets por DJ.
- **El audio de los sets no está**, solo sus tracklists y los enlaces a grabaciones.

## Términos de uso

- Privado, solo para la investigación de Gabriel. No se redistribuye ni se publica, ni entero ni en partes.
- El audio pertenece a sus dueños: sellos y artistas.
- Beatport declara `Content-Signal: ai-train=no`. Es una señal de política y no una licencia; antes de entrenar con sus
  etiquetas, lo decide Gabriel (plan § 6, C1).
- 1001Tracklists no publica términos para este uso.
- El repositorio es público: acá va el código, nunca los datos.

## Prácticas que sigue

- **Tablas planas con ids estables y audio nombrado por id**, como FMA (`tracks.csv`) y MTG-Jamendo (TSV, carpetas
  `00`-`99`).
- **Shards `.tar` y un comando para desempaquetar**, como la descarga de MTG-Jamendo (`--unpack`) y WebDataset.
- **Índice con md5 de cada archivo y `validate()`**, como los índices de mirdata (Bittner et al., ISMIR 2019).
- **Partición definida** y explicada, como los `splits/` de MTG-Jamendo.
- **Ficha con procedencia, sesgos y términos**, como «Datasheets for Datasets» (Gebru et al.). `dataset_info.json`
  lleva versión, conteos y commit; es una versión mínima de lo que formaliza Croissant (MLCommons).
- **Versiones inmutables** (`djsets_v1`, `djsets_v2`...).
- **Lector sin dependencias**, que viaja con los datos.

## Versiones

- **v1 (2026-11-13):** primera versión: 1549 sets, 36 611 temas, 33 321 con audio.

## Código y tests

- `djsets.py`: el lector.
- `build_djsets.py`: el armador. Depende del repositorio TRAKTOR ML.
- `tests/test_djsets.py`: arma un D5/D6 en miniatura y prueba el lector, la validación y el desempaquetado.

```
python -m pytest datasets/djsets/tests -q
```
