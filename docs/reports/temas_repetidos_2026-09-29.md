# Temas repetidos en la biblioteca (2026-09-29)

Disparador: en la app, «Josh Wink – Higher State Of Consciousness» aparecía dos veces, en dos puntos
distintos del mapa. Protocolo `auditar`.

**Coordenadas de todos los datos:** organización `biblioteca` v2 (1786 temas, `clap_full` + BPM
peso 0.3, `orgs/biblioteca/`), catálogo `musica` (1823 filas), commit base `034a969`. Coseno =
producto de embeddings CLAP normalizados, alineados por `representations/clap_full/track_uids.json`.
Los números salen de un script exploratorio (fuera del repo) que corrió en esta sesión.

## Veredicto

Los dos Josh Wink son **versiones distintas**, y los dos archivos ya traen la mezcla en el tag
Remixer (ID3 TPE4). La app no la mostraba. Desde este cambio la muestra: «Josh Wink – Higher State Of
Consciousness (Adana Twins Remix Two)». Quedan en puntos distintos porque, para CLAP, suenan bastante
distinto. Aparte, el análisis encontró **duplicados**: el mismo tema en dos o más archivos. La regla
acústica (capa 2, más abajo) da 16 grupos, 34 temas y 18 copias de más. Por el nombre, los 16 grupos
son el mismo tema, pero solo «Hide U» está verificado byte a byte; el resto es INFERENCIA por nombre,
coseno y duración. Todas las copias caen en la misma playlist que su original, así que los exports
repiten el tema (N1 trae «Hide U» cuatro veces).

## Josh Wink

| | Tweekin Acid Funk Mix | Adana Twins Remix Two |
| :--- | :--- | :--- |
| Archivo | `#1 BIBO/PRO/Nuevitas/TECHNO/… [Tweekin Acid Funk Mix] 1998.mp3` | `2020 old/old 5/… (Adana Twins Remix Two).mp3` |
| Tag Remixer (TPE4) | Tweekin Acid Funk Mix | Adana Twins Remix Two |
| Duración / BPM del tag | 375 s / 126 | 416 s / 123 |
| Sello y fecha (tags) | Nervous Dog, 1996 | Strictly Rhythm, 2020-01-17 |
| ISRC del match en `features/beatport.csv` | USSR30764005 | QMRSZ1901716 |
| Playlist | J2 Techno / Tech House (122-129) | D1 Techno / Indie Dance (120-128) |

La tercera copia (`2020 new - copia/…Adana Twins Remix Two.mp3`) tiene el mismo audio byte a byte.
Phase 0 ya la descartó (`duplicates.csv`).

| Afirmación | Etiqueta | Evidencia |
| :--- | :--- | :--- |
| Son grabaciones distintas | EVIDENCIA OBSERVADA | TPE4, duración, ISRC y track_uid distintos (tabla de arriba) |
| Coseno CLAP entre ambos 0.825, por debajo del vecino más cercano del 95 % de la biblioteca | EVIDENCIA OBSERVADA | vecino más cercano de cada tema: p5 0.886, mediana 0.936 |
| Para el Tweekin, el Adana es el vecino más parecido (#1). Para el Adana, el Tweekin es el #6 | EVIDENCIA OBSERVADA | el #1 del Adana es RDNK - Guess the Number (0.858) |
| En el mapa están a 0.37. El vecino típico está a 0.06 y el mapa mide ≈ 7.5 de ancho | EVIDENCIA OBSERVADA | UMAP de `biblioteca` v2 |
| La separación viene del sonido, no del tempo | INFERENCIA | en el espacio del agrupamiento la distancia es 0.61. CLAP solo da √(2 − 2·0.825) = 0.59, así que el BPM aporta ≈ 0.16 |
| Que suenen distinto para CLAP no dice si en una fecha se usan igual | HIPÓTESIS | lo decide Gabriel escuchando las dos |

## Duplicados reales

**Causa 1, WAV/AIFF.** `tags.payload_range` quita los tags de MP3 y FLAC, pero en WAV/AIFF hashea el
archivo entero. Si dos copias del mismo audio tienen distintos chunks `LIST`, reciben track_uid
distintos. Hoy afecta a un solo tema: «Sandy Rivera, Rae - Hide U (Chicola Extended Remix).wav», con
cuatro copias con el mismo chunk `data` y cuatro uid (EVIDENCIA OBSERVADA: se hasheó el chunk `data`
de los 32 WAV/AIFF).

**Causa 2, el mismo tema en otro archivo** (otra descarga u otra codificación). El hash no lo
detecta, y a veces el nombre tampoco: «Trommelmaschine» / «Trommel Machine», «Flight Of Birds -
Bedouin» / «Bedouin - Flight of the birds». También «Floyd Lavine - Masala.mp3», que por dentro es el
Pablo Fierro Remix: su TPE4 lo dice, el coseno es 0.996 y la diferencia de duración, 0.3 s.

### Cómo detectarlos (tres capas)

1. **Hash de audio exacto** (ya existe). Falta que en WAV/AIFF hashee solo el chunk de audio
   (`data` / `SSND`). Cambiar el track_uid de esos 31 archivos invalida sus embeddings y sus lugares
   en la organización. La alternativa es una columna aparte, solo para duplicados, que deje el uid
   intacto.
2. **Acústica: coseno CLAP ≥ 0.98 y diferencia de duración ≤ 5 s.** Da 21 pares (16 grupos) y, por el
   nombre, los 21 son el mismo tema. El umbral se eligió mirando estos mismos datos, así que es
   exploratorio. Justo por debajo aparecen falsos positivos: Techyon / Egorythmia (0.972, tramos de
   un mismo set continuo) y Joe Red «Orange» / «Blue» (0.964). **La duración es imprescindible:**
   Circulation «Lemon» Mix 1 / Mix 2 tienen coseno 0.989 pero 65 s de diferencia, y son dos versiones.
3. **Nombre: mismo artista, título y mezcla, con diferencia de duración ≤ 2 s.** Atrapa lo que CLAP
   deja pasar: Dennis Cruz «Bad Behaviour» (0.966), Jaydee «Plastic Dreams» (0.954), Wally Lopez
   «American Icon» (0.928), nthng «Untitled» (0.941) y Jacob Mikesh «Philipp Dolphia» (0.957).
   HIPÓTESIS: mismo audio con otro master o bitrate. Lo decide escuchar, o una huella de audio
   (Chromaprint).

Las versiones distintas no son duplicados. Hay 35 pares con el mismo artista y título pero otra
mezcla: coseno mediano 0.884, diferencia de duración mediana 58 s, y solo el 31 % cae en la misma
playlist. Con la mezcla en la etiqueta, en la app quedan 4 grupos de nombres idénticos (10 temas):
Hide U ×4, Barbatuques «Baianá», Jaydee «Plastic Dreams» y Markus Homm «Dance With Me». Este último
tiene 85 s de diferencia de duración y está sin decidir.

### Cómo manejarlos en el mapa (propuesta, decide Gabriel)

- **Duplicado (mismo audio): un solo punto.** Se queda una copia, primero por calidad (WAV/FLAC, luego
  MP3 320, luego el resto) y, entre iguales, la que está fuera de las carpetas «copia» u «old», como
  ya hace Phase 0. Las otras copias aparecen en el tooltip como «copias», sin entrar en playlists ni
  exports. No se borra ningún archivo. Antes de aplicarlo, Gabriel confirma los grupos de la capa 3.
- **Versión (otra mezcla): un punto por versión**, con la mezcla en la etiqueta (ya está). Si hace
  falta, al pasar el mouse se traza una línea fina hacia las otras versiones. Cuando dos versiones
  caen en la misma playlist, se decide si está bien o si hay que separarlas.

## Cambios de esta sesión

- `tools/playlist_review/build_review_page.py`: campo `m` con la mezcla de cada tema. Sale del tag
  Remixer; si falta, del paréntesis del título o del nombre del archivo que nombra una versión. No se
  muestra «Original Mix». Hoy 406 de los 1786 temas muestran una mezcla. Leer los tags suma ≈ 1.4 s
  a la primera carga de la app (de 0.6 a 2.0 s). Después se usa la caché. Si el catálogo trae una
  columna `tag_remixer`, se usa esa y no se leen los archivos.
- `tools/playlist_review/template.html`: la mezcla en el tooltip, la tabla, el menú y la búsqueda.
  Zoom con + / −, con Ctrl + rueda o con las teclas + / − / 0, y arrastre para moverse. ⤢ vuelve a
  mostrar todo. «Ancho del mapa» (20 a 200 %) angosta o ensancha solo el eje horizontal y queda
  guardado en el navegador. Se probó en Chrome headless: zoom, arrastre sin reproducir, clic que
  reproduce, Ctrl + rueda (la rueda sola no hace zoom) y ancho, sin errores de página.

## Pendiente

- `rekordbox.xml` y `traktor.nml` no llevan el Remixer, aunque los archivos sí lo tienen. HIPÓTESIS:
  Rekordbox y Traktor lo leen del archivo. Se verifica al validar el import (STATUS, ítem 1). Si no
  aparece, se agrega `Remixer` al XML y `REMIXER` al NML.
- Decisión de Gabriel sobre la propuesta de duplicados. Con eso se implementa la detección de tres
  capas y el punto único.

## Candidatos para el veredicto de Gabriel (2026-09-29)

Capas 1 y 2 (grupos 1 a 16) y capa 3 (17 a 21): 21 grupos, 44 archivos, 23 copias de más, todas en
la misma playlist que su original. [REEMPLAZADO 2026-09-29 por la regla de Gabriel (DECISIONS
2026-09-29, punto 9): se queda la MP3 de 320; si no, la de mejor calidad; empate, se pregunta] ★ = la
copia que se quedaría con la regla propuesta (mejor calidad; si empatan, fuera de «copia»/«old»; si no,
la ruta más corta). Similitud = coseno CLAP; Δ = diferencia máxima de duración. [REEMPLAZADO 2026-09-29
por el veredicto de Gabriel, sección siguiente] Veredicto: pendiente.

| # | Capa | Similitud | Δ | Archivos (calidad) |
| ---: | :--- | ---: | ---: | :--- |
| 1 | sonido | 1.000 | 0.0 s | `Nuevitas 12 (cachengue)/Sandy Rivera, Rae - Hide U (Chicola Extended Remix).wav` (WAV 16bit/44k, 7:38)<br>`Parte 2 - Misterio Melódico/Sandy Rivera, Rae - Hide U (Chicola Extended Remix).wav` (WAV 16bit/44k, 7:38)<br>`2020 new - copia/Sandy Rivera, Rae - Hide U (Chicola Extended Remix).wav` (WAV 16bit/44k, 7:38)<br>★ `Vocal/Sandy Rivera, Rae - Hide U (Chicola Extended Remix).wav` (WAV 16bit/44k, 7:38) |
| 2 | sonido | 1.000 | 0.0 s | `Parte 4 - Techno Comercial/Der Dritte Raum - Trommelmaschine (Martin Landsky Remix).mp3` (MP3 320k, 6:22)<br>★ `Nuevitas 6/Der Dritte Raum - Trommel Machine (Martin Landsky remix).mp3` (MP3 320k, 6:22) |
| 3 | sonido | 0.999 | 0.0 s | `Parte 3 - Solomun Explota/Fiberroot - Roccodrillo (Shall Ocin Remix).mp3` (MP3 320k, 6:11)<br>★ `Nuevitas 5/Fiberroot - Roccodrillo (Shall Ocin Remix).wav` (WAV 16bit/44k, 6:11) |
| 4 | sonido | 0.997 | 1.4 s | `Milo 5/dubfire-oliver-huntemann-terra-joseph-capriati-remix-senso-sounds.mp3` (MP3 259k, 7:34)<br>★ `John Digweed - Live at Music is Revolution Space Ibiza 2016/03. (6A) Dubfire & Oliver Huntemann - Terra (Joseph Capriati Remix).mp3` (MP3 320k, 7:32) |
| 5 | sonido | 0.996 | 0.3 s | ★ `TROPICALES/Floyd Lavine - Masala.mp3` (MP3 320k, 6:58)<br>`Milo 8/Floyd Lavine - Masala (Pablo Fierro Remix).mp3` (MP3 320k, 6:59) |
| 6 | sonido | 0.996 | 0.1 s | `Milo 13/Forever (Original Mix).mp3` (MP3 192k, 6:44)<br>★ `Milo 13/forever-original-mix.mp3` (MP3 265k, 6:44) |
| 7 | sonido | 0.995 | 3.6 s | `Parte 1 - Tech House/Barbatuques - Baianá (Jack Back Club Remix).mp3` (MP3 320k, 5:37)<br>★ `Nuevitas 9/Barbatuques - Baianá (Jack Back Club Remix).mp3` (MP3 320k, 5:33) |
| 8 | sonido | 0.993 | 1.6 s | `Milo 6/Rodrigo Gallardo feat. Fernando Milagros — El Abuelo (Los Suruba  Marcelo Burlon Remix).mp3` (MP3 192k, 5:14)<br>★ `Nuevitas 6/Rodrigo Gallardo feat. Fernando Milagro - El Abuelo (Los Suruba & Marcelo Burlon Remix).mp3` (MP3 320k, 5:12) |
| 9 | sonido | 0.992 | 1.7 s | ★ `Milo 6/Syap - Moving on.mp3` (MP3 258k, 6:13)<br>`Milo 4/SYAP - Moving On.mp3` (MP3 192k, 6:14) |
| 10 | sonido | 0.992 | 1.8 s | `Milo 6/Benoit  Sergio - The Way You Get.mp3` (MP3 192k, 7:21)<br>★ `Parte 1 - House/Benoit & Sergio - The Way You Get.mp3` (MP3 320k, 7:20) |
| 11 | sonido | 0.991 | 1.4 s | `old 6/Joris Voorn - Goodbye Fly.mp3` (MP3 320k, 6:56)<br>★ `Milo 10/Joris Voorn - Goodbye Fly (Original Mix).mp3` (MP3 320k, 6:57) |
| 12 | sonido | 0.990 | 2.4 s | `Milo 3/Inaky Garcia - Afrovita (Original Mix).mp3` (MP3 192k, 6:30)<br>★ `Parte 1 - Afro House/Iñaky Garcia - Afrovita.mp3` (MP3 192k, 6:27) |
| 13 | sonido | 0.989 | 0.4 s | `Milo 6/Flashmob- The lone brazilian.mp3` (MP3 264k, 6:36)<br>★ `Parte 1 - Tech House/Flashmob - The Lone Brazilian.flac` (FLAC 16bit/44k, 6:36) |
| 14 | sonido | 0.988 | 0.5 s | `NEW/Victor Ruiz - Never Mind (Oliver Huntemann Remix).mp3` (MP3 192k, 7:11)<br>★ `Parte 4 - Techno Comercial/Victor Ruiz - Nevermind (Oliver Huntemann Remix).mp3` (MP3 320k, 7:10) |
| 15 | sonido | 0.987 | 4.1 s | ★ `EGIPTO/Flight Of Birds - Bedouin.mp3` (MP3 320k, 9:13)<br>`Milo 6/Bedouin - Flight of the birds.mp3` (MP3 128k, 9:09) |
| 16 | sonido | 0.986 | 2.1 s | ★ `Parte 3 - Solomun Explota/Oliver Koletzki - Iyéwaye.mp3` (MP3 320k, 7:40)<br>`Milo 3/Oliver Koletzki - Iyewaye (Original Mix).mp3` (MP3 192k, 7:42) |
| 17 | nombre | 0.966 | 0.1 s | `Milo 15/Dennis Cruz - Bad Behaviour (Original Mix).mp3` (MP3 128k, 6:46)<br>★ `Milo 11/Dennis Cruz - Bad Behaviour (Original Mix).mp3` (MP3 320k, 6:46) |
| 18 | nombre | 0.957 | 1.5 s | ★ `MELODICAS _ HOUSE/Jacob Mikesh Filburt - Philipp Dolphia.mp3` (MP3 320k, 7:14)<br>`Milo 3/Jacob Mikesh Filburt - Philipp Dolphia.mp3` (MP3 192k, 7:16) |
| 19 | nombre | 0.954 | 0.1 s | `old 2/Jaydee - Plastic Dreams (Nicole Moudaber Renaissance Remix).mp3` (MP3 320k, 9:07)<br>★ `Milo 11/Jaydee - Plastic Dreams (Nicole Moudaber Renaissance Remix).mp3` (MP3 320k, 9:07) |
| 20 | nombre | 0.941 | 1.3 s | ★ `Techno Trance/nthng - Untitled (Human Pt. II).mp3` (MP3 320k, 8:50)<br>`(1A) nthng - Untitled (Human Pt.II) [LT029.5].mp3` (MP3 192k, 8:52) |
| 21 | nombre | 0.928 | 0.1 s | `Nuevitas 6/Wally Lopez - American Icon (Original Mix).mp3` (MP3 320k, 9:25)<br>★ `Milo 7/Wally Lopez - American Icon.mp3` (MP3 320k, 9:25) |

Casos límite, fuera de los grupos: Benno Blome «Abotha» / «Abotha - Mihai Popoviciu Rmx» (0.979, 2 s:
probable duplicado con otro nombre); otro corte del mismo tema: Boris Brejcha «I am a Maschine» (6:13) /
«(Original Mix)» (7:45) y Markus Homm «Dance With Me» (7:35 / 6:10); versiones con nombre propio:
Circulation «Lemon» Mix 1/Mix 2, Kotelett Bonus Mix / Unconditional Love Version, Chus & Ceballos «The Sun» /
«(Algarve Mix)», Alex Dimou / Cevin Fisher Remix, Kevin De Vries «Sciamachy» / Konstantin Sibold Remix;
temas distintos: Joe Red «Orange» / «Blue» y dos pares de tramos de un set continuo de psytrance.

## Veredicto de Gabriel y regla (2026-09-29)

Gabriel escuchó los 24 grupos en `Música/_duplicados_para_escuchar.html` (página local con ▶ por copia
y veredicto por grupo). Decisiones y regla: `docs/DECISIONS.md`, 2026-09-29, puntos 8 a 12.

| Afirmación | Etiqueta | Evidencia |
| :--- | :--- | :--- |
| 21 grupos son el mismo tema (1-10, 12-21, A) | EVIDENCIA OBSERVADA | veredicto de Gabriel escuchando |
| La regla de calidad elige la misma copia que Gabriel en los 10 grupos donde la calidad difiere | EVIDENCIA OBSERVADA | `dedupe.py candidates` antes de importar: 10 «se resuelve sola», las 10 iguales a su elección |
| En los empates prefiere carpetas de año, después Nuevitas, después Milo | HIPÓTESIS | 7 de 7 empates (4 elegidos a mano, 3 aceptados); se confirma preguntándole |
| El código reproduce los 21 grupos revisados | EVIDENCIA OBSERVADA | `dedupe.py candidates`: grupos 1-21 con los mismos miembros |
| Abotha (0.979) queda en la franja «parecido» (0.97-0.98), que se pregunta | EVIDENCIA OBSERVADA | en esta biblioteca la franja trae 2 pares: Abotha (el mismo) y dos tramos de un set continuo (0.972, distintos por nombre) |
| `biblioteca` v10: 1786 → 1763 temas, 51 playlists, orden relativo intacto | EVIDENCIA OBSERVADA | comparación de v9 y v10 en esta sesión |

Código: `src/v4/common/duplicates.py` (regla, decisiones y detección por capas: sonido, nombre,
parecido, corte), `src/v4/pipeline/dedupe.py` (candidates / decide / import / apply / list),
`organize.remove_tracks` y `Library` sin las copias descartadas (un build desde cero no las vuelve a
meter). Decisiones en `artifacts/v4/datasets/musica/duplicate_decisions.json`. La app muestra en el
tooltip y en la tabla las copias de cada tema que quedaron fuera de las playlists.

