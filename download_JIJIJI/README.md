# Descarga automática desde Spotify

## La app (sin línea de comandos)

Doble clic en **«Descargar playlists»** del Escritorio. El acceso directo se crea una vez por
ordenador:

```powershell
powershell -ExecutionPolicy Bypass -File download_JIJIJI\app\crear_acceso_directo.ps1
```

Abre una ventana propia (Edge en modo app) servida por `app/descargas_app.py` solo en
`127.0.0.1`, con tres pestañas arriba a la izquierda:

* **Descargas**: bajar playlists (todo lo que sigue en esta sección).
* **Colección**: qué música forma tu colección y cuál se elige para generar playlists. Lista la
  carpeta de destino de las descargas, la biblioteca organizada (si está en este ordenador:
  `Música/` del repo, según `config/v4.yaml`) y las carpetas que agregues con «Agregar carpeta»
  («Quitar de la lista» no borra nada). De cada subcarpeta (una por playlist bajada, por ejemplo)
  muestra temas, tamaño y último cambio, y una casilla para elegirla; lo elegido queda en
  `collection.json`, junto a los ajustes. «Organizar», en una carpeta elegida, abre la pestaña
  Organización con «Agregar música nueva» para esa carpeta. Los temas sueltos (fuera de
  subcarpetas) se cuentan pero no se eligen, y no cuentan `_copias/` (temas repetidos) ni las
  carpetas ocultas.
* **Organización**: la app de carpetas y playlists por estilo (`src/v4/ui/review_app.py`, la misma
  de `Abrir TRAKTOR ML.bat`). Se lanza al entrar en la pestaña, con el mismo Python, y se muestra
  dentro de ella; se cierra cuando se cierra la app, pero después de la tarea en curso (agregar
  música, fusiones, preparar para Rekordbox/Traktor). Necesita las dependencias del pipeline
  (`requirements_v4.txt`: usa el Python de la app o, si no las tiene, el `python` del PATH) y la
  organización `biblioteca` (`Música/` y `artifacts/v4/datasets/musica/`, que no están en Git:
  se copian desde el ordenador que las tiene); si falta algo, la pestaña lo dice. Sin `Música/`
  (como en el PC de escritorio), con solo `artifacts/v4/datasets/musica/` los temas salen del
  paquete del pendrive (`<unidad>:\TRAKTOR ML`, primero en la unidad de esos datos), buscados por track_uid
  en su `_organizacion.csv`, y la organización queda solo para ver y escuchar: los cambios se
  hacen donde está la biblioteca. En el PC de escritorio la biblioteca ya es el paquete
  (`package_library.py adopt`: `Música/` es un junction a `D:\TRAKTOR ML\Musica`), así que la
  organización se puede cambiar, y «Preparar para Rekordbox y Traktor» y «Agregar música nueva»
  rehacen también las playlists del pendrive sin mover la música.
  Con una descarga que puede estar grabando, avisa que no se reproduzca nada.

En la pestaña Descargas:

* pegar el enlace de una playlist y ver portada y temas (la primera vez, «Conectar» pide permiso
  a Spotify en el navegador; es una autorización propia de la app y no toca la del descargador).
  Toda autorización de Spotify abre el navegador predeterminado y además muestra el enlace con
  «Copiar enlace», para pegarlo en el navegador donde están las contraseñas (5 minutos);
* elegir la **salida de audio** de Windows (abajo a la derecha). Es la salida predeterminada:
  por ahí suena Spotify y eso es lo que graba el loopback. «Probar sonido» toca un ding-dong y lo
  graba para medirlo; «Medir lo que suena» muestra el nivel mientras suena Spotify. Con una
  descarga en curso no se puede cambiar ni probar, porque cortaría o ensuciaría la grabación;
* elegir la **carpeta de destino** (`--library-root`) y, en avanzado, la de trabajo (`--output-root`);
* **Descargar**, **Detener** (como un Ctrl+C: el run queda interrumpido y se cierran también
  `sockseek` y `ffmpeg`, nunca Spotify), **Continuar** o **Empezar de cero** (`--fresh`);
* **posponer la grabación** (`--defer-recording`): con «Grabar después» al empezar, o con
  «Posponer la grabación» en cualquier momento de la descarga (si está grabando, la tanda se corta
  en el acto, lo ya grabado se conserva y Spotify se pausa), se baja lo de Soulseek, va ya a la
  carpeta de destino y la playlist queda en «Grabación pendiente». Desactivarlo durante la descarga
  vuelve a grabar.
  **«Grabar ahora»** graba solo lo pendiente (sin volver a buscar ni bajar) y lo agrega a la
  misma carpeta. Desde la terminal: la misma orden sin `--defer-recording`;
* **revisar un tema ya bajado**: clic en la fila (o ▶ para escucharlo). Muestra la verificación
  contra Spotify (título, artistas e ISRC del archivo que llegó de Soulseek, duración y Spotify Track
  ID), de dónde salió (usuario, archivo original, por qué se aceptó y qué se descartó, o el tramo
  grabado), las propiedades del audio y todas sus etiquetas, y abre su ubicación o el reproductor.
  No se puede escuchar mientras una descarga pueda grabar (sonaría en la grabación), salvo con la
  grabación pospuesta. «‹ Anterior» / «Siguiente ›» recorren la lista y «Seguir escuchando los
  siguientes» pasa al próximo tema con archivo. Un tema sin archivo muestra qué devolvió Soulseek
  (y por qué se descartó cada archivo) y cada intento de descarga;
* **filtrar la lista**: los chips de estado («Por grabar», «No se pudieron»…) muestran solo esos
  temas, y el buscador filtra por título, artista o álbum;
* **«⋯ Más» o clic derecho en una playlist**: abrir su carpeta, abrirla en Spotify, copiar el
  enlace, ver el registro de la descarga (la salida completa o la búsqueda de sockseek), «Quitar
  de la lista» (no borra nada; vuelve si se baja de nuevo) y «Borrar archivos de trabajo» (las
  carpetas de `runs/`, avisando lo que no está guardado; nunca la carpeta de la música);
* **cola**: con una descarga en curso, otra playlist se pone en cola y empieza sola al terminar
  la anterior (si la detienes tú, la cola espera a «Empezar la cola ahora»);
* **avisos de Windows** cuando una descarga termina, queda incompleta, en pausa, con la
  grabación pendiente o cuando Spotify pide autorización, aunque la ventana esté cerrada;
* en la carpeta de cada playlist queda `<playlist>.m3u8` en el orden de Spotify (rutas
  absolutas, como el export del proyecto) para importarla en Rekordbox o Traktor;
* seguir cada tema: en espera, buscando, bajando, bajado, se grabará (también lo que no se pudo
  bajar, con el motivo), grabando, grabado o «no se pudo». También se siguen los runs lanzados desde la
  terminal. Un run «Incompleta» ya guardó lo que consiguió y dice qué falta;
* si un run terminó con error, **«Guardar los temas listos»** mueve a la carpeta de destino los
  MP3 que sí están, con la misma validación de la consolidación (audio, tamaño, antivirus,
  Spotify Track ID) y sin duplicar.

Cerrar la ventana no corta la descarga: la app sigue y el acceso directo la vuelve a abrir.
Si el código cambió desde que se abrió (un `git pull`, por ejemplo, también de la Organización),
el acceso directo la reinicia, salvo que haya una descarga o una tarea de la Organización en curso.
Sin ventana, descarga ni tarea de la Organización, se apaga sola. Ajustes, caché de portadas, token de la vista previa y
`app.log` van en `%APPDATA%\traktor_descargas`; con el Python de la Microsoft Store, esa carpeta
está virtualizada en
`%LOCALAPPDATA%\Packages\PythonSoftwareFoundation.Python.3.11_*\LocalCache\Roaming\`.
`app.log` anota la app (aperturas, cierres, errores con su motivo). La salida completa de cada
descarga (orquestador, Soulseek, grabación), con hora, queda en
`<carpeta de runs>\logs\<fecha-hora>_<playlist>.log`; la ventana muestra solo las últimas líneas.
El registro propio de sockseek de cada búsqueda se acumula en `sockseek_preflight.log`, dentro de
la carpeta de esa descarga.

## Línea de comandos

Después de configurar las credenciales y dependencias una sola vez, el flujo
completo se ejecuta pasando únicamente la URL de una playlist:

```powershell
.venv\Scripts\python.exe download_JIJIJI\spotify_soulseek_orchestrator.py "https://open.spotify.com/playlist/ID"
```

El orquestador va **tema a tema**, como cuando se busca a mano en Soulseek (2026-11-06):

1. el descargador busca cada tema en Soulseek por separado (un proceso de sockseek por
   búsqueda) y, si está, lo baja enseguida, antes de buscar el siguiente. El preflight en lote
   de antes gastaba las 34 búsquedas permitidas en menos de 2 minutos y quedaba otros 2 parado,
   una y otra vez: eso era lo que parecía «trancado»;
2. lo que no está en Soulseek, o está pero ningún archivo pasó la verificación, va a la **cola
   de grabación** (`recording_queue.jsonl`) apenas se sabe;
3. mientras la búsqueda sigue, el orquestador graba esa cola de Spotify en **tandas**: cada
   tanda toma lo que haya en la cola (hasta 10 temas), crea una playlist privada de Spotify solo
   con esos temas, abre el grabador WASAPI loopback y reproduce la playlist. La primera tanda
   empieza con el primer tema que haya que grabar, sin esperar al resto de la búsqueda;
4. si la cola queda vacía y la búsqueda sigue, espera el próximo tema (Spotify se pausa, para
   que su reproducción automática no siga sonando); cuando entra uno, abre otra tanda. Antes de
   abrir el grabador de cada tanda también se pausa Spotify: lo que siga sonando no puede ser lo
   que el grabador tome como comienzo del primer tema;
5. cuando la búsqueda terminó y no queda nada por grabar, mueve todos los MP3 a
   `Playlists_DOWNLOAD/<nombre de playlist>/`, elimina duplicados por Spotify Track ID, escribe
   `<playlist>.m3u8` y limpia los intermedios del run.

Soulseek bloquea la cuenta 30 minutos si se pasan 34 búsquedas cada 220 s. Como ahora cada
búsqueda es un proceso, la cuenta la lleva el descargador en
`%APPDATA%\traktor_soulseek_searches.json`, compartida por todas las descargas del ordenador:
si se llega al límite, espera su turno (la app lo muestra en el tema). Tema a tema el límite casi
no se toca, porque bajar cada tema lleva tiempo.

Una búsqueda que falla (tiempo agotado, sesión) se reintenta una vez. Si sigue fallando, ese tema
se manda a grabar recién cuando la búsqueda del siguiente funciona (el problema era del tema, no
de la búsqueda). Con 3 temas seguidos sin búsqueda, la descarga se corta sin mandarlos a grabar:
el problema es la búsqueda (por ejemplo, la cuenta de Soulseek abierta en otra PC). Lo que ya
estaba en la cola se graba igual y el run termina `partial` con lo conseguido.

Si Spotify se detiene durante una tanda (otro dispositivo tomó la reproducción, una pausa), o no
arranca, o no se pudo crear la playlist de la tanda, **la grabación se detiene pero la descarga
sigue**: al terminar, lo bajado va a la carpeta y el run queda «En pausa» (código 75) con lo que
falta grabar; «Continuar» (o el mismo comando) graba solo eso. Si el reproductor termina sin
llegar a reproducir (Spotify recién abierto que tarda en aparecer), tiene un segundo intento; si
se agota su tiempo, no se reintenta (pudo empezar a sonar y correría los cortes) y Spotify se
pausa.

Si aun así falta algún tema, igual se guarda en la carpeta todo lo que se consiguió: el run queda
como `partial`, con los Spotify IDs que faltan en `orchestration_report.json`, y el orquestador
termina con código 3. «Volver a intentar» (o el mismo comando) hace una descarga nueva; lo que ya
está en la carpeta no se duplica.

## Elección del archivo en Soulseek

En la búsqueda, sockseek exige el artista en la ruta, el formato y el bitrate, pero no
el título literal: el título y la versión de cada archivo los juzga el descargador con
las mismas reglas de la verificación, leyendo el nombre del archivo como si fuera la
etiqueta (sin número de pista, «Artista - » ni «- Artista 124» de los pools de DJ,
etiquetas entre corchetes ni sufijo de grupo scene, y con «Artista - Disco - 06 -
Título»). Con `--strict-title`, sockseek pedía el «Título - Versión» de Spotify al pie
de la letra y perdía «Título (Versión)»: en una playlist de 152 temas, 29 de los 46 «no
disponibles» estaban. La tolerancia de duración de sockseek llega hasta 1,9× (una
Extended); el rango de cada tema lo aplica el descargador.

Una descarga nueva de una playlist trabaja solo sobre lo que falta: lo que ya está en su
carpeta de música (`--have-dir`, la que usó una descarga anterior dentro del destino
actual) no se busca, no se baja y cuenta como guardado; y los MP3 terminados de una
descarga anterior que se cortó antes de guardar (`--reuse-dir`) se vuelven a verificar
(320 kbps, Spotify Track ID, carátula) y se traen sin buscarlos, igual que los grabados de una
descarga anterior que no se llegaron a guardar. Por eso «Continuar
descarga», «Volver a intentar», «Volver a buscar lo que falta» (con la grabación
pendiente) y «Buscar temas nuevos» buscan y bajan solo el resto.

De cada tema se prueban hasta 4 archivos de los que encontró la búsqueda, en este
orden: la Extended explícita (si Spotify no apunta ya a una), los de la duración de
Spotify (±8 s) y después el resto. Dentro de cada grupo van primero los que llevan el
título en el nombre del archivo, luego los de una carpeta con el nombre del álbum de
Spotify y luego los de duración más cercana. Cada archivo bajado se verifica (título,
artista, duración, variantes e ISRC) y, si no pasa, se borra y se prueba el siguiente.
La etiqueta de artista se compara también separada (`A & B`, `A e B`, `A, B`,
`feat.`), y un ISRC idéntico al de Spotify acepta la grabación aunque el artista esté
escrito de otra forma. La Extended inferida (sin «extended» en el nombre y al menos
20 s más larga) solo se acepta si el artista coincide sin separar.

Los MP3 grabados y los descargados por Soulseek pasan por el mismo finalizador:
usan el nombre `Artista principal - Título.mp3`, las etiquetas oficiales de
Spotify (álbum, artistas, fecha, número de pista/disco e ISRC), los identificadores
y enlaces esenciales para auditoría, y el cover oficial embebido. Se conserva un
perfil ID3v2.3 reducido, comprobado tanto en Windows Media Player Legacy como en
el Reproductor multimedia actual de Windows.

## Seguridad de las descargas Soulseek

La consulta previa rechaza nombres peligrosos (ejecutables, archivos comprimidos,
dobles extensiones, rutas relativas y caracteres de control) y solo admite MP3,
FLAC y WAV. Exige tamaño y duración anunciados válidos; sin estos datos el tema
queda como faltante. Descarga únicamente el archivo exacto de un candidato
aprobado, sin repetir una búsqueda abierta ni bajar carpetas completas.

Para MP3 320 kbps, el tamaño esperado es `segundos × 320000 / 8`:
se permite ±15 % y hasta 2 MiB adicionales para etiquetas y carátula. Un tema
de 5 minutos espera 12 MB de audio. Para WAV se calcula a partir de frecuencia,
profundidad y canales (Soulseek no anuncia canales: se contempla mono/estéreo).
FLAC tiene un rango más amplio porque su compresión depende del contenido.
Se rechazan parámetros PCM desconocidos, tamaños superiores a 512 MiB y
duraciones incompatibles con el tema o su Extended. Son reglas conservadoras:
pueden excluir archivos legítimos con metadata incompleta o carátulas grandes.

Antes de transferir se comprueba que el antivirus puede analizar un archivo de
prueba. Los archivos recibidos permanecen en una carpeta temporal aislada:
se verifica la cabecera y el tamaño real, se exige un análisis antivirus sin
detecciones y después se comprueban los datos de audio reales. Se vuelve a
analizar el MP3 final con la carátula antes de moverlo a la salida. Los candidatos
rechazados y las transferencias incompletas se eliminan; detección, error o
timeout del antivirus bloquean la incorporación a la biblioteca. La consolidación
revalida audio, tamaño y antivirus tanto de los MP3 del run como de los duplicados
existentes antes de borrar una copia; una etiqueta Spotify válida sola no basta.

Durante la transferencia se comprueban los bytes recibidos cada 100 ms, el espacio
libre y la salida del proceso. Si se supera el tamaño anunciado, se agota el margen
de disco o pasan 360 segundos, se cancela el proceso antes de limpiar los archivos.
Es un control por sondeo: puede haber un exceso breve entre comprobaciones.

La conversión WAV/FLAC tiene un timeout de 180 segundos, dos hilos, límite de
asignación individual de 64 MiB, duración y tamaño de salida acotados. ffmpeg solo
abre archivos locales y el formato de entrada se fija explícitamente. Se rechaza
un MP3 truncado. Estos límites no equivalen a una cuota total de memoria del proceso.

Las carátulas requieren HTTPS, antivirus y contenido JPEG/PNG/WebP decodificable,
con hasta 15 MiB, 16 millones de píxeles y un solo fotograma. Pillow verifica y
reconstruye un JPEG de hasta 2048×2048 y 2 MiB, descartando metadata y datos
añadidos al final; también se analiza esa imagen antes de incorporarla al MP3.

Windows requiere Microsoft Defender; Linux requiere `clamscan` con sus bases
instaladas. No se instala ni se desactiva ningún antivirus automáticamente.
La configuración de Sockseek debe tener `username` y `password` en la sección
global: el script usa una copia temporal que conserva solo esas credenciales,
para impedir que perfiles, comandos `on-complete` o descargas alternativas
heredadas ejecuten acciones sobre los archivos recibidos.

Estos controles reducen el riesgo, pero no certifican ausencia de malware:
un servidor puede mentir sobre un archivo y un antivirus puede no detectar una
amenaza nueva. No se puede asegurar que nunca lleguen bytes maliciosos a la
carpeta temporal; sí se bloquean los archivos detectados antes de incorporarlos.
Mantener actualizados el antivirus, Sockseek, Mutagen, Pillow y ffmpeg.

## Reanudación después de una interrupción

El grabador codifica y registra cada MP3 apenas termina el tema, sin esperar al final de la
tanda. Cada tanda tiene su carpeta, `recordings/NN/`, con sus temas y su playlist
(`session.json`) y el checkpoint del grabador (`recording_progress.json`). La cola de grabación
y los reportes de la búsqueda y de las descargas se escriben tema a tema.

Un watchdog consulta el estado real de Spotify y detecta una pausa sostenida; como respaldo, el
loopback también detecta 10 segundos de silencio digital. En ese caso el tema actual se considera
incompleto y la tanda queda reanudable.

Al ejecutar nuevamente exactamente el mismo comando (o «Continuar» / «Grabar ahora» en la app),
el orquestador retoma el run más reciente de la playlist si no terminó (interrumpido, en pausa,
con la grabación pospuesta o cortado):

1. si la búsqueda no había terminado, la sigue donde quedó: lo ya buscado, bajado o puesto en la
   cola no se repite;
2. retoma la tanda sin terminar desde su primer tema pendiente (Spotify arranca en ese tema y
   desde el segundo cero) y después graba en tandas nuevas lo que quedó en la cola;
3. si los MP3 de una tanda ya no están en el run (por ejemplo, «Guardar los temas listos» los
   llevó a la carpeta), lo que le faltaba vuelve a la cola para otra tanda. Los MP3 se buscan por
   nombre dentro de la carpeta de la tanda, así que un run que se movió (otra carpeta de runs)
   conserva lo grabado.

Los runs anteriores al 2026-11-06 (preflight en lote) no se retoman: una descarga nueva de la
playlist reutiliza lo que bajaron o grabaron sin guardarlo (`--reuse-dir`).

El fragmento del tema interrumpido no se reutiliza: solo se conservan temas completos, y por eso
la captura WAV parcial se elimina (y un MP3 que quedó a medio escribir no va a la carpeta). Para
ignorar deliberadamente un run anterior y crear otro puede usarse `--fresh`.

Cada ejecución queda aislada en `download_JIJIJI/runs/<fecha>_<playlist-id>/`,
pero al completarse ya no conserva audio. En la raíz quedan el resumen, la cola de
grabación y el reporte de consolidación; los detalles se agrupan bajo `diagnostics/` (las tandas,
en `diagnostics/recordings/NN/`). La música terminada queda en
`download_JIJIJI/Playlists_DOWNLOAD/<nombre de playlist>/`.

Puede cambiarse la raíz de runs mediante `SPOTIFY_DOWNLOAD_ROOT` y la biblioteca
consolidada mediante `SPOTIFY_PLAYLISTS_ROOT`, como variables de entorno o como
líneas del `.env`.

## Configuración inicial

Para instalar todo en otra PC paso a paso (claves de Spotify y Soulseek,
sockseek, ffmpeg y las autorizaciones de Spotify), seguir
[CONFIGURAR_OTRO_ORDENADOR.md](CONFIGURAR_OTRO_ORDENADOR.md). Resumen:

Instalar los paquetes Python:

```powershell
.venv\Scripts\python.exe -m pip install -r download_JIJIJI\requirements.txt
```

También deben estar disponibles `ffmpeg` y `sockseek`. El orquestador busca
`sockseek.exe` tanto en `PATH` como en `tools/sockseek/sockseek.exe`; para
`ffmpeg.exe` también reconoce automáticamente la instalación de WinGet de
`Gyan.FFmpeg`, aunque el enlace de WinGet todavía no esté disponible en `PATH`.

La entrada se normaliza a una URL canónica. Por eso también tolera que PowerShell
reciba accidentalmente un enlace Markdown con la forma `[URL](URL)`.

Definir en el archivo `.env` ignorado por Git:

```dotenv
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
SPOTIFY_REDIRECT_URI=http://127.0.0.1:48721/callback
```

La aplicación de Spotify debe tener registrado `http://127.0.0.1:48721/callback`,
que la lectura de playlists usa siempre (está fijo en el código), y también el
valor de `SPOTIFY_REDIRECT_URI` si es otro, porque lo usa el control de la
reproducción. En la PC actual ese valor es `http://127.0.0.1:8888/callback`. Durante
la grabación conviene desactivar otros sonidos del sistema y el crossfade de
Spotify.
