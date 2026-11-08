# TRAKTOR ML — Lessons Learned

## Clustering (HDBSCAN / PCA)

### PCA pre-HDBSCAN es necesario para embeddings de alta dimensión
**Contexto:** MERT produce embeddings de 1024 dims. HDBSCAN no estima densidad correctamente en espacios de alta dimensión (curse of dimensionality). Con N=239 tracks y HDBSCAN directamente en 1024D, los mejores resultados eran 2 clusters con 31% noise, o 3 clusters con 80% noise.
**Solución:** Aplicar PCA (sklearn) antes de HDBSCAN. Con pca_dim=50 se retiene 93.7% de varianza y HDBSCAN produce 8 clusters en `test_20` (N=239).
[REEMPLAZADO 2026-09-29 por DECISIONS 2026-09-29: la organización vigente es CLAP + BPM (peso 0.3) con Ward; en CLAP, HDBSCAN deja casi todo como ruido (V4_USAGE, sección 0b). Estos parámetros valen solo para MERT + HDBSCAN.] **Parámetros óptimos para test_20 (N=239):** `pca_dim=50, l1_min_cluster_size=6, l1_min_samples=1` → 8 clusters L1, ~51% noise, tamaños [33,24,20,11,8,8,7,6].
**Nota:** Con N~239 no es posible obtener <30% noise con ≥3 clusters simultáneamente. El límite de ruido mejora con más tracks. `mert_full` como L1 no ofrece ventaja sobre `mert_perc`.

### Separabilidad del espacio MERT-v1-330M con test_20
Para N≤300 tracks de techno/tech house, esperar noise rates de 40-60% en L1. Esto es estructural del dataset pequeño, no un bug. Los tracks asignados (~116 de 239) sí tienen estructura musical real.

## Tags de audio

### Guardar un ID3 v2.3 une con «/» los campos que traían varios valores
**Contexto (2026-09-29):** al escribir los tags de Beatport, mutagen conserva la versión ID3 del archivo.
Un archivo v2.3 («Len Faki - BX 3») traía un título con dos valores ("BX 3", "BX 3 (original mix)"), algo
que v2.3 no admite; al guardarlo quedó "BX 3/BX 3 (original mix)". Fue 1 de 1087 archivos escritos: los
demás campos no cambiaron (verificado contra el catálogo).
**Regla:** después de escribir tags, comparar todos los campos contra el catálogo, no solo los escritos.

## Detección de voz

### Silero VAD no detecta canto sobre la mezcla completa
**Contexto (2026-09-29):** Silero VAD v5.1.2 (TorchScript, 16 kHz) sobre el tema entero, en 63 temas de
`musica`: en las carpetas «Vocal» de Gabriel (23 temas, todos con voz) encontró voz en >= 1 % del tema
solo en el 13 % y nunca llegó al 5 %; CLAP zero-shot marcó el 87 %. Es un detector de habla entrenado
para ignorar música, y el canto sobre la base cuenta como música.
**Regla:** no usarlo sobre la mezcla. Solo tendría sentido sobre la voz separada (Demucs).

### CLAP zero-shot sobre 90 s del medio pierde voces
**Contexto (2026-09-29):** con 3 tramos de 30 s del medio (≈ 20 % de un tema de 7 min) el detector marcó
55 % de 20 temas con «feat.»; con ventanas de 10 s cada 20 s sobre todo el tema, 75 %, y en 20 temas
Dub/Instrumental bajó de 30 % a 25 %. Muestras chicas (etiquetas débiles: «feat.» no garantiza voz
audible y un dub puede tener recortes de voz). Ningún umbral sobre los 90 s arreglaba las dos cosas.
**Regla:** para la voz, mirar el tema entero antes de ajustar el umbral.

### Al cambiar un método, cambiar también a quien lo llama
**Contexto (2026-11-04):** la etiqueta Vocal pasó a `clap_probe` el 2026-10-05, pero «Agregar música
nueva» de la app y `organize.py ingest` seguían llamando `tag_vocals.py --method clap --write-tags`,
que recalcula y escribe toda la biblioteca: el primer agregado le habría quitado la marca a 473 temas
y puesto a 94. **Regla:** al reemplazar un método, buscar todas sus llamadas (`grep -rn` en `src/`,
`tools/`, `docs/`) y fijar el método vigente en un test.

### «2020 new - copia» no es un respaldo: es la carpeta de test_20
**Contexto (2026-09-29):** `data/raw_audio/test_20` es un enlace (junction) a `Música/2020 new - copia`.
Las reglas que ven «copia» como respaldo (Phase 0 al elegir entre archivos idénticos, `location_rank`)
la ponen última. Al mover a `_copias` una copia repetida de esa carpeta, test_20 perdió un tema
(`test_block1_common` esperaba 243). **Regla:** nunca mover ni borrar archivos de la carpeta de otro
dataset. `dedupe.move_copies` las protege con `protected_roots`.

## Descargas (`download_JIJIJI/`)

### Sockseek limita las búsquedas: un tope fijo de preflight rompe las playlists largas

Sockseek hace como máximo 34 búsquedas cada 220 s (Soulseek banea 30 minutos si se excede;
`--searches-per-time`, `--searches-renew-time`). El preflight lanza una búsqueda por tema en un
único proceso, así que 152 temas necesitan al menos 4 esperas de 220 s (880 s). Con el tope fijo
de 600 s, los dos intentos del 2026-10-30 agotaron el tiempo, perdieron todo el lote y los 152
temas quedaron como error. [REEMPLAZADO 2026-11-06 por la búsqueda tema a tema: un proceso de
sockseek por tema (tope 90 s) y el límite de 34/220 s lo lleva `SearchLimiter`, en un archivo del
ordenador; el límite de sockseek no sirve entre procesos.] El tope ahora se calcula con
`preflight_timeout_for(n)`. Un preflight que falla para todos los temas es un problema de la
búsqueda: no se manda la playlist a grabar.

### Sockseek con una lista busca a ráfagas y después se queda parado

Con un CSV, sockseek abre hasta `--concurrent-jobs` (20 por defecto) trabajos juntos y busca de a
`--concurrent-searches` (2): en el log del preflight del 2026-11-02 (152 temas), 20 temas quedaron
esperando búsqueda en el mismo segundo, las 34 búsquedas de la ventana se gastaron en menos de 2
minutos y el log quedó mudo 104-119 s, cuatro veces. Es lo que Gabriel vio como «Soulseek se
tranca». Buscando tema a tema y bajando cada uno enseguida, las búsquedas quedan espaciadas. Un
proceso nuevo de sockseek empieza su limitador de cero: la cuenta de búsquedas tiene que llevarse
fuera de sockseek (`SearchLimiter`).

### `--strict-title` de sockseek pide el título de Spotify al pie de la letra

En una playlist de 152 temas (2026-11-02), 29 de los 46 «no disponibles» estaban en Soulseek con la
versión correcta: `--strict-title` exigía «Título - Versión» (como lo escribe Spotify) dentro del
nombre del archivo y perdía «Título (Versión)», los nombres scene con guiones bajos, etc. Contraste
de una sola variable sobre 17 de esos temas: con la configuración real, 2 con resultados; sin
`--strict-title`, 16; con 15 s de búsqueda en vez de 5, 1; los 3 de control aparecieron siempre.
Buscar por el título base no agregó ninguno. El título ahora lo juzga el descargador con las reglas
de la verificación (`_availability_title_error`): la búsqueda nueva dio 132 disponibles contra 106.
Ojo al leer el log de sockseek: «N results found» cuenta lo que pasa sus filtros, no lo que llegó.

### Varias cosas a la vez en Soulseek: el daemon de sockseek, no varios procesos

Cada proceso de sockseek abre su sesión, y la misma cuenta conectada dos veces corta la primera:
no se puede buscar y bajar en paralelo con procesos sueltos. sockseek 3.0.5 trae un daemon
(`sockseek daemon --server-port N`, API REST en 127.0.0.1) con una sola sesión, y los procesos de
sockseek con `--remote=http://127.0.0.1:N` le pasan el trabajo con las mismas opciones de siempre:
la búsqueda devuelve el mismo JSON y la descarga cae en el `--output-dir` del cliente. La API tiene
`/api/openapi.json`, `/api/jobs` y `/api/jobs/{id}/cancel`. Para probar sin la cuenta:
`--mock-files-dir <carpeta>` (en el daemon o en el proceso) busca y «baja» archivos locales; el
daemon los indexa al arrancar y las descargas simuladas aparecen enteras de golpe, así que no sirven
para probar una transferencia que se traba. Verificado el 2026-11-06 con el daemon simulado.

### Con Traktor abierto en el DJM, el loopback del DJM no se puede grabar

Traktor usa el DJM-750MK2 en exclusiva: abrir el loopback de «Line (DJM-750MK2)» (la salida
predeterminada de Windows en el escritorio) da `OSError [Errno -9996] Invalid device`, y Spotify
tampoco puede sonar por ahí. El 2026-11-06 eso tiró el grabador con un traceback a mitad de una
descarga. Ahora el grabador lo dice y sale con el código 76, y `playlist_loopback_recorder.py
--check-device` lo prueba antes de cada tanda. Para grabar con Traktor abierto, Spotify tiene que
sonar por otra salida.

### Al final de una playlist, Spotify se detiene en el progreso 0: no es una pausa

Con la repetición apagada, al terminar el último tema Spotify se detiene y vuelve al primero de la
playlist con `progress_ms` 0. El watchdog, que solo daba por natural una parada a menos de 1 s del
final del tema, lo tomó como una pausa: el 2026-11-07 la tanda de un tema se cortó con 0/1 grabados
(`spotify_playback_status.json`: `progress_ms` 0, `duration_ms` 302065). Además, el grabador necesita
1,5 s de audio después del final esperado para cortar, y el watchdog esperaba 1,5 s: una carrera.
La corrida del 2026-10-30 tiene la misma huella y la ganó. Ahora la playlist de la tanda suena en
repetición (el último tema se corta como los demás) y el watchdog mide el final con el último estado
que vio sonando. Con tandas tema a tema, casi cada tema es el último de su tanda.

### Lo grabado de Spotify sale unos 9 dB más bajo: la normalización de volumen

Con «Normalizar volumen» activado (viene así; en `prefs` no aparece `audio.normalize_v2=false`),
Spotify baja los masters fuertes hacia −14 LUFS. Un tema grabado el 2026-11-07 quedó en −17 LUFS
con pico de −8,9 dBFS; los de Soulseek de la misma playlist están entre −6 y −10 LUFS (ffmpeg
`ebur128`). Es una baja pareja, sin recortes: el grabador sube cada tema hasta un pico de −1 dBFS
y recupera el nivel. Desactivar la normalización en Spotify evita perder resolución.

### En Windows, no dejar un hilo bloqueado leyendo la entrada estándar

La app de Organización sabía que la app de descargas se cerraba porque un hilo leía su entrada estándar (un pipe)
hasta el fin de archivo. El 2026-11-07 se trabó dos veces:
- un script lanzado por ella no arrancó (13 MB, 0 s de CPU a los 30 min);
- después la app entera se congeló al importar `dedupe` a mitad de una tarea.

En una prueba aislada, con un hilo así bloqueado, un hijo que hereda la entrada no arranca e importar módulos
queda colgado. En Windows, una lectura sincrónica pendiente frena otras operaciones sobre el mismo archivo.
Para saber si se cerró, mirar sin bloquear (`PeekNamedPipe`) o vigilar el proceso padre, y lanzar los hijos con
`stdin=DEVNULL`.

### Un dato que cambia en cada versión, guardado en cada archivo, obliga a reescribirlos todos

El lugar de cada tema en la organización va dentro del archivo (`file_embeddings.py`) e incluía la versión de la
organización. Cada «Agregar música» sube la versión. Por eso «Actualizar el pendrive» reescribía y verificaba,
con dos hashes de audio, los 1760 archivos para sumar 10 temas. La versión ya no cuenta como cambio y se
escriben solo los temas nuevos o movidos.

### Spotify puede rechazar con un 500 la transferencia al dispositivo que ya está activo

El 2026-11-07 a las 18:33, `PUT /v1/me/player` (`transfer_playback`) devolvía 500 «Server error» en cada
intento hacia el escritorio, que ya era el dispositivo activo. Siguió así por lo menos 10 minutos,
con o sin `play`. En cambio, `shuffle` y `repeat` sobre el mismo dispositivo respondían 200. spotipy
reintenta los 500 y lo informa como «rate/request limit» con un 429, que no es la causa. Los tres
arranques de la tanda fallaron y la grabación quedó en pausa. Si el dispositivo ya está activo, no se
transfiere; `start_playback` con el `device_id` alcanza. Una transferencia que falla no corta el
arranque.

### Una transferencia cancelada deja su archivo abierto un momento (WinError 32)

Al cancelar en el daemon de sockseek una transferencia trabada, el `.incomplete` sigue abierto unos
instantes. Borrar la carpeta del intento con `TemporaryDirectory` daba `PermissionError [WinError 32]`
y esa excepción cortaba el tema entero, sin probar el siguiente archivo («Human Safari - Amor»,
2026-11-07). Ahora la limpieza de cada intento tolera el error (`ignore_cleanup_errors=True`) y
lo que queda se borra con la carpeta de staging al final, ya con el daemon cerrado.

### Lo guardado se reconoce por la carpeta, no por la historia de corridas

«Borrar archivos de trabajo» (o borrar `runs/` a mano) se lleva la corrida que recordaba dónde quedó
la música de la playlist. La corrida siguiente buscó de nuevo 9 temas ya guardados y mandó dos a
grabar (2026-11-07). La carpeta se encuentra también por el nombre de la playlist
(`--library-root`), y un tema que ya está ahí no se graba aunque haya quedado en la cola.

### Un PID que ya terminó puede ser de otro programa: no adoptar a sus hijos

La app de descargas recordaba todos los procesos «hijos» de la descarga y al final cerraba los que
seguían vivos. Tomaba como hijo a cualquiera cuyo padre tuviera un PID que alguna vez fue de la
descarga. Windows reutiliza los PID: el 2026-11-07, al terminar una descarga de 23 minutos, se
cerraron `GoogleDriveFS.exe`, varios `msedgewebview2.exe` y el `crashpad_handler.exe` de Spotify. Un
proceso entra ahora solo si su padre sigue siendo el mismo (misma hora de creación) y nació antes que él.

### Spotify puede tardar en empezar a sonar: eso no es una pausa

El vigilante de Spotify daba por detenida la reproducción si a los 1,5 s no sonaba. El 2026-11-07 a las
17:41, con la PC cargando los modelos del postproceso, Spotify tardó más en arrancar: el vigilante
informó «sin tema, progreso 0», el grabador se cortó y la descarga dejó de grabar con 3 temas en cola.
Antes de verlo sonar por primera vez, el vigilante espera hasta 30 s (y repite la orden a los 8 s), y una
tanda cortada se reintenta sola en vez de bloquear la grabación de toda la descarga.

### Igualar volumen es igualar sonoridad, no picos

Subir los grabados a un pico de −1 dBFS los dejó en −8,4, −9,6 y −11,9 LUFS, contra −6,3 a −8,3 de los
mismos temas bajados de Soulseek. La versión de Spotify tiene más rango dinámico y toca el techo con
menos sonoridad. Lo que iguala es medir la sonoridad integrada (ffmpeg `ebur128`, 0,5 s por tema) y
subirla con un limitador; el limitador se come parte de la ganancia, así que se mide de nuevo después.

### Beatport puede pedir la verificación anti-robots de Cloudflare

El 2026-11-07, después de unos pocos pedidos seguidos (y de los reintentos de `BeatportClient`, que
esperan unos 5 minutos), `beatport.com/search` respondió 403 con `cf-mitigated: challenge` («Just a
moment…») a todo, también a consultas que minutos antes daban 200. `requests` no pasa esa
verificación. `tag_downloads.py` hace un pedido de prueba antes de buscar y, si está bloqueado, deja
los temas en espera en lugar de frenar el final de la descarga.

### La salida de una sola app (Spotify a Voicemeeter) usa una interfaz no documentada

Windows guarda una salida por programa (Mezclador de volumen), pero no hay API pública. EarTrumpet
usa `Windows.Media.Internal.AudioPolicyConfig` (`IAudioPolicyConfigFactory`, IID
`ab3d4648-e242-459f-b02f-541c70306324` desde Windows 10 21H2): 19 métodos sin usar y después
`SetPersistedDefaultAudioEndpoint` / `GetPersistedDefaultAudioEndpoint`, con el id envuelto en
`\\?\SWD#MMDEVAPI#<id>#{e6327cad-dcec-4949-ae8a-991e976a79d2}`. Se pide con un proceso abierto y
queda guardado para el programa. Verificado el 2026-11-07 en el escritorio (`app/audio_devices.ps1
-App/-SetApp`, ida y vuelta). Windows PowerShell 5.1 lee un `.ps1` sin BOM como ANSI y rompe los
acentos de los mensajes: el script va con BOM UTF-8.

### Probar en Soulseek mientras baja una descarga: otra cuenta y `--no-config`

Una segunda sesión con la misma cuenta corta la otra, y el descargador abre una por transferencia.
Para buscar en paralelo hace falta otra cuenta (se crea sola al conectarse con un nombre nuevo),
pasada con `--no-config --user=… --pass=…` y otro `--listen-port` (el 49998 lo usa la descarga).
Con `--config=<archivo>`, sockseek usó igual la cuenta de la configuración global; no se sabe por
qué, así que la configuración «segura» del descargador puede no aislarlo de esa configuración. Una
contraseña que empieza con «-» tiene que ir como `--pass=-…`.

### Leer el loopback con lecturas bloqueantes pierde audio en cada pausa, sin aviso

PyAudio (PortAudio WASAPI) en modo bloqueante, con `frames_per_buffer=1024`, guarda 42,7 ms (dos bloques).
Si el bucle que lee se atrasa más, porque la CPU está ocupada, lo que no entra se pierde, y PortAudio no lo
avisa ni con `exception_on_overflow=True`. Medido el 2026-11-07 sobre Voicemeeter Input: una pausa de
100 ms pierde de 56 a 78 ms, y una de 1 s pierde 957 ms. Agrandar el buffer no lo arregla: con 2 s, casi
lleno, entregó unos 70.000 frames por segundo de un stream de 48 kHz y no se recuperó. El modo callback
(`stream_callback` que pone cada bloque en una cola) no perdió nada con el bucle dormido 6 s. Síntoma en
Traktor: un BPM no entero y una grilla que calza en un tramo y no en otro. Para encontrar dónde se perdió,
comparar el reloj con el audio recibido (`capture_timing.csv`) y la fase de los bombos contra una grilla
fija (`tools/recording_audit/`).

### Spotify en otro dispositivo de la misma cuenta corta la grabación

Spotify reproduce en un solo dispositivo por cuenta. El 2026-11-07, a las 18:43:48, la cuenta pasó a
sonar un tema que no estaba en la playlist y se pausó; se cortaron dos intentos de grabación. Ninguna
sesión del agente había llamado a Spotify en ese momento. Mientras graba, no usar Spotify en el teléfono
ni en el televisor del comedor.

## Windows local

### El Python de la Microsoft Store virtualiza %APPDATA%

Lo que ese Python escribe en `%APPDATA%` (token `traktor_spotify_token.json`, ajustes de la app de
descargas) queda en `%LOCALAPPDATA%\Packages\PythonSoftwareFoundation.Python.3.11_*\LocalCache\Roaming\`.
Los procesos de ese Python lo ven en `%APPDATA%`; PowerShell y el Explorador, no. Al buscar esos
archivos fuera de Python, hay que mirar en la ruta virtualizada. Lo que se copia a `%APPDATA%` desde
fuera (`sockseek.conf`) sí se lee, porque la lectura cae a la carpeta real.

### Un Job Object no agrupa a los hijos del Python de la Store

`AssignProcessToJobObject` no captura a los procesos que lanza ese Python (ya corre dentro de su
propio contenedor), y al detener el orquestador quedó un `sockseek.exe` huérfano con la sesión de
Soulseek abierta. La app de descargas sigue el árbol por PID padre y guarda la hora de creación de
cada proceso para no cerrar uno que reutilizó el PID (`ProcessTree` en `descargas_app.py`).

### Las playlists de Rekordbox y Traktor guardan la letra de la unidad

`traktor.nml` (VOLUME="E:") y `rekordbox.xml` (file://localhost/E:/...) llevan la ruta absoluta con la
letra. El pendrive cambia de letra entre computadoras y un NML preparado en la laptop (pendrive en
`E:`, temas locales en `D:\MÚSICA`) dio todos los temas como faltantes en otra PC (2026-10-30). Las
playlists del paquete se rearman en cada PC con su `.bat` (`usb_package.py`), o se copia el paquete a
`C:\TRAKTOR ML` y se prepara desde ahí.

### Traktor 4.4: "Import Playlist" toma una sola playlist y no lee carátulas

Un NML con una carpeta y 51 playlists entró como una sola playlist (la primera), y sus temas sin
carátula: Traktor usa los datos del NML y solo lee la carátula del archivo (a su caché propia,
`Coverart/`) al agregarlo desde el disco. Lo que funciona: arrastrar la carpeta de música a Track
Collection y escribir las playlists en `collection.nml` con Traktor cerrado (`traktor_collection.py`).
Traktor a veces queda abierto sin ventana al cerrarse y sin "LOG CLOSE" en `Logs/Traktor.log`: una
vez no guardó la colección; otra la guardó y siguió colgado. Mirar la hora de `collection.nml`.

El análisis (BPM, grilla, tonalidad) queda en memoria hasta que Traktor guarda `collection.nml`, al
cerrarse. El 2026-11-06 analizó unos 1495 temas (17:33 a 20:08; las stripes en `Stripes/` sí se
escribieron), quedó abierto y Windows reinició al día siguiente con Traktor abierto: no hubo ningún
guardado y la colección siguió con 231 de 1717 temas de `C:\TRAKTOR ML` con BPM. Las sugerencias en
vivo copian las entradas de ese archivo y salieron con BPM 0.00. Después de analizar, cerrar Traktor y
comprobar la hora de `collection.nml`. Contar como analizado el `TEMPO` con atributo `BPM`: los temas
importados traen un `TEMPO` vacío (solo `BPM_QUALITY`).

### Preparar Traktor reconoce los temas por su nombre y carpeta originales

`traktor_prepare.ps1` busca en la colección local de Traktor cada tema del paquete por `nombre_original` y
`carpeta_original` de `_organizacion.csv`. Al regenerar el paquete hay que conservar esas columnas (y `origen`):
si pasan a la ruta del paquete, Traktor deja de reconocer los temas que ya analizó.

### Python no abre rutas de más de 260 caracteres en esta PC

Sin LongPathsEnabled, `Path.exists()` da False en silencio para rutas largas: un paquete probado
desde una carpeta de prueba profunda "perdió" 705 temas. Probar con la misma longitud de ruta que el
destino real (`subst R: <carpeta>` deja `R:\TRAKTOR ML`, como `D:\TRAKTOR ML`).

### El nodo iTunes de Traktor puede duplicar entradas de la colección

Traktor busca cada tema del XML de iTunes en su colección por la ruta. Los que reconoce los muestra
con los datos de la colección (cues, grid, BPM). Los que no reconoce los muestra con los datos del
XML (BPM 0.00, sin carátula), y al cargarlos en un deck agrega al final de `collection.nml` una
**segunda entrada con la misma ruta**, con lo que trae el XML (comentario, duración, contador de
reproducciones vacío). La entrada original queda intacta. Un "#" en la ruta (%23 en el XML) basta
para que no lo reconozca (prueba del 2026-11-04). Para revisar después de una prueba, contar las
entradas por ruta (`entry_key`), no solo las rutas distintas: un dict por ruta esconde el duplicado.

### Traktor relee un .nml del Explorer solo al entrar desde otra carpeta; el D2 mueve el árbol

El Explorer de Traktor 4.4.1 muestra como playlist los `.nml` de cualquier carpeta (los `.m3u8` no).
Si el archivo cambia, sigue mostrando lo que leyó hasta que se entra a la playlist viniendo de otra
carpeta: volver a elegir el mismo favorito no alcanza (prueba de Gabriel, 2026-11-04). Traktor no
tiene comando para volver a la posición anterior del árbol, pero el navegador de la pantalla del
Kontrol D2 (QML `Traktor.Browser`, también con un D2 virtual) es el de la ventana: `enterNode` y
`exitNode` mueven el árbol que ve el DJ y `currentPath` dice dónde está (`ApiBrowserProbe.qml`).
Entrar a un **tema** desde esa pantalla lo carga en un deck: la sonda solo entra a carpetas.
Esa pantalla vio vacía `C:\Users\bibbo` durante 6 s y tardó más de 2 s en dar los nombres de
`C:\Users` (2026-11-05): las sugerencias van en la raíz de la unidad y el "goto" espera cada nivel.
Al volver a una carpeta, Traktor conserva solo el tema elegido en la ventana: el `currentIndex` del D2
no hace falta (en `PRO` la sonda llegó con la lista aún vacía y no lo pudo poner, y el tema estaba).

### Probar "desde otro ordenador" con un clon nuevo, y comparar texto sin finales de línea

Para saber si algo se puede seguir desde GitHub, clonarlo en una carpeta vacía y arrancarlo ahí, con
un Python sin paquetes. El 2026-11-05 eso encontró tres cosas que los tests no veían:
- el lanzador suponía un `.venv` o el comando `py`, que esta PC no tiene;
- con una colección sin cues el programa se caía;
- aparecía un aviso falso de «sonda desactualizada».

El aviso salía porque Git con `core.autocrlf` deja los QML con CRLF en un clon de Windows y con LF en
otros lados. Por eso los archivos del repo se comparan con sus copias instaladas sin importar los
finales de línea (`instalar_api_client.same_text`). En un clon superficial (`--depth 1`),
`git pull` falla con "unrelated histories": para actualizarlo, `git fetch --depth 1 origin main` y
`git reset --hard origin/main`.

### Si un controlador deja de responder, ver primero si manda MIDI a alguien

El 2026-11-06 el K2 dejó de responder en Traktor justo al abrir las sugerencias en vivo. Parecía culpa
del programa, pero no lo era. Ni Traktor, ni el programa, ni un tercer oyente recibían nada del K2,
mientras la DJM seguía mandando su reloj por el mismo MIDI de Windows. El K2, o su driver, estaba
colgado. Se arregló desenchufándolo y volviéndolo a enchufar. Antes de culpar al software, abrir la
entrada con `midi_monitor.py` y comparar con otro aparato que mande MIDI solo.

### Si Traktor arranca sin la DJM, cambia solo el audio y el K2 deja de responder

El 2026-11-05 la DJM-750MK2 no estaba conectada al abrir Traktor: Traktor pasó solo de
`DJM-750MK2 ASIO` a la tarjeta interna (`Senary Audio`), borró el ruteo (Master 3/4, Monitor 1/2) y el
mapeo del K2 no hacía nada, aunque el K2 mandaba MIDI y el puerto no estaba tomado. Primero conectar y
encender la DJM, después abrir Traktor. Para volver: Preferences > Audio Setup, o copiar las entradas
`Audio.*` y `Routing.*Out.Internal.*` de un respaldo en `Backup\Settings\` con Traktor cerrado.

### Importar un .tsi completo reemplaza el Controller Manager: el Kontrol D2 tiene que estar en cada uno

El 2026-11-06 Gabriel cargó `DJM + K2 (external mix) (1).tsi` con Traktor abierto. Traktor escribió sus
ajustes (y un respaldo en `Backup\Settings\`) sin cerrarse, y la mezcla externa se vio a los 2 s. Pero
esa exportación es anterior a las sugerencias en vivo y no trae el Kontrol D2 virtual. Desde ese
momento no llegó ningún aviso de los decks, y cada pulsación dio «La pantalla del D2 no responde». La
sonda y el módulo están en los QML de Traktor, que no cambian; lo que se pierde es el dispositivo. Cada
configuración que se cargue tiene que tener el D2 en el Controller Manager: se agrega una vez y se
vuelve a exportar.

También reemplaza los Favoritos del navegador (2026-11-07). Se guardan en la entrada `Browser.Shortcuts`
del `.tsi`. La configuración externa traía unos Favoritos sin «TRAKTOR ML Sugerencias», y con una
carpeta de otro usuario de Windows (`C:\Users\usuario\...`). Después de importarla, el favorito de las
sugerencias ya no estaba. Lo mismo vale para el Controller Manager: dejar los Favoritos como se quieren y
volver a exportar.

### Dos sesiones en el mismo árbol: subir solo los bloques propios con contexto y verificar

Dos sesiones de agentes compartían el árbol y el índice de Git. Para subir solo sus bloques de un
documento compartido, una usó `git diff -U0` filtrado y `git apply --cached --unidiff-zero`. El
2026-11-06 un bloque sin contexto quedó en otro lugar: el punto 9 de DECISIONS cayó dentro de la
entrada 2026-11-04 y cortó una oración (commit bd0e074; la otra sesión lo arregló). Con contexto
(`-U3`), un bloque que no encaja falla en vez de caer mal. Después de prepararlo, `git diff -- <doc>`
tiene que mostrar solo los bloques ajenos. Si el otro no tiene nada sin commit, subir el archivo entero.

## Entorno HPC (ver también memory/MEMORY.md)

[REEMPLAZADO 2026-09-12 por DECISIONS 2026-09-12, punto 4: Surrey HPC ya no existe; se trabaja en CPU local (AGENTS.md). Se conserva como registro.]

- El env `traktor_ml` de conda debe crearse con Python 3.11 desde `/user/HS300/gb0048/anaconda3/`.
- Login node (`datamove1`) sí tiene acceso a internet (puede hacer pip install, git pull).
- El comando `source /user/HS300/gb0048/anaconda3/etc/profile.d/conda.sh` es necesario antes de `conda activate`.
- Siempre correr scripts con `PYTHONPATH=/mnt/fast/.../traktor` desde el repo root.

## 2026-11-07: audio continuo de sets mezclados (D7)

- **Un archivo elegido solo por duración puede ser otra mezcla.** En playlists de colección con decenas de
  sets de ~75 min, otro set dura lo mismo (±8 s) por azar: 53 de 96 discos «continua» de D7 eran de otro DJ.
  Validar por título (y por pistas en las bajadas tema por tema) antes de usar el audio; una verificación
  que use la misma magnitud que eligió el archivo (la duración) es circular y no detecta nada.
- **Toda medida de orden sobre audio mezclado lleva un control de límites falsos.** Dos tramos contiguos de
  cualquier mezcla se parecen; con límites desplazados, iguales o al azar el «tramo siguiente» ganó más
  (36-46 %) que con los límites reales (24-27 %). Si los falsos empatan o ganan, la medida es continuidad,
  no la elección del DJ. Vale para D8 (nivel de bloque) y para cualquier set grabado.
- **Al comparar vectores con distinto borde, igualar la cantidad de bloques.** Menos bloques baja el ranking
  aunque las diferencias de medianas se conserven; un control con los mismos bloques elegidos al azar
  separa el efecto del borde del ruido.

## 2026-11-08: bancos sobre los sets de Gabriel (D1)

- **Lo que Traktor analiza al cargar no es independiente de lo que sonó.** Traktor pone tonalidad (y BPM, cues)
  a lo que se carga en un deck; en un banco que repite sets, «sin tonalidad» se confunde con «nunca cargado». Un
  componente que castiga a esos temas mejora el acierto sin decir nada de armonía. Antes de medir un rasgo que
  viene del análisis de Traktor, completarlo para toda la biblioteca (tags, nombre del archivo, análisis) o
  informar la mejora como «temas que el DJ no toca».
- **Al comparar dos lados de un par, cambiar uno a la vez.** «Final del actual frente a comienzo del candidato»
  contra «comienzo del actual frente a final del candidato» cambia los dos lados; con un lado fijo se vio que
  lo que predecía era el comienzo del candidato y no el final del actual. El control de una consulta barajada
  (otro tema al azar) dice si el par importa o si gana el candidato solo.
- Un banco que mide si el tema siguiente del DJ cae en los 30 primeros no mide la calidad de una lista de
  parecidos: con unos 1700 candidatos y decenas de temas igual de buenos, todo da entre 9 y 12 % y nada se
  distingue. Sirve para descartar. La calidad de la lista se mide con la compacidad de la zona (sets
  profesionales) y con la escucha a ciegas de listas enteras de a dos, como el test de Vande Veire y De Bie
  2018 (17 de 18 oyentes). Antes de concluir «no mejora», preguntarse qué mide la medida.
- **Un umbral de distancia sobre una lista ya ordenada por esa distancia solo decide dónde cortarla.** No reordena nada:
  comparar la lista con su parte «segura» es compararla con su comienzo. Además, un radio sacado de pares tocados juntos
  (0,176 de mediana) deja adentro casi todo el top-30, que por construcción son los vecinos más cercanos (0,068 a 0,112).
- **Antes de comparar un BPM estimado con uno de tags, redondear.** El estimador cae en valores finos (125,9) que los tags
  no tienen, y el orden por cercanía prefiere a los temas estimados con el mismo método: la «pureza por BPM» dio 8,0 %
  en vez de 2,0 %.
- **En audio mezclado el BPM identifica el disco.** El DJ iguala tempos: el 72 % de los pares de un disco están a 1 BPM o menos
  (5,8 % entre discos). Cualquier medida de «mismo set» por BPM sobre mezclas gana por construcción.
