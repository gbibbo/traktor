# Descarga automática desde Spotify

## La app (sin línea de comandos)

Doble clic en **«Descargar playlists»** del Escritorio. El acceso directo se crea una vez por
ordenador:

```powershell
powershell -ExecutionPolicy Bypass -File download_JIJIJI\app\crear_acceso_directo.ps1
```

Abre una ventana propia (Edge en modo app) servida por `app/descargas_app.py` solo en
`127.0.0.1`. Desde ahí:

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
  «Posponer la grabación» mientras busca o baja (antes de que empiece a grabar), se baja lo de
  Soulseek, va ya a la carpeta de destino y la playlist queda en «Grabación pendiente».
  **«Grabar ahora»** graba solo lo pendiente (sin volver a buscar ni bajar) y lo agrega a la
  misma carpeta. Desde la terminal: la misma orden sin `--defer-recording`;
* seguir cada tema: en Soulseek, bajando, bajado, se grabará (también lo que no se pudo bajar,
  con el motivo), grabando, grabado o «no se pudo». También se siguen los runs lanzados desde la
  terminal. Un run «Incompleta» ya guardó lo que consiguió y dice qué falta;
* si un run terminó con error, **«Guardar los temas listos»** mueve a la carpeta de destino los
  MP3 que sí están, con la misma validación de la consolidación (audio, tamaño, antivirus,
  Spotify Track ID) y sin duplicar.

Cerrar la ventana no corta la descarga: la app sigue y el acceso directo la vuelve a abrir.
Si el código cambió desde que se abrió (un `git pull`, por ejemplo), el acceso directo la
reinicia, salvo que haya una descarga en curso.
Sin ventana ni descarga, se apaga sola. Ajustes, caché de portadas, token de la vista previa y
`app.log` van en `%APPDATA%\traktor_descargas`; con el Python de la Microsoft Store, esa carpeta
está virtualizada en
`%LOCALAPPDATA%\Packages\PythonSoftwareFoundation.Python.3.11_*\LocalCache\Roaming\`.
`app.log` anota la app (aperturas, cierres, errores con su motivo). La salida completa de cada
descarga (orquestador, Soulseek, grabación), con hora, queda en
`<carpeta de runs>\logs\<fecha-hora>_<playlist>.log`; la ventana muestra solo las últimas líneas.
La búsqueda en Soulseek muestra cuántos temas lleva buscados y encontrados, y deja el registro
propio de sockseek en `sockseek_preflight.log`, dentro de la carpeta de esa descarga.

## Línea de comandos

Después de configurar las credenciales y dependencias una sola vez, el flujo
completo se ejecuta pasando únicamente la URL de una playlist:

```powershell
.venv\Scripts\python.exe download_JIJIJI\spotify_soulseek_orchestrator.py "https://open.spotify.com/playlist/ID"
```

El orquestador:

1. ejecuta el preflight de Soulseek sobre toda la playlist. Sockseek permite 34
   búsquedas cada 220 s (más que eso le vale a la cuenta un bloqueo de 30 minutos), así
   que una playlist de 150 temas pasa unos 15 minutos solo esperando. El tope de tiempo se
   calcula según la cantidad de temas y el orquestador lo espera. Si la búsqueda falla
   para todos los temas, el run se corta sin grabar nada: el problema es la búsqueda;
2. pausa la descarga de los temas disponibles en una barrera de sincronización;
3. divide los faltantes en playlists privadas de Spotify de hasta 10 temas;
4. abre el grabador WASAPI loopback para el primer lote pendiente;
5. cuando el grabador confirma que está listo, reproduce ese lote en Spotify;
6. libera la descarga de Soulseek y procesa secuencialmente los demás lotes,
   creando un proceso nuevo de reproducción y grabación para cada grupo de 10;
7. cuando termina la descarga, graba de Spotify los temas que Soulseek tenía pero no
   se pudieron bajar (ningún archivo pasó la verificación) o cuya consulta falló:
   el descargador los deja en `soulseek_failed.json` con sus propias playlists
   privadas y se graban en `recorded_failed/`;
8. mueve todos los MP3 a `Playlists_DOWNLOAD/<nombre de playlist>/`, elimina
   duplicados por Spotify Track ID y limpia los intermedios del run.

Si aun así falta algún tema (por ejemplo, Spotify no pudo crear las playlists para
grabar), igual se guarda en la carpeta todo lo que se consiguió: el run queda como
`partial`, con los Spotify IDs que faltan en `orchestration_report.json`, y el
orquestador termina con código 3. «Volver a intentar» (o el mismo comando) hace una
descarga nueva; lo que ya está en la carpeta no se duplica.

## Elección del archivo en Soulseek

De cada tema se prueban hasta 4 archivos de los que encontró el preflight, en este
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

El grabador codifica y registra cada MP3 apenas termina el tema, sin esperar al
final de toda la playlist. Además, cada proceso de captura termina al completar
su lote de hasta 10 temas; el siguiente lote comienza con procesos nuevos. El
estado durable global queda en
`recorded_missing/recording_progress.json`.

Un watchdog consulta el estado real de Spotify y detecta una pausa sostenida;
como respaldo, el loopback también detecta 10 segundos de silencio digital. En
ese caso el tema actual se considera incompleto y el proceso termina con un
mensaje de sesión reanudable.
Al ejecutar nuevamente exactamente el mismo comando, el orquestador:

1. encuentra la sesión incompleta más reciente para esa playlist fuente;
2. valida los MP3 ya terminados y no los sobrescribe;
3. selecciona la playlist de 10 temas que contiene la primera pista pendiente;
4. inicia Spotify desde esa pista dentro del lote y desde el segundo cero.

Si lo que se cortó fue la grabación de los temas que no se pudieron bajar (la descarga
de Soulseek ya había terminado), no se vuelve a descargar: se sigue grabando desde el
primero pendiente de `recorded_failed/` y después se arma la carpeta.

El fragmento del tema interrumpido no se reutiliza: solo se conservan temas
completos, y por eso la captura WAV parcial se elimina. Para ignorar
deliberadamente un checkpoint anterior y crear otra sesión puede usarse
`--fresh`.

Cada ejecución queda aislada en `download_JIJIJI/runs/<fecha>_<playlist-id>/`,
pero al completarse ya no conserva audio. En la raíz quedan el resumen, el JSON
de faltantes y el reporte de consolidación; los detalles se agrupan bajo
`diagnostics/`. La música terminada queda en
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
