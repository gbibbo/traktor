# Descarga automática desde Spotify

Después de configurar las credenciales y dependencias una sola vez, el flujo
completo se ejecuta pasando únicamente la URL de una playlist:

```powershell
.venv\Scripts\python.exe download_JIJIJI\spotify_soulseek_orchestrator.py "https://open.spotify.com/playlist/ID"
```

El orquestador:

1. ejecuta el preflight de Soulseek sobre toda la playlist;
2. pausa la descarga de los temas disponibles en una barrera de sincronización;
3. divide los faltantes en playlists privadas de Spotify de hasta 10 temas;
4. abre el grabador WASAPI loopback para el primer lote pendiente;
5. cuando el grabador confirma que está listo, reproduce ese lote en Spotify;
6. libera la descarga de Soulseek y procesa secuencialmente los demás lotes,
   creando un proceso nuevo de reproducción y grabación para cada grupo de 10;
7. mueve todos los MP3 a `Playlists_DOWNLOAD/<nombre de playlist>/`, elimina
   duplicados por Spotify Track ID y limpia los intermedios del run.

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
timeout del antivirus bloquean la incorporación a la biblioteca.

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
Mantener actualizados el antivirus, Sockseek, Mutagen y ffmpeg.

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
consolidada mediante `SPOTIFY_PLAYLISTS_ROOT`.

## Configuración inicial

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

La aplicación de Spotify debe tener exactamente el mismo redirect URI. Durante
la grabación conviene desactivar otros sonidos del sistema y el crossfade de
Spotify.
