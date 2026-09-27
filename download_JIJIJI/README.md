# Descarga automática desde Spotify

Después de configurar las credenciales y dependencias una sola vez, el flujo
completo se ejecuta pasando únicamente la URL de una playlist:

```powershell
.venv\Scripts\python.exe download_JIJIJI\spotify_soulseek_orchestrator.py "https://open.spotify.com/playlist/ID"
```

El orquestador:

1. ejecuta el preflight de Soulseek sobre toda la playlist;
2. pausa la descarga de los temas disponibles en una barrera de sincronización;
3. espera el JSON definitivo y la URL de la playlist privada de faltantes;
4. abre el grabador WASAPI loopback;
5. cuando el grabador confirma que está listo, inicia esa playlist en Spotify;
6. libera la descarga de Soulseek y espera a que terminen ambas ramas.

Los MP3 grabados y los descargados por Soulseek pasan por el mismo finalizador:
usan el nombre `Artista principal - Título.mp3`, las etiquetas oficiales de
Spotify (álbum, artistas, fecha, número de pista/disco, ISRC, sello y campos de
auditoría) y el cover oficial embebido.

## Reanudación después de una interrupción

El grabador codifica y registra cada MP3 apenas termina el tema, sin esperar al
final de toda la playlist. El estado durable queda en
`recorded_missing/recording_progress.json`.

Un watchdog consulta el estado real de Spotify y detecta una pausa sostenida;
como respaldo, el loopback también detecta 10 segundos de silencio digital. En
ese caso el tema actual se considera incompleto y el proceso termina con un
mensaje de sesión reanudable.
Al ejecutar nuevamente exactamente el mismo comando, el orquestador:

1. encuentra la sesión incompleta más reciente para esa playlist fuente;
2. valida los MP3 ya terminados y no los sobrescribe;
3. abre la playlist de faltantes anterior;
4. inicia Spotify desde la primera pista pendiente y desde el segundo cero.

El fragmento del tema interrumpido no se reutiliza: solo se conservan temas
completos. Para ignorar deliberadamente un checkpoint anterior y crear otra
sesión puede usarse `--fresh`.

Cada ejecución queda aislada en `download_JIJIJI/runs/<fecha>_<playlist-id>/`,
con los MP3, JSON de auditoría y reporte final. Puede cambiarse la raíz mediante
`SPOTIFY_DOWNLOAD_ROOT`.

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
