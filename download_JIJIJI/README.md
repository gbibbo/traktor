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

Cada ejecución queda aislada en `download_JIJIJI/runs/<fecha>_<playlist-id>/`,
con los MP3, JSON de auditoría y reporte final. Puede cambiarse la raíz mediante
`SPOTIFY_DOWNLOAD_ROOT`.

## Configuración inicial

Instalar los paquetes Python:

```powershell
.venv\Scripts\python.exe -m pip install -r download_JIJIJI\requirements.txt
```

También deben estar disponibles `ffmpeg` y `sockseek`. El orquestador busca
`sockseek.exe` tanto en `PATH` como en `tools/sockseek/sockseek.exe`.

Definir en el archivo `.env` ignorado por Git:

```dotenv
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
SPOTIFY_REDIRECT_URI=http://127.0.0.1:48721/callback
```

La aplicación de Spotify debe tener exactamente el mismo redirect URI. Durante
la grabación conviene desactivar otros sonidos del sistema y el crossfade de
Spotify.
