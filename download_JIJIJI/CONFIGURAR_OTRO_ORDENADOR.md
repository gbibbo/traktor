# Configurar la descarga Spotify + Soulseek en otro ordenador

Guía paso a paso para dejar funcionando
`spotify_soulseek_orchestrator.py` en una PC nueva con Windows 10 u 11.
El flujo (grabación por loopback y control de Spotify de escritorio) solo
funciona en Windows. Lleva unos 30 minutos.

Los comandos son de PowerShell y suponen que el repo está en `C:\VS\traktor`,
igual que en la PC actual. Si lo clonas en otra carpeta, cambia esa ruta.

## Qué viene con el repo y qué no

El repo es **público**. Por eso las claves no están en Git y hay que llevarlas a mano.

| Qué | Dónde va en la PC nueva | Cómo se consigue |
|---|---|---|
| Código y esta guía | `C:\VS\traktor\` | `git clone` (paso 2) |
| `.env`: Client ID, Client secret y redirect URI de Spotify | `C:\VS\traktor\.env` | se copia por pendrive desde la PC actual (paso 0) |
| `sockseek.conf`: usuario y contraseña de Soulseek | `%APPDATA%\sockseek\sockseek.conf` | se copia por pendrive desde la PC actual (paso 0) |
| `sockseek.exe` 3.0.5 | `C:\VS\traktor\tools\sockseek\sockseek.exe` | se descarga de GitHub (paso 4) |
| ffmpeg, Python 3.12, Spotify de escritorio | instalados en Windows | winget (paso 1) |
| Autorizaciones de Spotify (tokens) | `C:\VS\traktor\.spotify_token_cache` y `%APPDATA%\traktor_spotify_token.json` | se crean solas al autorizar en el navegador (paso 8); no se copian |

## Paso 0. En la PC actual: copiar las dos claves a un pendrive

```powershell
$usb = "E:\traktor_claves"          # cambia E: por la letra del pendrive
New-Item -ItemType Directory -Force $usb | Out-Null
Copy-Item C:\VS\traktor\.env $usb
Copy-Item "$env:APPDATA\sockseek\sockseek.conf" $usb
```

Esos dos archivos son contraseñas. No los mandes por mail ni por chat, y no los
subas a Git. Cuando termines el paso 6, bórralos del pendrive.

## Paso 1. Programas base

En una terminal de PowerShell normal (sin administrador):

```powershell
winget install -e --id Git.Git
winget install -e --id Python.Python.3.12
winget install -e --id Gyan.FFmpeg
winget install -e --id Spotify.Spotify
```

Spotify también puede instalarse desde spotify.com o la Microsoft Store.
Al terminar, **cierra la terminal y abre una nueva** para que tome el PATH nuevo.

## Paso 2. Clonar el repo

```powershell
git clone https://github.com/gbibbo/traktor.git C:\VS\traktor
cd C:\VS\traktor
git config core.hooksPath tools/git_hooks
```

La última línea activa el chequeo que impide subir por error claves, audio o
archivos grandes. Si el repo ya estaba clonado, basta con `git pull`.

## Paso 3. Entorno de Python

Para descargar solo hacen falta los paquetes de `download_JIJIJI\requirements.txt`;
no hace falta `requirements_v4.txt` (eso es para el pipeline de clustering).

```powershell
cd C:\VS\traktor
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r download_JIJIJI\requirements.txt
.venv\Scripts\python.exe -c "import requests, mutagen, PIL, spotipy, dotenv, numpy, pyaudiowpatch; print('paquetes OK')"
```

Tiene que imprimir `paquetes OK`. Si `.venv` ya existía, sáltate la línea `py -3.12 -m venv .venv`.

## Paso 4. sockseek (cliente de Soulseek)

Usa la versión **3.0.5**, que es la que está probada con este flujo. Ya existe
la 3.0.6, pero no se probó.

```powershell
cd C:\VS\traktor
$ProgressPreference = "SilentlyContinue"   # sin esto la descarga es muy lenta
New-Item -ItemType Directory -Force tools\sockseek | Out-Null
Invoke-WebRequest https://github.com/fiso64/sockseek/releases/download/v3.0.5/sockseek_3.0.5_win-x64.zip -OutFile "$env:TEMP\sockseek_3.0.5.zip"
Expand-Archive "$env:TEMP\sockseek_3.0.5.zip" -DestinationPath tools\sockseek -Force
Get-FileHash tools\sockseek\sockseek.exe
tools\sockseek\sockseek.exe --version
```

* `Get-FileHash` tiene que dar
  `47B1D9ABB78DF23B66DA807ABA7610A2F3FBCC30D4D5FADFD550BAEE2F41A498`, el mismo
  hash que el ejecutable de la PC actual.
* `--version` tiene que decir `3.0.5`.

El orquestador busca `sockseek.exe` en esa carpeta (o en el PATH). La carpeta
`tools/sockseek/` está en `.gitignore`: el ejecutable pesa 114 MB y no se sube.

## Paso 5. Cuenta de Soulseek (`sockseek.conf`)

Copia el archivo que trajiste en el pendrive:

```powershell
New-Item -ItemType Directory -Force "$env:APPDATA\sockseek" | Out-Null
Copy-Item E:\traktor_claves\sockseek.conf "$env:APPDATA\sockseek\sockseek.conf"
```

Si prefieres escribirlo a mano (`notepad "$env:APPDATA\sockseek\sockseek.conf"`),
el contenido es:

```ini
username = <tu usuario de Soulseek>
password = <tu contraseña de Soulseek>
```

* Esas dos líneas tienen que ir al principio, antes de cualquier sección `[perfil]`.
  El script copia solo el usuario y la contraseña a una configuración temporal y
  descarta todo lo demás.
* **La misma cuenta en las dos PCs sirve, pero no al mismo tiempo**: Soulseek
  acepta una sola sesión por usuario y la segunda desconecta a la primera, con
  lo que la descarga en curso falla.
* Para tener una cuenta aparte en esta PC, escribe un usuario y una contraseña
  nuevos: si el nombre está libre, Soulseek crea la cuenta al primer inicio de sesión.

## Paso 6. Credenciales de Spotify (`.env` y la app de desarrollador)

### 6a. El archivo `.env`

```powershell
Copy-Item E:\traktor_claves\.env C:\VS\traktor\.env
```

Tiene que quedar en la raíz del repo, con este contenido (los valores son los
de la PC actual):

```dotenv
SPOTIFY_CLIENT_ID=<Client ID de la app de Spotify>
SPOTIFY_CLIENT_SECRET=<Client secret de la app de Spotify>
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8888/callback

# Opcional: guardar las descargas fuera del repo (por defecto quedan en
# download_JIJIJI\Playlists_DOWNLOAD y download_JIJIJI\runs)
# SPOTIFY_PLAYLISTS_ROOT=D:/Musica/Descargas
# SPOTIFY_DOWNLOAD_ROOT=D:/Musica/Descargas/_runs
```

Ya puedes borrar `E:\traktor_claves` del pendrive.

### 6b. La app en el dashboard de Spotify

Se usa **la misma app** de desarrollador que en la PC actual: el Client ID del
`.env` la identifica. Entra en https://developer.spotify.com/dashboard, abre
esa app, ve a **Settings** y comprueba que en **Redirect URIs** estén **los dos**:

| Redirect URI | Quién lo usa |
|---|---|
| `http://127.0.0.1:48721/callback` | `spotify_soulseek_download.py`: lee la playlist y crea las playlists privadas de faltantes. Este valor está fijo en el código: ignora el `.env`. |
| `http://127.0.0.1:8888/callback` | `spotify_playlist.py`: controla la reproducción en Spotify de escritorio. Usa `SPOTIFY_REDIRECT_URI` del `.env`. |

Tienen que coincidir exactamente: `127.0.0.1` (Spotify no acepta `localhost`),
`http` y sin barra final. Si falta alguno, agrégalo y pulsa **Save**.

Además:

* **La cuenta de Spotify tiene que ser Premium**: Spotify no permite controlar
  la reproducción por API con una cuenta gratuita.
* Si en la PC nueva usas **otra cuenta** de Spotify, distinta de la dueña de la
  app, agrégala en **User Management** de la app. En modo desarrollo solo pueden
  autorizarla los usuarios que figuran ahí.
* Solo si se perdió la app: **Create app**, cualquier nombre, los dos Redirect URIs
  de arriba y la API **Web API**. Después, en **Settings**, copia el Client ID y el
  Client secret (**View client secret**) al `.env`.

## Paso 7. Spotify de escritorio y audio de Windows

1. Abre Spotify de escritorio e inicia sesión con la cuenta Premium.
2. En la configuración de Spotify, **desactiva el crossfade** (fundido) y pon la
   calidad de transmisión en **Muy alta**. Lo que no se encuentra en Soulseek se
   graba de lo que suena, así que esa calidad es la que queda en el MP3.
3. El grabador captura la **salida de audio predeterminada de Windows**, que tiene
   que ser estéreo, y Spotify tiene que sonar por ella. Todo lo que suene durante
   la grabación queda grabado: silencia las notificaciones (No molestar) y no
   reproduzcas otra cosa.

## Paso 8. Autorizar Spotify (una sola vez por PC)

Hay dos autorizaciones independientes. Cada una abre el navegador una vez y
guarda su propio archivo:

| Autorización | Archivo que crea |
|---|---|
| Reproducción (`spotify_playlist.py`) | `C:\VS\traktor\.spotify_token_cache` |
| Lectura de playlists (`spotify_soulseek_download.py`) | `%APPDATA%\traktor_spotify_token.json` |

**8a. Reproducción. Hazla antes de la primera descarga.** Dentro del orquestador,
el reproductor tiene solo 30 segundos para arrancar; no alcanzan para iniciar
sesión en el navegador. Con Spotify de escritorio abierto:

```powershell
cd C:\VS\traktor
.venv\Scripts\python.exe download_JIJIJI\spotify_playlist.py "https://open.spotify.com/playlist/<ID de cualquier playlist>" --start-only
```

Se abre el navegador. Acepta los permisos y Spotify empieza a reproducir esa
playlist. El script termina con `Playback started at playlist position 0.` y
después puedes pausar Spotify. Esto también comprueba que Spotify de escritorio
aparece como dispositivo y que la cuenta puede controlarse por API.

**8b. Lectura de playlists.** Se pide sola en la primera descarga (paso 10): se
abre el navegador y hay 3 minutos para aceptar. Si el navegador no se abre, la
consola muestra la URL para pegarla a mano.

No copies estos archivos de la otra PC: cada PC tiene que tener su propia
autorización. Un token copiado puede dejar de valer cuando la otra PC lo renueva.

## Paso 9. Antivirus

El flujo analiza cada archivo con Microsoft Defender (`MpCmdRun.exe`). Si no
puede hacerlo, no descarga nada. Comprueba que Defender esté activo:

```powershell
Get-MpComputerStatus | Select-Object AntivirusEnabled, RealTimeProtectionEnabled
```

`AntivirusEnabled` tiene que ser `True`. Si hay otro antivirus que desactivó
Defender, el flujo se detiene con un error de antivirus.

## Paso 10. Primera prueba con una playlist corta

Arma en Spotify una playlist de 2 o 3 temas y lanza el flujo **desde la raíz del repo**:

```powershell
cd C:\VS\traktor
.venv\Scripts\python.exe download_JIJIJI\spotify_soulseek_orchestrator.py "https://open.spotify.com/playlist/<ID>"
```

Tiene que terminar con `=== FLUJO COMPLETADO ===` y dejar los MP3 en
`download_JIJIJI\Playlists_DOWNLOAD\<nombre de la playlist>\` (o en
`SPOTIFY_PLAYLISTS_ROOT`, si lo definiste). Abre un MP3: tiene que sonar entero y
tener carátula. Recién entonces pasa a playlists largas.

## Uso diario

Es el mismo comando del paso 10, con la URL de la playlist. Si se corta
(`PAUSADO: ...`), relanza **exactamente el mismo comando** y sigue desde el primer
tema pendiente; `--fresh` empieza de cero. El detalle del flujo y de los controles
de seguridad está en `download_JIJIJI/README.md`.

## Problemas frecuentes

| Mensaje o síntoma | Causa | Solución |
|---|---|---|
| `Faltan variables de entorno: SPOTIFY_...` | No hay `.env` en la raíz del repo o tiene valores vacíos | Paso 6a |
| `Faltan paquetes Python: ...` | No se instalaron los requisitos en `.venv` | Paso 3 |
| `No se encontró 'sockseek' en PATH ni en tools/sockseek/.` | Falta `sockseek.exe` | Paso 4 |
| `No se encontró 'ffmpeg' en PATH ni en tools/ffmpeg/.` | Falta ffmpeg o la terminal es anterior a la instalación | Paso 1 y abrir una terminal nueva |
| `falta sockseek.conf con username/password globales` | No existe `%APPDATA%\sockseek\sockseek.conf` | Paso 5 |
| `sockseek.conf debe tener username/password en la sección global` | Faltan las líneas o están debajo de una sección `[...]` | Paso 5 |
| La descarga de Soulseek se corta a mitad | La misma cuenta de Soulseek se abrió en la otra PC | Usar una sola PC a la vez o una cuenta por PC (paso 5) |
| El navegador dice `INVALID_CLIENT: Invalid redirect URI` | Falta uno de los dos Redirect URIs en la app | Paso 6b |
| Error `403` al reproducir, o la autorización se rechaza | Cuenta no Premium, o cuenta que no está en User Management | Paso 6b |
| `Spotify Desktop did not appear as an available device.` | Spotify de escritorio cerrado o con otra cuenta | Abrir Spotify con la cuenta autorizada. Si cambiaste de cuenta, borrar `.spotify_token_cache` y repetir 8a |
| `El reproductor de Spotify no quedó listo dentro del tiempo esperado.` | Falta la autorización de reproducción o Spotify estaba cerrado | Paso 8a, con Spotify abierto |
| `WinError 10048` al autorizar | Otro programa usa el puerto 48721 u 8888 (Jupyter usa el 8888) | Cerrar ese programa y repetir |
| `antivirus requerido ...` o `no se pudo completar el análisis antivirus` | Defender desactivado o reemplazado | Paso 9 |
| `Could not find the WASAPI loopback for the Windows default output.` | No hay una salida de audio predeterminada activa | Elegir altavoces o auriculares como salida predeterminada en Configuración de sonido |
| `DLL load failed` al importar un paquete | Smart App Control bloquea DLL nuevas (pasó en la laptop) | Probar una versión anterior del paquete que falla |

## Lo que nunca hay que hacer

* Subir a Git `.env`, `sockseek.conf`, `.spotify_token_cache` o
  `traktor_spotify_token.json`. El repo es público. El chequeo de `tools/git_hooks/`
  bloquea `.env` y las cachés de tokens, pero no hay que depender de eso.
* Si una clave se filtró: genera un Client secret nuevo en el dashboard de Spotify,
  cambia la contraseña de Soulseek y actualiza `.env` y `sockseek.conf` en las dos PCs.
