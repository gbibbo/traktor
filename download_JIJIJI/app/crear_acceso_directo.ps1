# PURPOSE: Crea en el Escritorio el acceso directo "Descargar playlists" que abre la app de descargas
#          (download_JIJIJI/app/descargas_app.py) con el pythonw del .venv del repo, sin ventana de consola.
#          Uso (una vez por ordenador, desde la raíz del repo):
#            powershell -ExecutionPolicy Bypass -File download_JIJIJI\app\crear_acceso_directo.ps1
# CHANGELOG:
#   - 2026-10-30: Creación inicial.
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$pythonw = Join-Path $repo '.venv\Scripts\pythonw.exe'
if (-not (Test-Path $pythonw)) {
    throw "No existe $pythonw. Crea primero el .venv (CONFIGURAR_OTRO_ORDENADOR.md, paso 3)."
}
$desktop = [Environment]::GetFolderPath('Desktop')
$link = Join-Path $desktop 'Descargar playlists.lnk'
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($link)
$shortcut.TargetPath = $pythonw
$shortcut.Arguments = '"' + (Join-Path $repo 'download_JIJIJI\app\descargas_app.py') + '"'
$shortcut.WorkingDirectory = $repo
$shortcut.IconLocation = (Join-Path $repo 'download_JIJIJI\app\icon.ico') + ',0'
$shortcut.Description = 'Bajar playlists de Spotify (Soulseek + grabación)'
$shortcut.Save()
Write-Output "Acceso directo creado: $link"
