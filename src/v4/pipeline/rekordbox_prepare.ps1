# PURPOSE: "Preparar Rekordbox" del paquete portatil (lo copia usb_package.py como
#          _preparar_rekordbox.ps1). Escribe rekordbox.xml con la unidad y la carpeta donde esta el
#          paquete en esta computadora (la letra del pendrive cambia de una PC a otra), a partir de
#          _plantilla_rekordbox.xml, y cuenta cuantos temas del XML existen de verdad.
# CHANGELOG:
#   - 2026-10-30: Creacion inicial (Rekordbox con controladora desde el pendrive).
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$vol = Split-Path -Qualifier $here
$rest = $here.Substring($vol.Length).Trim('\')
$root = $vol
if ($rest) { foreach ($p in $rest.Split('\')) { $root += '/' + [Uri]::EscapeDataString($p) } }
$enc = New-Object System.Text.UTF8Encoding $false

$tpl = [System.IO.File]::ReadAllText((Join-Path $here '_plantilla_rekordbox.xml'), $enc)
$text = $tpl.Replace('@@ROOT@@', [System.Security.SecurityElement]::Escape($root))
$doc = New-Object System.Xml.XmlDocument
$doc.LoadXml($text.Substring($text.IndexOf('<DJ_PLAYLISTS')))
$prefix = 'file://localhost/'
$total = 0; $missing = New-Object System.Collections.ArrayList
foreach ($t in $doc.SelectNodes('/DJ_PLAYLISTS/COLLECTION/TRACK')) {
  $total++
  $loc = $t.GetAttribute('Location')
  $path = [Uri]::UnescapeDataString($loc.Substring($prefix.Length)).Replace('/', '\')
  if (-not (Test-Path -LiteralPath $path)) { [void]$missing.Add($path) }
}
[System.IO.File]::WriteAllText((Join-Path $here 'rekordbox.xml'), $text, $enc)

Write-Host ''
Write-Host "Paquete en: $here"
Write-Host "Temas en rekordbox.xml: $total"
Write-Host "  Archivos encontrados: $($total - $missing.Count)"
Write-Host "  Archivos que faltan: $($missing.Count)"
if ($missing.Count -gt 0) {
  Write-Host 'Faltan estos archivos (copia la carpeta Musica completa):' -ForegroundColor Yellow
  $missing | Select-Object -First 20 | ForEach-Object { Write-Host "  $_" }
} else {
  Write-Host 'Todo bien.' -ForegroundColor Green
}
Write-Host ''
Write-Host 'Listo. En Rekordbox: Preferencias > Avanzado > Base de datos > rekordbox xml: elegir'
Write-Host "  $(Join-Path $here 'rekordbox.xml')"
Write-Host 'Despues: arbol de la izquierda > rekordbox xml > Playlists > clic derecho en la carpeta > Importar playlist.'
