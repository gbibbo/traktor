# PURPOSE: "Preparar Traktor" del paquete portatil (lo copia usb_package.py como
#          _preparar_traktor.ps1). Arma traktor.nml para la computadora donde se ejecuta: cada tema
#          de la organizacion se busca en la coleccion de Traktor de esa computadora
#          (collection.nml), primero por nombre de archivo (con duracion parecida) y despues por
#          artista y titulo. Si esta, la playlist usa ese archivo y su ficha tal cual (analisis,
#          grilla, cues); si no, usa la copia del paquete. No modifica collection.nml.
#          Deja el detalle tema por tema en _informe.txt.
# CHANGELOG:
#   - 2026-10-10: Creacion inicial (temas locales ya analizados en la otra computadora).
param([string]$Collection = '')
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$vol = Split-Path -Qualifier $here
$rest = $here.Substring($vol.Length).Trim('\')
$dir = '/:'
if ($rest) { foreach ($p in $rest.Split('\')) { $dir += $p + '/:' } }
$enc = New-Object System.Text.UTF8Encoding $false

function Norm([string]$s) {
  if (-not $s) { return '' }
  $d = $s.ToLowerInvariant().Normalize([Text.NormalizationForm]::FormD)
  return (($d -replace '\p{Mn}', '') -replace '[^\p{L}\p{N}]', '')
}
function DriveOf([string]$v) {
  if ($v -match '^[A-Za-z]:$') { return $v }
  foreach ($d in [System.IO.DriveInfo]::GetDrives()) {
    try { if ($d.IsReady -and $d.VolumeLabel -eq $v) { return $d.Name.TrimEnd('\') } } catch {}
  }
  return $null
}
function Num([string]$s) {
  $x = 0.0
  if ($s -and [double]::TryParse($s, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$x)) { return $x }
  return $null
}
function DurOk($a, $b, [double]$tol) {
  if ($a -eq $null -or $b -eq $null) { return $true }
  return ([math]::Abs($a - $b) -le $tol)
}

# 1. Coleccion de Traktor de esta computadora (la mas reciente si hay varias versiones)
if (-not $Collection) {
  $base = Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'Native Instruments'
  $found = @(Get-ChildItem -LiteralPath $base -Directory -Filter 'Traktor*' -ErrorAction SilentlyContinue |
    ForEach-Object { Get-Item -LiteralPath (Join-Path $_.FullName 'collection.nml') -ErrorAction SilentlyContinue } |
    Sort-Object LastWriteTime -Descending)
  if ($found.Count -gt 0) { $Collection = $found[0].FullName }
}
$byFile = @{}; $byName = @{}; $nCol = 0
if ($Collection) {
  $col = New-Object System.Xml.XmlDocument
  $col.Load($Collection)
  foreach ($e in $col.SelectNodes('/NML/COLLECTION/ENTRY')) {
    $loc = $e.SelectSingleNode('LOCATION')
    if ($loc -eq $null) { continue }
    $file = $loc.GetAttribute('FILE')
    if (-not $file) { continue }
    $v = $loc.GetAttribute('VOLUME'); $d = $loc.GetAttribute('DIR')
    $drive = DriveOf $v
    if ($drive -and -not (Test-Path -LiteralPath ($drive + $d.Replace('/:', '\') + $file))) { continue }
    $info = $e.SelectSingleNode('INFO')
    $dur = $null
    if ($info -ne $null) {
      $dur = Num $info.GetAttribute('PLAYTIME_FLOAT')
      if ($dur -eq $null) { $dur = Num $info.GetAttribute('PLAYTIME') }
    }
    $rec = [pscustomobject]@{ E = $e; Key = $v + $d + $file; Dir = $d.Replace('/:', '\').ToLowerInvariant(); Dur = $dur }
    $nCol++
    $k = $file.ToLowerInvariant()
    if (-not $byFile.ContainsKey($k)) { $byFile[$k] = New-Object System.Collections.ArrayList }
    [void]$byFile[$k].Add($rec)
    $nk = (Norm $e.GetAttribute('ARTIST')) + '|' + (Norm $e.GetAttribute('TITLE'))
    if ($nk -ne '|') {
      if (-not $byName.ContainsKey($nk)) { $byName[$nk] = New-Object System.Collections.ArrayList }
      [void]$byName[$nk].Add($rec)
    }
  }
}

# 2. Buscar cada tema de la organizacion
$rows = Import-Csv -LiteralPath (Join-Path $here '_organizacion.csv') -Encoding UTF8
$match = @{}; $report = New-Object System.Collections.ArrayList
$missing = New-Object System.Collections.ArrayList; $seen = @{}
$nFile = 0; $nName = 0; $nPkg = 0
foreach ($r in $rows) {
  if ($seen.ContainsKey($r.track_uid)) { continue }
  $seen[$r.track_uid] = $true
  $pkgKey = $vol + $dir + $r.archivo.Replace('\', '/:')
  $dur = Num $r.duracion_s
  $pick = $null; $how = ''
  $c = @()
  $fk = ([string]$r.nombre_original).ToLowerInvariant()
  if ($fk -and $byFile.ContainsKey($fk)) { $c = @($byFile[$fk] | Where-Object { DurOk $_.Dur $dur 5 }) }
  if ($c.Count -gt 0) {
    if ($r.carpeta_original) {
      $want = ('\' + $r.carpeta_original + '\').ToLowerInvariant()
      $same = @($c | Where-Object { $_.Dir.EndsWith($want) })
      if ($same.Count -gt 0) { $c = $same }
    }
    $pick = $c | Sort-Object { if ($_.Dur -eq $null -or $dur -eq $null) { 0 } else { [math]::Abs($_.Dur - $dur) } } | Select-Object -First 1
    $how = 'nombre de archivo'; $nFile++
  } else {
    $na = Norm $r.artista; $nt = Norm $r.titulo
    $nk = $na + '|' + $nt
    if ($na -and $nt -and $dur -ne $null -and $byName.ContainsKey($nk)) {
      $c = @($byName[$nk] | Where-Object { $_.Dur -ne $null -and [math]::Abs($_.Dur - $dur) -le 3 })
      if ($c.Count -gt 0) {
        $pick = $c | Sort-Object { [math]::Abs($_.Dur - $dur) } | Select-Object -First 1
        $how = 'artista y titulo'; $nName++
      }
    }
  }
  if ($pick) {
    $match[$pkgKey] = $pick
    [void]$report.Add("$how`t$($r.archivo)`t$($pick.Key)")
  } elseif (Test-Path -LiteralPath (Join-Path $here $r.archivo)) {
    $nPkg++
    [void]$report.Add("copia del paquete`t$($r.archivo)`t")
  } else {
    [void]$missing.Add($r.archivo)
    [void]$report.Add("NO ENCONTRADO`t$($r.archivo)`t")
  }
}

# 3. traktor.nml: plantilla con la ubicacion de este paquete; los temas encontrados usan su ficha local
$tpl = [System.IO.File]::ReadAllText((Join-Path $here '_plantilla.nml'), $enc)
$text = $tpl.Replace('@@VOL@@', [System.Security.SecurityElement]::Escape($vol)).Replace('@@DIR@@', [System.Security.SecurityElement]::Escape($dir))
$doc = New-Object System.Xml.XmlDocument
$doc.LoadXml($text.Substring($text.IndexOf('<NML')))
$colNode = $doc.SelectSingleNode('/NML/COLLECTION')
$used = @{}
foreach ($e in @($colNode.SelectNodes('ENTRY'))) {
  $loc = $e.SelectSingleNode('LOCATION')
  $k = $loc.GetAttribute('VOLUME') + $loc.GetAttribute('DIR') + $loc.GetAttribute('FILE')
  if ($match.ContainsKey($k)) {
    $lk = $match[$k].Key
    if ($used.ContainsKey($lk)) { [void]$colNode.RemoveChild($e) }
    else { [void]$colNode.ReplaceChild($doc.ImportNode($match[$k].E, $true), $e); $used[$lk] = $true }
  }
}
$colNode.SetAttribute('ENTRIES', [string]$colNode.SelectNodes('ENTRY').Count)
foreach ($pk in $doc.SelectNodes('//PRIMARYKEY')) {
  $k = $pk.GetAttribute('KEY')
  if ($match.ContainsKey($k)) { $pk.SetAttribute('KEY', $match[$k].Key) }
}
$sb = New-Object System.Text.StringBuilder
$ws = New-Object System.Xml.XmlWriterSettings
$ws.OmitXmlDeclaration = $true; $ws.Indent = $true; $ws.IndentChars = '  '
$w = [System.Xml.XmlWriter]::Create($sb, $ws)
$doc.Save($w); $w.Close()
$out = '<?xml version="1.0" encoding="UTF-8" standalone="no" ?>' + "`n" + $sb.ToString() + "`n"
[System.IO.File]::WriteAllText((Join-Path $here 'traktor.nml'), $out, $enc)
[System.IO.File]::WriteAllLines((Join-Path $here '_informe.txt'), [string[]]$report, $enc)

$total = $seen.Count
Write-Host ''
Write-Host "Paquete en: $here"
if ($Collection) { Write-Host "Coleccion de Traktor: $Collection ($nCol temas)" }
else { Write-Host 'No encontre la coleccion de Traktor: se usan las copias del paquete.' -ForegroundColor Yellow }
Write-Host "Temas de la organizacion: $total"
Write-Host "  Archivo de esta computadora (ya analizado): $($nFile + $nName)   (por nombre: $nFile, por artista y titulo: $nName)"
Write-Host "  Copia del paquete (Traktor la analiza al cargarla): $nPkg"
Write-Host "  No encontrados: $($missing.Count)"
if ($missing.Count -gt 0) {
  Write-Host 'No estan ni en esta computadora ni en el paquete (copia la carpeta Musica completa):' -ForegroundColor Yellow
  $missing | Select-Object -First 20 | ForEach-Object { Write-Host "  $_" }
} else {
  Write-Host 'Todo bien.' -ForegroundColor Green
}
Write-Host 'Detalle tema por tema: _informe.txt'
Write-Host ''
Write-Host 'Listo. En Traktor: clic derecho en Playlists > Import Playlist > traktor.nml de esta carpeta'
