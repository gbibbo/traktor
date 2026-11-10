<#
PURPOSE
  Copia la seccion de rclone del despachador de gpuq (la Raspberry de Gabriel, contenedor `gpuq`) al
  rclone.conf de esta PC, sin mostrarla en pantalla. Asi esta PC usa el mismo remoto de Google Drive que
  los trabajos de gpuq (alcance drive.file: solo ve lo que creo ese mismo cliente de rclone).
  Lo corre Gabriel: las credenciales no pasan por el agente (AGENTS.md, computo pesado).

USO (PowerShell, desde la raiz del repo)
  powershell -ExecutionPolicy Bypass -File tools\gpuq\rclone_desde_raspberry.ps1 -Pi gbibbo@<ip de la Raspberry> -Key $HOME\.ssh\id_ed25519_rp5
  -Prueba   solo muestra los nombres de las claves de la seccion, sin valores, y no escribe nada.

CHANGELOG
  2026-11-09  Primera version (puesta en marcha de gpuq en este repo).
#>
param(
    [string]$Pi = "asr-rp5",
    [string]$Key = "",
    [string]$Remote = "erclm",
    [switch]$Prueba
)
$ErrorActionPreference = "Stop"

$conf = Join-Path $env:APPDATA "rclone\rclone.conf"
if (-not $Prueba -and (Test-Path $conf) -and (Select-String -Path $conf -Pattern "^\[$Remote\]" -Quiet)) {
    Write-Host "Ya hay una seccion [$Remote] en $conf. No toco nada."
    exit 0
}

$modo = if ($Prueba) { "keys" } else { "full" }
$py = @"
import base64, configparser, io, sys
from gpuq import secrets
c = secrets.load_credentials().get("RCLONE_CONF") or ""
if c and "[" not in c:
    c = base64.b64decode(c).decode("utf-8")
cp = configparser.ConfigParser(interpolation=None)
cp.read_string(c)
if not cp.has_section("$Remote"):
    sys.exit("the dispatcher has no [$Remote] section; it has: " + ", ".join(cp.sections()))
if "$modo" == "keys":
    print("[$Remote] keys: " + ", ".join(sorted(cp["$Remote"])))
    sys.exit(0)
out = configparser.ConfigParser(interpolation=None)
out.add_section("$Remote")
for k, v in cp["$Remote"].items():
    out["$Remote"][k] = v
buf = io.StringIO()
out.write(buf)
sys.stdout.write(buf.getvalue())
"@

$sshArgs = @("-o", "BatchMode=yes", "-o", "ConnectTimeout=15")
if ($Key) { $sshArgs += @("-i", $Key, "-o", "IdentitiesOnly=yes") }
$lineas = $py | ssh @sshArgs $Pi "docker exec -i -e PYTHONPATH=/app gpuq python -"
if ($LASTEXITCODE -ne 0) { throw "ssh o docker exec fallo (codigo $LASTEXITCODE). Revisa -Pi y -Key." }

if ($Prueba) {
    $lineas | ForEach-Object { Write-Host $_ }
    exit 0
}

$texto = ($lineas -join "`r`n") + "`r`n"
if (-not ($texto -match "(?m)^\[$Remote\]" -and $texto -match "(?m)^type = drive")) {
    throw "Lo recibido no parece una seccion de rclone de Google Drive. No escribi nada."
}
New-Item -ItemType Directory -Force (Split-Path $conf) | Out-Null
[IO.File]::AppendAllText($conf, "`r`n" + $texto, [Text.Encoding]::ASCII)
Write-Host "Listo: seccion [$Remote] agregada a $conf (sin mostrarla)."

$rclone = (Get-Command rclone -ErrorAction SilentlyContinue).Source
if (-not $rclone) {
    $rclone = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter rclone.exe -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
}
if ($rclone) {
    Write-Host "Carpetas en ${Remote}: (deberia listar ERC-LM-2027/ y gpuq-test/)"
    & $rclone lsf --max-depth 1 "${Remote}:"
} else {
    Write-Host "No encontre rclone en esta PC; instalalo con: winget install Rclone.Rclone"
}
