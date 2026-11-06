@echo off
rem Deja las sugerencias en vivo en inicio automatico: arrancan solas cada vez que se abre Traktor,
rem sin ventanas (tools\traktor_live\sugerencias_servicio.py). Hace falta una sola vez.
rem Para quitarlo: este mismo archivo con "quitar". Para ver el estado: con "estado".
rem Guia: docs\SUGERENCIAS_EN_VIVO.md.
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Primera vez: creo el entorno de Python en .venv ...
  py -3.11 -m venv .venv 2>nul || python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] No encontre Python 3. Instalar Python 3.11 ^(python.org o Microsoft Store^) y volver a abrir esto.
  pause
  exit /b 1
)
set "ACCION=%~1"
if "%ACCION%"=="" set "ACCION=instalar"
".venv\Scripts\python.exe" "tools\traktor_live\sugerencias_servicio.py" %ACCION%
pause
