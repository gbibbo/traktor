@echo off
rem Sugerencias en vivo en Traktor con un boton del controlador (tools\traktor_live\sugerencias_vivo.py).
rem Primero encender la DJM y abrir Traktor. Esta ventana tiene que quedar abierta mientras se toca.
rem Guia: docs\SUGERENCIAS_EN_VIVO.md. Acepta las mismas opciones, p. ej. --aprender o --midi K2.
chcp 65001 >nul
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=py -3.11"
echo Sugerencias en vivo: Ctrl+C o cerrar esta ventana para salir.
%PY% "tools\traktor_live\sugerencias_vivo.py" %*
if errorlevel 1 pause
