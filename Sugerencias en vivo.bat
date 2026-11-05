@echo off
rem Sugerencias en vivo en Traktor con un boton del controlador (tools\traktor_live\sugerencias_vivo.py).
rem Primero encender la placa de sonido y abrir Traktor. Esta ventana tiene que quedar abierta mientras se toca.
rem La primera vez crea el entorno de Python en .venv (sin paquetes extra en Windows).
rem Guia: docs\SUGERENCIAS_EN_VIVO.md. Acepta las mismas opciones, p. ej. --aprender o --midi K2.
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
echo Sugerencias en vivo: Ctrl+C o cerrar esta ventana para salir.
".venv\Scripts\python.exe" "tools\traktor_live\sugerencias_vivo.py" %*
if errorlevel 1 pause
