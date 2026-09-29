@echo off
rem Abre la app local de TRAKTOR ML en el navegador (src\v4\ui\review_app.py).
rem Esta ventana tiene que quedar abierta mientras se usa la app; cerrarla la apaga.
chcp 65001 >nul
cd /d "%~dp0"
echo Abriendo TRAKTOR ML en el navegador... (cerra esta ventana para salir)
".venv\Scripts\python.exe" "src\v4\ui\review_app.py" %*
if errorlevel 1 pause
